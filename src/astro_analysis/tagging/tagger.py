"""Stage 2: Gemini-powered tagging.

Tags one whole session per API call (all its turns), which naturally gives the
model full within-session context (well beyond the 2-3 prior-turn minimum) and
keeps call volume to ~1 per session rather than ~1 per message.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import pandas as pd
from tqdm import tqdm

from astro_analysis import config
from astro_analysis.ingestion.parser import parse_user_profile
from astro_analysis.tagging.gemini_client import GeminiCachedClient
from astro_analysis.tagging.prompt_loader import load_system_instruction

ASTRO_FIELDS = [
    "concrete_dated_prediction", "specific_open_thread_named",
    "callback_prior_session", "callback_within_session",
    "claim_certainty", "specificity",
    "validation", "self_correction", "reframe", "repair_after_pushback",
]
USER_FIELDS = [
    "emotional_disclosure", "pushback_disagreement", "enthusiastic_engagement",
    "disengagement_signal", "neutral_factual",
]
ALL_MESSAGE_FIELDS = ASTRO_FIELDS + USER_FIELDS

_MESSAGE_TAG_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "message_order": {"type": "integer"},
        "role": {"type": "string", "enum": ["astrologer", "user"]},
        "concrete_dated_prediction": {"type": "boolean", "nullable": True},
        "specific_open_thread_named": {"type": "boolean", "nullable": True},
        "callback_prior_session": {"type": "boolean", "nullable": True},
        "callback_within_session": {"type": "boolean", "nullable": True},
        "claim_certainty": {"type": "string", "nullable": True,
                            "enum": ["hedged", "absolute", "not_applicable"]},
        "specificity": {"type": "string", "nullable": True,
                        "enum": ["generic_broad", "personalized_specific", "not_applicable"]},
        "validation": {"type": "boolean", "nullable": True},
        "self_correction": {"type": "boolean", "nullable": True},
        "reframe": {"type": "boolean", "nullable": True},
        "repair_after_pushback": {"type": "string", "nullable": True,
                                  "enum": ["defer", "offer_options", "repeat",
                                           "none_of_above", "not_applicable"]},
        "emotional_disclosure": {"type": "boolean", "nullable": True},
        "pushback_disagreement": {"type": "boolean", "nullable": True},
        "enthusiastic_engagement": {"type": "boolean", "nullable": True},
        "disengagement_signal": {"type": "boolean", "nullable": True},
        "neutral_factual": {"type": "boolean", "nullable": True},
    },
    "required": ["message_order", "role"],
}

# A single API call tags a BATCH of sessions at once (grouped to hit
# config.TAGGING_BATCH_SIZE_MESSAGES total messages, sessions never split
# across calls) to cut down on request count while each session still gets
# its own full transcript as context.
RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "sessions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "session_id": {"type": "string"},
                    "messages": {"type": "array", "items": _MESSAGE_TAG_ITEM_SCHEMA},
                    "session": {
                        "type": "object",
                        "properties": {
                            "ending_type": {"type": "string",
                                            "enum": ["resolved_closed", "open_thread", "abrupt_trail_off"]},
                            "wall_hit": {"type": "boolean"},
                        },
                        "required": ["ending_type", "wall_hit"],
                    },
                },
                "required": ["session_id", "messages", "session"],
            },
        },
    },
    "required": ["sessions"],
}


def _is_profile_message(row) -> bool:
    return row["role"] == "user" and bool(parse_user_profile(row["message"]))


def _build_turns_payload(session_msgs: pd.DataFrame, target_orders: Optional[set] = None) -> list[dict]:
    """target_orders, if given, restricts which message_orders actually need
    tags back (used to chunk oversized sessions: earlier/later turns are
    still sent as context but marked skip_tagging so the model doesn't have
    to re-emit tags for them, bounding the response size)."""
    turns = []
    for _, row in session_msgs.iterrows():
        order = int(row["message_order"])
        needs_tag = target_orders is None or order in target_orders
        turns.append({
            "message_order": order,
            "role": row["role"],
            "text": row["message"],
            "skip_tagging": _is_profile_message(row) or not needs_tag,
        })
    return turns


def _build_batch_prompt(sessions_payload: list[dict]) -> str:
    """sessions_payload: list of {session_id, source_folder, session_number,
    astrologer_id, turns}."""
    return (
        f"You will be given {len(sessions_payload)} session(s) to tag in this single "
        "request. Tag each session independently using its own turns as context "
        "(do not let one session's content bleed into another's tags).\n\n"
        f"Sessions:\n{json.dumps(sessions_payload, ensure_ascii=False, indent=2)}\n\n"
        "Some turns are marked \"skip_tagging\": true — these exist purely to give you "
        "context (either the opening profile message, or, when a session has been split "
        "into chunks for length, turns outside the current chunk) and must be READ for "
        "context but NOT included in your output at all.\n\n"
        "Return JSON with a top-level \"sessions\" array, one entry per session_id given "
        "above (same session_id string, in any order), each with a \"messages\" array "
        "containing ONLY the turns where skip_tagging is false or absent (one object per "
        "such turn, in the same order, carrying its message_order/role plus every tag "
        "field listed in the system instruction's schema, set to null where not "
        "applicable to that role) and a \"session\" object with ending_type and wall_hit "
        "for that session as a whole (judged from the full context, even if the tagged "
        "chunk doesn't cover the whole session)."
    )


def tag_session_batch(client: GeminiCachedClient, system_instruction: str,
                       batch: list[tuple]) -> dict:
    """batch: list of (session_id, source_folder, session_number, astrologer_id, session_msgs)."""
    sessions_payload = [
        {
            "session_id": session_id,
            "source_folder": source_folder,
            "session_number": session_number,
            "astrologer_id": astrologer_id,
            "turns": _build_turns_payload(session_msgs),
        }
        for session_id, source_folder, session_number, astrologer_id, session_msgs in batch
    ]
    prompt = _build_batch_prompt(sessions_payload)
    return client.generate_json(prompt, system_instruction=system_instruction,
                                 response_schema=RESPONSE_SCHEMA)


def tag_large_session(client: GeminiCachedClient, system_instruction: str,
                       session_id: str, source_folder: str, session_number: int,
                       astrologer_id: str, session_msgs: pd.DataFrame,
                       chunk_size: int = config.LARGE_SESSION_CHUNK_SIZE) -> dict:
    """For a session too large to tag (and get tagged back) in one response:
    call once per chunk of message_orders, each time sending the FULL session
    transcript as context but asking for tags on only that chunk's orders
    (other turns marked skip_tagging=True). Keeps every call's input context
    complete while bounding each response to `chunk_size` tagged messages."""
    orders = sorted(int(o) for o in session_msgs["message_order"])
    chunks = [orders[i:i + chunk_size] for i in range(0, len(orders), chunk_size)]

    merged_messages = {}
    session_level = {}
    for chunk_orders in chunks:
        turns = _build_turns_payload(session_msgs, target_orders=set(chunk_orders))
        sessions_payload = [{
            "session_id": session_id,
            "source_folder": source_folder,
            "session_number": session_number,
            "astrologer_id": astrologer_id,
            "turns": turns,
        }]
        prompt = _build_batch_prompt(sessions_payload)
        result = client.generate_json(prompt, system_instruction=system_instruction,
                                       response_schema=RESPONSE_SCHEMA)
        by_session = {s["session_id"]: s for s in result.get("sessions", [])}
        session_result = by_session.get(session_id, {})
        for m in session_result.get("messages", []):
            if m.get("message_order") in set(chunk_orders):
                merged_messages[m["message_order"]] = m
        if session_result.get("session"):
            session_level = session_result["session"]

    return {"sessions": [{
        "session_id": session_id,
        "messages": list(merged_messages.values()),
        "session": session_level,
    }]}


def _make_batches(target_ids: list, sessions_index: pd.DataFrame,
                   grouped_msgs: dict, batch_size_messages: int) -> list[list[tuple]]:
    batches = []
    large_items = []
    current: list[tuple] = []
    current_count = 0
    for session_id in target_ids:
        session_msgs = grouped_msgs.get(session_id)
        if session_msgs is None or session_msgs.empty:
            continue
        meta = sessions_index.loc[session_id]
        item = (session_id, meta["source_folder"], int(meta["session_number"]),
                 meta["astrologer_id"], session_msgs)

        if len(session_msgs) > config.LARGE_SESSION_THRESHOLD:
            large_items.append(item)
            continue

        if current and current_count + len(session_msgs) > batch_size_messages:
            batches.append(current)
            current, current_count = [], 0
        current.append(item)
        current_count += len(session_msgs)
    if current:
        batches.append(current)
    return batches, large_items


def tag_all(messages_df: pd.DataFrame, sessions_df: pd.DataFrame,
            session_ids: Optional[list] = None,
            client: Optional[GeminiCachedClient] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    client = client or GeminiCachedClient()
    system_instruction = load_system_instruction(config.TAGGING_PROMPT_PATH)

    target_ids = session_ids if session_ids is not None else sessions_df["session_id"].tolist()
    sessions_index = sessions_df.set_index("session_id")

    tagged_message_rows = []
    session_level_tags = {}
    failures = []
    failures_lock = threading.Lock()

    grouped_msgs = {sid: g.sort_values("message_order")
                    for sid, g in messages_df[messages_df["session_id"].isin(target_ids)].groupby("session_id")}

    batches, large_items = _make_batches(target_ids, sessions_index, grouped_msgs,
                                          config.TAGGING_BATCH_SIZE_MESSAGES)

    def _work(batch):
        result = tag_session_batch(client, system_instruction, batch)
        by_session = {s["session_id"]: s for s in result.get("sessions", [])}
        return batch, by_session

    def _work_large(item_batch):
        (session_id, source_folder, session_number, astrologer_id, session_msgs), = item_batch
        result = tag_large_session(client, system_instruction, session_id, source_folder,
                                    session_number, astrologer_id, session_msgs)
        by_session = {s["session_id"]: s for s in result.get("sessions", [])}
        return item_batch, by_session

    all_jobs = [(_work, batch) for batch in batches] + [(_work_large, [item]) for item in large_items]

    with ThreadPoolExecutor(max_workers=config.GEMINI_CONCURRENCY) as pool:
        futures = {pool.submit(fn, job): job for fn, job in all_jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="tagging session batches"):
            batch = futures[fut]
            try:
                batch, by_session = fut.result()
            except Exception as e:  # noqa: BLE001
                with failures_lock:
                    for session_id, *_ in batch:
                        failures.append({"session_id": session_id, "error": str(e)})
                continue

            for session_id, source_folder, session_number, astrologer_id, session_msgs in batch:
                session_result = by_session.get(session_id)
                if session_result is None:
                    with failures_lock:
                        failures.append({"session_id": session_id, "error": "missing from batch response"})
                    continue

                by_order = {m["message_order"]: m for m in session_result.get("messages", [])}
                for _, row in session_msgs.iterrows():
                    tags = by_order.get(int(row["message_order"]), {})
                    merged = row.to_dict()
                    for field in ALL_MESSAGE_FIELDS:
                        merged[field] = tags.get(field)
                    tagged_message_rows.append(merged)

                session_level_tags[session_id] = session_result.get("session", {})

    messages_tagged = pd.DataFrame(tagged_message_rows)

    sessions_tagged = sessions_df.copy()
    sessions_tagged["ending_type"] = sessions_tagged["session_id"].map(
        lambda sid: session_level_tags.get(sid, {}).get("ending_type"))
    sessions_tagged["wall_hit"] = sessions_tagged["session_id"].map(
        lambda sid: session_level_tags.get(sid, {}).get("wall_hit"))
    sessions_tagged["archetype"] = None  # filled in later by Stage 4.9 clustering

    if failures:
        fail_df = pd.DataFrame(failures)
        fail_path = config.LOGS_DIR / "tagging_failures.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"WARNING: {len(failures)} sessions failed tagging -> {fail_path}")

    print(f"Gemini stats: {client.stats}")
    return messages_tagged, sessions_tagged


def run(sample_sessions: Optional[int] = None, source_folder_filter: Optional[str] = None):
    messages_df = pd.read_parquet(config.MESSAGES_TABLE_PATH)
    sessions_df = pd.read_parquet(config.SESSIONS_TABLE_PATH)

    if source_folder_filter:
        sessions_df = sessions_df[sessions_df["source_folder"] == source_folder_filter]
        messages_df = messages_df[messages_df["session_id"].isin(sessions_df["session_id"])]

    session_ids = sessions_df["session_id"].tolist()
    if sample_sessions:
        session_ids = session_ids[:sample_sessions]

    messages_tagged, sessions_tagged = tag_all(messages_df, sessions_df, session_ids=session_ids)

    if sample_sessions:
        out_msg = config.PROCESSED_DIR / "messages_tagged_sample.parquet"
        out_sess = config.PROCESSED_DIR / "sessions_tagged_sample.parquet"
    else:
        out_msg = config.MESSAGES_TAGGED_PATH
        out_sess = config.SESSIONS_TAGGED_PATH

    messages_tagged.to_parquet(out_msg, index=False)
    sessions_tagged.to_parquet(out_sess, index=False)
    print(f"Tagged messages -> {out_msg} ({len(messages_tagged)} rows)")
    print(f"Tagged sessions -> {out_sess} ({len(sessions_tagged)} rows)")
    return messages_tagged, sessions_tagged


if __name__ == "__main__":
    run()
