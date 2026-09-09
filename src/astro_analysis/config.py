"""Central configuration: paths, source groups, model settings, tag definitions."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Data sources ---------------------------------------------------------
SOURCE_FOLDERS = {
    "Sitara-1": PROJECT_ROOT / "Sitara-1",
    "Sitara-2": PROJECT_ROOT / "Sitara-2",
    "Human_Astro": PROJECT_ROOT / "Human_Astro",
    "Human_Single": PROJECT_ROOT / "Human_single",  # note: actual dir is lowercase "single"
}

# System / sender conventions
SYSTEM_SENDER_ID = "0"
SESSION_START_MARKERS = ("welcome to astrolokal",)
SESSION_END_MARKERS = ("chat has ended",)
UNMARKED_SESSION_GAP_HOURS = 2.0  # internal gap this large with no marker -> flag as a likely hidden boundary

# --- Output locations ------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
LOGS_DIR = DATA_DIR / "logs"

for d in (PROCESSED_DIR, CACHE_DIR, RESULTS_DIR, LOGS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MESSAGES_TABLE_PATH = PROCESSED_DIR / "messages.parquet"
SESSIONS_TABLE_PATH = PROCESSED_DIR / "sessions.parquet"
MESSAGES_TAGGED_PATH = PROCESSED_DIR / "messages_tagged.parquet"
SESSIONS_TAGGED_PATH = PROCESSED_DIR / "sessions_tagged.parquet"
PARSE_ISSUES_LOG = LOGS_DIR / "parse_issues.csv"

# --- Gemini / tagging -------------------------------------------------------
GEMINI_MODEL = "gemini-2.5-flash"
TAGGING_PROMPT_VERSION = "v1"
TAGGING_PROMPT_PATH = PROMPTS_DIR / f"tagging_prompt_{TAGGING_PROMPT_VERSION}.md"
CONTEXT_TURNS_BEFORE = 3  # min prior turns of context given to the tagger

GEMINI_MAX_RETRIES = 5
GEMINI_BASE_DELAY_SECONDS = 2.0
GEMINI_REQUESTS_PER_MINUTE = 800  # Tier 1 paid ceiling; adjust to your tier
GEMINI_CONCURRENCY = 25  # thread-pool workers for tagging calls
TAGGING_BATCH_SIZE_MESSAGES = 150  # target messages per API call; sessions are grouped, never split, to hit this
LARGE_SESSION_THRESHOLD = 200  # sessions with more messages than this are tagged in chunks instead of one call
LARGE_SESSION_CHUNK_SIZE = 20  # messages tagged per call when chunking an oversized session

# --- Tag schema --------------------------------------------------------------
ASTROLOGER_MESSAGE_TAGS = [
    "concrete_dated_prediction",
    "specific_open_thread_named",
    "callback_prior_session",
    "callback_within_session",
    "claim_certainty",         # hedged / absolute / not_applicable
    "specificity",             # generic_broad / personalized_specific / not_applicable
    "validation",
    "self_correction",
    "reframe",
    "repair_after_pushback",   # defer / offer_options / repeat / none_of_above / not_applicable
]

USER_MESSAGE_TAGS = [
    "emotional_disclosure",
    "pushback_disagreement",
    "enthusiastic_engagement",
    "disengagement_signal",
    "neutral_factual",
]

SESSION_LEVEL_TAGS = [
    "ending_type",   # resolved_closed / open_thread / abrupt_trail_off
    "wall_hit",
    "archetype",     # filled in later by clustering (Stage 4.9)
]

CLAIM_CERTAINTY_VALUES = ["hedged", "absolute", "not_applicable"]
SPECIFICITY_VALUES = ["generic_broad", "personalized_specific", "not_applicable"]
REPAIR_AFTER_PUSHBACK_VALUES = ["defer", "offer_options", "repeat", "none_of_above", "not_applicable"]
ENDING_TYPE_VALUES = ["resolved_closed", "open_thread", "abrupt_trail_off"]

# --- User metadata fields parsed out of the opening message ------------------
USER_PROFILE_FIELDS = ["Name", "Gender", "DOB", "TOB", "POB"]

RANDOM_SEED = 42
