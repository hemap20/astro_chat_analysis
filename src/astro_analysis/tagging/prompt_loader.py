"""Parses the versioned tagging_prompt_*.md into (system_instruction, user_template)."""
from __future__ import annotations

import re
from pathlib import Path

_CODE_BLOCK = re.compile(r"```\n(.*?)\n```", re.DOTALL)


def load_system_instruction(path: Path) -> str:
    """The first fenced code block in the prompt file is the system instruction.
    (The second block is a human-readable schema example only; the actual
    request schema is built programmatically in tagger.py.)"""
    text = path.read_text()
    blocks = _CODE_BLOCK.findall(text)
    if not blocks:
        raise ValueError(f"expected at least 1 fenced code block in {path}")
    return blocks[0]
