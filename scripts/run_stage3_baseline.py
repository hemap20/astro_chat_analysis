#!/usr/bin/env python3
"""CLI entry point for Stage 3: baseline construction from Human_Single."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astro_analysis.baseline import build_baseline  # noqa: E402

if __name__ == "__main__":
    build_baseline.run()
