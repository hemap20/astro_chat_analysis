#!/usr/bin/env python3
"""CLI entry point for Stage 2: Gemini tagging + spot-check sample.

Usage:
  PYTHONPATH=src python3 scripts/run_stage2_tagging.py --sample 5
  PYTHONPATH=src python3 scripts/run_stage2_tagging.py --sample 5 --folder Sitara-1
  PYTHONPATH=src python3 scripts/run_stage2_tagging.py            # full run
  PYTHONPATH=src python3 scripts/run_stage2_tagging.py --spotcheck-only --n 50
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astro_analysis.tagging import tagger, spotcheck  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None, help="tag only the first N sessions (validation run)")
    ap.add_argument("--folder", type=str, default=None, help="restrict to one source_folder")
    ap.add_argument("--spotcheck-only", action="store_true", help="skip tagging, just resample review CSV")
    ap.add_argument("--n", type=int, default=50, help="spot-check sample size")
    args = ap.parse_args()

    if not args.spotcheck_only:
        tagger.run(sample_sessions=args.sample, source_folder_filter=args.folder)

    spotcheck_input = None
    if args.sample:
        from astro_analysis import config
        spotcheck_input = config.PROCESSED_DIR / "messages_tagged_sample.parquet"
    spotcheck.run(n=args.n, input_path=spotcheck_input)


if __name__ == "__main__":
    main()
