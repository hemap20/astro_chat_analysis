#!/usr/bin/env python3
"""CLI entry point for Stage 1: ingestion & session parsing.

Usage: PYTHONPATH=src python3 scripts/run_stage1_ingestion.py [--sample N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astro_analysis.ingestion import parser  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    parser.run()


if __name__ == "__main__":
    main()
