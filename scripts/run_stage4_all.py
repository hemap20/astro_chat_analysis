#!/usr/bin/env python3
"""CLI entry point for Stage 4: run every analysis module. Each module can
also be re-run independently (astro_analysis.analysis.<module>.run())."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from astro_analysis.analysis import (  # noqa: E402
    tag_rates, sequential_patterns, markov, survival, sentiment,
    paradise, diff_in_diff, process_mining, clustering,
)

MODULES = [
    ("4.1 tag rates", tag_rates),
    ("4.2 sequential patterns", sequential_patterns),
    ("4.3 Markov transitions", markov),
    ("4.4 survival analysis", survival),
    ("4.5 sentiment trajectories", sentiment),
    ("4.6 PARADISE regression", paradise),
    ("4.7 diff-in-diff scaffold", diff_in_diff),
    ("4.8 process mining", process_mining),
    ("4.9 tag-sequence clustering", clustering),
]

if __name__ == "__main__":
    for label, module in MODULES:
        print(f"\n=== Stage {label} ===")
        module.run()
