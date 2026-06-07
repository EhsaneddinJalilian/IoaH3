#!/usr/bin/env python
"""
scripts/run_ioah3.py
====================
Command-line entry point for the IOAH3 pipeline.

Usage
-----
    python scripts/run_ioah3.py
    python scripts/run_ioah3.py --viz-only
    python scripts/run_ioah3.py --vienna-only
"""

import argparse
import sys
from pathlib import Path

# Allow running from repo root without install
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Ioah3.config as cfg
from Ioah3.pipeline import run


def main():
    parser = argparse.ArgumentParser(description="Run the IOAH3 pipeline.")
    parser.add_argument("--viz-only",    action="store_true",
                        help="Load checkpoint and re-render map only (no pipeline).")
    parser.add_argument("--vienna-only", action="store_true",
                        help="Restrict domain to Vienna bounding box.")
    parser.add_argument("--force-elev",  action="store_true",
                        help="Force re-computation of elevation cache.")
    args = parser.parse_args()

    if args.viz_only:
        cfg.VIZ_ONLY = True
    if args.vienna_only:
        cfg.USE_VIENNA_ONLY = True
    if args.force_elev:
        cfg.FORCE_RECOMPUTE_ELEV = True

    substrate = run()
    if substrate is not None:
        print("\n✅ Done.")
        for r in sorted(substrate["resolution"].unique()):
            print(f"  Res {r}: {(substrate['resolution'] == r).sum():,} cells")


if __name__ == "__main__":
    main()
