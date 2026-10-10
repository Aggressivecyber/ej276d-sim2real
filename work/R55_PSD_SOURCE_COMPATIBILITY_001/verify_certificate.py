"""Recompute the rational certificate; no fitting, waveform generation or LP solve."""
from pathlib import Path
import argparse
import json
import sys
sys.dont_write_bytecode = True
from certificate_core import certify

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--matrix', type=Path, default=Path(__file__).resolve().parent / 'LP_PRIVATE.npz')
    args = parser.parse_args()
    print(json.dumps(certify(args.matrix), indent=2))
