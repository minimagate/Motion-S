"""Streamlit entry point kept separate so package-relative imports stay valid."""

import argparse

from motions.visualize import run_app

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--checkpoint", required=True)
arguments, _ = parser.parse_known_args()
run_app(arguments.checkpoint)
