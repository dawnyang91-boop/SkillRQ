#!/usr/bin/env python3
"""Build Tool/API capability recommendation data files."""

from __future__ import annotations

import sys

from skillrq.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["capability", "build", *sys.argv[1:]]))
