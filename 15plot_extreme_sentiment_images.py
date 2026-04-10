#!/usr/bin/env python3

import runpy
from pathlib import Path


# Compatibility entrypoint: step 15 now delegates to step 14 implementation.
if __name__ == "__main__":
    target = Path(__file__).resolve().with_name("14plot_extreme_sentiment_images.py")
    runpy.run_path(str(target), run_name="__main__")
