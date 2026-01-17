import os
import sys

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from compare_models import run_comparison


def main():
    run_comparison(include_deep=False)


if __name__ == "__main__":
    main()
