"""
Compatibility wrapper for training runs.

The maintained command-line entry point is run.py. This file remains so older
course links or shell commands that invoke run_training.py still work.
"""

from run import args_parser, main


if __name__ == '__main__':
    main(args_parser())
