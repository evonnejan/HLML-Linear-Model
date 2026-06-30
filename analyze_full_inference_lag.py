"""Backward-compatible entrypoint.

This module is kept temporarily so existing commands still work.
Please migrate to: analyze_full_inference.py
"""

from analyze_full_inference import main


if __name__ == "__main__":
    main()
