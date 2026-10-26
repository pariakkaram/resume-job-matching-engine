"""Logging setup and privacy-safe descriptions of text.

Resumes contain personal information (spec section 17): code must log IDs, counts, lengths and
hashes, never document text. ``describe_text`` is the sanctioned way to refer to a text in a log.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a module logger; the first call configures a stderr handler on the root 'src' logger.

    The level comes from ``RJM_LOG_LEVEL`` (default INFO).
    """
    global _configured
    if not _configured:
        root = logging.getLogger("src")
        if not root.handlers:
            handler = logging.StreamHandler(sys.stderr)
            handler.setFormatter(logging.Formatter(_FORMAT, datefmt="%H:%M:%S"))
            root.addHandler(handler)
        root.setLevel(os.environ.get("RJM_LOG_LEVEL", "INFO").upper())
        root.propagate = False
        _configured = True
    return logging.getLogger(name if name.startswith("src") else f"src.{name}")


def describe_text(text: str | None) -> str:
    """Summarise a text for logs without revealing it: length in characters/words and a short hash."""
    if text is None:
        return "<text: none>"
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"<text: {len(text)} chars, {len(text.split())} words, sha1 {digest}>"
