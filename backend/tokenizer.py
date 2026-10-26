"""Corpus/query tokenization for the sparse BM25 index.

`simple` reproduces the tokenizer the original pipeline used (str.lower().split()),
kept so the legacy hybrid baseline can be reproduced exactly.
`standard` is the tokenizer used by the new global BM25 index.
"""
from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_.][a-z0-9]+)*")

# A short, domain-agnostic stopword list. Deliberately small: scientific queries
# are noun-phrase heavy, so aggressive stopping mostly removes nothing useful.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by
with without within into over under is are was were be been being am do does did
we our us you your it its as via using use used such can could may might will would
shall should not no nor only also both each other some any more most much many
""".split())


def simple_tokens(text: str) -> list[str]:
    return text.lower().split()


def standard_tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower())
            if len(t) > 1 and t not in STOPWORDS]


TOKENIZERS = {"simple": simple_tokens, "standard": standard_tokens}
