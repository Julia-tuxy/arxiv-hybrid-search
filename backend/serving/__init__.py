"""Serving infrastructure shared by the search API and the embedding service.

Everything in this package is dependency-light (no model, no index) so it can be
unit-tested on its own: see eval/test_serving.py.
"""
