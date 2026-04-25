"""
models/labeling.py
Re-exports labeling primitives so models/ directory is self-contained.
The canonical implementation lives in labeling/barriers.py.
"""
from labeling.barriers import triple_barrier, select_label, MetaLabeler

__all__ = ["triple_barrier", "select_label", "MetaLabeler"]
