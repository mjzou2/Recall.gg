from typing import Dict


def row_to_dict(row) -> Dict:
    """Convert a database row to a dict.

    With RealDictCursor this is already a dict-like object,
    so we just ensure a plain dict is returned.
    """
    return dict(row)
