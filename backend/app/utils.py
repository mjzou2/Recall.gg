import sqlite3
from typing import Dict


def row_to_dict(row: sqlite3.Row) -> Dict:
    return {k: row[k] for k in row.keys()}
