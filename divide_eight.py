#!/usr/bin/env python3
"""
Split rows into 8 files based on exact (case-sensitive) combinations of:
  outcome LLM ∈ {"positive affect", "Well-being", "negative affect", "perceived level of safety"}
  direction   ∈ {"positive", "negative"}

Anything that doesn't match these exact strings goes to 'mixed.csv'.

Input:  set INPUT_FILE below (e.g., 'removed_none.csv' or your master CSV)
Output: CSV files named like:
  positive_affect_positive.csv
  positive_affect_negative.csv
  well_being_positive.csv
  well_being_negative.csv
  negative_affect_positive.csv
  negative_affect_negative.csv
  perceived_level_of_safety_positive.csv
  perceived_level_of_safety_negative.csv
  mixed.csv
"""

import os
import pandas as pd
from pathlib import Path

# ---- Config ----
INPUT_FILE = Path("removed_none.csv")  # change if needed

# ---- Checks ----
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_cols = {"outcome LLM", "direction"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {sorted(missing)}")

# ---- Exact-match (case-sensitive) definitions ----
outcomes = [
    "positive affect",
    "Well-being",
    "negative affect",
    "perceived level of safety",
]
directions = ["positive", "negative"]

# Map outcome to safe filename token
def safe_token(s: str) -> str:
    return (
        s.strip()
         .lower()
         .replace(" ", "_")
         .replace("-", "_")
    )

# Track which rows have been assigned to one of the 8 buckets
assigned_mask = pd.Series(False, index=df.index)

# ---- Create 8 files ----
for outcome in outcomes:
    for direction in directions:
        mask = (df["outcome LLM"] == outcome) & (df["direction"] == direction)
        bucket = df[mask]
        assigned_mask |= mask

        out_name = f"{safe_token(outcome)}_{safe_token(direction)}.csv"
        # Ensure file is created even if empty, preserving headers
        bucket.to_csv(out_name, index=False)
        print(f"Saved {len(bucket):>6} rows to {out_name}")

# ---- Mixed (everything else) ----
mixed = df[~assigned_mask]
mixed.to_csv("mixed.csv", index=False)
print(f"Saved {len(mixed):>6} rows to mixed.csv")
