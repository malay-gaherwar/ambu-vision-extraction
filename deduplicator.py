#!/usr/bin/env python3
import os
import re
import pandas as pd
from openai import OpenAI

# ---- Config ----
INPUT_CSV = "positive_affect_positive.csv"
OUTPUT_CSV = "grouped.csv"
MODEL_NAME = os.environ.get("MODEL_NAME", "GPT-OSS-120B")
BASE_URL   = os.environ.get("OPENAI_BASE_URL", "http://pluto/v1/")
API_KEY    = os.environ.get("VIRTUAL_API_KEY")

if not API_KEY:
    raise ValueError("❌ Missing VIRTUAL_API_KEY environment variable.")

LIMIT = 3000
CATEGORY_COL = "Category"

# ---------- LLM call ----------
def ask_llm_for_groups(client: OpenAI, categories):
    cats = [str(c).strip() for c in categories if str(c).strip()]
    unique_cats = sorted(set(cats), key=lambda x: cats.index(x))
    comma_list = ", ".join(unique_cats)

    system = (
        "You are a precise taxonomy normalizer. "
        "Group near-duplicate or synonymous labels together based on meaning. "
        "Number groups sequentially (Group 1, Group 2, ...). "
        "If a label has no synonyms, put it alone in its own group. "
        "Do not group opposite categories together. "
        "Return the groups between the exact markers, one group per line. "
        "Come up with a concise CategoryName for each group."
    )
    user = f"""Cluster these labels by meaning. Use the format:

<<GROUPS-BEGIN>>
Group 1 CategoryName: labelA, labelB
Group 2 CategoryName: labelC
...
<<GROUPS-END>>

Rules:
- Use the original labels verbatim.
- Keep first-seen order as much as possible.
- Do NOT add commentary or tables.
- Every label must appear exactly once.
- Singles should appear as a solo group.

Labels (comma-separated):
{comma_list}
"""

    print(f"🧮 Input length ≈ {len(comma_list)} characters (~{len(comma_list)//4} tokens)")

    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
        temperature=0.1,
        max_completion_tokens=40096,  # safer upper bound
        reasoning_effort="low",
        extra_body={"allowed_openai_params": ["reasoning_effort"]},
    )

    content = (resp.choices[0].message.content or "").strip()
    print(f"📝 Raw model output length: {len(content)} characters")

    start, end = content.find("<<GROUPS-BEGIN>>"), content.find("<<GROUPS-END>>")
    if start != -1 and end != -1 and end > start:
        extracted = content[start + len("<<GROUPS-BEGIN>>"):end].strip()
    else:
        print("⚠️ Warning: Missing markers or truncated output!")
        extracted = content

    print("🔎 First 10 lines of extracted output:")
    print("\n".join(extracted.splitlines()[:10]))

    return extracted

# ---------- Helpers ----------
def groups_text_to_rows(groups_text: str):
    rows = []
    pattern = re.compile(r"^Group\s+(\d+)\s+(.+?):\s*(.+?)\s*$")
    for raw_line in groups_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = pattern.match(line)
        if not m:
            continue
        group_num = int(m.group(1))
        group_name = m.group(2).strip()
        labels_blob = m.group(3).strip()
        labels = [lab.strip() for lab in labels_blob.split(",") if lab.strip()]
        for lab in labels:
            rows.append({"Group": group_num, "Group Name": group_name, "Label": lab})
    return rows

def pick_and_pop_row(df: pd.DataFrame, category_col: str, label: str):
    matches = df.index[df[category_col] == label]
    if len(matches) == 0:
        return None, df
    idx = matches[0]
    row = df.loc[idx].copy()
    df_new = df.drop(index=idx)
    return row, df_new

# ---------- Main ----------
def main():
    client = OpenAI(base_url=BASE_URL, api_key=API_KEY)
    df_full = pd.read_csv(INPUT_CSV)
    df_source = df_full  # or .head(LIMIT) if you want smaller tests

    categories = df_source[CATEGORY_COL].astype(str).tolist()
    groups_text = ask_llm_for_groups(client, categories)
    print("✅ LLM grouping response received.")

    group_rows = groups_text_to_rows(groups_text)
    print(f"📦 Parsed {len(group_rows)} grouped labels from model output.")

    if len(group_rows) == 0:
        print("⚠️ No groups parsed — likely truncated or malformed output. Exiting early.")
        return

    picked_rows = []
    working_df = df_source.copy()

    for i, item in enumerate(group_rows, start=1):
        label = item["Label"]
        group_num = item["Group"]
        group_name = item["Group Name"]
        print(f"🔍 [{i}/{len(group_rows)}] Matching label '{label}' (Group {group_num}: {group_name})")

        row, working_df = pick_and_pop_row(working_df, CATEGORY_COL, label)
        if row is None:
            row = pd.Series(index=df_source.columns, dtype="object")
            row[CATEGORY_COL] = label

        row_out = row.to_dict()
        row_out["Group"] = group_num
        row_out["Group Name"] = group_name
        picked_rows.append(row_out)

    out_df = pd.DataFrame(picked_rows)

    if CATEGORY_COL not in out_df.columns:
        print("⚠️ Category column missing in output — cannot reorder columns safely.")
        out_df.to_csv(OUTPUT_CSV, index=False)
        return

    # Reorder columns so Group and Group Name come right after Category
    cols = list(out_df.columns)
    cat_idx = cols.index(CATEGORY_COL)
    new_order = (
        cols[:cat_idx+1] +
        ["Group", "Group Name"] +
        [c for c in cols if c not in [CATEGORY_COL, "Group", "Group Name"]]
    )
    out_df = out_df[new_order]

    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✅ Saved grouped results to {OUTPUT_CSV}")
    print(out_df.head(10))

if __name__ == "__main__":
    main()
