#!/usr/bin/env python3
from pathlib import Path
import os
import csv
import json
import requests
from openai import OpenAI

# ----------------------
# Config
# ----------------------
INPUT_DIR = Path("artifacts/epmc_fulltext")
OUTPUT_DIR = Path("artifacts/visual_factors")
OUTPUT_CSV = OUTPUT_DIR / "all_visual_associations.csv"
MAX_FILES = 100
TRUNCATE_CHARS = int(os.environ.get("TRUNCATE_CHARS", "350000"))

MODEL_NAME = os.environ.get("MODEL_NAME", "GPT-OSS-120B")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://pluto/v1/")
API_KEY = os.environ.get("VIRTUAL_API_KEY", "VIRTUAL_API_KEY")

EPMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

SYSTEM_MSG = (
    "You are an expert research assistant for evidence extraction. "
    "Return ONLY valid JSON with no extra commentary."
)

# --- Prompt (your version, with outcome_raw kept) ---
PROMPT_HEADER_TEMPLATE = """Task:
From the paper below, identify every reported association between visually observable contextual factors
(i.e., elements that could plausibly be recognized in a single smartphone photo by a vision–language model)
and affective, emotional, stress-related, or well-being outcomes in humans. Do not report associations which have been found in studies relying solely on animals or cell/molecular lab contexts, cages/mesh platforms, Morris water maze, gene/protein markers.

STRICT SCENE SCOPE (must be visible in a single photo of human everyday life):
- Valid factors: for example proportion of green/blue (vegetation/water), proportion of grey/built surfaces, trees/water features, animals/pets,
  number of people/crowding, companions (alone/with others), litter/clutter, indoor/outdoor, location type
  (home/office/park/street/transport/shop), daylight/lighting, uniforms/badges/signage, food/alcohol/cigarettes, screens/devices, clear room type cues.
- Exclude if the factor requires multi-step events or specialized contexts not evident in a photo
  (e.g., hospital transfer, traveling to another city, policy status, questionnaires, logistics).
- Exclude VR/AR-only stimuli as the "factor" (unless the factor is the visible physical viewing setup/room).

SIMPLE OUTCOME VOCAB ONLY:
Set "outcome" to EXACTLY ONE of: ["positive affect", "negative affect", "Well-being", "perceived level of safety"].
Use your own reasoning to map the paper’s outcome wording to the closest one of these. Do not output any other label.

For each distinct association, return objects with keys:
- "factor" (short, human-readable, concrete and visible)
- "outcome_raw" (the affective state wording used in the paper, verbatim/near-verbatim)
- "outcome" (exactly one of the four allowed labels above)
- "direction" ("positive" | "negative" | "curvilinear" | "mixed/none")
- "vlm_detectability" ("likely" | "possible" | "unclear")
- "notes" (≤20 words, optional)

Validation rules:
1) If factor is not directly visible in a single photo, discard it.
2) Only keep items whose "outcome" is one of the four allowed labels above.
3) Prefer Abstract/Results/Discussion; avoid Introduction/speculation.

Output: ONLY a JSON array of objects with exactly the keys above.

Paper ID: {paper_id}

---BEGIN FULL TEXT---
"""
PROMPT_FOOTER = """
---END FULL TEXT---
"""

# Post-filters for obviously non-photo or lab contexts
BANNED_FACTOR_KEYWORDS = (
    "transfer", "travel", "airport", "border", "passport", "visa",
    "virtual", "vr", "ar", "online", "internet", "screening video",
    "mouse", "mice", "rat", "rodent", "cage", "mesh", "maze", "morris", "platform",
    "lab", "laboratory", "gene", "mrna", "immuno", "iba-1", "qpcr",
    "mri", "eeg", "ecg"
)

ALLOWED_NORMALIZED = {
    "positive affect", "negative affect", "Well-being", "perceived level of safety"
}

def factor_is_visual(factor: str) -> bool:
    t = (factor or "").lower()
    return bool(t) and not any(k in t for k in BANNED_FACTOR_KEYWORDS)

def sanitize(s: str) -> str:
    return (s or "").replace("\n", " ").replace("\r", " ").strip()

def build_prompt(paper_id: str, text: str) -> str:
    if len(text) > TRUNCATE_CHARS:
        text = text[:TRUNCATE_CHARS]
    header = PROMPT_HEADER_TEMPLATE.format(paper_id=paper_id)  # safe: not formatting 'text'
    return f"{header}{text}{PROMPT_FOOTER}"

def ask_llm_extract(client: OpenAI, paper_id: str, text: str):
    prompt = build_prompt(paper_id, text)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":SYSTEM_MSG},
                  {"role":"user","content":prompt}],
        temperature=0.1, top_p=1.0,
        max_completion_tokens=8192, max_tokens=8192,
    )
    content = resp.choices[0].message.content
    try:
        data = json.loads(content)
        return data if isinstance(data, list) else []
    except Exception:
        return []

def ask_llm_map_outcomes(client: OpenAI, phrases: list[str]) -> dict[int, str]:
    """
    Let the LLM map each phrase to exactly one of the 4 labels.
    Returns {index: label}. Items not mapped to allowed labels are omitted.
    """
    if not phrases:
        return {}

    numbered = [{"i": i, "phrase": p} for i, p in enumerate(phrases)]
    mapping_prompt = {
        "role": "user",
        "content": (
            "Map each outcome phrase to EXACTLY ONE of: "
            '["positive affect","negative affect","Well-being","perceived level of safety"]. '
            "Use your own reasoning. Return ONLY a JSON array of objects with keys "
            '`i` (the provided index) and `label` (one of the four). '
            "Here are the items:\n" + json.dumps(numbered, ensure_ascii=False)
        )
    }
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role":"system","content":"You are a careful, terse classifier. Output valid JSON only."},
                  mapping_prompt],
        temperature=0.0, top_p=1.0,
        max_completion_tokens=1024, max_tokens=1024,
    )
    try:
        arr = json.loads(resp.choices[0].message.content)
        out: dict[int, str] = {}
        for obj in arr if isinstance(arr, list) else []:
            try:
                i = int(obj.get("i"))
                label = str(obj.get("label") or "").strip()
                if label.lower() in {"well-being", "wellbeing"}:
                    label = "Well-being"
                if label in ALLOWED_NORMALIZED and 0 <= i < len(phrases):
                    out[i] = label
            except Exception:
                continue
        return out
    except Exception:
        return {}

def epmc_metadata_for_pmcid(pmcid: str):
    """Return (title, citation, doi) for a PMCID using Europe PMC."""
    params = {"query": f"PMCID:{pmcid}", "format": "json", "pageSize": 1, "resultType": "lite"}
    r = requests.get(EPMC_SEARCH_URL, params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    item = data["resultList"]["result"][0]
    title = (item.get("title") or "").strip()
    year = (item.get("pubYear") or "").strip()
    first_chunk = (item.get("authorString") or "").split(";")[0].strip()
    surname = (first_chunk.split(",")[0] if "," in first_chunk else first_chunk.split()[0]).strip()
    citation = f"{surname} et al. {year}"
    doi = (item.get("doi") or "").strip()
    return (title, citation, doi)

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    files = sorted(INPUT_DIR.glob("*.txt"))[:MAX_FILES]
    print(f"Processing {len(files)} files from {INPUT_DIR} ...")

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "factor",
            "outcome_raw",  # paper wording
            "outcome",      # LLM-mapped (one of 4)
            "study_title",
            "citation",
            "DOI",          # <-- new column
            "direction",
            "vlm_detectability",
            "notes",
        ])

        total_rows = 0
        for idx, f in enumerate(files, 1):
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                pmcid = f.stem

                # 1) Extract
                items = ask_llm_extract(client, pmcid, text)
                prepared = []
                outcome_phrases = []
                phrase_to_index = {}
                for it in items:
                    if not isinstance(it, dict):
                        continue
                    factor = sanitize(it.get("factor", ""))
                    if not factor or not factor_is_visual(factor):
                        continue
                    outcome_raw = sanitize(it.get("outcome_raw", "")) or sanitize(it.get("outcome", ""))
                    if not outcome_raw:
                        continue
                    direction = sanitize(it.get("direction", ""))
                    if direction not in {"positive","negative","curvilinear","mixed/none"}:
                        direction = ""
                    vlm = sanitize(it.get("vlm_detectability", ""))
                    if vlm not in {"likely","possible","unclear"}:
                        vlm = "possible"
                    notes = sanitize(it.get("notes", ""))[:200]

                    if outcome_raw not in phrase_to_index:
                        phrase_to_index[outcome_raw] = len(outcome_phrases)
                        outcome_phrases.append(outcome_raw)

                    prepared.append({
                        "factor": factor,
                        "outcome_raw": outcome_raw,
                        "direction": direction,
                        "vlm": vlm,
                        "notes": notes,
                    })

                if not prepared:
                    print(f"[{idx}/{len(files)}] {f.name}: 0 items")
                    continue

                # 2) Map outcome_raw -> one of four (LLM decides)
                idx_to_label = ask_llm_map_outcomes(client, outcome_phrases)

                # 3) Write rows for successfully mapped items
                title, citation, doi = epmc_metadata_for_pmcid(pmcid)
                if not title:
                    title = sanitize(text.splitlines()[0] if text.splitlines() else "")

                seen = set()
                written = 0
                for rec in prepared:
                    i = phrase_to_index[rec["outcome_raw"]]
                    label = idx_to_label.get(i)
                    if not label or label not in ALLOWED_NORMALIZED:
                        continue
                    rowkey = (rec["factor"].lower(), rec["outcome_raw"].lower(), label.lower(), rec["direction"].lower())
                    if rowkey in seen:
                        continue
                    seen.add(rowkey)
                    writer.writerow([
                        rec["factor"],
                        rec["outcome_raw"],
                        label,
                        title,
                        citation,
                        doi,                 # <-- write DOI here
                        rec["direction"],
                        rec["vlm"],
                        rec["notes"],
                    ])
                    written += 1
                total_rows += written
                print(f"[{idx}/{len(files)}] {f.name}: {written} rows")

            except Exception as e:
                print(f"[{idx}/{len(files)}] {f.name}: ERROR ({e})")

    print(f"Done. Wrote rows: {total_rows}, Output: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
