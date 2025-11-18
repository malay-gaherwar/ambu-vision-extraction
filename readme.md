from pathlib import Path

readme_content = """# AmbuVision — Categorization Pipeline

A reproducible pipeline to **retrieve papers**, **extract visually detectable factors linked to emotions**, **clean & categorize factors**, and **produce deduplicated group lists** for downstream VLM scoring.

---

## Table of contents
- [Overview](#overview)
- [Directory layout](#directory-layout)
- [Prerequisites](#prerequisites)
- [Quick start](#quick-start)
- [Step-by-step](#step-by-step)
  - [Step 1 — Retrieve files from EPMC](#step-1--retrieve-files-from-epmc)
  - [Step 2 — LLM extraction of factor–outcome associations](#step-2--llm-extraction-of-factoroutcome-associations)
  - [Step 3 — Clean & filter factors](#step-3--clean--filter-factors)
  - [Step 4 — Split by psychological outcome & direction](#step-4--split-by-psychological-outcome--direction)
  - [Step 5 — Deduplicate categories with LLM grouping](#step-5--deduplicate-categories-with-llm-grouping)
  - [Step 6 — Build group lists](#step-6--build-group-lists)
- [Outputs](#outputs)
- [Environment variables](#environment-variables)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Overview

**Goal:** identify **visually observable contextual factors** reported in psychology papers that are associated with **affect, emotion, stress, safety, and well-being**, then normalize and deduplicate these factors into clean lists.

**High-level pipeline**
1) Retrieve full texts/records from Europe PMC  
2) Use an LLM to extract factor → outcome associations  
3) Filter out invalid/`None` categories  
4) Split into six bins by outcome × direction  
5) LLM-based deduplication & grouping  
6) Export per-group CSVs and a master unique list

---

## Directory layout

