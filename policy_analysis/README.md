# Policy Analysis

Pipeline and data for analyzing AI-related policies (GW and non‑GW sources).

## Structure
- `gwu/` GW policy corpus, scripts, and outputs.
- `non-gwu/` Non‑GW policy corpus, scripts, and outputs.

Each subfolder contains:
- `src/` Python scripts for PDF/text extraction, chunking, keywording, embedding, and clustering.
- `dat/` Input data (PDF/TXT/CSV).
- `out/` Generated outputs (CSV/plots).

## Run (typical flow)
Run scripts in numerical order within each `src/` folder:
1. `01_*`
2. `02_*`
3. `03_*`
4. `04_*`
5. `05_*`
6. `06_*`
7. `07_*` (if present)

## Notes
- Scripts resolve paths relative to their module location, not your CWD.
- Some scripts require API keys via environment variables.
