# Inventory

Scripts and data for building an AI-related faculty/staff inventory for the GW School of Business.

## Structure
- `src/` Python scripts for collection, joining, summarization, and verification.
- `dat/` Input datasets (CSV).
- `out/` Generated outputs (CSV).
- `cv/` CV storage.

## Run (typical flow)
1. `src/01_fac_profile_inventory.py`
2. `src/02_fac_per_site_inventory.py`
3. `src/03_fac_cv_inventory.py`
4. `src/04_fac_manual_inventory.py`
5. `src/05_fac_join_inventory.py`
6. `src/06_fac_summary_inventory.py`
7. `src/07_fac_verify_inventory.py`

## Notes
- Some scripts expect `shared/` utilities at the repo root.
- Configure any required API keys via environment variables before running.
