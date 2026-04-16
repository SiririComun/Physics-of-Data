# Skill: Decoupled Architecture
- **Logic Isolation:** All computational logic and classes must reside in `src/`.
- **Notebooks as Observatories:** Jupyter notebooks are strictly for data loading, calling `src` functions, visualization, and narrative.
- **Data Immutability:** Never modify files in `data/raw/`.
- **Artifacts:** Persist all statistical results and distribution parameters in `artifacts/` as JSON files.