# Epic 4 Milestone 1 retrieval notes

- **Small (`kennethreitz/records`)**: 18 files, 69 chunks, 41.4 s. CSV export retrieved the README export section first (0.621); the supported-databases question was weak, with a generic README introduction first (0.360).
- **Medium (`pallets/flask`)**: 216 files, 1,602 chunks, 135.3 s. Route variables and blueprints ranked their documentation first (0.575 and 0.569); configuration loaded ranked an extension-development page first, with the dedicated configuration page third (0.463 top score).
- **Docs-heavy (`jlevy/the-art-of-command-line`)**: after excluding translated README duplicates, 4 files, 119 chunks, 34.7 s. File-name search retrieved the English `find` guidance first (0.436); port lookup was weak (process-tree advice ranked first), and JSON command-line guidance appeared second at 0.322.

The docs-heavy filter removed translated README copies from results. Remaining weak queries are recorded for later retrieval tuning; the guide identifies reranking and hybrid search as later improvements.
