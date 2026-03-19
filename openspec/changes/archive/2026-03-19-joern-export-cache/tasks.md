## 1. Configuration

- [x] 1.1 Add `CPG_CACHE_DIR` and `JOERN_WORKERS` to `graph_vul/config.py`

## 2. Cache Logic

- [x] 2.1 Add cache helper functions to `joern_export.py`: `_cache_key(func_code)` (SHA-256 of stripped source), `_cache_hit(cache_dir, key)` (check + copy), `_cache_store(cache_dir, key, source_dir)` (copy result into cache)
- [x] 2.2 Integrate cache lookup and store into the per-sample export flow

## 3. Parallel Processing

- [x] 3.1 Extract per-sample processing into a standalone function suitable for use with ProcessPoolExecutor
- [x] 3.2 Replace sequential loop with `ProcessPoolExecutor.submit` + `as_completed`, using `JOERN_WORKERS` for `max_workers`
- [x] 3.3 Add per-future error handling: log failures, continue processing, report summary at end

## 4. Finalize

- [x] 4.1 Add `data/cpg_cache/` to `.gitignore` (already covered by `/data/` pattern)
- [x] 4.2 Update progress reporting to work with parallel execution (aggregate counters after completion)
