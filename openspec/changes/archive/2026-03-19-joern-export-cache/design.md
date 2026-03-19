## Context

The `export_cpgs` function in `graph_vul/pipeline/joern_export.py` processes samples sequentially, invoking Joern CLI for each function. It has basic idempotency — it skips samples whose output directory already exists — but this is scoped to a single pipeline run's output directory. When we change sampling parameters (subset size, CWE filter, balance ratio), the subset CSV changes and many previously-exported functions get re-processed from scratch. Each Joern invocation takes ~2 minutes, making re-exports very expensive. Processing is also single-threaded.

## Goals / Non-Goals

**Goals:**
- Avoid redundant Joern invocations across experiment runs by caching results keyed on function source code content
- Reduce wall-clock time by parallelizing Joern invocations across multiple worker processes
- Maintain backward compatibility with existing pipeline and DVC stages

**Non-Goals:**
- Distributed/remote cache (e.g., S3-backed) — local disk cache is sufficient for now
- Caching intermediate `.bin` CPG files — only the final exported CPG directory is cached
- Changing Joern CLI invocation flags or export format

## Decisions

### Content-addressed cache keyed by SHA-256 of function source code

The cache directory (`data/cpg_cache/`) stores exported CPG results in subdirectories named by the SHA-256 hex digest of the normalized (stripped) function source code. On cache hit, the cached directory is copied to the target output location. On cache miss, Joern runs normally and the result is copied into the cache.

**Why SHA-256 over sample_id**: Sample IDs are dataset-specific. The same function can appear with different IDs across subsets. Hashing the source code ensures deduplication regardless of which subset references it.

**Why copy instead of symlink**: Symlinks would break if the cache is moved or cleaned. Copies are self-contained and compatible with DVC tracking. The CPG export files are small (a few KB per sample), so disk overhead is minimal.

### Parallel processing with ProcessPoolExecutor

Use `concurrent.futures.ProcessPoolExecutor` to run multiple Joern invocations in parallel. Each worker handles one sample independently (check cache → run Joern → populate cache → write output). A configurable `JOERN_WORKERS` setting controls the degree of parallelism, defaulting to `min(cpu_count, 4)` to avoid overwhelming the system.

**Why ProcessPoolExecutor over multiprocessing.Pool**: Cleaner API, better exception handling, and future-compatible. Since Joern is a subprocess call, GIL is not a concern — processes are used to run multiple Joern CLI instances concurrently.

**Why cap at 4 by default**: Joern is memory-intensive (JVM-based). Running too many instances concurrently can cause OOM or excessive GC pressure.

### Cache structure

```
data/cpg_cache/
  <sha256_hex>/
    nodes.csv
    edges.csv
    ...  (whatever Joern exports)
```

Each hash directory mirrors the structure that `_run_joern` produces. A completed cache entry is identified by the directory existing and being non-empty (same logic as current idempotency check).

## Risks / Trade-offs

- **Hash collisions**: SHA-256 collisions are practically impossible. No mitigation needed.
- **Stale cache entries**: If Joern version changes, cached CPGs may be from an older format. → Mitigation: Users can delete `data/cpg_cache/` to force re-export. A version marker file could be added later if needed.
- **Disk usage**: Cache grows monotonically. → CPG exports are small (KB each); thousands of cached samples will use < 1 GB. Manual cleanup is sufficient.
- **Parallel failure handling**: One worker failure should not crash the entire batch. → Use `as_completed` with per-future exception handling; log failures and continue.
