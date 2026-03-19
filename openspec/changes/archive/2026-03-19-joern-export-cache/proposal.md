## Why

Joern CPG export is the slowest stage in the pipeline (~2 min per sample). When we change sampling methods between experiments (different subsets, different CWE filters, different balance ratios), many of the same functions get re-processed because the current idempotency check is tied to the output directory of a specific pipeline run. A content-addressed local cache would let us skip re-exporting any function we've already processed, regardless of which subset it belongs to. Additionally, exports currently run sequentially — parallelizing Joern invocations across multiple workers would further reduce wall-clock time.

## What Changes

- Add a local CPG cache directory (`data/cpg_cache/`) that stores Joern export results keyed by a hash of the function source code
- Modify `export_cpgs` to check the cache before invoking Joern — on a cache hit, copy the cached result to the output directory instead of re-running Joern
- After a successful Joern export, store the result in the cache for future reuse
- Parallelize Joern export across multiple worker processes to reduce total processing time
- Add config entries for the cache directory path and worker count

## Capabilities

### New Capabilities
- `joern-export-cache`: Local content-addressed cache for Joern CPG export results, keyed by function source code hash, with parallel export workers

### Modified Capabilities
- `data-pipeline`: The CPG generation requirement gains a cache-lookup step before invoking Joern and parallel processing

## Impact

- **Code**: `graph_vul/pipeline/joern_export.py` (main changes), `graph_vul/config.py` (cache dir path, worker count)
- **Data**: New `data/cpg_cache/` directory storing cached CPG exports; should be added to `.gitignore` and optionally tracked by DVC
- **Dependencies**: No new dependencies (uses stdlib `hashlib`, `shutil`, `concurrent.futures`)
