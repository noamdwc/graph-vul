## ADDED Requirements

### Requirement: Content-addressed CPG cache
The system SHALL maintain a local cache directory that stores Joern CPG export results keyed by the SHA-256 hash of the normalized (whitespace-stripped) function source code. Before invoking Joern for a sample, the system SHALL check the cache and copy the cached result to the output directory on a hit.

#### Scenario: Cache hit
- **WHEN** `export_cpgs` processes a function whose SHA-256 hash matches an existing non-empty cache entry
- **THEN** the system SHALL copy the cached CPG directory to the sample's output location without invoking Joern

#### Scenario: Cache miss
- **WHEN** `export_cpgs` processes a function whose SHA-256 hash has no cache entry
- **THEN** the system SHALL invoke Joern, write the result to the output directory, and copy the result into the cache for future reuse

#### Scenario: Cache directory does not exist
- **WHEN** `export_cpgs` runs and the configured cache directory does not exist
- **THEN** the system SHALL create the cache directory automatically

### Requirement: Parallel CPG export
The system SHALL process Joern exports in parallel using multiple worker processes. The number of workers SHALL be configurable via `JOERN_WORKERS` in config, defaulting to `min(os.cpu_count(), 4)`.

#### Scenario: Parallel processing
- **WHEN** `export_cpgs` is invoked with multiple samples to process
- **THEN** the system SHALL distribute Joern invocations across the configured number of worker processes

#### Scenario: Worker failure isolation
- **WHEN** a Joern invocation fails in one worker process
- **THEN** the system SHALL log the error with the sample identifier and continue processing remaining samples in other workers without terminating the batch

### Requirement: Cache configuration
The system SHALL expose a `CPG_CACHE_DIR` configuration value in `graph_vul/config.py` pointing to the cache directory (default: `data/cpg_cache/`), and a `JOERN_WORKERS` value controlling parallelism (default: `min(os.cpu_count(), 4)`).

#### Scenario: Custom cache directory
- **WHEN** `CPG_CACHE_DIR` is set to a custom path
- **THEN** the system SHALL use that path for all cache reads and writes
