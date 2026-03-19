"""Export Code Property Graphs for each sample using the Joern CLI."""

import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from graph_vul.config import SUBSET_CSV, RAW_DIR, CPG_CACHE_DIR, JOERN_WORKERS
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
JOERN_HOME = Path("/opt/joern/joern-cli")
HOMEBREW_OPENJDK_HOME = Path("/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home")


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_key(func_code: str) -> str:
    """Return SHA-256 hex digest of the stripped function source code."""
    return hashlib.sha256(func_code.strip().encode("utf-8")).hexdigest()


def _cache_hit(cache_dir: Path, key: str, dest: Path) -> bool:
    """If a cached CPG exists for *key*, copy it to *dest* and return True."""
    cached = cache_dir / key
    if cached.exists() and any(cached.iterdir()):
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(cached, dest)
        return True
    return False


def _cache_store(cache_dir: Path, key: str, source_dir: Path) -> None:
    """Copy an exported CPG directory into the cache."""
    cached = cache_dir / key
    if cached.exists():
        shutil.rmtree(cached)
    shutil.copytree(source_dir, cached)


# ---------------------------------------------------------------------------
# Per-sample worker (top-level so it's picklable for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_sample(
    sample_id: str,
    func_code: str,
    output_dir: Path,
    cache_dir: Path,
) -> str:
    """Process a single sample. Returns one of 'cached', 'processed', 'skipped', 'failed'."""
    sample_dir = output_dir / sample_id

    # Already in output directory
    if sample_dir.exists() and any(sample_dir.iterdir()):
        return "skipped"

    if not isinstance(func_code, str) or not func_code.strip():
        logger.warning("Sample %s has empty function code, skipping", sample_id)
        return "failed"

    key = _cache_key(func_code)

    # Try cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    if _cache_hit(cache_dir, key, sample_dir):
        return "cached"

    # Run Joern
    try:
        _run_joern(func_code, sample_dir)
    except Exception as exc:
        logger.error("Joern failed for sample %s: %s", sample_id, exc)
        if sample_dir.exists():
            shutil.rmtree(sample_dir)
        return "failed"

    # Store in cache
    _cache_store(cache_dir, key, sample_dir)
    return "processed"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def export_cpgs(
    subset_csv: str | None = None,
    output_dir: str | None = None,
) -> None:
    _ensure_dependencies()

    import pandas as pd

    src = subset_csv or str(SUBSET_CSV)
    dst = Path(output_dir) if output_dir else RAW_DIR
    dst.mkdir(parents=True, exist_ok=True)

    cache_dir = CPG_CACHE_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    total = len(df)

    # Build work items
    items = [
        (str(row["sample_id"]), row.get("func_before", ""))
        for _, row in df.iterrows()
    ]

    counters = {"processed": 0, "cached": 0, "skipped": 0, "failed": 0}

    with ProcessPoolExecutor(max_workers=JOERN_WORKERS) as executor:
        futures = {
            executor.submit(_process_sample, sid, code, dst, cache_dir): sid
            for sid, code in items
        }

        with tqdm(total=total, desc="Exporting CPGs", unit="sample") as pbar:
            for future in as_completed(futures):
                sid = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    logger.error("Unexpected error for sample %s: %s", sid, exc)
                    result = "failed"
                counters[result] += 1
                pbar.set_postfix(**counters)
                pbar.update(1)

    logger.info(
        "CPG export complete: %d processed, %d cached, %d skipped (existing), %d failed out of %d total",
        counters["processed"],
        counters["cached"],
        counters["skipped"],
        counters["failed"],
        total,
    )


def _ensure_dependencies() -> None:
    try:
        import pandas as pd  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "joern_export requires pandas. Install project dependencies first, "
            "for example with `python3 -m pip install -r requirements.txt`."
        ) from exc

    missing = [name for name in ("joern-parse", "joern-export") if _resolve_joern_tool(name) is None]
    if missing:
        raise RuntimeError(
            "Missing Joern CLI tools on PATH: "
            + ", ".join(missing)
            + ". Install Joern and ensure its binaries are available."
        )

    joern_env = _build_joern_env()

    try:
        subprocess.run(
            [_resolve_joern_tool("joern-parse"), "--help"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=joern_env,
        )
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "joern-parse failed during preflight. "
            f"stdout={exc.stdout.strip() or '<empty>'}; stderr={exc.stderr.strip() or '<empty>'}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"joern-parse preflight timed out after {exc.timeout} seconds") from exc


def _resolve_joern_tool(name: str) -> str | None:
    direct = JOERN_HOME / "bin" / name
    if direct.exists():
        return str(direct)

    path_tool = shutil.which(name)
    if path_tool:
        return path_tool

    return None


def _build_joern_env() -> Mapping[str, str]:
    env = os.environ.copy()

    if _java_available(env):
        return env

    brew_java = HOMEBREW_OPENJDK_HOME / "bin" / "java"
    if brew_java.exists():
        env["JAVA_HOME"] = str(HOMEBREW_OPENJDK_HOME)
        env["PATH"] = f"{HOMEBREW_OPENJDK_HOME / 'bin'}:{env.get('PATH', '')}"
        if _java_available(env):
            return env

    raise RuntimeError(
        "Joern requires a working Java runtime. Install a JRE/JDK or expose "
        "Homebrew OpenJDK via JAVA_HOME/PATH."
    )


def _java_available(env: Mapping[str, str]) -> bool:
    try:
        subprocess.run(
            ["java", "-version"],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
            env=env,
        )
        return True
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def _run_joern(func_code: str, output_dir: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
        f.write(func_code)
        tmp_path = f.name
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        cpg_path = f.name

    if output_dir.exists():
        shutil.rmtree(output_dir)

    try:
        joern_env = _build_joern_env()

        # Step 1: Parse the file into a CPG
        subprocess.run(
            [_resolve_joern_tool("joern-parse"), tmp_path, "-o", cpg_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env=joern_env,
        )

        # Step 2: Export the CPG
        subprocess.run(
            [_resolve_joern_tool("joern-export"), cpg_path, "--repr", "cpg14", "--out", str(output_dir)],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env=joern_env,
        )

        # Clean up the binary CPG file
        bin_path = Path(cpg_path)
        if bin_path.exists():
            bin_path.unlink()

    except subprocess.CalledProcessError as exc:
        stdout = exc.stdout.strip() if exc.stdout else ""
        stderr = exc.stderr.strip() if exc.stderr else ""
        raise RuntimeError(
            f"{exc.cmd[0]} failed with exit code {exc.returncode}. "
            f"stdout={stdout or '<empty>'}; stderr={stderr or '<empty>'}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{exc.cmd[0]} timed out after {exc.timeout} seconds") from exc
    finally:
        os.unlink(tmp_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export CPGs via Joern CLI")
    parser.add_argument("--input", type=str, default=None, help="Path to subset CSV")
    parser.add_argument("--output", type=str, default=None, help="Output directory for CPG exports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    export_cpgs(subset_csv=args.input, output_dir=args.output)
