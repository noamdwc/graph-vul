"""Export Code Property Graphs for each sample using the Joern CLI."""

import logging
import os
import shutil
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path

from graph_vul.config import SUBSET_CSV, RAW_DIR
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
JOERN_HOME = Path("/opt/joern/joern-cli")
HOMEBREW_OPENJDK_HOME = Path("/opt/homebrew/opt/openjdk/libexec/openjdk.jdk/Contents/Home")


def export_cpgs(
    subset_csv: str | None = None,
    output_dir: str | None = None,
) -> None:
    _ensure_dependencies()

    import pandas as pd

    src = subset_csv or str(SUBSET_CSV)
    dst = Path(output_dir) if output_dir else RAW_DIR
    dst.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    total = len(df)
    skipped = 0
    failed = 0

    progress = tqdm(
        df.iterrows(),
        total=total,
        desc="Exporting CPGs",
        unit="sample",
    )

    for idx, row in progress:
        sample_id = str(row["sample_id"])
        sample_dir = dst / sample_id

        # Idempotency: skip if export already exists and is non-empty
        if sample_dir.exists() and any(sample_dir.iterdir()):
            skipped += 1
            progress.set_postfix(processed=idx + 1 - skipped - failed, skipped=skipped, failed=failed)
            continue

        func_code = row.get("func_before", "")
        if not isinstance(func_code, str) or not func_code.strip():
            logger.warning("Sample %s has empty function code, skipping", sample_id)
            failed += 1
            progress.set_postfix(processed=idx + 1 - skipped - failed, skipped=skipped, failed=failed)
            continue

        try:
            _run_joern(func_code, sample_dir)
        except Exception as exc:
            logger.exception("Joern failed for sample %s: %s", sample_id, exc)
            # Clean up partial output
            if sample_dir.exists():
                shutil.rmtree(sample_dir)
            raise RuntimeError(f"Joern export failed for sample {sample_id}") from exc

        progress.set_postfix(processed=idx + 1 - skipped - failed, skipped=skipped, failed=failed)

    processed = total - skipped - failed
    logger.info(
        "CPG export complete: %d processed, %d skipped (existing), %d failed out of %d total",
        processed,
        skipped,
        failed,
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
        parse_proc = subprocess.run(
            [_resolve_joern_tool("joern-parse"), tmp_path, "-o", cpg_path],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
            env=joern_env,
        )

        # Step 2: Export the CPG
        export_proc = subprocess.run(
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
