"""Small artifact identities for reproducible runs and checked exports."""
import hashlib
import json
from pathlib import Path
import subprocess


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path):
    digest = hashlib.sha256()
    root = Path(path)
    for item in sorted(root.rglob("*")):
        if item.is_file():
            digest.update(str(item.relative_to(root)).encode())
            digest.update(file_sha256(item).encode())
    return digest.hexdigest()


def code_identity():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    source_files = [
        *root.glob("graph_vul/**/*.py"),
        *root.glob("scripts/*.py"),
        *root.glob("configs/*.json"),
        root / "requirements.txt",
    ]
    for path in sorted(source_files):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return {
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_sha256": digest.hexdigest(),
        "dirty": bool(
            subprocess.check_output(
                [
                    "git",
                    "status",
                    "--porcelain",
                    "--",
                    "graph_vul",
                    "scripts",
                    "configs",
                    "requirements.txt",
                ],
                text=True,
            )
        ),
    }


def preparation_identity(root):
    root = Path(root)
    manifest = json.loads((root / "preparation.json").read_text())
    for name, expected in manifest["hashes"].items():
        if file_sha256(root / name) != expected:
            raise ValueError(f"Preparation artifact changed: {name}")
    for name, expected in manifest.get("directories", {}).items():
        if directory_sha256(root / name) != expected:
            raise ValueError(f"Preparation artifact directory changed: {name}")
    return manifest


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
