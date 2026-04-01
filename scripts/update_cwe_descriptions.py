"""Update CWE descriptions from dataset CWEs using the MITRE CWE REST API.

The script reads unique `cwe_id` values from a CSV file, fetches each CWE's
official description text from MITRE, and writes a JSON mapping like:

    {
      "CWE-79": "The product does not neutralize ...",
      ...
    }

MITRE's API uses different endpoints for weakness/category/view entries, so the
script first resolves the entry type via `/api/v1/cwe/{id}` and then follows
the corresponding content endpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


API_ROOT = "https://cwe-api.mitre.org/api/v1"
DEFAULT_DATASET_CANDIDATES = (
    Path("data/func_subset.csv"),
    Path("data/subset.csv"),
    Path("data/bigvul.csv"),
)
DEFAULT_OUTPUT_PATH = Path("data/cwe_descriptions.json")
SPECIAL_CWE_DESCRIPTIONS = {
    "NVD-CWE-Other": "Fallback label used by NVD when a vulnerability does not map cleanly to a more specific CWE entry.",
    "NVD-CWE-noinfo": "Fallback label used by NVD when insufficient information is available to assign a specific CWE entry.",
}
TYPE_TO_ENDPOINT = {
    "category": "category",
    "view": "view",
}
TYPE_TO_CONTAINER = {
    "category": "Categories",
    "view": "Views",
}
DESCRIPTION_KEYS = ("Description", "Summary")
_CWE_ID_RE = re.compile(r"^CWE-(\d+)$")


def _http_get_json(url: str, timeout: float) -> object:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _normalize_cwe_id(raw: str) -> str | None:
    value = raw.strip()
    if not value:
        return None
    match = _CWE_ID_RE.fullmatch(value)
    if match is None:
        return None
    return f"CWE-{int(match.group(1))}"


def _load_dataset_cwes(dataset_path: Path) -> list[str]:
    with dataset_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "cwe_id" not in reader.fieldnames:
            raise ValueError(f"{dataset_path} does not contain a cwe_id column")

        cwe_ids = set()
        for row in reader:
            raw_value = (row.get("cwe_id", "") or "").strip()
            if not raw_value:
                continue
            normalized = _normalize_cwe_id(raw_value)
            if normalized is not None:
                cwe_ids.add(normalized)
                continue
            if raw_value in SPECIAL_CWE_DESCRIPTIONS:
                cwe_ids.add(raw_value)

    return sorted(cwe_ids, key=_sort_key)


def _sort_key(cwe_id: str) -> tuple[int, str]:
    match = _CWE_ID_RE.fullmatch(cwe_id)
    number = int(match.group(1)) if match else 10**9
    return (number, cwe_id)


def _resolve_entry_type(cwe_number: int, timeout: float) -> str:
    payload = _http_get_json(f"{API_ROOT}/cwe/{cwe_number}", timeout)
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"Unexpected type payload for CWE-{cwe_number}: {payload!r}")

    entry = payload[0]
    if not isinstance(entry, dict) or "Type" not in entry:
        raise ValueError(f"Missing Type for CWE-{cwe_number}: {entry!r}")

    entry_type = str(entry["Type"]).strip().lower()
    if not entry_type:
        raise ValueError(f"Empty Type for CWE-{cwe_number}")
    return entry_type


def _fetch_entry_payload(cwe_number: int, entry_type: str, timeout: float) -> dict:
    endpoint = TYPE_TO_ENDPOINT.get(entry_type, "weakness")
    container = TYPE_TO_CONTAINER.get(entry_type, "Weaknesses")
    payload = _http_get_json(f"{API_ROOT}/cwe/{endpoint}/{cwe_number}", timeout)

    if not isinstance(payload, dict) or container not in payload:
        raise ValueError(
            f"Unexpected content payload for CWE-{cwe_number} ({entry_type}): {payload!r}"
        )

    items = payload[container]
    if not isinstance(items, list) or not items:
        raise ValueError(f"Empty {container} payload for CWE-{cwe_number}")

    entry = items[0]
    if not isinstance(entry, dict):
        raise ValueError(f"Unexpected entry payload for CWE-{cwe_number}: {entry!r}")
    return entry


def _extract_description(entry: dict, cwe_id: str) -> str:
    for key in DESCRIPTION_KEYS:
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"No description field found for {cwe_id}")


def fetch_cwe_description(cwe_id: str, timeout: float = 30.0, retries: int = 3) -> str:
    cwe_number = int(cwe_id.split("-", 1)[1])
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            entry_type = _resolve_entry_type(cwe_number, timeout)
            entry = _fetch_entry_payload(cwe_number, entry_type, timeout)
            return _extract_description(entry, cwe_id)
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(0.5 * attempt)

    raise RuntimeError(f"Failed to fetch {cwe_id}: {last_error}") from last_error


def _fallback_description(cwe_id: str, reason: str | None = None) -> str:
    if cwe_id in SPECIAL_CWE_DESCRIPTIONS:
        return SPECIAL_CWE_DESCRIPTIONS[cwe_id]
    suffix = f" MITRE API lookup failed: {reason}." if reason else ""
    return f"Placeholder description for {cwe_id}. This CWE appeared in the dataset but an official MITRE description could not be fetched automatically.{suffix}"


def _default_dataset_path() -> Path:
    for candidate in DEFAULT_DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in DEFAULT_DATASET_CANDIDATES)
    raise FileNotFoundError(f"No dataset CSV found. Checked: {joined}")


def update_cwe_descriptions(
    dataset_path: Path,
    output_path: Path,
    timeout: float = 30.0,
    retries: int = 3,
) -> dict[str, str]:
    cwe_ids = _load_dataset_cwes(dataset_path)
    descriptions: dict[str, str] = {}
    if output_path.exists():
        with output_path.open() as handle:
            existing = json.load(handle)
        if isinstance(existing, dict):
            descriptions.update({str(k): str(v) for k, v in existing.items() if str(v).strip()})

    for cwe_id in cwe_ids:
        if cwe_id in descriptions:
            continue
        if cwe_id in SPECIAL_CWE_DESCRIPTIONS:
            descriptions[cwe_id] = SPECIAL_CWE_DESCRIPTIONS[cwe_id]
            continue
        try:
            descriptions[cwe_id] = fetch_cwe_description(cwe_id, timeout=timeout, retries=retries)
        except RuntimeError as exc:
            descriptions[cwe_id] = _fallback_description(cwe_id, str(exc))

    descriptions = {
        cwe_id: descriptions[cwe_id]
        for cwe_id in sorted(descriptions.keys(), key=_sort_key)
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(descriptions, handle, indent=2, sort_keys=False)
        handle.write("\n")

    return descriptions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch official CWE descriptions from MITRE for all CWEs in a dataset CSV"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="CSV path containing a cwe_id column (default: first existing of func_subset/subset/bigvul)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSON path",
    )
    parser.add_argument("--timeout", type=float, default=30.0, help="Per-request timeout in seconds")
    parser.add_argument("--retries", type=int, default=3, help="Retries per CWE request")
    args = parser.parse_args()

    dataset_path = args.dataset or _default_dataset_path()
    descriptions = update_cwe_descriptions(
        dataset_path=dataset_path,
        output_path=args.output,
        timeout=args.timeout,
        retries=args.retries,
    )
    print(
        f"Wrote {len(descriptions)} CWE descriptions from {dataset_path} to {args.output}"
    )


if __name__ == "__main__":
    main()
