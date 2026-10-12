"""Build the inference catalog from the fixed MITRE CWE release, never dataset labels."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

ARCHIVE_URL = "https://cwe.mitre.org/data/xml/cwec_v4.20.xml.zip"
ARCHIVE_SHA256 = "3976f599e5e5200219a3108bb896d06e2a88fbb293369e1883cb423a5e9d7d50"


def build_catalog(archive_path, output_path):
    archive_path = Path(archive_path)
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise ValueError("Archive is not the fixed MITRE CWE 4.20 release")
    with zipfile.ZipFile(archive_path) as archive:
        root = ET.fromstring(archive.read("cwec_v4.20.xml"))
    namespace = {"cwe": "http://cwe.mitre.org/cwe-7"}
    descriptions = {}
    counts = {}
    # Categories are retained because they are valid coarse CWE annotation targets.
    # Include every entry, including deprecated ones; views organize the taxonomy and are excluded.
    for group, entry_tag, description_tag in [
        ("Weaknesses", "Weakness", "Description"),
        ("Categories", "Category", "Summary"),
    ]:
        entries = root.findall(f"cwe:{group}/cwe:{entry_tag}", namespace)
        counts[group] = len(entries)
        for entry in entries:
            description = entry.find(f"cwe:{description_tag}", namespace)
            if description is None:
                raise ValueError(f'Missing description for CWE-{entry.attrib["ID"]}')
            text = " ".join(" ".join(description.itertext()).split())
            if not text:
                raise ValueError("CWE descriptions must be nonempty")
            descriptions["CWE-" + entry.attrib["ID"]] = (
                entry.attrib["Name"] + ". " + text
            )
    catalog = {
        "version": "mitre-cwe-4.20-weaknesses-categories-v1",
        "source": {
            "url": ARCHIVE_URL,
            "archive_sha256": digest,
            "release": root.attrib["Version"],
            "date": root.attrib["Date"],
            "scope": counts,
        },
        "copyright": "Copyright © 2006–2026, The MITRE Corporation.",
        "license": Path(__file__)
        .resolve()
        .parents[1]
        .joinpath("configs/CWE_LICENSE.txt")
        .read_text(),
        "descriptions": dict(sorted(descriptions.items())),
    }
    Path(output_path).write_text(
        json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    )
    return catalog


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, required=True, help="Downloaded CWE 4.20 XML ZIP"
    )
    parser.add_argument(
        "--output", type=Path, default=Path("configs/cwe_catalog_v1.json")
    )
    args = parser.parse_args()
    catalog = build_catalog(args.archive, args.output)
    print(f"Catalog: {len(catalog['descriptions'])} fixed CWE entries")
