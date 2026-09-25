#!/usr/bin/env python3
"""Download patient-matched open GDC ASCAT2 gene-level CNV files for TCGA-UCEC.

The script deliberately keeps raw GDC downloads outside ``data/processed`` and
does not read test labels.  It uses the existing four-stage target file only to
select CNV files from the same 470 RNA patients.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECTS_ROOT = REPO_ROOT.parent
GDC_FILES_URL = "https://api.gdc.cancer.gov/files"
GDC_DATA_URL = "https://api.gdc.cancer.gov/data"
WORKFLOW = "ASCAT2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label-file",
        type=Path,
        default=REPO_ROOT / "data/processed/tcga_ucec_rna/target_file.csv",
        help="Existing RNA+stage target table; used only for patient matching.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECTS_ROOT / "practicum/data/tcga_ucec_gdc/cnv_ascat2",
        help="Directory for raw GDC CNV files and the selected manifest.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Write the selected manifest without downloading data files.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def md5(path: Path) -> str:
    digest = hashlib.md5()  # nosec B324: GDC publishes MD5 for file integrity.
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_manifest() -> list[dict[str, object]]:
    payload = {
        "filters": {
            "op": "and",
            "content": [
                {"op": "=", "content": {"field": "cases.project.project_id", "value": "TCGA-UCEC"}},
                {"op": "=", "content": {"field": "data_type", "value": "Gene Level Copy Number"}},
                {"op": "=", "content": {"field": "analysis.workflow_type", "value": WORKFLOW}},
                {"op": "=", "content": {"field": "access", "value": "open"}},
            ],
        },
        "format": "JSON",
        "fields": (
            "file_id,file_name,cases.case_id,cases.submitter_id,"
            "cases.samples.submitter_id,data_type,data_format,"
            "analysis.workflow_type,file_size,md5sum"
        ),
        "size": 5000,
    }
    request = Request(
        GDC_FILES_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=60) as response:  # nosec B310: fixed official API URL.
        result = json.load(response)
    hits = result["data"]["hits"]
    if result["data"]["pagination"]["total"] != len(hits):
        raise RuntimeError("GDC manifest did not return every requested ASCAT2 file")
    return hits


def select_patient_matched_files(
    manifest: list[dict[str, object]], label_file: Path
) -> tuple[pd.DataFrame, list[str]]:
    labels = pd.read_csv(label_file, usecols=["Cell_line"])
    if labels.Cell_line.duplicated().any():
        raise ValueError("RNA target file has duplicate sample IDs")
    sample_by_patient = {
        str(sample)[:12]: str(sample) for sample in labels["Cell_line"].tolist()
    }
    if len(sample_by_patient) != len(labels):
        raise ValueError("RNA target file must have one sample per patient")

    candidates: dict[str, list[dict[str, object]]] = {}
    for item in manifest:
        cases = item.get("cases", [])
        if len(cases) != 1 or "submitter_id" not in cases[0]:
            raise ValueError(f"Unexpected GDC case metadata for {item.get('file_id')}")
        patient_id = str(cases[0]["submitter_id"])
        candidates.setdefault(patient_id, []).append(item)

    rows: list[dict[str, object]] = []
    missing_patients: list[str] = []
    for patient_id, sample_id in sorted(sample_by_patient.items()):
        patient_candidates = candidates.get(patient_id, [])
        if not patient_candidates:
            missing_patients.append(patient_id)
            continue

        def sort_key(item: dict[str, object]) -> tuple[int, str, str]:
            samples = item["cases"][0].get("samples", [])
            sample_ids = {str(sample["submitter_id"]) for sample in samples}
            exact_sample_match = sample_id in sample_ids
            return (0 if exact_sample_match else 1, str(item["file_name"]), str(item["file_id"]))

        selected = sorted(patient_candidates, key=sort_key)[0]
        samples = selected["cases"][0].get("samples", [])
        selected_sample_ids = ";".join(sorted(str(sample["submitter_id"]) for sample in samples))
        rows.append(
            {
                "patient_id": patient_id,
                "rna_sample_id": sample_id,
                "file_id": selected["file_id"],
                "file_name": selected["file_name"],
                "md5sum": selected["md5sum"],
                "file_size": selected["file_size"],
                "workflow_type": selected["analysis"]["workflow_type"],
                "associated_sample_ids": selected_sample_ids,
                "candidate_files_for_patient": len(patient_candidates),
                "exact_rna_sample_match": sample_id in selected_sample_ids.split(";"),
            }
        )
    return pd.DataFrame(rows), missing_patients


def download_one(row: dict[str, object], files_dir: Path, retries: int) -> str:
    file_id = str(row["file_id"])
    destination = files_dir / f"{file_id}.tsv"
    expected_md5 = str(row["md5sum"])
    if destination.is_file() and md5(destination) == expected_md5:
        return "already_verified"

    temporary = destination.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            request = Request(f"{GDC_DATA_URL}/{file_id}")
            with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:  # nosec B310
                while block := response.read(1024 * 1024):
                    handle.write(block)
            if md5(temporary) != expected_md5:
                temporary.unlink(missing_ok=True)
                raise ValueError("MD5 mismatch")
            temporary.replace(destination)
            return "downloaded"
        except (HTTPError, URLError, OSError, TimeoutError, ValueError) as error:
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"{file_id} failed after {retries} attempts: {error}") from error
    raise AssertionError("unreachable")


def main() -> None:
    args = parse_args()
    if args.workers < 1 or args.retries < 1:
        raise ValueError("workers and retries must be positive")
    if not args.label_file.is_file():
        raise FileNotFoundError(args.label_file)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest()
    selected, missing_patients = select_patient_matched_files(manifest, args.label_file)
    if selected.empty:
        raise RuntimeError("No GDC ASCAT2 CNV files matched the RNA target table")
    selected.to_csv(args.output_dir / "selected_manifest.tsv", sep="\t", index=False)
    (args.output_dir / "missing_rna_patients_without_cnv.txt").write_text(
        "\n".join(missing_patients) + "\n"
    )

    metadata = {
        "gdc_project": "TCGA-UCEC",
        "data_type": "Gene Level Copy Number",
        "workflow_type": WORKFLOW,
        "rna_target_file": str(args.label_file.resolve()),
        "rna_target_sha256": sha256(args.label_file),
        "manifest_files": len(manifest),
        "matched_rna_patients": len(selected),
        "missing_rna_patients": len(missing_patients),
        "downloaded_test_labels": False,
    }
    (args.output_dir / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Matched {len(selected)} RNA patients; {len(missing_patients)} have no {WORKFLOW} CNV file.",
        flush=True,
    )
    if args.manifest_only:
        return

    files_dir = args.output_dir / "files"
    files_dir.mkdir(exist_ok=True)
    statuses: list[str] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(download_one, row, files_dir, args.retries)
            for row in selected.to_dict("records")
        ]
        for index, future in enumerate(as_completed(futures), start=1):
            statuses.append(future.result())
            if index % 25 == 0 or index == len(futures):
                print(f"Verified {index}/{len(futures)} CNV files", flush=True)
    metadata["download_status_counts"] = pd.Series(statuses).value_counts().to_dict()
    (args.output_dir / "download_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
