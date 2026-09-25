"""Audit public molecular subtype labels and prepare a separate RNA cohort.

Run after downloading the cBioPortal PanCancer Atlas patient and sample tables.
No stage label is required, no missing subtype is imputed, and no model is fit.
"""
import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/ucec_tcga_pan_can_atlas_2018/"
LABELS = {
    "UCEC_CN_HIGH": "CN_HIGH", "UCEC_CN_LOW": "CN_LOW",
    "UCEC_MSI": "MSI", "UCEC_POLE": "POLE",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patient-file", type=Path, required=True)
    parser.add_argument("--sample-file", type=Path, required=True)
    args = parser.parse_args()
    out = ROOT / "data/processed/tcga_ucec_subtypes"
    raw = out / "source"
    raw.mkdir(parents=True, exist_ok=True)
    provenance = []
    for source, name in [(args.patient_file, "data_clinical_patient.txt"),
                         (args.sample_file, "data_clinical_sample.txt")]:
        target = raw / name
        if source.resolve() != target.resolve():
            shutil.copyfile(source, target)
        provenance.append({"file": name, "url": SOURCE + name,
                           "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})
    patients = pd.read_csv(raw / "data_clinical_patient.txt", sep="\t", comment="#")
    if patients.PATIENT_ID.duplicated().any():
        raise ValueError("Duplicate source patients require manual review")
    unknown = set(patients.SUBTYPE.dropna()) - set(LABELS)
    if unknown:
        raise ValueError("Unexpected subtype labels: " + repr(unknown))
    original = ROOT / "data/processed/tcga_ucec_rna"
    samples = pd.read_csv(original / "selected_sample_sheet.tsv", sep="\t")
    if samples.case_submitter_id.duplicated().any() or samples.sample_submitter_id.duplicated().any():
        raise ValueError("Expected one selected sample per patient")
    audit = samples[["sample_submitter_id", "case_submitter_id"]].rename(
        columns={"sample_submitter_id": "Cell_line"})
    audit = audit.merge(patients[["PATIENT_ID", "SUBTYPE"]], how="left",
                        left_on="case_submitter_id", right_on="PATIENT_ID", validate="one_to_one")
    audit["molecular_subtype"] = audit.SUBTYPE.map(LABELS)
    audit["label_status"] = "labeled"
    audit.loc[audit.SUBTYPE.isna(), "label_status"] = "source_subtype_missing"
    audit.loc[audit.PATIENT_ID.isna(), "label_status"] = "patient_not_in_source"
    stage = pd.read_csv(original / "tcga_ucec_labels_all_candidates.tsv", sep="\t")
    audit = audit.merge(stage[["Cell_line", "stage_main", "tumor_grade"]], on="Cell_line", validate="one_to_one")
    cnv_path = ROOT / "data/processed/tcga_ucec_rna_cnv/data_file_all_matched_samples.csv"
    cnv_ids = set(pd.read_csv(cnv_path, usecols=[0]).iloc[:, 0])
    audit["in_existing_rna_cnv_matrix"] = audit.Cell_line.isin(cnv_ids)
    audit.to_csv(out / "patient_label_audit.tsv", sep="\t", index=False)
    selected = audit[audit.molecular_subtype.notna()].set_index("Cell_line")
    expression = pd.read_csv(original / "data_file_all_selected_samples.csv", index_col=0)
    if not expression.index.is_unique or not expression.columns.is_unique:
        raise ValueError("Duplicate expression identifiers")
    expression.loc[selected.index].to_csv(out / "data_file.csv", index_label="Cell_line")
    selected[["molecular_subtype"]].rename(columns={"molecular_subtype": "Cancer_type"}).to_csv(out / "target_file.csv")
    counts = selected.molecular_subtype.value_counts().rename_axis("subtype").to_frame("rna_patients")
    counts["existing_rna_cnv_patients"] = selected[selected.in_existing_rna_cnv_matrix].molecular_subtype.value_counts().reindex(counts.index, fill_value=0)
    counts.to_csv(out / "class_counts.csv")
    pd.crosstab(selected.molecular_subtype, selected.stage_main.fillna("Missing_stage")).to_csv(out / "subtype_by_stage.csv")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "source": provenance,
               "source_patients": len(patients), "source_labeled": int(patients.SUBTYPE.notna().sum()),
               "rna_patients": len(audit), "matched_labeled": len(selected),
               "label_status": audit.label_status.value_counts().to_dict(),
               "rna_genes": expression.shape[1], "classes": counts.to_dict("index"),
               "stage_missing_but_subtype_available": int(selected.stage_main.isna().sum()),
               "split_created": False, "model_trained": False,
               "notes": ["Existing CNV matrix was restricted to stage-labeled patients; absence here does not mean GDC has no CNV.",
                         "Cancer_type is the loader-compatible column name; values are molecular subtypes.",
                         "Subtype definitions use genomic features. CNV prediction of CN-based labels needs explicit interpretation.",
                         "Plan RNA-only as primary screen. Reserve a new patient-level stratified test set before fitting; fit preprocessing within training folds."]}
    (out / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
