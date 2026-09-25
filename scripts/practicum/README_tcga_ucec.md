# TCGA-UCEC main-stage training

Run commands from the DeePathNet repository root. Use the preprocessing script
with the practicum Python environment; use deepathnet_env for training.

## Data location and Git policy

Raw TCGA-UCEC files are intentionally stored outside this repository. They are
currently located at:

```text
/Users/yinglin/projects/practicum/data/tcga_ucec_gdc
```

Do not commit raw GDC files, generated matrices, model checkpoints, or complete
training directories. The repository tracks the preparation/training scripts,
configuration, tests, and small result summaries needed to reproduce the work.

The raw directory contains:

```text
clinical/             GDC clinical and biospecimen exports
metadata/             RNA manifests, file IDs, sample sheet, and checksums
rna_star_counts/      per-sample STAR-Counts TSV files
cnv_ascat2/           open ASCAT2 gene-level CNV files and download manifest
```

Prepare the RNA inputs by explicitly supplying the external raw-data location:

```bash
python scripts/practicum/prepare_tcga_ucec_deepathnet_inputs.py \
  --gdc-root /Users/yinglin/projects/practicum/data/tcga_ucec_gdc
```

CNV files can be queried and downloaded reproducibly with:

```bash
python scripts/practicum/download_tcga_ucec_cnv.py \
  --output-dir /Users/yinglin/projects/practicum/data/tcga_ucec_gdc/cnv_ascat2
```

Generated model-ready files are written under `data/processed/`, and training
outputs are written under `work_dirs/`. Both directories are ignored by Git.
The compact, non-patient-level result summary intended for version control is
available at `results/practicum/tcga_ucec/README.md`.

## Input and split

The existing train files contain 376 development samples. The training entrypoint
splits them into 300 training and 76 validation samples, stratified by stage with
seed 1. The original 94 test samples remain held out. Each patient must have one
selected sample, and any patient overlap causes an error.

Labels are Stage I, Stage II, Stage III and Stage IV. Substages are collapsed in
stage_main; the original figo_stage remains in the candidate-label table.
All 3,012 RNA features overlap the pathway file, so cancer_only is true to avoid
an empty non-pathway branch. This flag refers to pathway membership, not a
restriction to a subset of cancer patients.

## Smoke test

```bash
/opt/anaconda3/envs/deepathnet_env/bin/python scripts/practicum/train_tcga_ucec_stage.py configs/practicum/tcga_ucec_rna/deepathnet_tcga_ucec_stage_main_rna.json --smoke-test
```

This uses the configured full model, two training batches of four samples and
all 76 validation samples. It saves and reloads a checkpoint, but does not run
test predictions. Smoke-test metrics are diagnostic only.

## Full training

```bash
/opt/anaconda3/envs/deepathnet_env/bin/python scripts/practicum/train_tcga_ucec_stage.py configs/practicum/tcga_ucec_rna/deepathnet_tcga_ucec_stage_main_rna.json
```

Normal runs use the configured 100 epochs and batch size 64. Optional --epochs
and --batch-size override these values. Use validation Macro-F1 to select the
best epoch; the first epoch always saves a checkpoint. The selected checkpoint
is reloaded before a single final test evaluation. Do not choose subsequent
hyperparameters based on the held-out test results.

## Outputs

Each invocation creates a distinct smoke_* or train_* directory under
work_dirs/practicum/DeePathNet/tcga_ucec_stage_main_rna/.

- split_assignments.csv: sample ID, patient ID, split and stage.
- run_metadata.json: effective configuration, class mapping, feature order,
  pathway membership, input hashes, parameter count and environment.
- history.csv and training.log: training loss, validation metrics and timings.
- best_model.pth: state dictionary selected using validation Macro-F1.
- validation_predictions.csv: sample IDs, true/predicted stages and probabilities
  at the selected epoch; matching metrics, classification report and confusion
  matrix are saved alongside it.
- test_predictions.csv and matching reports: normal runs only, after selection.
- run_summary.json: selected epoch, runtime and whether test was evaluated.

The confusion matrix uses true stages as rows and predicted stages as columns.
Macro-F1 includes all four classes; ROC-AUC uses macro one-versus-one averaging.
Model probabilities are not calibrated clinical probabilities. Stage IV has
only 3 validation and 4 test samples, so its metrics are particularly unstable.
Training does not automatically run pathway attribution.

This entrypoint replaces deepathnet_independent_test.py for this task, since
that older entrypoint selects checkpoints using the provided test set.
