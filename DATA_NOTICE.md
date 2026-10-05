# Data Notice

This repository releases code only.

The original project used clinical CBCT scans, panoramic X-ray images derived from CBCT, tooth masks, mesh labels, and intermediate preprocessing outputs. These assets may contain private medical information or may be governed by institutional access rules. They are therefore not redistributed in this repository.

## Not Included

- raw CBCT volumes;
- panoramic X-ray images generated from private CBCT data;
- tooth masks and label images;
- STL/PLY/OBJ mesh labels;
- patient or case identifiers;
- model checkpoints;
- training logs, visualization outputs, and experiment tracker files.

## Anonymization

For public release, local paths and case identifiers were replaced with placeholders:

- `xxx` denotes a private local path or project-specific data root;
- `CASE_ID` denotes an anonymized case identifier.

Users should replace these placeholders with their own local data paths after obtaining appropriate data access.

## Intended Use

The released code is intended for research reproduction, method inspection, and extension. It is not intended for clinical diagnosis or direct deployment in medical workflows.
