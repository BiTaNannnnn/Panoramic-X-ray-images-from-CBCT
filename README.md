# 🦷 PX2Tooth

<p align="center">
  <b>Reconstructing 3D tooth point clouds from a single panoramic X-ray</b>
</p>

<p align="center">
  <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2411.03725-b31b1b">
  <img alt="Task" src="https://img.shields.io/badge/Task-2D%20Panoramic%20X--ray%20%E2%86%92%203D%20Teeth-blue">
  <img alt="Framework" src="https://img.shields.io/badge/Framework-PyTorch-orange">
  <img alt="Code" src="https://img.shields.io/badge/Release-Code%20Only-green">
  <img alt="License" src="https://img.shields.io/badge/License-MIT-purple">
</p>

PX2Tooth is a cleaned research-code release for the paper:

> **PX2Tooth: Reconstructing the 3D Point Cloud Teeth from a Single Panoramic X-ray**  
> arXiv: [2411.03725](https://arxiv.org/abs/2411.03725)

The project studies a dental reconstruction pipeline that first extracts tooth structures from panoramic X-ray images and then reconstructs tooth-level 3D point clouds from the 2D panoramic input.

---

## ✨ Highlights

| Feature | Description |
| --- | --- |
| 🦷 Panoramic-to-3D reconstruction | Reconstructs 3D tooth point clouds from a single panoramic X-ray input. |
| 🧩 Two-stage pipeline | Combines 2D tooth segmentation with PointNet-style 3D point-cloud generation. |
| 🩻 CBCT-derived preprocessing | Includes scripts for generating panoramic representations and labels from CBCT/mesh data. |
| 🧠 Segmentation + generation code | Provides U-Net-style segmentation and 3D generation modules in PyTorch. |
| 🔐 Privacy-aware release | Releases code only; clinical images, masks, meshes, checkpoints, and logs are not redistributed. |

---

## 🧭 Pipeline Overview

```text
CBCT volume / dental mesh
        │
        ▼
Panoramic projection + arch-curve preprocessing
        │
        ▼
2D panoramic tooth segmentation
        │
        ▼
Tooth-level feature extraction
        │
        ▼
3D tooth point-cloud reconstruction
        │
        ▼
Evaluation with segmentation and point-cloud metrics
```

The released code is organized around two main stages:

| Stage | Main Goal | Representative Files |
| --- | --- | --- |
| 🛠️ Preprocessing | Generate panoramic projections, tooth masks, arch curves, target labels, and mesh/point-cloud inputs | `data_process/*.py` |
| 🖼️ 2D Segmentation | Segment tooth regions from panoramic X-ray images | `px2tooth/train_seg.py`, `px2tooth/test_Unet.py`, `px2tooth/unet/` |
| 🧬 3D Generation | Reconstruct tooth-level point clouds from panoramic features | `px2tooth/train_all.py`, `px2tooth/test_all.py`, `px2tooth/PointNet_3d/` |
| 📏 Evaluation | Compute segmentation and 3D reconstruction quality metrics | `px2tooth/evaluate.py`, `px2tooth/utils/IOU.py`, `px2tooth/utils/evaluation_metrics.py` |

---

## 📦 Repository Layout

```text
PX2Tooth/
├── data_process/                 # CBCT-to-panorama and label preprocessing scripts
├── px2tooth/
│   ├── train_seg.py              # U-Net segmentation training entry
│   ├── test_Unet.py              # U-Net segmentation inference/evaluation
│   ├── train_all.py              # Joint segmentation + 3D generation training
│   ├── test_all.py               # Joint model inference/evaluation
│   ├── predict_Unet.py           # Segmentation prediction utility
│   ├── evaluate.py               # Evaluation helper
│   ├── unet/                     # 2D U-Net modules
│   ├── PointNet_3d/              # PointNet-style 3D modules
│   └── utils/                    # Dataloaders, metrics, mesh and tensor utilities
├── baseline_generation/
│   └── model/                    # Baseline generation-model components
├── DATA_NOTICE.md                # Data privacy and release policy
├── requirements.txt              # Main dependency list
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start

Create and activate an environment:

```bash
conda create -n px2tooth python=3.8 -y
conda activate px2tooth
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install PyTorch according to your CUDA version from the official PyTorch website. The original experiments were developed with PyTorch on CUDA GPUs.

---

## 🗂️ Configure Local Data Paths

This is a **code-only release**. Clinical CBCT data and derived assets are not included. In the released scripts, private local paths are anonymized as `xxx`.

Before running training or inference, replace placeholders with your own local data layout, for example:

```text
xxx/CBCT
xxx/panoramic_images
xxx/tooth_masks
xxx/mesh_labels
xxx/checkpoints
```

See [`DATA_NOTICE.md`](DATA_NOTICE.md) for the full data and privacy policy.

---

## 🧪 Common Commands

### 1. Train panoramic tooth segmentation

```bash
cd px2tooth
python train_seg.py \
  --epochs 100 \
  --batch-size 1 \
  --classes 48
```

### 2. Run U-Net segmentation inference

```bash
cd px2tooth
python test_Unet.py \
  --model path/to/checkpoint.pth \
  --input path/to/panoramic_images \
  --output path/to/output_masks \
  --classes 48
```

### 3. Train the joint segmentation-generation pipeline

```bash
cd px2tooth
python train_all.py \
  --batch_size 1 \
  --npoint 4096 \
  --classes 48
```

### 4. Run joint inference / evaluation

```bash
cd px2tooth
python test_all.py \
  --batch_size 1 \
  --npoint 4096 \
  --classes 48
```

> These commands expose the original research-code entry points. You will need to configure local data roots, checkpoint paths, GPU IDs, and experiment directories for your own environment.

---

## 🧰 Script Map

| File / Folder | Purpose |
| --- | --- |
| `data_process/Get_Panoramic_Mean.py` | Panoramic projection / mean image generation helper |
| `data_process/Get_Panoramic_Label.py` | Panoramic label generation |
| `data_process/Get_Teeth_Mask.py` | Tooth-mask extraction |
| `data_process/Get_Teeth_curve.py` | Dental arch-curve estimation |
| `data_process/Get_target_label.py` | Target label construction |
| `data_process/nrrd2mesh.py` | NRRD-to-mesh conversion utility |
| `data_process/points2mesh.py` | Point-to-mesh conversion utility |
| `px2tooth/train_seg.py` | Segmentation training |
| `px2tooth/test_Unet.py` | Segmentation testing/inference |
| `px2tooth/train_all.py` | Joint segmentation + 3D generation training |
| `px2tooth/test_all.py` | Joint model testing/evaluation |
| `px2tooth/PointNet_3d/train_semseg.py` | PointNet-style 3D training entry |
| `baseline_generation/model/` | Baseline generation-model building blocks |

---

## 🔐 Data and Privacy Policy

The following assets are **not included** in this public repository:

| Not Released | Reason |
| --- | --- |
| Raw CBCT volumes | Clinical data may be private or institutionally restricted. |
| Derived panoramic X-ray images | Generated from private CBCT data. |
| Tooth masks and label images | Derived clinical annotations are not redistributed. |
| STL / PLY / OBJ mesh labels | Mesh labels may contain private or restricted information. |
| Patient or case identifiers | Removed for privacy. |
| Model checkpoints | Not part of the cleaned code release. |
| Training logs and visualizations | Experiment artifacts are excluded. |

Public-release placeholders:

| Placeholder | Meaning |
| --- | --- |
| `xxx` | Private local path or project-specific data root |
| `CASE_ID` | Anonymized case identifier |

---

## 📌 Notes

- This repository is a cleaned research-code release, not a turnkey clinical software package.
- Some scripts reflect the original experimental workflow and require local path configuration before running.
- No raw medical images, patient identifiers, model checkpoints, or training logs are included.
- The code is released to support method inspection, reproducibility, and extension.

---

## 📚 Citation

If you use this code, please cite the associated paper:

```bibtex
@article{px2tooth2024,
  title   = {PX2Tooth: Reconstructing the 3D Point Cloud Teeth from a Single Panoramic X-ray},
  year    = {2024},
  journal = {arXiv preprint arXiv:2411.03725}
}
```

---

## ⚖️ License

The released code is provided under the MIT License. Data, annotations, clinical images, checkpoints, and derived private assets are not redistributed.

See [`LICENSE`](LICENSE) and [`DATA_NOTICE.md`](DATA_NOTICE.md) for details.
