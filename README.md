# PX2Tooth Code Release

✨ **Code release for generating panoramic X-ray representations from CBCT data and reconstructing 3D tooth point clouds from a single panoramic X-ray.**

This repository contains the cleaned research code associated with the paper:

> **PX2Tooth: Reconstructing the 3D Point Cloud Teeth from a Single Panoramic X-ray**  
> arXiv: [2411.03725](https://arxiv.org/abs/2411.03725)

The project studies a two-stage dental reconstruction pipeline: first segmenting teeth from panoramic X-ray images, and then reconstructing tooth-level 3D point clouds from the 2D panoramic input.

---

## 🌟 What Is Included

| Component | Description |
|---|---|
| `data_process/` | Preprocessing scripts for panoramic image generation, tooth-mask extraction, arch-curve estimation, target-label construction, mesh conversion, and scale adjustment. |
| `px2tooth/` | Main PyTorch code for segmentation and 3D tooth point-cloud generation, including U-Net, PointNet-style generation modules, training scripts, testing scripts, dataloaders, and metrics. |
| `baseline_generation/model/` | Baseline generation-model code adapted from the original baseline implementation used in the project. |
| `DATA_NOTICE.md` | Data privacy and release policy. |
| `requirements.txt` | Minimal Python package list for setting up the code environment. |

---

## 🧩 Repository Structure

```text
.
├── baseline_generation/
│   └── model/                  # Baseline generation modules
├── data_process/               # CBCT-to-panorama and label preprocessing
├── px2tooth/
│   ├── train_all.py            # Joint segmentation + generation training entry
│   ├── test_all.py             # Joint model testing / inference entry
│   ├── train_seg.py            # U-Net segmentation training
│   ├── test_Unet.py            # U-Net segmentation inference
│   ├── unet/                   # 2D segmentation network
│   ├── PointNet_3d/            # 3D point-cloud generation modules
│   └── utils/                  # Dataloaders, losses, metrics, mesh utilities
├── DATA_NOTICE.md
├── LICENSE
└── requirements.txt
```

---

## 🚀 Quick Start

Create an environment:

```bash
conda create -n px2tooth python=3.8 -y
conda activate px2tooth
pip install -r requirements.txt
```

Install PyTorch following your CUDA version from the official PyTorch website. The original experiments were developed with PyTorch and CUDA GPUs.

Prepare your local data paths. Because the clinical CBCT data are private and cannot be redistributed, paths in the released scripts have been anonymized as `xxx`. Replace these placeholders with your own local folders, for example:

```text
xxx/CBCT
xxx/panoramic_images
xxx/tooth_masks
xxx/mesh_labels
xxx/checkpoints
```

Run segmentation training:

```bash
cd px2tooth
python train_seg.py --epochs 100 --batch-size 1 --classes 48
```

Run the joint segmentation-generation training pipeline:

```bash
cd px2tooth
python train_all.py --batch_size 1 --npoint 4096 --classes 48
```

Run joint inference / evaluation:

```bash
cd px2tooth
python test_all.py --batch_size 1 --npoint 4096 --classes 48
```

> The commands above show the original code entry points. Users need to configure local data paths, checkpoint paths, and GPU settings for their own environment.

---

## 🔬 Main Pipeline

1. **CBCT preprocessing**  
   Scripts in `data_process/` generate panoramic projections, estimate the dental arch curve, extract tooth masks, and prepare target labels from CBCT and mesh annotations.

2. **Panoramic tooth segmentation**  
   `px2tooth/train_seg.py` trains a U-Net-style segmentation model for tooth-region prediction on panoramic X-ray images.

3. **3D tooth point-cloud generation**  
   `px2tooth/train_all.py` connects the segmentation output with PointNet-style generation modules to reconstruct 3D tooth point clouds.

4. **Evaluation**  
   `px2tooth/test_all.py` and utility metrics compute segmentation and 3D reconstruction quality, including IoU and point-cloud distance metrics.

---

## 🔐 Data Policy

Clinical CBCT scans, derived panoramic images, tooth masks, mesh labels, checkpoints, and experiment logs are **not included** in this repository because they may contain private or license-restricted medical data.

All private paths, case identifiers, and local server information have been replaced with `xxx` or `CASE_ID`.

See [`DATA_NOTICE.md`](DATA_NOTICE.md) for details.

---

## 📌 Notes

- This is a cleaned research-code release, not a turnkey clinical software package.
- Some scripts reflect the original experimental workflow and may require local path configuration before running.
- No raw medical images, patient identifiers, model checkpoints, or training logs are included.
- The code is released to support reproducibility of the method design and implementation logic.

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
