# FFDL_GCS

**Official Codebase for the Paper**  
**"A Fluid Flow‐Based Deep Learning (FFDL) Architecture for Subsurface Flow Systems With Application to Geologic CO₂ Storage"**  
Published in *Water Resources Research*, 2025
Official implementation of the paper:

> **Zhen Qin, Yingxiang Liu, Fangning Zheng, Behnam Jafarpour**  
> *A Fluid Flow‐Based Deep Learning (FFDL) Architecture for Subsurface Flow Systems With Application to Geologic CO₂ Storage*  
> *Water Resources Research*, 2025  
> DOI: [10.1029/2024WR037953](https://doi.org/10.1029/2024WR037953)

---

## 🧠 Overview

This repository provides the official implementation of **FFDL** (Fluid Flow‐Based Deep Learning), a novel deep learning architecture for subsurface flow modeling, particularly designed for **geologic CO₂ storage** (CCS).

---

## 📌 Key Contributions

- Proposes a novel deep learning architecture, **FFDL**, designed for simulating subsurface multiphase flow in **geologic CO₂ storage** (GCS).
- Addresses the limitations of standard numerical simulators (high computational cost) and conventional DL models (lack of physical interpretability).
- Incorporates **physical causality** through a physics-guided encoder and a **residual-based processor**, ensuring more physically consistent predictions.
- Demonstrates superior performance over baseline models in field-scale CO₂ injection experiments.
- Enables efficient **real-time forecasting**, **decision support**, and **inverse modeling** in CCS workflows.

---

## 📁 Repository Structure

```
FFDL_GCS/
│
├── source/                 # Main codebase
│   ├── model.py/           # Model definitions (FFDL core modules)
│   └── utils.py            # Utility functions for training and data loading, etc.
├── data (to do...)/        # Processed datasets (not uploaded here)
├── main_train.py           # Scripts to generate data from raw .mat files
├── requirements.txt        # Required packages (to be uploaded...)
└── README.md               # Project description (this file)
```

---

## 📄 Citation

Please cite the following paper if you use this code or models in your research:

```bibtex
@article{qin_fluid_2025,
  title = {A {Fluid} {Flow}-{Based} {Deep} {Learning} ({FFDL}) {Architecture} for {Subsurface} {Flow} {Systems} {With} {Application} to {Geologic} {CO}$_2$ {Storage}},
  volume = {61},
  issn = {0043-1397, 1944-7973},
  doi = {10.1029/2024WR037953},
  number = {1},
  journal = {Water Resources Research},
  author = {Qin, Zhen and Liu, Yingxiang and Zheng, Fangning and Jafarpour, Behnam},
  month = jan,
  year = {2025},
  pages = {e2024WR037953},
}
```

---
