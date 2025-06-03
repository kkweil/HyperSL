# HyperSL

Official implementation of "**HyperSL:** A Spectral Foundation Model for Hyperspectral Image Interpretation".

## 🔗 Pretrained Weights

You can download the pretrained weights from Baidu Netdisk:

* **Link**: [https://pan.baidu.com/s/11uuzhKs-dtFExlnph1IYTQ](https://pan.baidu.com/s/11uuzhKs-dtFExlnph1IYTQ?pwd=27mh)
* **Extraction Code**: `27mh`

## 📖 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@ARTICLE{10981753,
  author={Kong, Weili and Liu, Baisen and Bi, Xiaojun and Yu, Changdong and Li, Xinyao and Chen, Yushi},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={HyperSL: A Spectral Foundation Model for Hyperspectral Image Interpretation}, 
  year={2025},
  volume={63},
  number={5513119},
  pages={1-19},
  keywords={Hyperspectral imaging;Data models;Sensors;Adaptation models;Training;Representation learning;Data mining;Autoencoders;Vectors;Transformers;Cross-scenario;foundation model;hyperspectral image;knowledge transfer},
  doi={10.1109/TGRS.2025.3566205}}

```

## 📎 Overview

HyperSL is designed to:

* Learn transferable spectral representations across heterogeneous hyperspectral sensors
* Accept arbitrary spectral input dimensions (bands, range)
* Generalize without retraining and modifying network structure

## 🛠️ Features

* Masked self-supervised learning strategy tailored for spectral data
* Unified spectral token representation
* Wavelength-aware positional encoding
* Trained on over 300 million spectra from multiple sensors

