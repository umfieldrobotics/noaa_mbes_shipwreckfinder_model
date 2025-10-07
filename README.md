## NOAA MBES ShipwreckFinder Model Training + Eval

This repo accompanies the ShipwreckFinder QGIS Plugin repo, which can be found here: https://github.com/umfieldrobotics/ShipwreckFinderQGISPlugin

The project/tutorial website can be found here: https://sites.google.com/umich.edu/oceans2025-tutorial

The accompanying paper, presented at OCEANS 2025, can be found here: https://arxiv.org/pdf/2509.21386

### Usage

The plugin repo linked above contains instructions for installing the plugin into your QGIS environment, while this repo contains the training and evaluation code for the pre-trained models released as a part of the plugin. 

If you wish to train your own model as the backbone of the QGIS plugin, then take a look at the four "train_" scripts in this repo (four scripts for four different model architectures).

BASNet code is adapted from https://github.com/xuebinqin/BASNet. HRNet code is adapted from https://github.com/HRNet/HRNet-Semantic-Segmentation.

### BibTeX

If you find this work useful, please cite us at:

```bibtex
@inproceedings{shep2025shipwreckfinder,
  title={ShipwreckFinder: A QGIS Tool for Shipwreck Detection in Multibeam Sonar Data},
  author={Sheppard, Anja and Smithline, Tyler and Scheffer, Andrew and Smith, David and Sethuraman, Advaith V. and Bird, Ryan and Lin, Sabrina and Skinner, Katherine A.},
  booktitle={Proceedings of OCEANS Great Lakes 2025},
  year={2025},
  organization={IEEE}
}