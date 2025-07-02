# MWCL: Memory-driven and Mapping Alignment with Weighted Contrastive Learning for Radiology Reports

Existing radiology report generation methods often suffer from biased visual-textual representations and inefficient cross-modal interactions, hindering the detection of fine-grained medical abnormalities.

#  Important Notice

 This code is **directly associated with our manuscript submitted to the _Multimedia Systems**.  
If you use this repository in your research, **please cite the corresponding paper** (see citation section below).  
We encourage transparency and reproducibility in medical AI. This repository provides **full implementation**, **setup instructions**, and **evaluation tools** to replicate our results.

#  Key Features
MWCL addresses these issues by introducing an adaptive feature refinement, a memory-driven alignment mechanism, and weighted contrastive learning:

* **Adaptive Feature Refinement:** Visual features through spatial and channel-wise attention, enabling the abnormal structures critical to diagnosis.
* **Memory-driven Alignment:** Enhances contextual consistency between modalities using learned memory modules.
* **Weighted Contrastive Learning:** Refines image-text features by emphasizing critical visual-textual pairs.

This framework significantly improves **abnormality recognition** and **image-report alignment**, setting a new benchmark in medical vision-language tasks.

##  Requirements
Ensure you have the following installed:
- Python ≥ 3.8  
- PyTorch ≥ 1.7  
- `transformers`, `numpy`, `matplotlib`, `scikit-learn`, `opencv-python`

You can install dependencies with:

```bash
pip install -r requirements.txt
```
  ##  Dataset

Download the following datasets and place them under the `data/` directory:

* [IU X-Ray Dataset](https://iuhealth.org/find-medical-services/x-rays)
* [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr-jpg/2.0.0/)

Expected directory structure:

```
MWCL/
├── config/
├── data/
│   ├── iu_xray/
│   └── mimic_cxr/
├── models/
├── modules/
│   ├── dataloader/
│   ├── modal/
    ├── ..../
│   ├── weighted/
│   ├── metrics/
│   ├── tokenizer/
│   └── utils/
├── pycocoevalcap/
├── main_train.py
├── main_test.py
├── test_iu_xray/
├── test_mimic_cxr/
├── train_iu_xray/
└── README.md

```
## Train

Run `bash train_iu_xray.sh` to train a model on the IU X-Ray data.

Run `bash train_mimic_cxr.sh` to train a model on the MIMIC-CXR data.

## Test

Run `bash test_iu_xray.sh` to test a model on the IU X-Ray data.

Run `bash test_mimic_cxr.sh` to test a model on the MIMIC-CXR data.
# Results
| Model    | BLEU-4    | METEOR    | ROUGE-L   | CIDEr     |
| -------- | --------- | --------- | --------- | --------- |
| **IU-Xray** | **0.196** | **0.213** | **0.409** | **0.484** |
| **MIMIC** | **0.120** | **0.160** | **0.286** | **0.230** |

(Refer to the paper for full comparison)

##  Configuration

Edit configuration files inside the `maintrain/` directory to set:

* Dataset paths
* Training hyperparameters
* Model saving/loading options

---

# Acknowledgments

This work is supported by a grant from the **Natural Science Foundation of China (Grant No. 62072070)**.  <br><br>

We would also like to express our gratitude to all the source code contributors, especially the authors of **R2GenCNM**, whose work inspired parts of this implementation.


## Citation 
```
If you use this code or findings, please cite:  

@article{usman2025MWCL,  
  title = {MWCL: Memory-driven and mapping alignment with weighted contrastive learning for radiology},  
  author = {Usman, M. and [Coauthors]},  
  journal = {Multimedia Systems},  
  year = {2025},  
  doi = {10.5281/zenodo.15771095},  
  note = {Code: \url{https://github.com/Usmannooh/MWCL}}  
}
*This repository accompanies the manuscript under review at [Multimedia Systems].*

```
