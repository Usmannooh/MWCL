# MWCL: Memory-driven and mapping alignment  for radiology Reports
# Overview
Existing methods struggle with biased visual-textual representations and inefficient cross-modal interactions, limiting their ability to capture fine-grained medical abnormalities. To address these challenges, we introduce memory-driven  (MWCL), a radiology report generation model that leverages weighted contrastive learning for feature refinement and memory-driven alignment for contextual consistency. Our approach improves abnormality recognition and strengthens the alignment between image and text representations.
##  Requirements
* Python >= 3.6
* PyTorch >= 1.7
* torchvision
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
│   ├── loss/
│   ├── metrics/
│   ├── tokenizer/
│   └── utils/
├── preprocess/
├── pycocoevalcap/
├── main_train.py
├── main_test.py
└── README.md
```
##  Configuration

Edit configuration files inside the `maintrain/` directory to set:

* Dataset paths
* Training hyperparameters
* Model saving/loading options

---

##  Training

To train MWCL on either dataset, run:

```bash
python main_train.py --config config/<your_config_file>.yaml --gpu <gpu_id>
```
