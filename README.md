# MWCL: Memory-driven and Mapping Alignment with Weighted Contrastive Learning for Radiology Reports
This paper focuses on the task of automated radiology report generation (ARRG) from chest X-ray images, a critical area in medical AI that enhances diagnostic precision, reduces radiologist workload, and promotes timely clinical decision-making. Despite advancements in vision-language modeling, current approaches struggle with three core limitations: Despite advancements in vision-language modeling, current approaches struggle with three core limitations: (1) poor alignment between visual and textual features, (2) underrepresentation of rare and subtle abnormalities, and (3) limited contextual consistency in generated reports due to data imbalance and shallow multimodal interactions.


#  Important Notice

 This code is **directly associated with our manuscript submitted to the _Multimedia Systems**.  
If you use this repository in your research, **please cite the corresponding paper** (see citation section below).  
We encourage transparency and reproducibility in medical AI. This repository provides **full implementation**, **setup instructions**, and **evaluation tools** to replicate our results.

#  Key Features
Existing radiology report generation methods often suffer from biased visual-textual representations and inefficient cross-modal interactions, hindering the detection of fine-grained medical abnormalities.MWCL addresses these issues by: 

| Module      | Purpose                                                                          |
| ----------- | -------------------------------------------------------------------------------- |
| **AFRM**    | Enhances focus on abnormal regions using spatial and channel attention           |
| **AEMF**    | Fuses visual and textual features with self-attention                            |
| **MDAM**    | Aligns image-text features via a dynamic memory bank                             |
| **WCL**     | Optimizes contrastive learning using structured similarity and auxiliary signals |
| **Decoder** | Generates medical reports from memory-aligned image features                     |

This framework significantly improves **abnormality recognition** and **image-report alignment**, setting a new benchmark in medical vision-language tasks.
###  MWCL Pseudocode (Simplified Algorithm Overview)

```python
# Input: Radiology image(s) I
# Output: Generated Radiology Report Y

# Step 1: Feature Extraction
F = ResNet101(I)                    # Extract visual features from image

# Step 2: Adaptive Feature Refinement (AFRM)
G = Conv3x3(ReLU(Conv3x3(F)))       # Enhance features with convolution
AC = ChannelWiseAttention(G)        # Channel attention
AS = SpatialFeatureEnhancement(G)   # Spatial attention
F_refined = F + sigmoid(AC + AS) * G  # Recalibrated feature map

# Step 3: Attention-Enhanced Multimodal Fusion (AEMF)
T = TextEmbedding()                 # Text token embeddings
R = concat(F_refined, T)           # Fuse visual and textual features
R_fused = SelfAttention(R)         # Refine fusion with attention

# Step 4: Memory-Driven Alignment Module (MDAM)
Q = Linear(F_refined + PositionalEncoding)
S = softmax(Q @ MemoryKeys.T / sqrt(d))   # Similarity to memory slots
F_memory = S @ MemoryValues              # Retrieve relevant memory
Memory = UpdateMemory(F_memory)         # Update memory dynamically

# Step 5: Weighted Contrastive Learning (WCL)
F1 = Augment(F_memory)
F2 = Augment(F_memory)
Z1 = Project(F1)
Z2 = Project(F2)
loss_WCL = ContrastiveLoss(Z1, Z2, temperature=τ)

# Step 6: Report Generation (Transformer Decoder)
H = TransformerEncoder(F_memory, Memory)
Y = TransformerDecoder(H)         # Generate report text

# Final Output
return Y, loss_WCL
```
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
  note = {Code: \url{https://github.com/Usmannooh/MWCL}}  
}
*This repository accompanies the manuscript under review at [Multimedia Systems].*

```
