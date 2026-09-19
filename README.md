# 🦷 Dental Condition Classification from Intraoral Images

## Clinical Motivation
As an aspiring dental student, I wanted to explore how computer vision could 
assist in preliminary dental screening — especially in areas with limited access 
to dentists. This project builds and evaluates a classifier for 6 common dental 
conditions from intraoral photographs.

## 🚀 Live Demo
**[Try the model →](https://c76e7d55642fd5af0d.gradio.live)** *(link active ~72 hours)*

## 📊 Dataset
- **Source:** [Oral Diseases Dataset](https://www.kaggle.com/datasets/miraosama/oral-diseases-dataset) (Kaggle)
- **Classes (6):** Calculus, Caries, Gingivitis, Hypodontia, Mouth Ulcer, Tooth Discoloration
- **Total images:** 12,320 (after cleaning and dropping a YOLO-annotated subset)
- **Split:** 80% train / 15% validation / 5% test (stratified)

## 🛠️ Methods
- **Model:** EfficientNet-B0, pretrained on ImageNet (transfer learning)
- **Augmentation:** Random resized crop, horizontal flip, rotation ±15°, color jitter
- **Loss:** Cross-entropy with class weights (balanced for class imbalance)
- **Optimizer:** AdamW (lr 1e-3, weight decay 1e-4) + cosine annealing
- **Early stopping:** patience 4 epochs
- **Framework:** PyTorch on Google Colab (T4 GPU)

## 📈 Results
- **Best validation accuracy:** 93.56%
- **Test accuracy:** 92.37%
- **Macro F1:** 0.910

### Per-Class Performance
| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Calculus | 0.631 | 0.815 | 0.711 | 65 |
| Caries | 1.000 | 0.985 | 0.992 | 130 |
| Gingivitis | 0.871 | 0.746 | 0.804 | 118 |
| Hypodontia | 0.967 | 0.952 | 0.959 | 62 |
| Mouth_Ulcer | 1.000 | 1.000 | 1.000 | 140 |
| Tooth_Discoloration | 0.990 | 1.000 | 0.995 | 101 |

## 🩺 Clinical Interpretation

The model achieved **92.37% test accuracy** across 6 dental conditions, with 
strong performance on visually distinct pathologies (Mouth Ulcer: 100% F1, 
Tooth Discoloration: 99.5% F1, Caries: 99.2% F1). The interesting failures 
cluster in two classes: **Calculus (precision 0.63)** and **Gingivitis (recall 0.75)**.

This is clinically meaningful. Calculus (calcified plaque) and Gingivitis 
(gum inflammation) both present at the gingival margin, and in intraoral 
photographs they share overlapping visual features — yellowish deposits, 
redness, and irregular gum contours. The model's confusion between them 
mirrors a real diagnostic challenge: distinguishing supragingival calculus 
from early gingival inflammation often requires tactile examination with a 
periodontal probe, not just visual inspection.

The model is 100% precise on Mouth Ulcer — never falsely predicts it — 
which is appropriate since ulcers are visually distinctive. However, its 
Calculus precision of 0.63 means it over-predicts Calculus, likely labeling 
gingivitis-affected sites as calculus due to similar coloration.

This is not a flaw unique to this model — it reflects the inherent difficulty 
of 2D image-based diagnosis. It's exactly why dentists use magnification, 
probing, and patient history alongside visual inspection.

## ⚠️ Limitations
- Trained on a curated dataset; may not generalize to real-world clinical lighting
- Not validated on diverse populations
- **Not a clinical diagnostic tool.** This is a learning project.

## 📁 Files
- `train.py` — training and evaluation script
- `results/best_dental_model.pth` — trained model weights
- `results/classes.json` — class label mapping
- `results/metrics.json` — evaluation metrics
- `results/confusion_matrix.png` — raw confusion matrix
- `results/confusion_matrix_normalized.png` — per-class recall
- `results/training_curves.png` — loss and accuracy over epochs
- `results/sample_predictions.png` — sample predictions on test images

## 🏃 How to Run
```bash
pip install -r requirements.txt
python train.py
```

## 📚 References
- Oral Diseases Dataset (Kaggle): miraosama/oral-diseases-dataset
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
