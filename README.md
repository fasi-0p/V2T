# Video-to-Text (V2T) Captioning Model

A self-supervised **Video-to-Text (V2T)** framework built using **PyTorch** that learns visual-temporal patterns from videos and generates natural language descriptions.  

This repository includes:
- `train_v2t.py` — model training and checkpoint saving  
- `inference_v2t.py` — model loading and caption generation  
- Modular design for easy extension and experimentation  

---

## 🚀 Features

- Frame-level encoding using **ResNet-50**
- Temporal sequence modeling with **Bi-LSTM + Attention** you could use tempral transformer but as a MVP I chose BiLSTM
- Multi-head descriptor for **motion**, **scene**, and **action**
- Self-supervised training (no labels required)
- Human-readable caption generation

---

## 🧠 Architecture Overview

- **Video** → **FrameEncoder** (ResNet-50) → **TemporalEncoder** (BiLSTM + Attention)
→ **VideoDescriptor** (Motion / Scene / Action Heads)→ **Description Generator** (Top-k word selection)



Each video is represented by extracted frames, encoded into spatio-temporal features, clustered into visual concepts, and converted into descriptive text.

---

## 📦 Requirements

| Library | Version (recommended) |
|----------|----------------------|
| Python | 3.8+ |
| PyTorch | 2.0+ |
| TorchVision | 0.15+ |
| OpenCV | 4.x |
| NumPy | latest |
| tqdm | latest |

Install dependencies:
```bash
pip install torch torchvision opencv-python numpy tqdm


📁 v2t/
│
├── train_v2t.py         # Training pipeline
├── inference_v2t.py     # Inference (caption generation)
├── v2t_model.pt         # Saved model weights (after training)
└── README.md            # This file
```

---


## 🧬 Model Components
- FrameEncoder	Extracts spatial features from frames using pretrained ResNet-50
- TemporalEncoder	Models sequence-level temporal dependencies with BiLSTM + Attention
- VideoDescriptor	Produces separate embeddings for motion, scene, and action
- generate_description()	Converts visual embeddings into a descriptive sentence

##  🧱 Future Enhancements
- Integrate Transformer-based temporal encoders
- Add multimodal support (audio/text)
- Improve caption grammar via fine-tuned LLM
- Extend to multi-sentence video summaries

## 🧩 Quick Start Summary
```bash
# Clone repo
git clone https://github.com/yourusername/v2t-captioning.git
cd v2t-captioning

# Install dependencies
pip install torch torchvision opencv-python numpy tqdm

# Train model
python train_v2t.py

# Run inference
python inference_v2t.py
```
