# ⚽ Ball Action Spotting with SoccerNet 2024

This project explores dense temporal event detection in soccer videos, focusing on the **Ball Action Spotting** task introduced in the [SoccerNet 2024 challenge](https://www.soccer-net.org/challenges/2024). The goal is to automatically detect fine-grained ball-related actions (e.g., pass, shot, tackle) in untrimmed broadcast videos using deep spatiotemporal models.

---

## 🧠 Overview

> **Objective:**  
Spot 12 classes of ball-related events in long untrimmed soccer broadcasts with only 7 labeled games.

> **Core Challenge:**  
Scarce annotations, dense action distribution, and subtle motion cues.

> **Approach:**  
Fine-tune a pretrained spatiotemporal network using lightweight augmentations and robust sampling strategies, while optimizing for precision under low-resource constraints.

---

## 📖 Key Highlights

- 2D EfficientNetV2-B0 as spatial encoder (pretrained on ImageNet).
- 4-layer inverted residual 3D block for temporal modeling.
- Focal loss + class-aware sampling for robust training.
- Test-time augmentation + Gaussian peak detection for inference.

---

## 🗂️ Project Structure
project-root/
├── src/                 # Core model and components
│   ├── model.py         # 2D+3D ActionSpottingModel
│   └── utils.py         # Logger, losses, etc.
├── train.py             # Fine-tuning script
├── evaluate.py          # Evaluation + mAP computation
├── prepare_data.py      # Download and extract ball spotting dataset
├── configs/             # YAML configs for hyperparameters
├── data/                # Preprocessed features, labels
├── visualize/           # Visualization
├── requirements.txt     # Python dependencies
└── README.md            # You're here

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
git clone https://github.com/timc1325/ActionSpotting.git
cd ActionSpotting
pip install -r requirements.txt
```

### 2. Download Data
pip install SoccerNet

```python
from SoccerNet.Downloader import SoccerNetDownloader
d = SoccerNetDownloader(LocalDirectory="data/")
d.downloadGames(files=["Labels-ball.json"], split=["train", "test"])
```

### 3. Train the model
```bash
python train.py --config configs/ball_action.yaml
```

### 4. Run evaluation
```bash
python evaluate.py --predictions results/predictions.json --split test
```