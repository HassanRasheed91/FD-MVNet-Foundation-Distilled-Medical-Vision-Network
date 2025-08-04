# 🏥 Foundation Model-Enhanced COVID Detection System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 **97.59% COVID Detection Accuracy with 297x Parameter Reduction**

A cutting-edge **foundation model ensemble** with **knowledge distillation** framework for COVID-19 detection from chest CT scans. This system combines **BiomedCLIP**, **DINOv2**, and **OpenAI CLIP** foundation models to create lightweight, deployable models suitable for resource-constrained clinical environments.

---

## 🚀 **Quick Start**

### **Option 1: Kaggle (Recommended)**
```bash
# 1. Upload this repository as a Kaggle dataset
# 2. Create new notebook with GPU enabled
# 3. Run these commands:

!pip install -r /kaggle/input/covid-foundation-models/requirements.txt
!cp /kaggle/input/covid-foundation-models/*.py ./
!python main.py
```

### **Option 2: Local Setup**
```bash
git clone https://github.com/HassanRasheed91/FD-MVNet-Foundation-Distilled-Medical-Vision-Network.git
cd FD-MVNet-Foundation-Distilled-Medical-Vision-Network
pip install -r requirements.txt
python main.py
```

---

## 🏆 **Key Achievements**

| Metric | Teacher Ensemble | Lightweight Student | Improvement |
|--------|------------------|---------------------|-------------|
| **Accuracy** | 98.66% | **97.59%** | -1.07% |
| **Parameters** | 435.0M | **1.5M** | **297x reduction** |
| **Model Size** | 1740 MB | **6.0 MB** | **290x smaller** |
| **AUC-ROC** | 0.9985 | **0.9958** | Maintained |
| **Inference Speed** | 45ms | **8ms** | **5.6x faster** |

### 🎯 **Clinical Performance**
- **Sensitivity**: 96.8% (COVID detection)
- **Specificity**: 97.6% (Non-COVID detection)  
- **Deployment Ready**: Edge device compatible
- **Real-time**: <10ms inference on CPU

---

## 🔬 **Technical Innovation**

### **Foundation Model Ensemble Teacher**
```
┌─────────────────────────────────────────────────────────┐
│  🧠 Foundation Model Teacher Ensemble (435.0M params)  │
├─────────────────────────────────────────────────────────┤
│  🔬 BiomedCLIP     │  Medical-specific representations  │
│  🎯 DINOv2         │  Self-supervised visual features   │  
│  👁️ OpenAI CLIP    │  General vision understanding      │
└─────────────────────────────────────────────────────────┘
                              │
                    📚 Knowledge Distillation
                              │
                              ▼
┌─────────────────────────────────────────────────────────┐
│     💡 Lightweight Student (1.5M params)               │
├─────────────────────────────────────────────────────────┤
│  🏗️ Medical-Optimized Architecture                     │  
│  ⚡ Efficient Blocks with SE Attention                 │
│  🎯 Depthwise Separable Convolutions                   │
│  📱 Mobile-Friendly Design                             │
└─────────────────────────────────────────────────────────┘
```

### **Advanced Knowledge Distillation**
- **Multi-Teacher Ensemble**: Combines knowledge from multiple foundation models
- **Temperature Scaling**: Soft target generation with T=3.0
- **Ensemble Consistency**: Ensures student aligns with all teachers
- **Feature-Level Transfer**: Deep feature alignment beyond just predictions

---

## 📊 **Research Results**

### **Published Performance** (From Paper: *Computers in Biology and Medicine*, 2021)
Based on the **Sugeno fuzzy integral ensemble** with four pre-trained models:
- **Paper Accuracy**: 98.93%
- **Paper Sensitivity**: 98.93%
- **Models Used**: VGG-11, GoogLeNet, SqueezeNet v1.1, Wide ResNet-50-2
- **Dataset**: SARS-COV-2 CT-scan dataset (2,481 images)

### **Current Implementation Results**
Our **foundation model enhanced** version achieves:
- **Student Accuracy**: 97.59%
- **Teacher Accuracy**: 98.66%
- **Parameter Reduction**: 297x
- **Model Compression**: 290x size reduction
- **Maintained AUC**: 0.9958 vs 0.9985

### **Key Improvements Over Paper**
1. **Foundation Model Integration**: Added BiomedCLIP, DINOv2, OpenAI CLIP
2. **Extreme Compression**: 297x parameter reduction vs traditional ensemble
3. **Real-time Deployment**: 6MB model size for edge devices
4. **Medical Optimization**: Enhanced augmentations and architecture

---

## 🏗️ **Architecture Overview**

### **File Structure**
```
foundation-covid-detection/
├── 📄 main.py              # Main execution pipeline
├── 🧠 models.py            # Foundation models & architectures  
├── 🎓 training.py          # Knowledge distillation framework
├── 📊 visualization.py     # Research visualization engine
├── 📁 dataset.py           # Enhanced data pipeline
├── 📋 requirements.txt     # Dependencies
├── 📖 README.md           # This file
└── 📁 results/            # Generated outputs
    ├── training_curves.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── performance_summary.csv
    └── foundation_models.pth
```

## 🛠️ **Installation & Setup**

### **System Requirements**
- **Python**: 3.8+
- **GPU**: NVIDIA GPU with 8GB+ VRAM (recommended)
- **CPU**: 8+ cores for CPU-only training
- **RAM**: 16GB+ recommended
- **Storage**: 10GB for models and datasets

### **Dependencies**
```bash
# Core ML Framework
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0

# Foundation Models
open_clip_torch==2.23.0
transformers==4.35.2
timm>=0.9.0

# Computer Vision & Data Processing
albumentations>=1.3.0
opencv-python>=4.8.0
Pillow>=9.5.0

# Scientific Computing
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
scipy>=1.10.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Utilities
tqdm>=4.65.0
tensorboard>=2.13.0
```

### **Quick Installation**
```bash
# Method 1: Using pip
pip install -r requirements.txt

# Method 2: Using conda
conda env create -f environment.yml
conda activate covid-detection
```

---

## 📊 **Dataset Setup**

### **Expected Directory Structure**
```
dataset_split/
├── train/
│   ├── COVID/          # COVID-positive CT scans
│   │   ├── img001.png
│   │   ├── img002.png
│   │   └── ...
│   └── NonCOVID/       # COVID-negative CT scans
│       ├── img001.png
│       ├── img002.png
│       └── ...
├── val/
│   ├── COVID/
│   └── NonCOVID/
└── test/
    ├── COVID/
    └── NonCOVID/
```

### **Supported Datasets**
- **SARS-COV-2 CT-scan Dataset** ([Kaggle](https://www.kaggle.com/plameneduardo/sarscov2-ctscan-dataset))
- **Custom datasets** following the above structure

---

## 🚀 **Usage**

### **Basic Usage**
```python
# Run complete experiment
python main.py

# Expected output:
# 🎓 FOUNDATION MODEL COVID DETECTION EXPERIMENT
# ================================================================================
# 1️⃣ Setting up enhanced data pipeline...
# 2️⃣ Initializing foundation models...
# 3️⃣ Stage 1: Training Foundation Teacher (25 epochs)...
# 4️⃣ Stage 2: Training Student with Distillation (40 epochs)...
# 5️⃣ Comprehensive Evaluation & Visualization...
# 🏆 SUCCESS! ACHIEVED 97%+ ACCURACY!
```

### **Custom Configuration**
```python
from main import run_foundation_model_experiment

# Custom training configuration
results = run_foundation_model_experiment(
    data_dir='/path/to/dataset',
    device=torch.device('cuda'),
    teacher_epochs=50,      # More epochs for better teacher
    student_epochs=100      # More epochs for better student
)

print(f"Final Accuracy: {results['student_accuracy']:.2f}%")
print(f"Parameter Reduction: {results['reduction_ratio']:.1f}x")
```

### **Individual Components**
```python
# Use individual components
from models import FoundationModelTeacher, LightweightMedicalStudent
from training import FoundationModelTrainer

# Initialize models
teacher = FoundationModelTeacher(num_classes=2)
student = LightweightMedicalStudent(num_classes=2, width_multiplier=1.2)

# Train with knowledge distillation
trainer = FoundationModelTrainer(device='cuda')
trainer.train_teacher(teacher, train_loader, val_loader, epochs=25)
trainer.train_student(student, teacher, train_loader, val_loader, epochs=40)
```

---

## 📈 **Training Details**

### **Training Strategy**
1. **Stage 1**: Foundation Model Teacher Training
   - Multi-rate learning: 1e-5 for foundation models, 1e-4 for classical
   - Cosine annealing schedule
   - Early stopping with patience=10
   - Gradient clipping (max_norm=1.0)

2. **Stage 2**: Knowledge Distillation  
   - OneCycleLR scheduler (max_lr=1e-3)
   - Advanced distillation loss (α=0.8, T=3.0)
   - Extended patience=30 for student training
   - Feature-level alignment

### **Loss Functions**
```python
# Multi-component distillation loss
L_total = (1-α) × L_hard + α × L_soft + λ × L_ensemble

Where:
- L_hard: Cross-entropy with ground truth
- L_soft: KL divergence with teacher predictions  
- L_ensemble: Consistency across all teacher models
- α = 0.8, λ = 0.1, Temperature = 3.0
```

### **Data Augmentation**
```python
# Medical-optimized augmentations
- Geometric: HorizontalFlip, Rotate(±10°), ShiftScaleRotate
- Intensity: RandomBrightnessContrast, CLAHE, HSV
- Robustness: GaussNoise, GaussianBlur, Compression
- Regularization: CoarseDropout
- Normalization: ImageNet statistics for foundation models
```

---

## 📊 **Evaluation Metrics**

### **Classification Metrics**
- **Accuracy**: Overall classification accuracy
- **Precision**: Class-wise and weighted precision
- **Recall/Sensitivity**: COVID detection rate
- **Specificity**: Non-COVID detection rate  
- **F1-Score**: Harmonic mean of precision/recall
- **AUC-ROC**: Area under ROC curve
- **AUC-PR**: Area under Precision-Recall curve

### **Efficiency Metrics**
- **Parameter Count**: Total trainable parameters
- **Model Size**: Storage requirements (MB)
- **Inference Time**: Forward pass latency
- **FLOPS**: Floating point operations
- **Memory Usage**: Peak GPU/CPU memory

---

## 📁 **Output Files**

### **Generated Results**
```
results/
├── 📊 Visualizations
│   ├── training_curves.png              # Training progress
│   ├── model_comparison.png             # Performance comparison  
│   ├── confusion_matrices.png           # Classification results
│   ├── roc_curves.png                   # ROC analysis
│   ├── precision_recall_curves.png      # PR analysis
│   ├── comprehensive_metrics.png        # Metrics table
│   └── knowledge_distillation_analysis.png
│
├── 📈 Performance Data
│   ├── performance_summary.csv          # Model comparison
│   ├── comprehensive_metrics.csv        # Detailed metrics
│   └── research_highlights.json         # Key achievements
│
├── 🧠 Trained Models
│   ├── foundation_models.pth            # Teacher & student weights
│   ├── teacher_best.pth                 # Best teacher checkpoint
│   └── student_best.pth                 # Best student checkpoint
│
├── 📝 Documentation
│   ├── research_summary.md              # Publication summary
│   ├── training.log                     # Detailed training logs
│   └── foundation_experiment_results.pth # Complete results
```

---

## 🔬 **Research Applications**

### **Clinical Deployment**
```python
# Load trained student model for inference
import torch
from models import LightweightMedicalStudent

# Load lightweight model (6.0 MB)
model = LightweightMedicalStudent(num_classes=2)
checkpoint = torch.load('results/student_best.pth')
model.load_state_dict(checkpoint)
model.eval()

# Real-time inference
def predict_covid(ct_scan_image):
    with torch.no_grad():
        prediction = model(ct_scan_image)
        probability = torch.softmax(prediction, dim=1)
        return probability[0][1].item()  # COVID probability

# Edge deployment ready
covid_prob = predict_covid(patient_scan)
print(f"COVID-19 Probability: {covid_prob:.3f}")
```

### **Research Extensions**
- **Multi-class Classification**: Extend to Normal/Pneumonia/COVID
- **Segmentation**: Add lesion segmentation capabilities  
- **Cross-modal**: Integrate with clinical text data
- **Federated Learning**: Distributed training across hospitals
- **Explainability**: Add Grad-CAM visualization
- **Uncertainty Quantification**: Bayesian neural networks

---

### **Related Work**
```bibtex
@article{biomedclip2024,
  title={BiomedCLIP: Large-scale biomedical vision-language pre-training},
  author={Zhang, Sheng and Xu, Yanbo and Usuyama, Naoto and others},
  journal={arXiv preprint arXiv:2303.00915},
  year={2024}
}

@article{dinov2023,
  title={DINOv2: Learning Robust Visual Representations without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and others},
  journal={arXiv preprint arXiv:2304.07193},
  year={2023}
}
```

---

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/HassanRasheed91/FD-MVNet-Foundation-Distilled-Medical-Vision-Network.git
cd FD-MVNet-Foundation-Distilled-Medical-Vision-Network

# Create development environment
conda create -n covid-dev python=3.9
conda activate covid-dev
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### **Contribution Guidelines**
1. **Fork** the repository
2. **Create** feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** changes (`git commit -m 'Add amazing feature'`)
4. **Push** to branch (`git push origin feature/amazing-feature`)
5. **Open** Pull Request

---

## 🆘 **Support & FAQ**

### **Common Issues**

**Q: Out of Memory Error**
```python
# Reduce batch size in main.py
batch_size = 8  # instead of 16

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
```

**Q: Foundation Models Not Loading**
```python
# Check internet connection for model downloads
# Verify transformers and open_clip versions
pip install --upgrade transformers open_clip_torch

# Use fallback classical models
# Set use_foundation_models=False in config
```

**Q: Dataset Path Issues**
```python
# Update dataset path in main.py
DATASET_PATH = '/your/actual/dataset/path'

# Verify directory structure
python -c "from dataset import ResearchGradeCOVIDDataset; ResearchGradeCOVIDDataset('path', 'train')"
```

### **Performance Optimization**
- **GPU**: Use Tesla V100 for fastest training
- **CPU**: 16+ cores recommended for CPU-only training  
- **Mixed Precision**: Reduces memory usage by 50%
- **DataLoader**: Increase num_workers for faster data loading

### **Getting Help**
- 📧 **Email**: 221980038@gift.edu.pk
- 💬 **Issues**: [GitHub Issues](https://github.com/HassanRasheed91/FD-MVNet-Foundation-Distilled-Medical-Vision-Network/issues)

---

## 📜 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2024 Hassan Rasheed

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🌟 **Acknowledgments**

- **Foundation Models**: BiomedCLIP, DINOv2, OpenAI CLIP teams
- **Datasets**: SARS-COV-2 CT-scan dataset contributors
- **Computing**: Thanks to Kaggle for GPU resources
- **Community**: PyTorch and Hugging Face ecosystems
- **Inspiration**: Knowledge distillation and medical AI research

---

## 📊 **Project Stats**

**📈 Impact Metrics:**
- **97.59% Accuracy** - High-performance COVID detection
- **297x Compression** - Massive parameter reduction
- **<10ms Inference** - Real-time clinical deployment
- **6.0 MB Model** - Edge device compatible

---

<div align="center">

### 🎯 **Ready to achieve 97%+ COVID detection accuracy?**

**[⭐ Star this repo](https://github.com/HassanRasheed91/FD-MVNet-Foundation-Distilled-Medical-Vision-Network)** • **[🍴 Fork it](https://github.com/HassanRasheed91/FD-MVNet-Foundation-Distilled-Medical-Vision-Network/fork)** 

**Built with ❤️ for advancing medical AI research**

</div>
