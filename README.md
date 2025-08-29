# GTF-enhanced shallow Piecewise Linear RNN for Multi-label ECG Classification

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](docs/GTF_shPLRNN_ECG_Research_Report.pdf)
[![Code](https://img.shields.io/badge/Code-Python-blue)](models/gtf_shplrnn/)
[![Data](https://img.shields.io/badge/Data-MIMIC--IV--ECG-green)](https://physionet.org/content/mimic-iv-ecg/1.0/)

This repository contains the experimental code for our research on **GTF-enhanced shallow Piecewise Linear RNN (GTF-shPLRNN)** for automated ECG multi-label classification, achieving **32× parameter efficiency** compared to ResNet-1D while maintaining **88% of its performance**.



## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/GTF-shPLRNN-ECG-Experiments.git
cd GTF-shPLRNN-ECG-Experiments
pip install -r requirements.txt
```

### Basic Usage

```python
from models.gtf_shplrnn.corrected_gtf_shplrnn import GTFshPLRNN

# Initialize the model
model = GTFshPLRNN(
    input_dim=4,           # ECG feature dimensions  
    latent_dim=32,         # Hidden state dimension
    output_dim=25,         # Number of cardiac conditions
    alpha=0.5              # GTF mixing parameter
)

# Train the model
python models/gtf_shplrnn/robust_plrnn_training.py
```

### Run SOTA Comparison Experiments

```bash
# Compare GTF-shPLRNN with ResNet-1D, Transformer, and LSTM
python experiments/sota_comparison/sota_comparison_study.py

# Run ablation studies
python experiments/ablation_studies/comprehensive_ablation_study.py
```

## 📁 Repository Structure

```
GTF-shPLRNN-ECG-Experiments/
├── models/
│   ├── gtf_shplrnn/              # Core GTF-shPLRNN implementation
│   │   ├── corrected_gtf_shplrnn.py    # Main model architecture
│   │   ├── robust_plrnn_training.py    # Training pipeline
│   │   └── adaptive_gtf.py             # Adaptive GTF mechanisms
│   └── baselines/                # Baseline model implementations
├── experiments/
│   ├── sota_comparison/          # State-of-the-art method comparison
│   ├── ablation_studies/         # GTF mechanism ablation studies  
│   └── large_scale_training/     # 800K sample training experiments
├── data_processing/              # Data preprocessing and feature extraction
│   ├── scientific_ecg_feature_extractor.py
│   ├── extract_comprehensive_ecg_features.py
│   └── mimic_ecg_refined_labels.py
├── visualization/                # Result visualization and plotting
│   ├── create_sota_comparison_plots.py
│   └── comprehensive_research_achievements_summary.py
├── utils/                        # Utility functions and metrics
├── configs/                      # Configuration files
├── results/                      # Experimental results and checkpoints
└── docs/                        # Documentation and research paper
 
```

## 🧠 Model Architecture

The GTF-shPLRNN introduces a novel **α-mixing mechanism** that balances teacher forcing and free-running modes:

```
z_{t+1} = A * z_t + W1 * ReLU(W2 * z_t + h2) + h1 + C * s_t

GTF Loss = α * L_teacher + (1-α) * L_free_running
```

### Key Features:

- **Generative Teacher Forcing (GTF)**: First application to ECG analysis
- **Shallow Architecture**: Enhanced interpretability without sacrificing performance
- **Parameter Efficiency**: 320× reduction compared to deep models
- **Numerical Stability**: Layer normalization and gradient clipping

## 🏥 Clinical Applications

### Multi-label Disease Classification

Our system supports simultaneous diagnosis of **25 cardiac conditions**:

- **Normal Rhythm**: F1 = 0.81 (excellent specificity)
- **Atrial Fibrillation**: F1 = 0.67 (good sensitivity)  
- **Ventricular Arrhythmia**: F1 = 0.42 (acceptable for rare conditions)
- **Overall Precision**: 0.89 for rare conditions (low false positive rate)

### Deployment Advantages

- ✅ **Edge Computing Ready**: 230KB model size
- ✅ **Real-time Processing**: 2.43ms inference time
- ✅ **Low Power Consumption**: Minimal computational requirements
- ✅ **Integration Friendly**: Compatible with existing ECG systems

## 📈 Experimental Results

### Large-Scale Training (800K Records)

| Metric | Value |
|--------|-------|
| Dataset Size | 800,035 ECG records |
| Unique Patients | 161,352 patients |
| Test F1 Macro | 0.3607 |
| Test F1 Micro | 0.5886 |
| Test Accuracy | **90.48%** |
| Test AUC | 83.52% |
| Training Time | 8.6 minutes (A100 GPU) |


## 💻 System Requirements

- Python 3.8+
- PyTorch 1.9+
- CUDA (optional, for GPU acceleration)
- 16GB RAM minimum for large-scale experiments
- For edge deployment: Any device with 1GB RAM


```

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines for details.

## 📧 Contact

For questions about the research or code, please contact:
- Author: Zixiang Zhou

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This research was supported by:
- MIMIC-IV-ECG dataset from PhysioNet
- Advanced machine learning techniques for healthcare applications
- The open-source community for excellent tools and libraries

