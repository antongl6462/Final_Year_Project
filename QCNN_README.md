# Quantum CNN for Concrete Crack Classification

## Overview

This project implements a **Hybrid Quantum-Classical Convolutional Neural Network (QCNN)** for binary classification of concrete crack images. The model combines classical CNN feature extraction with a parameterized quantum circuit to leverage potential quantum advantages in pattern recognition.

## Architecture

### Hybrid Model Structure

The `HybridQuantumCNN` combines three components:

1. **Classical CNN Encoder**: Extracts spatial features from input images
2. **Quantum Circuit Layer**: 4-qubit variational quantum circuit with 2 entangling layers
3. **Classical Classifier**: Fusion layer for final binary prediction

```
Input Image (32×32) 
    ↓
Classical CNN (feature extraction)
    ↓
Quantum Embedding (dim=8)
    ↓
4-Qubit Variational Circuit (2 layers)
    ↓
Classical Hidden Layer (dim=64)
    ↓
Output (2 classes: cracked/not_cracked)
```

### Quantum Circuit Details

- **Framework**: PennyLane with PyTorch integration
- **Qubits**: 4 qubits (default configuration)
- **Quantum Layers**: 2 variational layers with trainable parameters
- **Entanglement**: Parameterized rotation gates + CNOT entangling gates
- **Measurement**: Expectation values of Pauli-Z operators

## Dataset

### Source
Concrete crack images for classification from the `raw_classification` directory containing:
- **Cracked**: 20,000 images
- **Not Cracked**: 20,000 images
- **Total**: 40,000 images

### Configuration (Notebook Settings)
- **Training Samples**: 7,000 (70%)
- **Validation Samples**: 1,500 (15%)
- **Test Samples**: 1,500 (15%)
- **Total Used**: 10,000 images (subset for computational efficiency)

### Preprocessing Pipeline
1. **Grayscale Conversion**: ✓ (reduces dimensionality)
2. **CLAHE (Contrast Limited Adaptive Histogram Equalization)**: ✓ (enhances crack visibility)
3. **Blackhat Transform**: ✗ (disabled)
4. **Quantum-Optimized Preprocessing**: ✓ (custom preprocessing for quantum input)
5. **Normalization**: Standard PyTorch ImageNet normalization
6. **Resize**: 32×32 pixels (optimized for quantum simulation speed)

## Training Configuration

### Hyperparameters
| Parameter | Value | Description |
|-----------|-------|-------------|
| Image Size | 32×32 px | Input dimensions |
| Batch Size | 32 | Samples per training batch |
| Epochs | 8 | Maximum training iterations |
| Learning Rate | 1e-3 | Initial learning rate (AdamW) |
| Weight Decay | 1e-4 | L2 regularization strength |
| Random Seed | 42 | For reproducibility |

### Quantum Architecture
| Parameter | Value | Description |
|-----------|-------|-------------|
| Number of Qubits | 4 | Quantum circuit qubits |
| Quantum Layers | 2 | Variational quantum layers |
| Quantum Embedding Dim | 8 | Dimension before quantum layer |
| Classical Hidden Dim | 64 | Classical feature dimension |

### Optimizer
- **Type**: AdamW (Adam with decoupled weight decay)
- **Learning Rate Schedule**: Static (no learning rate decay)
- **Early Stopping**: Best validation accuracy saved

## Notebook Structure

### `Quantum_CNN_Crack_Classification.ipynb`

Located in: `quantum_cnn/notebooks/`

#### Cells Overview:

1. **Environment Setup** (Cells 1-4)
   - Import dependencies (PyTorch, PennyLane, NumPy, matplotlib, sklearn)
   - Check Python environment and versions
   - Set random seeds for reproducibility
   - Detect GPU/CPU device

2. **Data Preparation** (Cells 5-8)
   - Navigate to project root (handles multiple execution contexts)
   - Validate dataset existence
   - Count images per class
   - Visualize sample images from each class

3. **Configuration** (Cell 10)
   - Initialize `QuantumCNNConfig` with all hyperparameters
   - Generate visual configuration table (pandas DataFrame)
   - Save configuration as PNG image (300 DPI)
   - Display table with pale blue/white alternating rows

4. **Model Building** (Cell 11)
   - Create PyTorch DataLoaders (train/val/test)
   - Initialize `HybridQuantumCNN` model
   - Move model to device (CPU/GPU)
   - Display model architecture summary

5. **Training** (Cell 13)
   - Train hybrid quantum CNN for 8 epochs
   - Track training and validation loss/accuracy
   - Save best model based on validation accuracy
   - Report total training time

6. **Evaluation** (Cells 15-18)
   - **Cell 15**: Test set evaluation with classification metrics
   - **Cell 16**: Confusion matrix visualization
   - **Cell 17**: ROC-AUC score (binary classification)
   - **Cell 18**: Training curves (loss & accuracy) with blue/orange color scheme

7. **Model Saving** (Cell 20)
   - Save best model weights to disk
   - Generate final summary with all metrics
   - Print output directory path

## Dependencies

### Core Libraries
```
torch>=2.0.0
pennylane>=0.44.0
numpy>=1.24.0
opencv-python>=4.8.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
```

### Python Environment
- **Recommended**: Python 3.10+
- **Virtual Environment**: `quantum_cnn/.venv` (dedicated environment)
- **Package Manager**: pip or conda

## File Structure

```
quantum_cnn/
├── notebooks/
│   └── Quantum_CNN_Crack_Classification.ipynb  # Main training notebook
└── .venv/  # Dedicated Python environment

src/
└── quantum_cnn/
    ├── __init__.py
    ├── config.py              # QuantumCNNConfig dataclass
    ├── data.py               # Dataset loading and preprocessing
    ├── hybrid_qcnn.py        # HybridQuantumCNN model definition
    ├── quantum_layer.py      # PennyLane quantum circuit layer
    ├── quantum_preprocessing.py  # Quantum-optimized preprocessing
    └── train_quantum_cnn.py  # Training loop implementation

runs/quantum_cnn/notebook_run/
├── best_hybrid_quantum_cnn.pt  # Saved model weights
└── quantum_cnn_config.png      # Configuration table image

raw_classification/
├── cracked/        # 20,000 cracked images
└── not_cracked/    # 20,000 non-cracked images
```

## Usage

### Running the Notebook

1. **Activate Environment**:
   ```bash
   cd quantum_cnn
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate     # Windows
   ```

2. **Open Notebook**:
   ```bash
   jupyter notebook notebooks/Quantum_CNN_Crack_Classification.ipynb
   # or use VS Code notebook interface
   ```

3. **Execute Cells Sequentially**:
   - Run cells 1-11 to load data and build model
   - Run cell 13 to train (expect 20-40 minutes on CPU with 10k images)
   - Run cells 15-18 to evaluate and visualize results
   - Run cell 20 to save model and generate summary

### Expected Performance

- **Training Time**: ~20-40 minutes (CPU, 10,000 images, 8 epochs)
- **Expected Accuracy**: 85-95% (depending on quantum circuit configuration)
- **Model Size**: ~500KB (saved PyTorch state dict)

## Key Features

### Quantum Advantages
- **Quantum Entanglement**: Captures complex correlations between features
- **Exponential Hilbert Space**: 4 qubits provide 2^4 = 16-dimensional quantum state space
- **Variational Optimization**: Trainable quantum parameters updated via backpropagation

### Computational Considerations
- **Simulation Overhead**: Quantum circuit simulation is computationally expensive
- **Scalability**: Simulation cost grows exponentially with number of qubits
- **Optimization**: Image size (32×32) and sample count (10k) optimized for reasonable training time
- **Future Work**: Real quantum hardware execution would eliminate simulation overhead

## Outputs

### Generated Files

1. **Model Checkpoint**: `runs/quantum_cnn/notebook_run/best_hybrid_quantum_cnn.pt`
   - PyTorch state dict with trained parameters
   - Can be loaded for inference or transfer learning

2. **Configuration Table**: `runs/quantum_cnn/notebook_run/quantum_cnn_config.png`
   - High-resolution (300 DPI) PNG image
   - Professional styling with pale blue alternating rows
   - Documents all hyperparameters for reproducibility

3. **Training Plots**: Generated in notebook (not saved by default)
   - Loss curves (train/val)
   - Accuracy curves (train/val)
   - Confusion matrix
   - Blue/orange color scheme, white background

### Evaluation Metrics

The notebook computes and displays:
- **Test Accuracy**: Overall classification accuracy on test set
- **Precision, Recall, F1-Score**: Per-class metrics
- **Confusion Matrix**: Visual heatmap of predictions vs. ground truth
- **ROC-AUC Score**: Binary classification performance metric
- **Training Time**: Total time for model training

## Comparison with Classical Approaches

This hybrid quantum CNN can be compared against:

1. **Classical CNN**: Pure PyTorch CNN without quantum layers
2. **YOLO Classification**: YOLOv8 classification model (separate notebook)
3. **Traditional ML**: SVM, Random Forest on handcrafted features

The quantum approach aims to demonstrate:
- Potential advantages in pattern recognition
- Feasibility of hybrid quantum-classical architectures
- Foundation for quantum machine learning research

## Troubleshooting

### Common Issues

1. **PennyLane NumPy Warning**:
   - Warning about NumPy < 2.0.0 is non-blocking
   - PennyLane 0.44.1 compatible with NumPy 1.26.4

2. **Out of Memory**:
   - Reduce batch size (e.g., 16 instead of 32)
   - Reduce max_train_samples (e.g., 5000 instead of 7000)
   - Use smaller image size (e.g., 28×28 instead of 32×32)

3. **Slow Training**:
   - Quantum simulation is CPU-intensive
   - Consider reducing n_qubits (e.g., 2-3 qubits)
   - Reduce n_q_layers to 1
   - Use GPU if available (limited quantum speedup)

4. **Module Import Errors**:
   - Ensure correct Python environment activated
   - Run from project root or notebook handles path resolution
   - Cell 11 includes module reload for development

## Research Context

### Motivation
Quantum machine learning explores potential quantum advantages in:
- **Feature Mapping**: Quantum states as high-dimensional feature spaces
- **Optimization**: Quantum algorithms for training neural networks
- **Pattern Recognition**: Quantum interference for classification

### Limitations
- **Current Stage**: Proof-of-concept on classical quantum simulators
- **NISQ Era**: Noisy Intermediate-Scale Quantum devices have limitations
- **Simulation Bottleneck**: Classical simulation limits scalability
- **Quantum Advantage**: Not yet demonstrated for practical applications

### Future Directions
- Test on real quantum hardware (IBM Quantum, Google Quantum AI)
- Explore quantum convolutional layers
- Investigate quantum data encoding strategies
- Compare with quantum kernel methods
- Scale to larger datasets with quantum acceleration

## References

### Frameworks
- **PennyLane**: Xanadu Quantum Technologies - https://pennylane.ai/
- **PyTorch**: Facebook AI Research - https://pytorch.org/

### Related Work
- Quantum Convolutional Neural Networks (QCNN)
- Variational Quantum Eigensolvers (VQE)
- Quantum Transfer Learning
- Hybrid Quantum-Classical Architectures

## License

This is a research project for academic purposes. Refer to the main project repository for licensing information.

## Contact & Contributions

For questions or contributions related to the quantum CNN implementation, please refer to the main project repository.

---

**Note**: This README documents the quantum CNN notebook (`Quantum_CNN_Crack_Classification.ipynb`) specifically. For classical approaches and overall project structure, see the main `README.md` in the project root.
