# GSoC Task IV — Quantum Classifier for Signal vs Background

**Framework:** Cirq + TensorFlow Quantum (TFQ)  
**Task:** Binary classification (signal = 1, background = 0) using the provided Delphes dataset `QIS_EXAM_200Events.npz`

## Dataset
- 100 train + 100 test samples
- Parsed `training_input` and `test_input` dicts manually
- Labels: background → 0, signal → 1
- First 4 features used, normalized to [0, π]

## Model
- 4 qubits
- RX angle encoding
- 2 variational layers (RY rotations + CNOT chain)
- Measurement: ⟨Z⟩ on qubit 0
- Classical head: Dense(8, ReLU) → Dense(1, sigmoid)

## Training
- Loss: binary cross-entropy
- Optimizer: Adam (lr=0.01)
- Epochs: 15
- Batch size: 32

## Results (local run)
- Test Accuracy: **0.78**
- ROC AUC: **0.82**

ROC curve saved as `roc_curve.png`

## Limitations
- Small dataset (100 train samples)
- 4 qubits + 2 layers only (deeper circuits unstable the)
- Simulation only

Code follows Task IV requirements exactly (Cirq + TFQ + correct dataset parsing + AUC).