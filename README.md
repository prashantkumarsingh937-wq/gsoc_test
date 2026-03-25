# Quantum Machine Learning for High Energy Physics

### GSoC 2026 Evaluation Tasks

This repository contains my submission for the GSoC 2026 evaluation tasks focused on applying Quantum Machine Learning (QML) techniques to High Energy Physics (HEP) problems. The work is organized into four independent tasks, each targeting a different layer of the problem — from foundational quantum circuits to graph-based learning and hybrid quantum-classical models.

The goal throughout is not only to implement working solutions, but to demonstrate clear reasoning, correct methodology, and an honest assessment of limitations.

---

## Repository Structure

```
gsoc-test/
├── task1_quantum_circuits/
├── task2_gnn_jet_classification/
├── task3_qml_hep_essay/
└── task4_qml_classifier/
```

---

## Task 1 — Quantum Circuits

This task implements two quantum circuits using PennyLane, focusing on correctness and clarity of construction.

* A 5-qubit circuit generating superposition and entanglement through Hadamard gates and a CNOT chain, followed by SWAP and local rotation.
* A Swap Test circuit used to estimate the overlap between two quantum states using an ancilla qubit and controlled-SWAP operations.

The implementation follows the required gate sequence precisely, and circuit diagrams are generated for verification.

---

## Task 2 — Graph Neural Networks for Jet Classification

This task addresses quark–gluon jet classification using Graph Neural Networks (GNNs), based on the ParticleNet dataset.

Key aspects:

* Jets are represented as graphs constructed using k-nearest neighbors in angular (η, φ) space.
* Node features include transverse momentum (pt) and spatial coordinates.
* Two architectures are implemented:

  * Graph Convolutional Network (GCN) as a baseline
  * Graph Attention Network (GAT) to learn weighted neighbor contributions

The implementation uses real HEP data and reports validation performance consistent with expectations for this dataset. The comparison highlights the effect of attention mechanisms in capturing local structure.

---

## Task 3 — Quantum Machine Learning in HEP (Essay)

This section provides a critical analysis of the role of machine learning in High Energy Physics, with a focus on where Quantum Machine Learning may offer advantages.

The discussion covers:

* Current success of classical methods (GNNs, ParticleNet, GANs)
* Challenges such as high-dimensional correlations, generalization, and scalability
* Theoretical motivations for QML, including expressibility and representation of complex distributions
* Practical limitations of current quantum hardware and training methods

The conclusion emphasizes the need for careful benchmarking and realistic evaluation of QML approaches.

---

## Task 4 — Hybrid Quantum-Classical Classifier

This task implements a hybrid model using Cirq and TensorFlow Quantum (TFQ) for binary classification of signal vs background events.

Key components:

* Dataset: Provided NPZ file with labeled signal and background samples
* Feature encoding using parameterized quantum circuits (RX rotations)
* Variational quantum circuit with entanglement layers
* Classical post-processing using dense layers
* Evaluation using accuracy and ROC AUC metrics

---

## Execution Notes (Task 4)

TensorFlow Quantum has strict environment requirements and is not supported on standard Windows or Python 3.12 environments.

The implementation is designed to run under:

* Linux (or WSL)
* Python 3.10
* TensorFlow 2.11

Due to these constraints, execution may require a compatible setup. The code is complete and structured accordingly.

---

## Approach and Design Philosophy

* Prioritized correctness and alignment with task requirements
* Avoided unnecessary abstraction or over-engineering
* Used realistic assumptions and simple, interpretable models
* Explicitly acknowledged limitations where applicable

---

## Summary

This repository demonstrates:

* Practical implementation of quantum circuits and measurement techniques
* Application of graph-based deep learning to particle physics data
* Critical evaluation of Quantum Machine Learning in a scientific context
* Construction of hybrid quantum-classical models under real-world constraints

---

**Evaluation tasks completed and organized as required.**
