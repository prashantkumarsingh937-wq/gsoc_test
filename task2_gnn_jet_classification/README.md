# GSoC Evaluation Task 2 — Jet Classification with GNNs

Classify quark vs gluon jets using two GNN architectures: GCN and GAT.

## Setup

```bash
pip install torch torch-geometric matplotlib numpy
python gsoc_gnn_jets.py
```

> For torch-geometric, check the official install page:
> https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html
> (version must match your torch + CUDA version)

---

## Dataset

Final results use the real quark-gluon dataset from Zenodo:
https://zenodo.org/record/3164691

Download `QG_jets.npz` and place it in the same directory as the script.
The file contains ~100k jets — we cap at 2500 for speed (controlled by `N_JETS`).

Features used per particle: `(pt, eta, phi)` — the 4th column (`pdgid`) is ignored.
Zero-padded particle slots (pt == 0) are filtered out before building the graph.

To run on synthetic data instead (no download needed), set `use_real_data = False` at the top of the script.

---

## Graph Construction

| Property | Choice |
|----------|--------|
| Nodes | Particles in the jet |
| Node features | (pt, eta, phi) |
| Edge method | k-nearest neighbors (k=5) in (η, φ) space |

k=5 is a reasonable default — enough connectivity without connecting particles that are physically far apart. (η, φ) is the natural angular space for jet physics; proximity there reflects actual collimation, unlike proximity in pt.

---

## Models

### GCN (`JetGCN`)
- 3 × `GCNConv` layers
- `global_mean_pool` to get jet-level representation
- Linear classifier head
- Dropout 0.3

### GAT (`JetGAT`)
- 3 × `GATConv` layers (4 attention heads on first two)
- Same pooling + head structure
- ELU activation (works better with attention than ReLU)
- Dropout 0.3 + internal attention dropout 0.2

---

## Output

| File | Description |
|------|-------------|
| `gnn_comparison.png` | Loss curves + val accuracy for both models |
| Terminal | Per-epoch metrics + final accuracy comparison |

---

## Results
Results may vary slightly across runs due to stochastic training, even with fixed seeds.

### Real dataset (Zenodo, 2500 jets, 40 epochs)

| Model | Val Accuracy |
|-------|-------------|
| GCN   | ~0.77        |
| GAT   | ~0.71        |

### Synthetic dataset (for reference)

| Model | Val Accuracy |
|-------|-------------|
| GCN   | ~0.78        |
| GAT   | ~0.82        |

Both models beat the 50% random baseline, which means the graph structure is capturing something real. GAT generally edges ahead — attention lets it down-weight particles that aren't relevant to the jet substructure. The gap is smaller on real data because the quark/gluon boundary is genuinely harder to separate than the synthetic distributions.

In this run, GCN slightly outperforms GAT. This is not unexpected — attention-based models can be more sensitive to training noise and dataset size, especially with limited statistics.

---

## Limitations

This implementation uses only (pt, eta, phi) as node features and does not include
particle identity (pdgid) or explicit edge features (e.g. ΔR). These are known to
improve performance and are left as future extensions.
