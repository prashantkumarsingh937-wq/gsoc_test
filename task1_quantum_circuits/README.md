# GSoC Evaluation Task — Quantum Circuits (PennyLane)

## Setup

```bash
pip install pennylane matplotlib
python gsoc_qml_task.py
```

---

## Circuit 1 — 5-qubit entangled circuit

**Wires:** 0, 1, 2, 3, 4

| Step | Gate | Purpose |
|------|------|---------|
| 1 | H on all qubits | Put each qubit in superposition |
| 2 | CNOT chain (0→1→2→3→4) | Entangle the full register |
| 3 | SWAP(0, 4) | Swap the two endpoints |
| 4 | RX(π/2) on qubit 2 | Local rotation to break symmetry |

**Output:** 32-element complex state vector (first 8 amplitudes printed)
**Diagram:** saved as `circuit1.png`

---

## Circuit 2 — Swap Test

**Wire layout:**

```
wire 0       →  ancilla
wires 1, 2   →  state |ψ⟩  (state A)
wires 3, 4   →  state |φ⟩  (state B)
```

**State preparation:**
- State A: H on wire 1, RX(π/3) on wire 2
- State B: H on wire 3, H on wire 4

**Swap test procedure:**
1. `H` on ancilla (wire 0)
2. `CSWAP(0, 1, 3)` — controlled-swap between qubit 1 (A) and qubit 3 (B)
3. `CSWAP(0, 2, 4)` — controlled-swap between qubit 2 (A) and qubit 4 (B)
4. `H` on ancilla again
5. Measure ancilla → get P(|0⟩) and P(|1⟩)

**Overlap formula:**

```
|<ψ|φ>|² = 1 - 2 * P(ancilla = |1⟩)
```

- P(|1⟩) ≈ 0   → states are nearly identical  
- P(|1⟩) ≈ 0.5 → states are orthogonal  

**Diagram:** saved as `circuit2_swaptest.png`

---

## Output files

| File | Description |
|------|-------------|
| `circuit1.png` | Circuit diagram for part 1 |
| `circuit2_swaptest.png` | Circuit diagram for swap test |
| Terminal output | State vector + ancilla probs + overlap value |
