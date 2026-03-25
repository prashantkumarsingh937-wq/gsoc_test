import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev1 = qml.device("default.qubit", wires=5)

@qml.qnode(dev1)
def circuit1():
    for i in range(5):
        qml.Hadamard(wires=i)

    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])

    qml.SWAP(wires=[0, 4])

    qml.RX(np.pi / 2, wires=2)

    return qml.state()

state = circuit1()
print("Circuit 1 — state vector (first 8 amplitudes):")
print(state[:8])
print()

fig1, ax1 = qml.draw_mpl(circuit1)()
ax1.set_title("Circuit 1: Entanglement + SWAP + RX")
fig1.tight_layout()
fig1.savefig("circuit1.png", dpi=120)
print("Circuit 1 diagram saved → circuit1.png")
plt.close(fig1)

dev2 = qml.device("default.qubit", wires=5)

@qml.qnode(dev2)
def swap_test():

    qml.Hadamard(wires=1)
    qml.RX(np.pi / 3, wires=2)           

    qml.Hadamard(wires=3)
    qml.Hadamard(wires=4)

    qml.Hadamard(wires=0)

    qml.CSWAP(wires=[0, 1, 3])   
    qml.CSWAP(wires=[0, 2, 4])   

    qml.Hadamard(wires=0)

    return qml.probs(wires=0)


probs = swap_test()
print("Swap Test — ancilla measurement probabilities:")
print(f"  P(|0⟩) = {probs[0]:.4f}")
print(f"  P(|1⟩) = {probs[1]:.4f}")
print()

overlap_sq = 1 - 2 * probs[1]
print(f"Estimated |<ψ|φ>|² = {overlap_sq:.4f}")
print()

fig2, ax2 = qml.draw_mpl(swap_test)()
ax2.set_title("Circuit 2: Swap Test (ancilla = wire 0)")
fig2.tight_layout()
fig2.savefig("circuit2_swaptest.png", dpi=120)
print("Circuit 2 diagram saved → circuit2_swaptest.png")
plt.close(fig2)

