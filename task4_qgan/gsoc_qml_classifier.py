import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import sympy
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import os

if not os.path.exists("QIS_EXAM_200Events.npz"):
    raise FileNotFoundError("QIS_EXAM_200Events.npz not found in current folder")

data = np.load("QIS_EXAM_200Events.npz", allow_pickle=True)

train_input = data["training_input"].item()
test_input  = data["test_input"].item()

X_train = np.concatenate([train_input["0"], train_input["1"]], axis=0)
y_train = np.concatenate([np.zeros(len(train_input["0"])), np.ones(len(train_input["1"]))])

X_test = np.concatenate([test_input["0"], test_input["1"]], axis=0)
y_test = np.concatenate([np.zeros(len(test_input["0"])), np.ones(len(test_input["1"]))])

print(f"Loaded {len(X_train)} train samples | {len(X_test)} test samples")
print(f"Signal (1) : {int(y_train.sum())} | Background (0) : {int(len(y_train) - y_train.sum())}")

if X_train.shape[1] > 4:
    X_train = X_train[:, :4]
    X_test  = X_test[:, :4]

max_vals = X_train.max(axis=0)
X_train = X_train / max_vals * np.pi
X_test  = X_test / max_vals * np.pi

qubits = cirq.LineQubit.range(4)

def create_circuit():
    circuit = cirq.Circuit()
    
    for i in range(4):
        circuit.append(cirq.rx(sympy.Symbol(f"x{i}")).on(qubits[i]))
    
    for i in range(4):
        circuit.append(cirq.ry(sympy.Symbol(f"theta{i}_1")).on(qubits[i]))
    for i in range(3):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    circuit.append(cirq.CNOT(qubits[3], qubits[0]))
    
    for i in range(4):
        circuit.append(cirq.ry(sympy.Symbol(f"theta{i}_2")).on(qubits[i]))
    for i in range(3):
        circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))
    circuit.append(cirq.CNOT(qubits[3], qubits[0]))
    
    return circuit

circuit = create_circuit()

def convert_to_circuits(X):
    circuits = []
    for x in X:
        c = cirq.Circuit()
        for i in range(4):
            c.append(cirq.rx(x[i]).on(qubits[i]))
        circuits.append(c)
    return tfq.convert_to_tensor(circuits)

X_train_c = convert_to_circuits(X_train)
X_test_c  = convert_to_circuits(X_test)

print("Converted features to quantum circuits")

pqc_layer = tfq.layers.PQC(
    circuit, 
    cirq.Z(qubits[0]),
    initializer=tf.keras.initializers.RandomUniform(minval=0, maxval=2*np.pi)
)

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(), dtype=tf.string),
    pqc_layer,
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

history = model.fit(
    X_train_c, y_train,
    validation_data=(X_test_c, y_test),
    epochs=15,
    batch_size=32,
    verbose=1
)

y_pred_prob = model.predict(X_test_c).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nTest Accuracy: {acc:.4f}")
print(f"ROC AUC      : {auc:.4f}")

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("roc_curve.png", dpi=120)
print("ROC curve saved → roc_curve.png")