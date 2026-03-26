import numpy as np
import tensorflow as tf
import cirq
import sympy
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


def load_hep_dataset(path: str):
    """
    Load NPZ dataset. Expected keys: 'x_train','y_train','x_test','y_test'
    OR a single 'data','labels' key — we handle both layouts.
    """
    data = np.load(path, allow_pickle=True)
    keys = list(data.keys())
    print(f"NPZ keys: {keys}")

    if 'x_train' in keys:
        x_train = data['x_train'].astype(np.float32)
        y_train = data['y_train'].astype(np.float32).ravel()
        x_test  = data['x_test'].astype(np.float32)
        y_test  = data['y_test'].astype(np.float32).ravel()
    else:
        # single-array layout — split 100/100
        X = data[keys[0]].astype(np.float32)
        y = data[keys[1]].astype(np.float32).ravel()
        x_train, y_train = X[:100], y[:100]
        x_test,  y_test  = X[100:200], y[100:200]

    print(f"Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Signal fraction train: {y_train.mean():.2f}, test: {y_test.mean():.2f}")
    return x_train, y_train, x_test, y_test


def preprocess(x_train, x_test):
    """
    StandardScaler then clip + rescale to [-π, π] for angle encoding.
    This rescaling is important because the angle embedding method requires a bound on the input.
    Also, the clipping at 3σ ensures that outlier qubits are not saturated at π.
    """
    scaler = StandardScaler()
    x_tr = scaler.fit_transform(x_train)
    x_te = scaler.transform(x_test)

    x_tr = np.clip(x_tr, -3, 3) * (np.pi / 3)
    x_te = np.clip(x_te, -3, 3) * (np.pi / 3)
    return x_tr.astype(np.float32), x_te.astype(np.float32), scaler


N_QUBITS     = 4                      
N_LAYERS     = 3    
LATENT_DIM   = 4    


def build_generator_circuit(qubits: list, n_layers: int):
    """
    Hardware efficient ansatz: Ry(noise) → Alternating Ry/Rz + CNOT Ladder
    Encoding: The noise is encoded as a rotation in 4D space, i.e., z ∈ [-π, π]^4, which is then represented as a series of Ry rotations.
    This is the standard choice for continuous input variables, as it's injective (different noise leads to different states) and gradient-friendly.
    Entanglement: CNOT Ladder (q0→q1→q2→q3) in each layer.
    Why CNOT Ladder and not All-To-All? With 4 qubits and 3 layers, a CNOT Ladder achieves entanglement reachability in 3 steps (q0 reaches q3 in layer 2).
    Returns: circuit, sympy symbols (noise + variational params)
    """
    circuit = cirq.Circuit()

    noise_syms = [sympy.Symbol(f'z_{i}') for i in range(N_QUBITS)]

    var_syms = [[sympy.Symbol(f'w_{l}_{i}_{g}')
                 for g in range(2)]          
                for l in range(n_layers)
                for i in range(N_QUBITS)]
    var_flat = [s for layer in var_syms for s in layer]
    circuit += [cirq.ry(noise_syms[i])(qubits[i]) for i in range(N_QUBITS)]

    idx = 0
    for l in range(n_layers):
        for i in range(N_QUBITS):
            circuit += cirq.ry(var_flat[idx])(qubits[i])
            circuit += cirq.rz(var_flat[idx+1])(qubits[i])
            idx += 2
        for i in range(N_QUBITS - 1):
            circuit += cirq.CNOT(qubits[i], qubits[i+1])

    return circuit, noise_syms, var_flat


def build_generator_model(qubits, n_layers, n_features_out):
    """
    Hybrid generator:
      noise z (4-dim) → PQC → 4 Pauli-Z expectations → classical head → n_features_out

    The classical head (single linear layer) maps 4 qubit expectations to
    the feature dimensionality of the real data. No activation — we want
    the output range unconstrained so WGAN-GP can operate properly.
    """
    circuit, noise_syms, var_syms = build_generator_circuit(qubits, n_layers)
    observables = [cirq.Z(q) for q in qubits]
    pqc_layer = tfq.layers.PQC(circuit, observables)
    n_var = len(var_syms)
    n_noise = len(noise_syms)
    noise_input = tf.keras.Input(shape=(n_noise,), name='noise_input')
    quantum_out = pqc_layer(noise_input) 

    gen_out = tf.keras.layers.Dense(
        n_features_out,
        name='gen_head',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(quantum_out)

    model = tf.keras.Model(inputs=noise_input, outputs=gen_out, name='quantum_generator')
    return model, circuit, noise_syms, var_syms

def build_discriminator(n_features: int) -> tf.keras.Model:
    """
    Classical MLP discriminator.
    Justification for classical: 100 training samples → quantum discriminator would need careful initialization to prevent barren plateaus; classical MLP is more sample-efficient for this case
   -WGAN-GP needs to compute the gradient of D with respect to the inputs; this is easy for classical networks but expensive for quantum circuits
   -Spectral normalization (constraint) regulates the Lipschitz constant; this is the WGAN-GP requirement
    Architecture: 3 layers; shrinking width; no batch norm: cannot use batch norm and WGAN-GP at the same time; this would violate the Lipschitz condition
    """
    inp = tf.keras.Input(shape=(n_features,), name='d_input')
    x = tf.keras.layers.Dense(
        64, activation='swish',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(inp)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(
        32, activation='swish',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(1, name='d_out')(x)

    model = tf.keras.Model(inputs=inp, outputs=out, name='discriminator')
    return model

LAMBDA_GP   = 10.0   
N_CRITIC    = 5      
LR_G        = 2e-3   
LR_D        = 1e-4  
BATCH_SIZE  = 16    
N_EPOCHS    = 100


class WGANGP_Trainer:
    def __init__(self, generator, discriminator, latent_dim):
        self.G = generator
        self.D = discriminator
        self.latent_dim = latent_dim

        self.opt_G = tf.keras.optimizers.Adam(LR_G, beta_1=0.0, beta_2=0.9)
        self.opt_D = tf.keras.optimizers.Adam(LR_D, beta_1=0.0, beta_2=0.9)

        self.g_losses = []
        self.d_losses = []
        self.gp_values = []

    def gradient_penalty(self, real_batch, fake_batch):
        """
        Compute gradient penalty term: E[(||∇D(x̂)||₂ − 1)²]
        x̂ = εx_real + (1−ε)x_fake, ε ~ Uniform[0,1]
        """
        batch_size = tf.shape(real_batch)[0]
        eps = tf.random.uniform([batch_size, 1], 0.0, 1.0)
        interp = eps * real_batch + (1 - eps) * fake_batch

        with tf.GradientTape() as tape:
            tape.watch(interp)
            pred = self.D(interp, training=True)

        grads = tape.gradient(pred, interp)
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=1) + 1e-8)
        gp = tf.reduce_mean((norm - 1.0) ** 2)
        return gp

    @tf.function
    def train_discriminator_step(self, real_batch):
        batch_size = tf.shape(real_batch)[0]
        z = tf.random.uniform([batch_size, self.latent_dim], -np.pi, np.pi)

        with tf.GradientTape() as tape:
            fake_batch = self.G(z, training=True)
            d_real = self.D(real_batch, training=True)
            d_fake = self.D(fake_batch, training=True)
            w_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
            gp = self.gradient_penalty(real_batch, fake_batch)
            d_loss = w_loss + LAMBDA_GP * gp

        grads = tape.gradient(d_loss, self.D.trainable_variables)
        self.opt_D.apply_gradients(zip(grads, self.D.trainable_variables))
        return d_loss, gp

    @tf.function
    def train_generator_step(self, batch_size):
        z = tf.random.uniform([batch_size, self.latent_dim], -np.pi, np.pi)

        with tf.GradientTape() as tape:
            fake_batch = self.G(z, training=True)
            d_fake = self.D(fake_batch, training=True)
            g_loss = -tf.reduce_mean(d_fake)

        grads = tape.gradient(g_loss, self.G.trainable_variables)
        self.opt_G.apply_gradients(zip(grads, self.G.trainable_variables))
        return g_loss

    def train(self, x_real: np.ndarray, epochs: int, verbose_every: int = 10):
        dataset = tf.data.Dataset.from_tensor_slices(
            x_real.astype(np.float32)
        ).shuffle(len(x_real), seed=42).batch(BATCH_SIZE, drop_remainder=True)

        for epoch in range(1, epochs + 1):
            epoch_d, epoch_g, epoch_gp = [], [], []

            for real_batch in dataset:
                # N_CRITIC discriminator steps
                for _ in range(N_CRITIC):
                    d_l, gp = self.train_discriminator_step(real_batch)
                    epoch_d.append(float(d_l))
                    epoch_gp.append(float(gp))

                g_l = self.train_generator_step(tf.shape(real_batch)[0])
                epoch_g.append(float(g_l))

            mean_d = np.mean(epoch_d)
            mean_g = np.mean(epoch_g)
            mean_gp = np.mean(epoch_gp)
            self.d_losses.append(mean_d)
            self.g_losses.append(mean_g)
            self.gp_values.append(mean_gp)

            if epoch % verbose_every == 0:
                print(f"Epoch {epoch:4d} | D_loss: {mean_d:+.4f} | "
                      f"G_loss: {mean_g:+.4f} | GP: {mean_gp:.4f}")

    def generate_samples(self, n: int) -> np.ndarray:
        z = tf.random.uniform([n, self.latent_dim], -np.pi, np.pi)
        return self.G(z, training=False).numpy()


def build_classifier(n_features: int) -> tf.keras.Model:
    """
    Small classifier trained on real + generated data.
    The QGAN Hypothesis: Adding training set with generated samples from
    the learned distribution improves test set AUC.
    """
    inp = tf.keras.Input(shape=(n_features,))
    x = tf.keras.layers.Dense(32, activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(1e-3))(inp)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = tf.keras.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate_classifier(x_tr, y_tr, x_te, y_te, label='baseline'):
    clf = build_classifier(x_tr.shape[1])
    clf.fit(x_tr, y_tr, epochs=50, batch_size=16,
            validation_split=0.1, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=10, restore_best_weights=True)])
    y_pred_proba = clf.predict(x_te, verbose=0).ravel()
    y_pred = (y_pred_proba > 0.5).astype(int)
    acc = accuracy_score(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_pred_proba)
    auc_score = auc(fpr, tpr)
    print(f"[{label}] Accuracy: {acc:.4f}  AUC: {auc_score:.4f}")
    return acc, auc_score, fpr, tpr, y_pred_proba


def classical_baseline(x_tr, y_tr, x_te, y_te):
    """
    Logistic regression on the original features.
    This gives us a floor: if QGAN augmentation doesn't improve on this,
    then the quantum generator is not providing useful information.
    """
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=42)
    clf.fit(x_tr, y_tr)
    y_pred = clf.predict(x_te)
    y_proba = clf.predict_proba(x_te)[:, 1]
    acc = accuracy_score(y_te, y_pred)
    fpr, tpr, _ = roc_curve(y_te, y_proba)
    auc_score = auc(fpr, tpr)
    print(f"[LogReg baseline] Accuracy: {acc:.4f}  AUC: {auc_score:.4f}")
    return acc, auc_score, fpr, tpr


def plot_training_curves(trainer):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(trainer.d_losses, color='tomato', label='D loss')
    axes[0].set_title('Discriminator loss (Wasserstein)')
    axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(trainer.g_losses, color='steelblue', label='G loss')
    axes[1].set_title('Generator loss')
    axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(trainer.gp_values, color='seagreen', label='Gradient penalty')
    axes[2].axhline(1.0, color='gray', linestyle='--', linewidth=1, label='target GP=1')
    axes[2].set_title('Gradient penalty')
    axes[2].set_xlabel('Epoch'); axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle('WGAN-GP Training Curves', fontsize=13)
    plt.tight_layout()
    plt.savefig('qgan_training.png', dpi=130)
    print("Saved: qgan_training.png")
    plt.close()


def plot_roc_comparison(results: dict):
    plt.figure(figsize=(7, 6))
    colors = ['tomato', 'steelblue', 'seagreen', 'orange']
    for i, (label, (fpr, tpr, auc_s)) in enumerate(results.items()):
        plt.plot(fpr, tpr, color=colors[i % len(colors)],
                 label=f'{label} (AUC={auc_s:.3f})', linewidth=2)
    plt.plot([0,1],[0,1],'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title('ROC Curves — Signal vs Background')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('qgan_roc.png', dpi=130)
    print("Saved: qgan_roc.png")
    plt.close()


def plot_feature_distributions(x_real, x_gen, feature_idx=0):
    """Visual sanity check: do generated samples look like real data?"""
    plt.figure(figsize=(8, 4))
    plt.hist(x_real[:, feature_idx], bins=20, alpha=0.6,
             color='tomato', density=True, label='Real')
    plt.hist(x_gen[:, feature_idx], bins=20, alpha=0.6,
             color='steelblue', density=True, label='Generated')
    plt.xlabel(f'Feature {feature_idx} (normalized)')
    plt.ylabel('Density')
    plt.title('Real vs Generated Distribution (Feature 0)')
    plt.legend(); plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('qgan_distributions.png', dpi=130)
    print("Saved: qgan_distributions.png")
    plt.close()


def main(npz_path: str = 'hep_data.npz'):
    print("=" * 60)
    print("  QGAN for HEP Signal/Background Classification")
    print("  Cirq + TensorFlow Quantum | WGAN-GP")
    print("=" * 60)

    x_train, y_train, x_test, y_test = load_hep_dataset(npz_path)
    x_tr, x_te, scaler = preprocess(x_train, x_test)
    n_features = x_tr.shape[1]
    print(f"\nFeature dimension: {n_features}")

    print("\n[Classical baseline — Logistic Regression]")
    lr_acc, lr_auc, lr_fpr, lr_tpr = classical_baseline(x_tr, y_train, x_te, y_test)

    print("\n[NN classifier — real data only]")
    base_acc, base_auc, base_fpr, base_tpr, _ = train_and_evaluate_classifier(
        x_tr, y_train, x_te, y_test, label='real-only'
    )

    print("\n[Building QGAN...]")
    qubits = cirq.LineQubit.range(N_QUBITS)

    circuit, noise_syms, var_syms = build_generator_circuit(qubits, N_LAYERS)
    observables = [cirq.Z(q) for q in qubits]

    n_var = len(var_syms)
    n_noise = len(noise_syms)

    noise_in = tf.keras.Input(shape=(LATENT_DIM,), name='z')
    params = tf.keras.layers.Dense(
        n_noise + n_var, activation='tanh',
        name='param_map'
    )(noise_in)
    params_scaled = params * np.pi

    all_syms = noise_syms + list(var_syms)
    pqc_layer = tfq.layers.Expectation()(
        tf.expand_dims(tfq.convert_to_tensor([circuit]), 0),
        symbol_names=[str(s) for s in all_syms],
        symbol_values=params_scaled,
        operators=tfq.convert_to_tensor([[cirq.Z(q) for q in qubits]])
    )

    generator = build_generator_functional(LATENT_DIM, n_features, qubits)
    discriminator = build_discriminator(n_features)

    print(f"Generator params: {generator.count_params()}")
    print(f"Discriminator params: {discriminator.count_params()}")
    print(f"\n[Training QGAN — {N_EPOCHS} epochs, WGAN-GP]")
    x_signal = x_tr[y_train == 1]
    print(f"Training on {len(x_signal)} signal samples")

    trainer = WGANGP_Trainer(generator, discriminator, LATENT_DIM)
    trainer.train(x_signal, epochs=N_EPOCHS, verbose_every=20)
    plot_training_curves(trainer)

    N_GEN = 200
    x_gen = trainer.generate_samples(N_GEN)
    y_gen = np.ones(N_GEN, dtype=np.float32) 

    x_aug = np.vstack([x_tr, x_gen]).astype(np.float32)
    y_aug = np.concatenate([y_train, y_gen]).astype(np.float32)

    print("\n[NN classifier — real + QGAN augmented]")
    aug_acc, aug_auc, aug_fpr, aug_tpr, _ = train_and_evaluate_classifier(
        x_aug, y_aug, x_te, y_test, label='augmented'
    )

    plot_feature_distributions(x_signal, x_gen, feature_idx=0)

    roc_results = {
        'LogReg baseline': (lr_fpr, lr_tpr, lr_auc),
        'NN (real only)':  (base_fpr, base_tpr, base_auc),
        'NN + QGAN aug':   (aug_fpr, aug_tpr, aug_auc),
    }
    plot_roc_comparison(roc_results)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Logistic Regression:    AUC = {lr_auc:.4f}")
    print(f"  NN (real data only):    AUC = {base_auc:.4f}")
    print(f"  NN + QGAN augmentation: AUC = {aug_auc:.4f}")
    delta = aug_auc - base_auc
    print(f"  QGAN improvement:       ΔAUC = {delta:+.4f}")
    print("=" * 60)

    return trainer, generator


def build_generator_functional(latent_dim, n_features, qubits):
    """
    Functional generator that does not use the calling convention of the Expectation layer in TFQ.
    Architecture: z (latent_dim) → Dense(32, tanh) → Dense(N_QUBITS*2, tanh)*π → PQC → Dense(n_features)
    The dense layer before the PQC layer maps the latent noise to Ry/Rz angles for each qubit.
    This gives the generator an inductive bias: the generator will learn which rotations encode signal features, not random angles.
    """
    n_params = N_QUBITS * 2 * N_LAYERS + N_QUBITS  

    qc, noise_s, var_s = build_generator_circuit(qubits, N_LAYERS)
    all_syms = noise_s + list(var_s)
    sym_names = [str(s) for s in all_syms]
    obs = [cirq.Z(q) for q in qubits]

    noise_input = tf.keras.Input(shape=(latent_dim,), name='noise')

    noise_angles = tf.keras.layers.Dense(
        N_QUBITS, activation='tanh', name='noise_encoder'
    )(noise_input) * np.pi

    controlled_pqc = tfq.layers.ControlledPQC(
        qc,
        operators=obs,
        repetitions=None,   
        name='controlled_pqc'
    )

    quantum_out = controlled_pqc(
        [tfq.convert_to_tensor([qc])], 
        noise_angles                     
    )
    gen_out = tf.keras.layers.Dense(
        n_features,
        name='output_head',
        kernel_regularizer=tf.keras.regularizers.l2(1e-4)
    )(quantum_out)

    model = tf.keras.Model(inputs=noise_input, outputs=gen_out,
                           name='quantum_generator')
    return model


if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else 'hep_data.npz'
    trainer, generator = main(path)

