import torch
import torch.nn.functional as F
from torch.nn import Linear
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

np.random.seed(42)
torch.manual_seed(42)

use_real_data = True

REAL_DATA_PATH = "QG_jets.npz"

N_JETS = 2500


def load_real_jets(path, n_jets=N_JETS):
    """
    Load quark/gluon jets from the Zenodo .npz file.
    X shape: (N, M, 4) — N jets, up to M particles, 4 features (pt, eta, phi, pdgid)
    y shape: (N,) — 0 = gluon, 1 = quark
    We only use pt, eta, phi (first 3 features).
    """
    data = np.load(path, allow_pickle=True)
    X = data["X"][:n_jets]  
    y = data["y"][:n_jets]  

    dataset = []
    for i in range(len(X)):
        particles = X[i]  
        mask = particles[:, 0] > 0
        particles = particles[mask]
        if len(particles) < 2:
            continue
        x_np = particles[:, :3].astype(np.float32)  
        dataset.append((x_np, int(y[i])))

    return dataset


def make_fake_jets(n_jets=2000, max_particles=30, seed=42):
    """
    Simulate a jet dataset. Each jet contains a maximum of 30 particles, 
    Each particle contains 3 features: pt, eta, phi. Label: 0 = gluon, 1 = quark Real 
    dataset: https://zenodo.org/record/3164691
    """
    rng = np.random.RandomState(seed)
    dataset = []

    for i in range(n_jets):
        label = rng.randint(0, 2) 

        n_particles = rng.randint(8, max_particles)
        
        if label == 0:
            pt  = rng.exponential(0.3, n_particles)
            eta = rng.normal(0, 0.5, n_particles)
            phi = rng.uniform(-np.pi, np.pi, n_particles)
        else:        
            pt  = rng.exponential(0.6, n_particles)
            eta = rng.normal(0, 0.3, n_particles)
            phi = rng.uniform(-np.pi, np.pi, n_particles)

        x = np.stack([pt, eta, phi], axis=1).astype(np.float32)
        dataset.append((x, label))

    return dataset


def build_graph(x_np, k=5):
    """
    Convert one jet (numpy array of particles) into a PyG Data object.
    Edges are built using k-nearest neighbors in (eta, phi) space —
    particles that are close in angle are likely related.
    """
    x = torch.tensor(x_np, dtype=torch.float)
    
    pos = x[:, 1:3].clone()   

    n = x.shape[0]
    k_actual = min(k, n - 1)  

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  
    dist = (diff ** 2).sum(-1)                    
    
    dist.fill_diagonal_(float('inf'))
    _, idx = dist.topk(k_actual, dim=1, largest=False)
    
    row = torch.arange(n).unsqueeze(1).expand(-1, k_actual).reshape(-1)
    col = idx.reshape(-1)
    edge_index = torch.stack([row, col], dim=0)

    return Data(x=x, edge_index=edge_index, pos=pos)


def load_dataset(k=5):
    if use_real_data:
        print(f"loading real dataset from {REAL_DATA_PATH} ...")
        raw = load_real_jets(REAL_DATA_PATH, n_jets=N_JETS)
    else:
        print("using synthetic dataset (use_real_data=False)")
        raw = make_fake_jets(n_jets=2000)

    graphs = []
    for x_np, label in raw:
        g = build_graph(x_np, k=k)
        g.y = torch.tensor([label], dtype=torch.long)
        graphs.append(g)
    return graphs


class JetGCN(torch.nn.Module):
    """
    Simple GCN — 3 conv layers then global mean pool + MLP head.
    Nothing fancy, just to see if graph structure helps.
    """
    def __init__(self, in_dim=3, hidden=64, n_classes=2):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.conv3 = GCNConv(hidden, hidden)
        self.head  = Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch) 
        x = F.dropout(x, p=0.3, training=self.training)
        return self.head(x)


class JetGAT(torch.nn.Module):
    """
    GAT — same structure but with attention heads.
    Idea: not all neighbor particles are equally relevant,
    attention should let the model figure that out.
    """
    def __init__(self, in_dim=3, hidden=32, heads=4, n_classes=2):
        super().__init__()
        self.conv1 = GATConv(in_dim,          hidden, heads=heads, dropout=0.2)
        self.conv2 = GATConv(hidden * heads,   hidden, heads=heads, dropout=0.2)
        self.conv3 = GATConv(hidden * heads,   hidden, heads=1,     dropout=0.2, concat=False)
        self.head  = Linear(hidden, n_classes)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.elu(self.conv1(x, edge_index))
        x = F.elu(self.conv2(x, edge_index))
        x = F.elu(self.conv3(x, edge_index))

        x = global_mean_pool(x, batch)
        x = F.dropout(x, p=0.3, training=self.training)
        return self.head(x)


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out  = model(batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        out  = model(batch)
        loss = F.cross_entropy(out, batch.y.squeeze())
        total_loss += loss.item() * batch.num_graphs
        pred = out.argmax(dim=1)
        correct += (pred == batch.y.squeeze()).sum().item()
    return correct / len(loader.dataset), total_loss / len(loader.dataset)


def run_training(model, train_loader, val_loader, device, epochs=40, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_acc, val_loss = evaluate(model, val_loader, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if epoch % 5 == 0:
            print(f"  epoch {epoch:3d}  |  train_loss: {tr_loss:.4f}  "
                  f"val_loss: {val_loss:.4f}  val_acc: {val_acc:.4f}")

    return history

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}\n")

    print("building jet graphs...")
    graphs = load_dataset(k=5)

    split  = int(0.8 * len(graphs))
    train_data = graphs[:split]
    val_data   = graphs[split:]

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=64, shuffle=False)
    print(f"train jets: {len(train_data)}, val jets: {len(val_data)}\n")

    print("=" * 45)
    print("Training GCN")
    print("=" * 45)
    gcn = JetGCN(in_dim=3, hidden=64).to(device)
    gcn_hist = run_training(gcn, train_loader, val_loader, device, epochs=40)
    gcn_acc, _ = evaluate(gcn, val_loader, device)
    print(f"\nGCN final val accuracy: {gcn_acc:.4f}\n")

    print("=" * 45)
    print("Training GAT")
    print("=" * 45)
    gat = JetGAT(in_dim=3, hidden=64, heads=4).to(device)
    gat_hist = run_training(gat, train_loader, val_loader, device, epochs=40)
    gat_acc, _ = evaluate(gat, val_loader, device)
    print(f"\nGAT final val accuracy: {gat_acc:.4f}\n")

    print("=" * 45)
    print(f"  GCN val accuracy : {gcn_acc:.4f}")
    print(f"  GAT val accuracy : {gat_acc:.4f}")
    winner = "GCN" if gcn_acc > gat_acc else "GAT"
    print(f"  → {winner} wins (on this run)")
    print("=" * 45)

    epochs_range = range(1, 41)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs_range, gcn_hist["train_loss"], label="GCN train", color="steelblue")
    axes[0].plot(epochs_range, gcn_hist["val_loss"],   label="GCN val",   color="steelblue", linestyle="--")
    axes[0].plot(epochs_range, gat_hist["train_loss"], label="GAT train", color="tomato")
    axes[0].plot(epochs_range, gat_hist["val_loss"],   label="GAT val",   color="tomato",    linestyle="--")
    axes[0].set_title("Loss curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs_range, gcn_hist["val_acc"], label="GCN", color="steelblue")
    axes[1].plot(epochs_range, gat_hist["val_acc"], label="GAT", color="tomato")
    axes[1].axhline(0.5, color="gray", linestyle=":", linewidth=1, label="random baseline")
    axes[1].set_title("Validation Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("GCN vs GAT — Quark/Gluon Jet Classification", fontsize=13)
    plt.tight_layout()
    plt.savefig("gnn_comparison.png", dpi=130)
    print("\nplot saved → gnn_comparison.png")
    plt.show()
