import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import functional_call, jacrev

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

class LayerWrapper(nn.Module):
    def __init__(self, layer_type, d=16):
        super().__init__()
        self.layer_type = layer_type
        if layer_type == "Linear (Bare)":
            self.net = nn.Linear(d, d)
        elif layer_type == "Linear+GELU+Lin":
            self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        elif layer_type == "Conv1d+GELU+Lin":
            # Trailing 1x1 conv removes GELU's -0.17 negative output floor
            self.net = nn.Sequential(
                nn.Conv1d(d, d, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(d, d, kernel_size=1),
            )
        elif layer_type == "DepthwiseConv1d":
            # Pure temporal filtering per channel (no cross-channel mixing)
            self.net = nn.Sequential(
                nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d),
                nn.GELU(),
                nn.Conv1d(d, d, kernel_size=3, padding=1, groups=d),
            )
        elif layer_type == "SelfAttention":
            self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=2, batch_first=True)
        elif layer_type == "CausalAttention":
            self.attn = nn.MultiheadAttention(embed_dim=d, num_heads=2, batch_first=True)
        elif layer_type == "GRU":
            self.gru = nn.GRU(input_size=d, hidden_size=d, batch_first=True)
        elif layer_type == "BiGRU":
            self.gru = nn.GRU(input_size=d, hidden_size=d // 2, batch_first=True, bidirectional=True)

    def forward(self, x):
        if self.layer_type in ("Linear (Bare)", "Linear+GELU+Lin"):
            return self.net(x)
        elif self.layer_type in ("Conv1d+GELU+Lin", "DepthwiseConv1d"):
            return self.net(x.transpose(1, 2)).transpose(1, 2)
        elif self.layer_type == "SelfAttention":
            out, _ = self.attn(x, x, x)
            return out
        elif self.layer_type == "CausalAttention":
            L = x.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
            out, _ = self.attn(x, x, x, attn_mask=causal_mask, is_causal=True)
            return out
        elif self.layer_type in ("GRU", "BiGRU"):
            out, _ = self.gru(x)
            return out

def generate_datasets(B=64, L=16, D=16):
    t = torch.linspace(0, 1, L, device=device).view(1, L, 1)
    datasets = {}

    # 1. Pointwise Nonlinear Mapping (Hard for Bare Linear & pure Attention)
    x_point = torch.randn(B, L, D, device=device)
    datasets["1. Pointwise Nonlin"] = (x_point, torch.sin(x_point))

    # 2. Local Temporal Shift (t-1) (Linear task -> Easy for Conv/GRU, impossible for Linear)
    phases = torch.rand(B, 1, 1, device=device) * 2 * torch.pi
    freqs = torch.arange(1, D + 1, device=device).view(1, 1, D) * 0.5
    x_wave = torch.sin(2 * torch.pi * freqs * t + phases)
    datasets["2. Local Shift (t-1)"] = (x_wave, torch.roll(x_wave, shifts=1, dims=1))

    # 3. Ill-Conditioned Scale (Feature 0 is 50x larger -> warps gradients)
    x_aniso = x_point.clone()
    x_aniso[:, :, 0] *= 50.0
    datasets["3. Ill-Conditioned"] = (x_aniso, torch.sin(x_point))

    # 4. Global Content Routing (Copy random needle token to all positions)
    x_sparse = torch.zeros(B, L, D, device=device)
    idx = torch.randint(0, L, (B,), device=device)
    needles = torch.randn(B, D, device=device)
    x_sparse[torch.arange(B), idx, :] = needles
    y_sparse = needles.unsqueeze(1).expand(-1, L, -1)
    datasets["4. Global Routing"] = (x_sparse, y_sparse)

    # 5. Anti-Causal Routing (Copy needle at t=14 backward to t=2 -> blocks CausalAttn & forward GRU)
    x_anticausal = torch.zeros(B, L, D, device=device)
    future_needle = torch.randn(B, D, device=device)
    x_anticausal[:, 14, :] = future_needle
    y_anticausal = torch.zeros(B, L, D, device=device)
    y_anticausal[:, 2, :] = future_needle
    datasets["5. Anti-Causal (14→2)"] = (x_anticausal, y_anticausal)

    # 6. Cross-Channel Interaction (y_c = x_c * x_{(c+1)%D} -> blocks DepthwiseConv1d)
    x_cross = torch.randn(B, L, D, device=device)
    y_cross = x_cross * torch.roll(x_cross, shifts=-1, dims=-1)
    datasets["6. Cross-Channel Mix"] = (x_cross, y_cross)

    return datasets

def analyze_ntk(model, x, y):
    params = dict(model.named_parameters())
    x_probe = x[:16]

    def compute_batch(params):
        return functional_call(model, params, (x_probe,))

    with torch.backends.cudnn.flags(enabled=False):
        jac_dict = jacrev(compute_batch)(params)

    J = torch.cat([
        jac_dict[name].reshape(x_probe.size(0), -1, param.numel())
        for name, param in params.items()
    ], dim=-1).detach()
    J_flat = J.reshape(-1, J.shape[-1])

    K = J_flat @ J_flat.T
    eigvals = torch.linalg.eigvalsh(K).clamp(min=1e-12)
    cond_number = (eigvals[-1] / eigvals[0]).item()

    y_flat = y[:16].reshape(-1, 1)
    alignment = ((y_flat.T @ K @ y_flat) / (
        torch.linalg.vector_norm(K) * (torch.linalg.vector_norm(y_flat) ** 2) + 1e-12
    )).item()
    return cond_number, alignment

def measure_convergence(layer_type, x, y, max_steps=500, target_loss=1e-2):
    model = LayerWrapper(layer_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    loss_history = []

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss_val = min(loss.item(), 1e4)
        loss_history.append(loss_val)

        if loss_val <= target_loss:
            loss_history.extend([loss_val] * (max_steps - step))
            return step, loss_history

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    return max_steps, loss_history

datasets = generate_datasets()
layers = [
    "Linear (Bare)",
    "Linear+GELU+Lin",
    "Conv1d+GELU+Lin",
    "DepthwiseConv1d",
    "SelfAttention",
    "CausalAttention",
    "GRU",
    "BiGRU",
]
cat_names = list(datasets.keys())

cond_matrix = np.zeros((len(layers), len(cat_names)))
align_matrix = np.zeros((len(layers), len(cat_names)))
steps_matrix = np.zeros((len(layers), len(cat_names)))
curves = {layer: {} for layer in layers}

print("Running expanded 8x6 layer landscape benchmark...")
for i, layer in enumerate(layers):
    for j, cat in enumerate(cat_names):
        x, y = datasets[cat]
        probe_model = LayerWrapper(layer).to(device)
        cond, align = analyze_ntk(probe_model, x, y)
        steps, history = measure_convergence(layer, x, y)

        cond_matrix[i, j] = np.log10(cond)
        align_matrix[i, j] = align
        steps_matrix[i, j] = steps
        curves[layer][cat] = history

# Visual Dashboard: 3 Heatmaps on Top, 8 Loss Curves (2x4) Below
plt.style.use("dark_background")
fig = plt.figure(figsize=(20, 14))
fig.suptitle("Expanded PyTorch Layer Landscape: Easy (Green) → Hard (Red)", fontsize=16, fontweight="bold", y=0.98)

def plot_heatmap(ax, data, title, cmap, fmt, vmin=None, vmax=None):
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(cat_names)))
    ax.set_yticks(np.arange(len(layers)))
    ax.set_xticklabels([c.split(". ", 1)[1] for c in cat_names], rotation=22, ha="right", fontsize=9.5)
    ax.set_yticklabels(layers, fontsize=10, fontweight="bold")
    ax.set_title(title, fontsize=12, pad=10)

    for r in range(data.shape[0]):
        for c in range(data.shape[1]):
            val = data[r, c]
            norm_val = (val - np.min(data)) / (np.max(data) - np.min(data) + 1e-8)
            txt_color = "black" if 0.25 < norm_val < 0.75 else "white"
            ax.text(c, r, fmt.format(val), ha="center", va="center", color=txt_color, fontweight="bold", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Row 1: Heatmaps
ax1 = fig.add_subplot(3, 3, 1)
plot_heatmap(ax1, steps_matrix, "Empirical Steps to Converge (↓ Easier)", "RdYlGn_r", "{:.0f}", vmin=0, vmax=500)

ax2 = fig.add_subplot(3, 3, 2)
plot_heatmap(ax2, cond_matrix, "Init NTK Log10 Condition # (↓ Smoother)", "RdYlGn_r", "1e{:.1f}")

ax3 = fig.add_subplot(3, 3, 3)
plot_heatmap(ax3, align_matrix, "Init NTK Target Alignment (↑ Better Fit)", "RdYlGn", "{:.3f}", vmin=0)

# Rows 2 & 3: 8 Loss Curves (2 rows x 4 cols)
palette = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]
for idx, layer in enumerate(layers):
    ax = fig.add_subplot(3, 4, 5 + idx)
    for c_idx, cat in enumerate(cat_names):
        ax.plot(curves[layer][cat], label=cat.split(". ", 1)[1], color=palette[c_idx], linewidth=1.8)
    ax.set_yscale("log")
    ax.axhline(1e-2, color="white", linestyle="--", alpha=0.5, label="Target (1e-2)")
    ax.set_title(f"{layer}", fontsize=11, fontweight="bold")
    if idx >= 4:
        ax.set_xlabel("Training Step", fontsize=9)
    if idx % 4 == 0:
        ax.set_ylabel("MSE Loss (Log)", fontsize=9.5)
    ax.grid(True, alpha=0.2)
    if idx == 3:
        ax.legend(loc="upper right", fontsize=7.5, ncol=2)

plt.subplots_adjust(left=0.09, right=0.97, top=0.92, bottom=0.07, wspace=0.35, hspace=0.45)
plt.savefig("layer_difficulty_landscape_8x6.png", dpi=150)
plt.show()
