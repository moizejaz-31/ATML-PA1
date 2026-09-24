"""
Method schematics drawn with matplotlib (no external tools), used by the Task 3 and
Task 4 notebooks. Every function returns the matplotlib Figure.

Colour use follows shared/src/plotting.py: entities keep their colour; text is ink.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

from shared.src.plotting import PALETTE, INK, INK_2, NEUTRAL, SURFACE, DOMAIN_COLORS

LIGHT = "#eef3fb"
MUTED = "#ecebe8"


def _canvas(w, h, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off"); ax.set_aspect("equal")
    return fig, ax


def box(ax, x, y, w, h, text, fc=LIGHT, ec=PALETTE[0], fs=9, weight="normal", color=INK, ls="-", lw=1.4):
    """Rounded box with its lower-left corner at (x, y)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight, wrap=True)
    return (x, y, w, h)


def arrow(ax, p, q, text=None, color=INK_2, fs=8, style="-|>", ls="-", rad=0.0, offset=(0, 0.12)):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(arrowstyle=style, color=color, lw=1.4, ls=ls,
                                                     shrinkA=2, shrinkB=2, connectionstyle=f"arc3,rad={rad}"))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text, ha="center", va="bottom",
                fontsize=fs, color=color)


def right(b):  return (b[0] + b[2], b[1] + b[3] / 2)
def left(b):   return (b[0], b[1] + b[3] / 2)
def top(b):    return (b[0] + b[2] / 2, b[1] + b[3])
def bottom(b): return (b[0] + b[2] / 2, b[1])


# ----------------------------------------------------------------------------- Task 3
def dg_protocol_diagram():
    """Task 3 information flow: three labelled sources in, Sketch locked until final evaluation."""
    fig, ax = _canvas(14, 6.2, (13, 5.8))
    ys = [4.6, 3.1, 1.6]
    src = []
    for y, d, name in zip(ys, ["photo", "art_painting", "cartoon"], ["Photo", "Art Painting", "Cartoon"]):
        c = DOMAIN_COLORS[d]
        src.append(box(ax, 0.2, y, 2.2, 0.9, f"{name}\n(labelled)", fc=SURFACE, ec=c, weight="semibold"))
        tr = box(ax, 3.0, y + 0.47, 1.6, 0.42, "train 80%", fc=SURFACE, ec=c, fs=8)
        va = box(ax, 3.0, y, 1.6, 0.42, "val 20%", fc=SURFACE, ec=c, fs=8)
        arrow(ax, right(src[-1]), left(tr)); arrow(ax, right(src[-1]), left(va))
    batch = box(ax, 5.3, 3.0, 2.0, 1.3, "domain-balanced\nbatch\n8 + 8 + 8", fc=LIGHT)
    for y in ys:
        arrow(ax, (4.6, y + 0.68), left(batch), rad=0.0)
    net = box(ax, 8.0, 3.0, 2.2, 1.3, "ResNet-18\n(ImageNet init,\nfrozen BN stats)\n→ 512-d F(x)", fc=LIGHT)
    arrow(ax, right(batch), left(net))
    obj = box(ax, 10.9, 3.35, 2.9, 1.9, "ERM:  CE\nDAN-DG:  CE + λ/3 Σ MMD²\nSAM:  max‖ε‖≤ρ CE(θ+ε)", fc=LIGHT, fs=8.5)
    arrow(ax, right(net), left(obj))
    sel = box(ax, 5.6, 0.5, 4.6, 1.2, "checkpoint selection: mean source-val macro-F1\n(validation splits only)", fc=SURFACE,
              ec=PALETTE[2], fs=8.5)
    for y, d in zip(ys, ["photo", "art_painting", "cartoon"]):
        arrow(ax, (4.6, y + 0.21), (5.6, 1.1), color=DOMAIN_COLORS[d], ls=":")
    arrow(ax, (10.2, 1.1), (10.9, 3.6), color=PALETTE[2], ls=":", rad=0.25)
    sk = box(ax, 10.9, 0.5, 2.9, 1.6, "Sketch (unseen target)\nlocked: loaded only in\n4_evaluation.ipynb", fc=MUTED,
             ec=NEUTRAL, color=INK_2, ls="--", fs=8.5)
    ax.text(7.75, 5.75, "Task 3: every training, diagnostic and selection step sees the three sources only", ha="center",
            fontsize=11, color=INK, weight="semibold")
    return fig


def dan_dg_diagram():
    """Pairwise source-domain MMD: three domain nodes, one MMD term per edge."""
    fig, ax = _canvas(12, 5.6, (11, 5.2))
    pos = {"photo": (1.6, 4.3), "art_painting": (4.6, 4.3), "cartoon": (3.1, 1.3)}
    names = {"photo": "F(X_Photo)", "art_painting": "F(X_Art)", "cartoon": "F(X_Cartoon)"}
    for d, (x, y) in pos.items():
        ax.add_patch(Circle((x, y), 0.78, fc=SURFACE, ec=DOMAIN_COLORS[d], lw=2.2))
        ax.text(x, y, names[d], ha="center", va="center", fontsize=9, color=INK)
    for a, b in [("photo", "art_painting"), ("photo", "cartoon"), ("art_painting", "cartoon")]:
        (x1, y1), (x2, y2) = pos[a], pos[b]
        v = np.array([x2 - x1, y2 - y1]); v = v / np.linalg.norm(v) * 0.82
        ax.annotate("", xy=(x2 - v[0], y2 - v[1]), xytext=(x1 + v[0], y1 + v[1]),
                    arrowprops=dict(arrowstyle="<|-|>", color=PALETTE[0], lw=1.6))
        ax.text((x1 + x2) / 2 + (0.35 if a != "photo" or b != "art_painting" else 0), (y1 + y2) / 2 + 0.18,
                "MMD²", fontsize=8.5, color=PALETTE[0], ha="center")
    box(ax, 6.6, 3.3, 5.1, 1.5, "L = CE(all source labels)\n+ (λ_DG / 3) · [MMD²(P,A) + MMD²(P,C) + MMD²(A,C)]",
        fc=LIGHT, fs=9)
    box(ax, 6.6, 1.0, 5.1, 1.7, "kernel = Task 2 multi-RBF\nbandwidths 0.5 / 1 / 2 × median sq. distance\n"
        "of each pair's own batch (8 + 8)\nno Sketch anywhere", fc=SURFACE, ec=NEUTRAL, fs=8.5, color=INK_2)
    ax.text(6, 5.3, "DAN-DG aligns the observed sources with each other, not with a target", ha="center",
            fontsize=11, weight="semibold", color=INK)
    return fig


def sam_toy_diagram(rho=0.35):
    """What SAM minimises, on a 1-D toy loss (illustration, not the real model), plus one SAM step."""
    w = np.linspace(-3, 3, 1200)
    def L(v):
        return 1.0 - 0.95 * np.exp(-((v + 1.4) ** 2) / 0.02) - 0.8 * np.exp(-((v - 1.2) ** 2) / 0.9) + 0.02 * v ** 2
    lw = L(w)
    k = int(round(rho / (w[1] - w[0])))
    sam = np.array([lw[max(0, i - k): i + k + 1].max() for i in range(len(w))])   # max_{|e|<=rho} L(w+e)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.35, 1]})
    a1.plot(w, lw, color=PALETTE[0], label="training loss L(w)")
    a1.plot(w, sam, color=PALETTE[3], label=f"SAM objective  max|ε|≤ρ L(w+ε),  ρ = {rho}")
    i_s, i_f = np.argmin(np.where(w < 0, lw, 9)), np.argmin(np.where(w > 0, lw, 9))
    for i, name in [(i_s, "sharp minimum"), (i_f, "flat minimum")]:
        a1.plot(w[i], lw[i], "o", color=INK, ms=6)
        a1.annotate(f"{name}\nL = {lw[i]:.2f},  L_SAM = {sam[i]:.2f}", (w[i], lw[i]), xytext=(0, -42),
                    textcoords="offset points", ha="center", fontsize=8.5, color=INK_2)
    a1.set_ylim(-0.25, 1.25); a1.set_xlabel("parameter w (toy, 1-D)"); a1.set_ylabel("loss")
    a1.set_title("SAM prefers minima whose neighbourhood is also low"); a1.legend(loc="upper center")

    a2.set_xlim(0, 10); a2.set_ylim(0, 7); a2.set_aspect("equal"); a2.axis("off")
    th = np.array([3.2, 2.6]); g = np.array([1.0, 0.6]); r = 2.2
    pert = th + r * g / np.linalg.norm(g)
    a2.add_patch(Circle(tuple(th), r, fc="none", ec=NEUTRAL, ls="--"))
    a2.text(th[0] - r + 0.1, th[1] + r - 0.2, "‖ε‖ ≤ ρ", fontsize=9, color=NEUTRAL)
    a2.plot(*th, "o", color=INK, ms=8); a2.text(th[0] - 0.35, th[1] - 0.45, "θ", fontsize=13)
    arrow(a2, tuple(th), tuple(pert), color=PALETTE[1])
    a2.text(pert[0] + 0.15, pert[1] + 0.1, "① ascend: θ + ε\nε = ρ ∇L(θ)/‖∇L(θ)‖", fontsize=8.5, color=PALETTE[1])
    arrow(a2, tuple(pert), tuple(pert + [0.0, -1.3]), color=PALETTE[3])
    a2.text(pert[0] + 0.15, pert[1] - 1.1, "② gradient at θ + ε", fontsize=8.5, color=PALETTE[3])
    arrow(a2, tuple(th), tuple(th + [0.0, -1.3]), color=PALETTE[3], ls="--")
    a2.text(th[0] + 0.15, th[1] - 1.6, "③ restore θ, AdamW step\n    with ∇L(θ + ε)", fontsize=8.5, color=PALETTE[3])
    a2.text(5, 6.5, "One SAM update = 2 forward/backward passes", ha="center", fontsize=11, weight="semibold")
    a2.text(5, 0.05, "BatchNorm running stats stay frozen in both passes", ha="center", fontsize=8.5, color=INK_2)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- Task 4
def cifar_resnet_diagram():
    """ImageNet stem vs the CIFAR stem required by the manual, with feature-map sizes."""
    fig, ax = _canvas(15, 5.4, (14, 5))
    ax.text(7.5, 5.05, "ResNet-18 for 32×32 CIFAR images", ha="center", fontsize=12, weight="semibold")
    ax.text(0.2, 4.3, "ImageNet stem (not used)", fontsize=9, color=INK_2)
    b0 = box(ax, 0.2, 3.3, 2.0, 0.8, "input\n3×32×32", fc=SURFACE, ec=NEUTRAL, color=INK_2)
    b1 = box(ax, 2.8, 3.3, 2.4, 0.8, "7×7 conv, stride 2\n→ 64×16×16", fc=MUTED, ec=NEUTRAL, color=INK_2)
    b2 = box(ax, 5.8, 3.3, 2.4, 0.8, "3×3 max-pool, s2\n→ 64×8×8", fc=MUTED, ec=NEUTRAL, color=INK_2)
    arrow(ax, right(b0), left(b1), color=NEUTRAL); arrow(ax, right(b1), left(b2), color=NEUTRAL)
    ax.text(8.5, 3.6, "4× smaller before layer1: too\nmuch resolution lost for 32×32", fontsize=8.5, color=INK_2)

    ax.text(0.2, 2.35, "CIFAR stem (used)", fontsize=9, color=PALETTE[0], weight="semibold")
    c0 = box(ax, 0.2, 1.1, 1.6, 0.9, "input\n3×32×32")
    c1 = box(ax, 2.2, 1.1, 2.0, 0.9, "3×3 conv, s1\nBN, ReLU\n64×32×32")
    specs = [("layer1", "64×32×32"), ("layer2", "128×16×16"), ("layer3", "256×8×8"), ("layer4", "512×4×4")]
    prev = c1
    for i, (n, s) in enumerate(specs):
        b = box(ax, 4.6 + i * 1.95, 1.1, 1.7, 0.9, f"{n}\n{s}", fc=LIGHT if n != "layer2" else "#fde7dc",
                ec=PALETTE[0] if n != "layer2" else PALETTE[1])
        arrow(ax, right(prev), left(b)); prev = b
    gp = box(ax, 12.45, 1.1, 1.3, 0.9, "avg-pool\nf ∈ ℝ⁵¹²")
    arrow(ax, right(prev), left(gp))
    fc = box(ax, 13.95, 1.1, 0.9, 0.9, "fc\n10")
    arrow(ax, right(gp), left(fc))
    arrow(ax, right(c0), left(c1))
    ax.text(7.5, 0.35, "PROSER mixes layer2 outputs (orange); f is the penultimate feature used by Mahalanobis",
            ha="center", fontsize=8.5, color=INK_2)
    return fig


def osr_pipeline_diagram():
    """Open-set decision rule shared by every method in Task 4."""
    fig, ax = _canvas(15, 5, (14, 4.6))
    ax.text(7.5, 4.6, "Open-set recognition: classify, then decide whether to trust the prediction", ha="center",
            fontsize=12, weight="semibold")
    k = box(ax, 0.1, 2.9, 2.7, 0.8, "known: CIFAR-10 test", fc=SURFACE, ec=PALETTE[0])
    n = box(ax, 0.1, 1.8, 2.7, 0.8, "near: 8 CIFAR-100\nclasses (800 images)", fc=SURFACE, ec=PALETTE[1])
    f = box(ax, 0.1, 0.7, 2.7, 0.8, "far: 8 CIFAR-100\nclasses (800 images)", fc=SURFACE, ec=PALETTE[2])
    net = box(ax, 3.3, 1.5, 2.2, 1.6, "trained model\n(frozen)\nlogits z, feature f")
    for b in (k, n, f):
        arrow(ax, right(b), left(net))
    sc = box(ax, 6.3, 1.5, 3.0, 1.6, "unknownness u(x)\nMSP · MLS · Energy\nMahalanobis · PROSER")
    arrow(ax, right(net), left(sc))
    th = box(ax, 10.1, 1.5, 1.9, 1.6, "u(x) ≤ τ ?", fc="#fff4d6", ec=PALETTE[3], weight="semibold")
    arrow(ax, right(sc), left(th))
    acc = box(ax, 12.7, 2.6, 2.1, 0.9, "accept\n→ argmax z (CSA)", fc=SURFACE, ec=PALETTE[5])
    rej = box(ax, 12.7, 1.0, 2.1, 0.9, "reject\n→ 'unknown'", fc=SURFACE, ec=PALETTE[7])
    arrow(ax, (12.0, 2.6), left(acc), text="yes", offset=(0, 0.05)); arrow(ax, (12.0, 2.0), left(rej), text="no", offset=(0, -0.35))
    tau = box(ax, 5.6, 0.1, 6.4, 0.8, "τ = 95th percentile of u on CIFAR-10 validation\n(no unknown image is involved)",
              fc=SURFACE, ec=NEUTRAL, fs=8.5, color=INK_2)
    arrow(ax, (11.0, 0.9), bottom(th), color=NEUTRAL)
    return fig


def proser_diagram():
    """PROSER training step: half the batch builds data placeholders, the other half trains CE + classifier placeholders."""
    fig, ax = _canvas(16, 6.4, (14.5, 5.9))
    ax.text(8, 6.05, "PROSER: one mini-batch of 128 known images", ha="center", fontsize=12, weight="semibold")
    b = box(ax, 0.2, 2.6, 1.9, 1.3, "batch\n128 images\n(CIFAR-10 only)")
    h1 = box(ax, 2.8, 4.2, 2.1, 0.9, "second half (64)", fc=SURFACE, ec=PALETTE[1])
    h2 = box(ax, 2.8, 1.4, 2.1, 0.9, "first half (64)", fc=SURFACE, ec=PALETTE[0])
    arrow(ax, right(b), left(h1)); arrow(ax, right(b), left(h2))
    pre = box(ax, 5.6, 4.2, 2.2, 0.9, "conv1 → layer2\nh = φ_pre(x)", fc="#fde7dc", ec=PALETTE[1])
    mix = box(ax, 8.4, 4.0, 2.6, 1.3, "h̃ = λh_i + (1−λ)h_j\ny_i ≠ y_j,  λ ~ Beta(2,2)", fc="#fde7dc", ec=PALETTE[1], fs=8.5)
    post = box(ax, 11.6, 4.2, 2.0, 0.9, "layer3 → layer4\n→ f̃", fc="#fde7dc", ec=PALETTE[1])
    arrow(ax, right(h1), left(pre)); arrow(ax, right(pre), left(mix)); arrow(ax, right(mix), left(post))
    l_data = box(ax, 13.9, 3.95, 2.0, 1.4, "ℓ_data = CE(\n[z, max d], K+1)\n× γ = 0.1", fc=SURFACE, ec=PALETTE[1], fs=8.5)
    arrow(ax, right(post), left(l_data))

    full = box(ax, 5.6, 1.4, 2.2, 0.9, "full network\n→ f", fc=LIGHT)
    heads = box(ax, 8.4, 1.1, 2.6, 1.5, "known head z ∈ ℝ¹⁰\ndummy head d ∈ ℝ⁵\nf̂ = [z, max d]", fc=LIGHT, fs=8.5)
    arrow(ax, right(h2), left(full)); arrow(ax, right(full), left(heads))
    l_ce = box(ax, 11.6, 1.95, 4.3, 0.85, "ℓ_CE = CE(f̂, y)", fc=SURFACE, ec=PALETTE[0], fs=8.5)
    l_clf = box(ax, 11.6, 0.6, 4.3, 1.05, "ℓ_clf = CE(f̂ with z_y masked, K+1) × β = 1\n'after the true class, a dummy wins'",
                fc=SURFACE, ec=PALETTE[0], fs=8.5)
    arrow(ax, right(heads), left(l_ce)); arrow(ax, right(heads), left(l_clf))
    ax.text(8, 0.05, "no CIFAR-100 image is used; the placeholders are built only from known classes",
            ha="center", fontsize=8.5, color=INK_2)
    return fig
