#!/usr/bin/env python3
"""Figuras del artículo sobre el apitxat de Gina (PDF vectorial, ancho de columna IEEE)."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AQUI = Path(__file__).parent
AZUL, NARANJA, AGUA, AMARILLO = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
TINTA, TINTA2, REJILLA = "#1f1f1e", "#5f5e5a", "#e4e3df"

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": REJILLA,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "pdf.fonttype": 42,
})
ANCHO = 3.45  # pulgadas, una columna IEEE

# ---------- Fig. 1: barrido de la mezcla de embeddings en inferencia (avl-v1, época 29) ----------
alfa = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8]
contraste = [3.1, 2.9, 3.4, 9.1, 13.7, 17.0]
sim = [0.949, 0.957, 0.931, 0.865, 0.702, 0.546]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(ANCHO, 1.75), constrained_layout=True)
a1.axhspan(16.1, 20.3, color=REJILLA, alpha=0.7, lw=0)
a1.text(0.02, 18.2, "DNV", color=TINTA2, fontsize=6.5, va="center")
a1.plot(alfa, contraste, color=AZUL, lw=1.5, marker="o", ms=4, mec="white", mew=0.8)
a1.set_xlabel(r"$\alpha$")
a1.set_ylabel("Contraste (dB)")
a1.set_ylim(0, 21)
a2.plot(alfa, sim, color=NARANJA, lw=1.5, marker="s", ms=4, mec="white", mew=0.8)
a2.set_xlabel(r"$\alpha$")
a2.set_ylabel("Similitud con Gina")
a2.set_ylim(0.5, 1.0)
for a, t in ((a1, "(a)"), (a2, "(b)")):
    a.set_title(t, loc="left", color=TINTA)
fig.savefig(AQUI / "fig_mezcla.pdf")

# ---------- Fig. 2: contraste por época (semilla fija) ----------
ep = [1, 3, 5, 7, 9, 11, 13, 15]
v2 = [10.1, 9.9, 15.9, 16.7, 15.4, 15.2, 11.7, 11.5]
v3 = [10.8, 11.8, 14.6, 16.0, 13.7, 12.9, 11.3, 10.9]
ep4 = list(range(1, 30, 2))
v4 = [16.3, 11.7, 15.1, 19.3, 12.2, 14.9, 17.9, 15.3, 14.2, 14.4, 17.5, 15.9, 16.0, 13.6, 16.2]
fig, ax = plt.subplots(figsize=(ANCHO, 2.0), constrained_layout=True)
ax.axhspan(16.1, 20.3, color=REJILLA, alpha=0.7, lw=0)
ax.text(29.6, 19.6, "locutores DNV", color=TINTA2, fontsize=6.5, ha="right", va="center")
ax.axhline(3.1, color=TINTA2, lw=0.8, ls=(0, (3, 2)))
ax.text(0.5, 1.6, "avl-v1 (referencia)", color=TINTA2, fontsize=6.5, ha="left")
ax.plot(ep, v2, color=AZUL, lw=1.5, marker="o", ms=3.5, mec="white", mew=0.6, label="v2: sin clips sonoros")
ax.plot(ep, v3, color=NARANJA, lw=1.5, marker="s", ms=3.5, mec="white", mew=0.6, label=r"v3: + mezcla $\alpha$=0,5")
ax.plot(ep4, v4, color=AGUA, lw=1.5, marker="D", ms=3.2, mec="white", mew=0.6, label="v4: + embedding congelado")
ax.set_xlabel("Época")
ax.set_ylabel("Contraste (dB)")
ax.set_xlim(0, 30)
ax.set_ylim(0, 21)
ax.legend(loc="lower right", frameon=False, bbox_to_anchor=(1.0, 0.19))
fig.savefig(AQUI / "fig_epocas.pdf")

# ---------- Fig. 3: caída LF por clase de sibilante ----------
clases = ["/z/", "/dz/", r"/ʒ/, /dʒ/", "sordas"]
sistemas = [
    ("DNV (mujer)", [-12.3, -18.2, -23.6, -32.7], AZUL, ""),
    ("Gina servidor", [-28.9, -29.8, -32.0, -32.7], NARANJA, "////"),
    ("avl-v1", [-32.8, -29.8, -35.3, -36.3], AMARILLO, "...."),
    ("avl-v4 e25", [-16.6, -20.0, -23.5, -35.7], AGUA, ""),
]
fig, ax = plt.subplots(figsize=(ANCHO, 2.35), constrained_layout=True)
w = 0.2
for i, (nom, vals, col, h) in enumerate(sistemas):
    xs = [k + (i - 1.5) * w for k in range(len(clases))]
    ax.bar(xs, [-v for v in vals], w * 0.9, color=col, hatch=h, edgecolor="white", lw=0.4, label=nom)
ax.set_xticks(range(len(clases)), clases)
ax.set_ylabel("Caída de energía < 400 Hz (dB)")
ax.set_ylim(0, 40)
ax.invert_yaxis()
ax.xaxis.tick_top()
fig.legend(loc="outside lower center", frameon=False, ncol=4, handlelength=1.4, columnspacing=0.9)
fig.savefig(AQUI / "fig_clases.pdf")
for f in ("fig_mezcla", "fig_epocas", "fig_clases"):
    print(AQUI / f"{f}.pdf")
