"""
visualize_samplers.py
Script visualisasi ilmiah komparatif 3 strategi neighbor sampling
(Uniform, Topology-Aware, Importance-Based) untuk paper publikasi.
Desain minimalis, palet akademik, ukuran font standar, mudah dibaca dalam cetak hitam-putih.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_node(ax, xy, label, sublabel=None, node_type="selected", radius=0.048):
    """Menggambar simpul graf dengan gaya visual akademik minimalis."""
    x, y = xy
    if node_type == "target":
        facecolor = "#1C2833"  # Deep Charcoal
        edgecolor = "#000000"
        textcolor = "#FFFFFF"
        linestyle = "-"
        linewidth = 2.0
    elif node_type == "intermediate":
        facecolor = "#EAEDED"  # Light Slate Tint
        edgecolor = "#2C3E50"
        textcolor = "#1C2833"
        linestyle = "-"
        linewidth = 1.8
    elif node_type == "selected":
        facecolor = "#2C3E50"  # Solid Dark Navy/Slate
        edgecolor = "#1C2833"
        textcolor = "#FFFFFF"
        linestyle = "-"
        linewidth = 1.8
    else:  # unselected / pruned
        facecolor = "#F8F9F9"  # Very Pale Gray
        edgecolor = "#BDC3C7"  # Light Muted Gray
        textcolor = "#95A5A6"
        linestyle = "--"
        linewidth = 1.2

    circle = patches.Circle(
        (x, y),
        radius,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        zorder=10,
    )
    ax.add_patch(circle)

    if sublabel:
        ax.text(x, y + 0.008, label, ha="center", va="center", color=textcolor, fontsize=10, fontweight="bold", zorder=11)
        ax.text(x, y - 0.018, sublabel, ha="center", va="center", color=textcolor, fontsize=7.5, zorder=11)
    else:
        ax.text(x, y, label, ha="center", va="center", color=textcolor, fontsize=10, fontweight="bold", zorder=11)


def draw_edge(ax, start, end, weight_label=None, is_selected=True, edge_width=1.8, node_radius=0.048, label_pos=0.5, label_offset=(0.0, 0.0)):
    """Menggambar panah perambatan pesan (bottom-to-top) dengan label bobot sampling yang ter-stagger rapi."""
    x1, y1 = start
    x2, y2 = end
    import math

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return

    # Hitung titik potong tepi lingkaran agar panah rapi di luar lingkaran
    ux = dx / dist
    uy = dy / dist
    sx = x1 + ux * node_radius
    sy = y1 + uy * node_radius
    ex = x2 - ux * (node_radius + 0.015)
    ey = y2 - uy * (node_radius + 0.015)

    if is_selected:
        color = "#2C3E50"
        linestyle = "-"
        alpha = 1.0
        zorder = 4
    else:
        color = "#BDC3C7"
        linestyle = "--"
        alpha = 0.8
        zorder = 2

    ax.annotate(
        "",
        xy=(ex, ey),
        xytext=(sx, sy),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=edge_width,
            linestyle=linestyle,
            mutation_scale=11,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=zorder,
    )

    if weight_label:
        mid_x = sx + (ex - sx) * label_pos + label_offset[0]
        mid_y = sy + (ey - sy) * label_pos + label_offset[1]
        ax.text(
            mid_x,
            mid_y,
            weight_label,
            ha="center",
            va="center",
            fontsize=8,
            color="#1C2833" if is_selected else "#7F8C8D",
            fontweight="bold" if is_selected else "normal",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="#FFFFFF", edgecolor="#D5D8DC" if is_selected else "none", linewidth=0.5, alpha=0.95),
            zorder=6,
        )


def create_figure(output_dir: Path):
    """Merender diagram komparatif 2-panel berkualitas publikasi (Default vs Weighted)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.6), dpi=300)

    # Definisi koordinat tata letak (diberi jarak vertikal optimal agar label bebas tabrakan)
    pos_target = (0.50, 0.80)
    pos_user = (0.28, 0.49)
    pos_merchant = (0.72, 0.49)

    # Tuple: (posisi, label, label_pos_t, label_offset)
    candidates_user = {
        "T1": ((0.12, 0.15), "$T_1$", 0.42, (-0.028, 0.0)),
        "T2": ((0.28, 0.15), "$T_2$", 0.68, (0.0, 0.0)),
        "T3": ((0.44, 0.15), "$T_3$", 0.36, (0.028, 0.0)),
    }
    candidates_merchant = {
        "T4": ((0.56, 0.15), "$T_4$", 0.36, (-0.028, 0.0)),
        "T5": ((0.72, 0.15), "$T_5$", 0.68, (0.0, 0.0)),
        "T6": ((0.88, 0.15), "$T_6$", 0.42, (0.028, 0.0)),
    }

    # Konfigurasi 2 panel: Default vs Weighted
    panel_configs = [
        {
            "title": "(a) Uniform Random Sampling (Default)",
            "subtitle": "Unweighted, equal selection probability",
            "desc": "Baseline: All candidate neighbors have equal probability ($P = 1/d$)",
            "user_weights": {
                "T1": (True, 1.8, "P=1/3"),
                "T2": (False, 1.0, "P=1/3"),
                "T3": (True, 1.8, "P=1/3"),
            },
            "merchant_weights": {
                "T4": (True, 1.8, "P=1/3"),
                "T5": (False, 1.0, "P=1/3"),
                "T6": (True, 1.8, "P=1/3"),
            },
        },
        {
            "title": "(b) Weighted Sampling (Topology & Importance)",
            "subtitle": "Prioritized by structural & attribute weights",
            "desc": "Proposed: Prioritizes high-weight candidates ($w(u, v)$) and prunes noise",
            "user_weights": {
                "T1": (False, 1.0, "w=0.21"),
                "T2": (True, 3.0, "w=0.95"),
                "T3": (True, 2.6, "w=0.85"),
            },
            "merchant_weights": {
                "T4": (True, 2.9, "w=0.90"),
                "T5": (False, 1.0, "w=0.25"),
                "T6": (True, 2.5, "w=0.81"),
            },
        },
    ]

    for ax, cfg in zip(axes, panel_configs):
        ax.set_xlim(0.02, 0.98)
        ax.set_ylim(0.02, 0.98)
        ax.set_aspect("equal")
        ax.axis("off")

        # Header panel (Tanpa rumus, menggunakan teks subjudul yang bersih dan elegan)
        ax.text(0.50, 0.96, cfg["title"], ha="center", va="top", fontsize=11, fontweight="bold", color="#1C2833")
        ax.text(0.50, 0.905, cfg["subtitle"], ha="center", va="top", fontsize=9.5, color="#566573", style="italic")

        # 1. Gambar Edge Hop 1 (Deterministik 1-ke-1)
        draw_edge(ax, pos_user, pos_target, weight_label=None, is_selected=True, edge_width=2.2)
        draw_edge(ax, pos_merchant, pos_target, weight_label=None, is_selected=True, edge_width=2.2)

        # 2. Gambar Edge Hop 2 (Sampling Tetangga Bipartit dengan Staggering Label)
        for t_key, (is_sel, ew, lbl) in cfg["user_weights"].items():
            t_pos, _, t_pos_frac, t_off = candidates_user[t_key]
            draw_edge(ax, t_pos, pos_user, weight_label=lbl, is_selected=is_sel, edge_width=ew, label_pos=t_pos_frac, label_offset=t_off)

        for t_key, (is_sel, ew, lbl) in cfg["merchant_weights"].items():
            t_pos, _, t_pos_frac, t_off = candidates_merchant[t_key]
            draw_edge(ax, t_pos, pos_merchant, weight_label=lbl, is_selected=is_sel, edge_width=ew, label_pos=t_pos_frac, label_offset=t_off)

        # 3. Gambar Simpul Target (T0)
        draw_node(ax, pos_target, "$T_0$", sublabel="Target", node_type="target", radius=0.052)

        # 4. Gambar Simpul Perantara (User & Merchant)
        draw_node(ax, pos_user, "$U_1$", sublabel="User", node_type="intermediate", radius=0.046)
        draw_node(ax, pos_merchant, "$M_1$", sublabel="Merch", node_type="intermediate", radius=0.046)

        # 5. Gambar Simpul Calon Tetangga Hop 2
        for t_key, (is_sel, _, _) in cfg["user_weights"].items():
            t_pos, t_lbl, _, _ = candidates_user[t_key]
            draw_node(ax, t_pos, t_lbl, node_type="selected" if is_sel else "unselected", radius=0.042)

        for t_key, (is_sel, _, _) in cfg["merchant_weights"].items():
            t_pos, t_lbl, _, _ = candidates_merchant[t_key]
            draw_node(ax, t_pos, t_lbl, node_type="selected" if is_sel else "unselected", radius=0.042)

        # Keterangan ringkas di bagian bawah panel
        ax.text(0.50, 0.05, cfg["desc"], ha="center", va="bottom", fontsize=8, color="#566573")

        # Garis pembatas panel halus
        rect = patches.FancyBboxPatch(
            (0.03, 0.03),
            0.94,
            0.94,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="none",
            edgecolor="#E5E7E9",
            linewidth=1.0,
        )
        ax.add_patch(rect)

    # Legenda Bersama di Bagian Bawah Gambar
    legend_elements = [
        patches.Patch(facecolor="#1C2833", edgecolor="#000000", label="Target Transaction ($T_0$)"),
        patches.Patch(facecolor="#EAEDED", edgecolor="#2C3E50", label="Intermediate Entity ($U, M$)"),
        patches.Patch(facecolor="#2C3E50", edgecolor="#1C2833", label="Sampled Neighbor ($k=2$)"),
        patches.Patch(facecolor="#F8F9F9", edgecolor="#BDC3C7", linestyle="--", label="Pruned / Unsampled"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D5D8DC",
        fontsize=9,
        bbox_to_anchor=(0.50, -0.02),
    )

    fig.tight_layout(rect=[0.0, 0.05, 1.0, 1.0])

    png_path = output_dir / "neighbor_sampling_comparison.png"
    pdf_path = output_dir / "neighbor_sampling_comparison.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUKSES] Diagram publikasi berhasil disimpan ke:")
    print(f"  • PNG (300 DPI): {png_path.resolve()}")
    print(f"  • PDF (Vektor) : {pdf_path.resolve()}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "plots"
    create_figure(out)
