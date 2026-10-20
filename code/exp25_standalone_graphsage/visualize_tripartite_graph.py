"""
visualize_tripartite_graph.py
Script visualisasi graf tripartit heterogen (User - Transaksi - Merchant)
untuk paper publikasi. Desain sederhana, minimalis, palet akademik monokromatik,
font standar, ramah cetak hitam-putih.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def draw_node(ax, xy, label, sublabel=None, node_type="user", radius=0.045):
    """Menggambar simpul pada partisi graf."""
    x, y = xy
    if node_type == "user":
        facecolor = "#EAEDED"  # Light Slate Tint
        edgecolor = "#2C3E50"  # Dark Slate
        textcolor = "#1C2833"
        linewidth = 1.8
        linestyle = "-"
    elif node_type == "merchant":
        facecolor = "#EAEDED"
        edgecolor = "#2C3E50"
        textcolor = "#1C2833"
        linewidth = 1.8
        linestyle = "-"
    elif node_type == "tx_normal":
        facecolor = "#2C3E50"  # Solid Dark Slate
        edgecolor = "#1C2833"
        textcolor = "#FFFFFF"
        linewidth = 1.8
        linestyle = "-"
    elif node_type == "tx_fraud":
        facecolor = "#1C2833"  # Deep Charcoal / Accent
        edgecolor = "#C0392B"  # Subtle Dark Red Border (atau kontras tebal)
        textcolor = "#FFFFFF"
        linewidth = 2.5
        linestyle = "-"

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
        ax.text(x, y + 0.009, label, ha="center", va="center", color=textcolor, fontsize=10, fontweight="bold", zorder=11)
        ax.text(x, y - 0.016, sublabel, ha="center", va="center", color=textcolor, fontsize=7.5, zorder=11)
    else:
        ax.text(x, y, label, ha="center", va="center", color=textcolor, fontsize=10, fontweight="bold", zorder=11)


def draw_edge(ax, start, end, node_radius=0.045, is_fraud=False):
    """Menggambar garis relasi tanpa panah arah (undirected/bidirectional)."""
    x1, y1 = start
    x2, y2 = end
    import math

    dx = x2 - x1
    dy = y2 - y1
    dist = math.hypot(dx, dy)
    if dist == 0:
        return

    ux = dx / dist
    uy = dy / dist
    sx = x1 + ux * node_radius
    sy = y1 + uy * node_radius
    ex = x2 - ux * node_radius
    ey = y2 - uy * node_radius

    color = "#1C2833" if is_fraud else "#566573"
    linewidth = 2.2 if is_fraud else 1.5
    linestyle = "-"

    ax.plot([sx, ex], [sy, ey], color=color, linewidth=linewidth, linestyle=linestyle, zorder=4)


def create_tripartite_figure(output_dir: Path):
    """Merender graf tripartit bersih dan elegan."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Rasio aspek 16:10 agar gambar proporsional pada paper dua kolom/satu kolom
    fig, ax = plt.subplots(figsize=(9.6, 5.2), dpi=300)
    ax.set_xlim(0.0, 1.6)
    ax.set_ylim(-0.02, 0.94)
    ax.set_aspect("equal")
    ax.axis("off")

    # Posisi Kolom Partisi Tripartit:
    # Kolom 1 (Kiri): User
    # Kolom 2 (Tengah): Transaction
    # Kolom 3 (Kanan): Merchant
    x_user = 0.28
    x_tx = 0.80
    x_merch = 1.32

    # Baris 1: Judul Partisi di Bagian Atas (jarak lebih rapat ke simpul di bawahnya)
    ax.text(x_user, 0.88, "Cardholders ($U$)", ha="center", va="center", fontsize=11, fontweight="bold", color="#1C2833")
    ax.text(x_tx, 0.88, "Transactions ($T$ - Target)", ha="center", va="center", fontsize=11, fontweight="bold", color="#1C2833")
    ax.text(x_merch, 0.88, "Merchants ($M$)", ha="center", va="center", fontsize=11, fontweight="bold", color="#1C2833")

    # Garis partisi vertikal halus (pembatas latar belakang kolom)
    for xc in [0.54, 1.06]:
        ax.plot([xc, xc], [0.17, 0.84], color="#EAEDED", linestyle=":", linewidth=1.2, zorder=1)

    # Baris 2: Visualisasi Simpul Graf
    # Users
    users = {
        "U1": ((x_user, 0.70), "$U_1$", "Cardholder 1"),
        "U2": ((x_user, 0.36), "$U_2$", "Cardholder 2"),
    }

    # Transactions
    txs = {
        "T1": ((x_tx, 0.78), "$T_1$", "Normal", "tx_normal"),
        "T2": ((x_tx, 0.60), "$T_2$", "Fraud", "tx_fraud"),
        "T3": ((x_tx, 0.42), "$T_3$", "Normal", "tx_normal"),
        "T4": ((x_tx, 0.24), "$T_4$", "Normal", "tx_normal"),
    }

    # Merchants
    merchants = {
        "M1": ((x_merch, 0.78), "$M_1$", "Retail"),
        "M2": ((x_merch, 0.51), "$M_2$", "Online / Hub"),
        "M3": ((x_merch, 0.24), "$M_3$", "Gas / Travel"),
    }

    # Definisi Edges (Relasi Bipartit: User - Tx dan Merchant - Tx)
    # Setiap transaksi selalu menghubungkan tepat satu User dan satu Merchant
    edges = [
        ("U1", "T1", False),
        ("T1", "M1", False),
        ("U1", "T2", True),   # Fraud transaction
        ("T2", "M2", True),   # Fraud transaction
        ("U2", "T3", False),
        ("T3", "M2", False),  # Shared merchant M2 menghubungkan U1 dan U2 melalui T2 & T3!
        ("U2", "T4", False),
        ("T4", "M3", False),
    ]

    # 1. Gambar Edges
    for src, dst, is_fr in edges:
        p1 = users[src][0] if src in users else (txs[src][0] if src in txs else merchants[src][0])
        p2 = users[dst][0] if dst in users else (txs[dst][0] if dst in txs else merchants[dst][0])
        draw_edge(ax, p1, p2, node_radius=0.048, is_fraud=is_fr)

    # 2. Gambar Simpul Users
    for key, (pos, lbl, _) in users.items():
        draw_node(ax, pos, lbl, sublabel="User", node_type="user", radius=0.050)

    # 3. Gambar Simpul Merchants
    for key, (pos, lbl, _) in merchants.items():
        draw_node(ax, pos, lbl, sublabel="Merch", node_type="merchant", radius=0.050)

    # 4. Gambar Simpul Transaksi
    for key, (pos, lbl, stat, ntype) in txs.items():
        draw_node(ax, pos, lbl, sublabel=stat, node_type=ntype, radius=0.048)

    # Anotasi Fitur di Bawah Simpul (termasuk dalam Baris 2)
    ax.text(x_user, 0.15, "Cardholder Features\n(Historical Profile)", ha="center", va="top", fontsize=8.5, color="#566573", style="italic")
    ax.text(x_tx, 0.15, "Transaction Features & Target\n(Amount, Time, MCC, Errors)", ha="center", va="top", fontsize=8.5, color="#566573", style="italic")
    ax.text(x_merch, 0.15, "Merchant Features\n(MCC, State Profile)", ha="center", va="top", fontsize=8.5, color="#566573", style="italic")

    # Baris 3: Legenda Bersama di Bagian Bawah Gambar (jarak rapat dan proporsional)
    legend_elements = [
        patches.Patch(facecolor="#EAEDED", edgecolor="#2C3E50", label="User Entity ($U$)"),
        patches.Patch(facecolor="#2C3E50", edgecolor="#1C2833", label="Normal Transaction ($T$)"),
        patches.Patch(facecolor="#1C2833", edgecolor="#C0392B", linewidth=2.0, label="Fraud Transaction ($T$)"),
        patches.Patch(facecolor="#EAEDED", edgecolor="#2C3E50", label="Merchant Entity ($M$)"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=4,
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor="#D5D8DC",
        fontsize=8.5,
        bbox_to_anchor=(0.50, 0.01),
    )

    fig.tight_layout(rect=[0.0, 0.02, 1.0, 1.0])

    png_path = output_dir / "tripartite_graph_structure.png"
    pdf_path = output_dir / "tripartite_graph_structure.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[SUKSES] Diagram graf tripartit berhasil disimpan ke:")
    print(f"  • PNG (300 DPI): {png_path.resolve()}")
    print(f"  • PDF (Vektor) : {pdf_path.resolve()}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "plots"
    create_tripartite_figure(out)
