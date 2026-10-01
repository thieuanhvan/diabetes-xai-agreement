"""Sinh ba hình khái niệm của Chương 3 luận văn (Hình 3.1, 3.2, 3.3).

Chạy: python make_thesis_figures.py
Đầu ra: figures/hinh_3_1.png, hinh_3_2.png, hinh_3_3.png (300 dpi)

Kích thước figsize được đặt bằng đúng kích thước in trên trang A4 (cột chữ
15,24 cm), nên cỡ chữ trong hình tính bằng point là cỡ chữ thật khi in.

Nội dung đã sửa so với bản cũ:
  - Bỏ "safety-by-design" (luận văn dùng "tuân thủ hệ phân loại theo thiết kế")
  - Bỏ "domain expert review" ở bước tri thức (thẩm định chuyên gia nằm ngoài
    phạm vi thực nghiệm)
  - Lớp Tri thức lấy nội dung từ HƯỚNG DẪN LÂM SÀNG, độc lập với đầu ra Kiểm
    định; quan hệ giữa hai giai đoạn vẽ bằng mũi tên NÉT ĐỨT "đối chiếu"
  - Toàn bộ chữ trong hình bằng tiếng Việt
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)

CM = 1 / 2.54

DATA = "#FAE3C6"
MODEL = "#D6E4F7"
AUDIT = "#DED3F0"
KNOW = "#F7CFDA"
ACTION = "#CDE9CE"
EDGE = "#5A5A5A"


def box(ax, x, y, w, h, title, lines, fc, ts=7.5, ls=6.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=0.7, edgecolor=EDGE, facecolor=fc,
        )
    )
    cx = x + w / 2
    if lines:
        ax.text(cx, y + h - 0.028, title, ha="center", va="top",
                fontsize=ts, fontweight="bold", color="#1A1A1A")
        ax.text(cx, y + h - 0.028 - 0.030, "\n".join(lines), ha="center",
                va="top", fontsize=ls, color="#333333", linespacing=1.35)
    else:
        ax.text(cx, y + h / 2, title, ha="center", va="center",
                fontsize=ts, fontweight="bold", color="#1A1A1A")


def arrow(ax, p0, p1, dashed=False, label=None, lx=0.0, ly=0.0, color=EDGE):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=8,
            linewidth=0.8, color=color,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=0, shrinkB=0,
        )
    )
    if label:
        mx = (p0[0] + p1[0]) / 2 + lx
        my = (p0[1] + p1[1]) / 2 + ly
        ax.text(mx, my, label, ha="center", va="center", fontsize=6.0,
                style="italic", color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))


def canvas(w_cm, h_cm):
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


# ── Hình 3.1 — quy trình tổng thể, 7 bước, hai giai đoạn ───────────────
def hinh_3_1():
    fig, ax = canvas(15.24, 8.7)

    w, h = 0.228, 0.265
    ytop, ybot = 0.625, 0.145
    xs = [0.008, 0.256, 0.504, 0.752]

    box(ax, xs[0], ytop, w, h, "(1) Dữ liệu BRFSS",
        ["2015 · 2021 · 2023", "N = 762.827"], DATA, ts=7.0, ls=6.0)
    box(ax, xs[1], ytop, w, h, "(2) Tiền xử lý",
        ["cấu trúc chung 17 đặc trưng", "giữ mất cân bằng tự nhiên"], DATA,
        ts=7.0, ls=6.0)
    box(ax, xs[2], ytop, w, h, "(3) Huấn luyện mô hình",
        ["LR · RF · XGBoost", "siêu tham số cố định"], MODEL, ts=7.0, ls=6.0)
    box(ax, xs[3], ytop, w, h, "(4) Kiểm định · Chương 4",
        ["Đồng thuận (cABC, SHAP/PI)", "Ổn định theo thời gian",
         "→ Core Group A"], AUDIT, ts=7.0, ls=6.0)

    box(ax, xs[0], ybot, w, h, "Hướng dẫn lâm sàng",
        ["ADA Standards of Care 2024", "và y văn can thiệp"], "#FFFFFF",
        ts=7.0, ls=6.0)
    box(ax, xs[1], ybot, w, h, "(5) Lớp tri thức K",
        ["hệ phân loại 5 lớp hướng", "can thiệp; K(f) mỗi đặc trưng"], KNOW,
        ts=7.0, ls=6.0)
    box(ax, xs[2], ybot, w, h, "(6) Bộ biên dịch Γ · Ch.5",
        ["Γ(K, xᵢ) → (Vᵢ, Pᵢ)", "ràng buộc theo truy vấn"], KNOW,
        ts=7.0, ls=6.0)
    box(ax, xs[3], ybot, w, h, "(7) Khuyến nghị phản thực",
        ["sinh qua DiCE dưới ràng buộc", "tuân thủ hướng can thiệp"], ACTION,
        ts=6.4, ls=6.0)

    ymid_t = ytop + h / 2
    ymid_b = ybot + h / 2
    for i in range(3):
        arrow(ax, (xs[i] + w, ymid_t), (xs[i + 1], ymid_t))
        arrow(ax, (xs[i] + w, ymid_b), (xs[i + 1], ymid_b))

    cxr = xs[3] + w / 2
    ax.add_patch(
        FancyArrowPatch((cxr, ytop), (cxr, ybot + h), arrowstyle="<|-|>",
                        mutation_scale=8, linewidth=0.8, color=EDGE,
                        linestyle=(0, (3, 2)), shrinkA=0, shrinkB=0))
    ax.text(cxr, (ytop + ybot + h) / 2, "đối chiếu", ha="center", va="center",
            fontsize=6.2, style="italic", color=EDGE,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none"))

    ax.text(0.008, ytop + h + 0.030, "GIAI ĐOẠN KIỂM ĐỊNH",
            fontsize=7.0, fontweight="bold", color="#4A3B7A", va="bottom")
    ax.text(0.008, ybot - 0.080, "GIAI ĐOẠN HÀNH ĐỘNG",
            fontsize=7.0, fontweight="bold", color="#2F6B33", va="bottom")
    ax.text(0.992, ybot - 0.080,
            "Nét liền: luồng xử lý.   Nét đứt: đối chiếu kết quả, "
            "không truyền ràng buộc.",
            fontsize=6.0, style="italic", color="#555555",
            ha="right", va="bottom")

    fig.savefig(OUT / "hinh_3_1.png", dpi=300, facecolor="white")
    plt.close(fig)


# ── Hình 3.2 — kiến trúc 5 lớp ─────────────────────────────────────────
def hinh_3_2():
    fig, ax = canvas(13.0, 9.4)

    w, h = 0.520, 0.150
    x = 0.185
    ys = [0.055, 0.240, 0.425, 0.615, 0.800]
    cw = 0.160
    cx = 0.760

    layers = [
        ("Lớp Dữ liệu", ["BRFSS 2015/2021/2023 · cấu trúc chung 17 đặc trưng"],
         DATA, "Chương 4"),
        ("Lớp Mô hình", ["Logistic Regression · Random Forest · XGBoost"],
         MODEL, "Chương 4"),
        ("Lớp Kiểm định", ["Đồng thuận (cABC, Jaccard) · Ổn định theo thời gian",
                           "→ Core Group A + cảnh báo hiệu ứng chọn mẫu"],
         AUDIT, "Chương 4"),
        ("Lớp Tri thức", ["hệ phân loại 5 lớp + bộ biên dịch Γ",
                          "nội dung lấy từ hướng dẫn lâm sàng"],
         KNOW, "Chương 5"),
        ("Lớp Hành động", ["sinh khuyến nghị phản thực qua DiCE"],
         ACTION, "Chương 5"),
    ]
    for (name, lines, fc, ch), y in zip(layers, ys):
        box(ax, x, y, w, h, name, lines, fc)
        box(ax, cx, y + 0.030, cw, h - 0.060, ch, [], "#FFF6D6", ts=6.8)

    for i in range(4):
        y0 = ys[i] + h
        y1 = ys[i + 1]
        dashed = (i == 2)  # Kiểm định → Tri thức là quan hệ đối chiếu
        arrow(ax, (x + w / 2, y0), (x + w / 2, y1), dashed=dashed,
              label="đối chiếu" if dashed else None)

    arrow(ax, (0.125, ys[3] + h / 2), (x, ys[3] + h / 2))
    ax.text(0.118, ys[3] + h / 2, "Hướng dẫn\nlâm sàng\n(ADA 2024)",
            fontsize=6.2, ha="right", va="center", color="#333333",
            linespacing=1.35)

    ax.text(0.999, 0.005,
            "Nét đứt: Lớp Tri thức độc lập với đầu ra Lớp Kiểm định; "
            "Core Group A chỉ dùng để đối chiếu.",
            fontsize=6.0, style="italic", color="#555555",
            ha="right", va="bottom")

    fig.savefig(OUT / "hinh_3_2.png", dpi=300, facecolor="white")
    plt.close(fig)


# ── Hình 3.3 — năm bước tri thức → hành động ───────────────────────────
def hinh_3_3():
    fig, ax = canvas(12.4, 8.9)

    w, h = 0.880, 0.148
    x = 0.060
    ys = [0.035, 0.240, 0.440, 0.640, 0.855]

    box(ax, x, ys[4], w, h - 0.030, "(1) Phát hiện kiểm định — Chương 4",
        ["Core Group A · cảnh báo hiệu ứng chọn mẫu"], AUDIT)
    box(ax, x, ys[3], w, h,
        "(2) Lớp tri thức lâm sàng",
        ["nội dung lấy từ hướng dẫn lâm sàng (ADA 2024) và y văn;",
         "thẩm định bởi hội đồng chuyên gia nằm ngoài phạm vi luận văn"], KNOW)
    box(ax, x, ys[2], w, h, "(3) Biểu diễn tri thức K(f)",
        ["mỗi đặc trưng một bộ sáu: (c_f, d_f, ℓ_f, u_f, τ_f, h_f)"], KNOW)
    box(ax, x, ys[1], w, h, "(4) Bộ biên dịch ràng buộc Γ",
        ["Γ(K, xᵢ) → (Vᵢ, Pᵢ): tham số DiCE theo từng truy vấn"], KNOW)
    box(ax, x, ys[0], w, h, "(5) Khuyến nghị lâm sàng",
        ["phản thực tuân thủ hệ phân loại theo thiết kế;",
         "bác sĩ rà soát trước khi trao đổi với người bệnh"], ACTION)

    arrow(ax, (x + w / 2, ys[4]), (x + w / 2, ys[3] + h), dashed=True,
          label="đối chiếu, không sinh ràng buộc")
    for i in (3, 2, 1):
        arrow(ax, (x + w / 2, ys[i]), (x + w / 2, ys[i - 1] + h))

    fig.savefig(OUT / "hinh_3_3.png", dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    hinh_3_1()
    hinh_3_2()
    hinh_3_3()
    for f in sorted(OUT.glob("hinh_3_*.png")):
        print("wrote", f)
