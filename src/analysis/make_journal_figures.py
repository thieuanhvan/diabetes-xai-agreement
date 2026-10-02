"""
Figures for the journal manuscript (JBI draft v1).

Every figure is drawn from the stored CSV outputs, never from hard-coded
numbers; a few asserts pin the values quoted in the manuscript text so a
silent change in the outputs breaks the build instead of the paper.

    Fig. 2  variance decomposition (pooled eta^2; share and rank responses)
    Fig. 3  feature-level attribution-share shifts, diagnosis vs HbA1c label
    Fig. 4  population-level vs patient-level agreement
    Graphical abstract (three key results)

Fig. 1 (study design) is drawn in TikZ inside the manuscript.

Writes outputs/journal_figures/fig{2,3,4}_*.pdf and graphical_abstract.pdf
Usage:
    python -m src.analysis.make_journal_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.analysis.manuscript_numbers import t4

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans", "font.size": 8,
    "axes.edgecolor": "#8a8984", "axes.linewidth": 0.6, "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.titlesize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
})

OUT = Path(__file__).resolve().parents[2] / "outputs"
FIG = OUT / "journal_figures"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # validated categorical slots 1-4
NOISE = "#b5b4ae"                                   # neutral for the seed residual
MODEL_LABEL = {"logistic_regression": "LR", "random_forest": "RF", "xgboost": "XGBoost"}


def fig_variance() -> None:
    panels = [("BRFSS\nas MAPR", "brfss_pipeline", None), ("BRFSS\nunweighted", "brfss_none", None),
              ("BRFSS\nbalanced", "brfss_balanced", None),
              ("NHANES\n17–20", "nhanes", "2017-2020"), ("NHANES\n21–23", "nhanes", "2021-2023")]
    groups = [("XAI method", ["method"]), ("Model", ["model"]),
              ("Year / label", ["year", "label"]), ("Interactions", None), ("Seed (residual)", ["residual (seed)"])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    for ax, resp in zip(axes, ["share", "rank"]):
        x = np.arange(len(panels))
        bottom = np.zeros(len(panels))
        vals = {g: [] for g, _ in groups}
        for _, stem, cyc in panels:
            d = pd.read_csv(OUT / "variance_decomposition" / f"{stem}_{resp}_pooled.csv")
            if cyc:
                d = d[d.cycle == cyc]
            s = d.set_index("source")["pooled_eta2"]
            used = 0.0
            for g, keys in groups:
                if keys is None:
                    continue
                v = float(sum(s.get(k, 0.0) for k in keys))
                vals[g].append(v); used += v
            vals["Interactions"].append(1.0 - used)
        if resp == "rank":
            assert abs(vals["Model"][1] - 0.303) < 1e-3 and abs(vals["XAI method"][1] - 0.166) < 1e-3
        for (g, _), col in zip(groups, C + [NOISE]):
            h = np.array(vals[g])
            ax.bar(x, h, bottom=bottom, width=0.62, color=col if g != "Interactions" else "#d9d8d2",
                   edgecolor="white", linewidth=1.0, label=g)
            bottom += h
        ax.set_xticks(x, [p[0] for p in panels], fontsize=6.8)
        ax.set_ylim(0, 1); ax.set_title(f"Response: attribution {resp}", color=INK, loc="left")
        ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    axes[0].set_ylabel("Pooled η² (share of variation)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(FIG / "fig2_variance_decomposition.pdf"); plt.close(fig)


def fig_label_shift() -> None:
    f = pd.read_csv(OUT / "label_axis" / "feature_shift.csv")
    f = f[(f.weighting == "unweighted") & (f.method == "SHAP") & (f.label_1 == "diag") & (f.label_2 == "lab")]
    hb = f[(f.feature == "HighBP") & (f.model == "xgboost") & (f.cycle == "2021-2023")]["diff"].iloc[0]
    assert abs(hb * 100 - 5.7) < 0.05
    order = (f.groupby("feature")["diff"].mean().sort_values()).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4), sharey=True)
    for ax, cyc in zip(axes, ["2017-2020", "2021-2023"]):
        g = f[f.cycle == cyc]
        ax.axvline(0, color="#8a8984", linewidth=0.8)
        for k, (m, col) in enumerate(zip(["logistic_regression", "random_forest", "xgboost"], C)):
            gm = g[g.model == m].set_index("feature").loc[order]
            y = np.arange(len(order)) + (k - 1) * 0.22
            sig = gm.q_bh < 0.05
            ax.scatter(gm["diff"][sig] * 100, y[sig.values], s=26, color=col, edgecolor="white", linewidth=0.8,
                       zorder=3, label=f"{MODEL_LABEL[m]} (q < 0.05)")
            ax.scatter(gm["diff"][~sig] * 100, y[~sig.values], s=26, facecolor="white", edgecolor=col,
                       linewidth=1.0, zorder=3, label=f"{MODEL_LABEL[m]} (n.s.)")
        ax.set_yticks(np.arange(len(order)), order, fontsize=7)
        ax.set_title(f"NHANES {cyc.replace('-', '–')}", color=INK, loc="left")
        ax.set_xlabel("Δ attribution share, diagnosis − HbA1c label (pp)\n"
                      "(← heavier under HbA1c | heavier under diagnosis →)", fontsize=7.5)
        ax.xaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=6, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(FIG / "fig3_label_shift.pdf"); plt.close(fig)


def fig_instance() -> None:
    inst = pd.read_csv(OUT / "instance_vs_population" / "instance_level.csv")
    summ = pd.read_csv(OUT / "instance_vs_population" / "instance_summary.csv")
    summ = summ[summ.metric == "spearman"]
    keys = [("BRFSS", "2015", "model: XGB vs LR", "BRFSS 2015\nXGB vs LR"),
            ("BRFSS", "2021", "model: XGB vs LR", "BRFSS 2021\nXGB vs LR"),
            ("BRFSS", "2023", "model: XGB vs LR", "BRFSS 2023\nXGB vs LR"),
            ("NHANES", "2017-2020", "model: XGB vs LR (diag)", "NHANES 17–20\nXGB vs LR"),
            ("NHANES", "2021-2023", "model: XGB vs LR (diag)", "NHANES 21–23\nXGB vs LR"),
            ("NHANES", "2017-2020", "label: diag vs lab (XGB)", "NHANES 17–20\ndiag vs HbA1c"),
            ("NHANES", "2021-2023", "label: diag vs lab (XGB)", "NHANES 21–23\ndiag vs HbA1c")]
    inst["cohort"] = inst["cohort"].astype(str); summ = summ.assign(cohort=summ["cohort"].astype(str))
    data, pops = [], []
    for ds, coh, pair, _ in keys:
        data.append(inst[(inst.dataset == ds) & (inst.cohort == coh) & (inst.pair == pair)]["spearman"].dropna().values)
        pops.append(float(summ[(summ.dataset == ds) & (summ.cohort == coh) & (summ.pair == pair)]["population"].iloc[0]))
    assert abs(pops[0] - 0.956) < 5e-4
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    bp = ax.boxplot(data, widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, linewidth=1.2), whiskerprops=dict(color=INK2, linewidth=0.8),
                    capprops=dict(color=INK2, linewidth=0.8))
    for b in bp["boxes"]:
        b.set(facecolor="#cfe0f6", edgecolor=C[0], linewidth=0.9)
    ax.scatter(np.arange(1, len(pops) + 1), pops, marker="D", s=30, color=C[1], edgecolor="white",
               linewidth=0.8, zorder=4, label="Population-level value")
    ax.plot([], [], color=INK, linewidth=1.2, label="Patient-level median (box: IQR, whiskers: 1.5 IQR)")
    ax.set_xticks(np.arange(1, len(keys) + 1), [k[3] for k in keys], fontsize=6.8)
    ax.set_ylabel("Spearman ρ of |SHAP| vectors")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "fig4_population_vs_patient.pdf"); plt.close(fig)


def fig_graphical_abstract() -> None:
    pairs = pd.read_csv(OUT / "metric_comparison" / "pairs.csv")
    cw = pd.read_csv(OUT / "brfss_multiseed" / "class_weighting_comparison" / "axis_summary.csv")
    cw = cw[(cw["axis"] == "model") & (cw["method"] == "SHAP")].set_index("class_weighting")["J_A"]
    sh = pd.read_csv(OUT / "label_axis" / "feature_shift.csv")
    sh = sh[(sh.weighting == "unweighted") & (sh.method == "SHAP") & (sh.label_1 == "diag") & (sh.label_2 == "lab")
            & (sh.model == "xgboost") & (sh.cycle == "2021-2023")].set_index("feature")["diff"] * 100
    assert abs(cw["none"] - 0.942) < 1e-3 and abs(cw["pipeline"] - 0.857) < 1e-3
    # JBI: 531 x 1328 px (h x w) or proportionally more, readable at 5 x 13 cm -> draw at 13 x 5 cm
    with plt.rc_context({"font.size": 5.6, "axes.titlesize": 5.8, "axes.labelsize": 5.4,
                         "xtick.labelsize": 5.0, "ytick.labelsize": 5.0}):
        _graphical_abstract_panels(pairs, cw, sh)


def _graphical_abstract_panels(pairs, cw, sh) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13 / 2.54, 5 / 2.54))
    ax = axes[0]
    ax.plot([0.4, 1.0], [0.4, 1.0], color=GRID, linewidth=1.0, zorder=1)
    ax.scatter(pairs["size_ratio_A"], pairs["J_A"], s=10, color=C[0], edgecolor="white", linewidth=0.8, zorder=3)
    n_diag = int((abs(pairs["J_A"] - pairs["size_ratio_A"]) < 1e-9).sum())
    n_contr = int(pairs["contradiction_A"].sum())
    assert (n_diag, n_contr, len(pairs)) == (44, 1, 45)
    ax.text(0.42, 0.96, f"{n_diag} of {len(pairs)} pairs: $J_A$ = size ratio", fontsize=4.8, color=INK2)
    c = pairs[pairs["contradiction_A"] == 1].iloc[0]
    ax.annotate("1 true contradiction", (c["size_ratio_A"], c["J_A"]), xytext=(0.6, 0.6), fontsize=4.8,
                color=INK2, arrowprops=dict(arrowstyle="-", color="#8a8984", linewidth=0.7))
    ax.set_xlabel("Group-A size ratio"); ax.set_ylabel("Group-A Jaccard $J_A$")
    ax.set_title("1  $J_A$ tracks depth", loc="left", color=INK)
    ax = axes[1]
    labs = ["as\nconference", "no\nweighting", "all\nweighted"]
    v = [cw["pipeline"], cw["none"], cw["balanced"]]
    ax.scatter(range(3), v, s=22, color=[C[1], C[0], C[1]], edgecolor="white", linewidth=1.0, zorder=3)
    for i, y in enumerate(v):
        ax.text(i, y + 0.008, t4(y), ha="center", va="bottom", fontsize=5.0, color=INK)
    ax.set_xticks(range(3), labs, fontsize=4.8); ax.set_xlim(-0.5, 2.5); ax.set_ylim(0.8, 1.0)
    ax.set_ylabel("Cross-model $J_A$ (SHAP)")
    ax.set_title("2  Weighting made the model gap", loc="left", color=INK)
    ax = axes[2]
    feats = ["HighBP", "HighChol", "Education", "BMI"]
    vals = [sh[f] for f in feats]
    ax.barh(range(len(feats)), vals, color=[C[0] if x > 0 else C[1] for x in vals], height=0.6)
    ax.axvline(0, color="#8a8984", linewidth=0.8)
    ax.set_yticks(range(len(feats)), feats); ax.invert_yaxis()
    ax.set_xlabel("Δ SHAP share, diag − HbA1c (pp)\nXGBoost, NHANES 2021–23")
    ax.set_title("3  Label moves explanations", loc="left", color=INK)
    for a in axes:
        a.grid(True, color=GRID, linewidth=0.5); a.set_axisbelow(True)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIG / "graphical_abstract.pdf"); fig.savefig(FIG / "graphical_abstract.png", dpi=300)
    fig.savefig(FIG / "graphical_abstract.tiff", dpi=300); plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig_variance(); fig_label_shift(); fig_instance(); fig_graphical_abstract()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
