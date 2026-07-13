from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import json
import logging
import sys
from datetime import datetime

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
AGREEMENT_DIR = OUTPUT_DIR / "xai_agreement"
LOGS_DIR = PROJECT_ROOT / "logs"

# Display labels + preferred ordering for known models. Anything not listed
# falls back to a title-cased folder token via pretty_model(), so a run that
# adds or drops a model still works without editing this file.
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "logistic_regression": "Logistic Regression",
}
PREFERRED_MODEL_ORDER = ["xgboost", "random_forest", "logistic_regression"]


def pretty_model(model: str) -> str:
    return MODEL_LABELS.get(model, model.replace("_", " ").title())

# method folder + file suffix for the per-feature importance CSVs
METHOD_FILES = {
    "SHAP": ("shap", "_shap_feature_importance.csv"),
    "PI": ("fi", "_feature_importance.csv"),
}


# ============================================================
# LOGGING
# ============================================================

def setup_run_logger(script_name: str = "run_cabc_export") -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    log_path = LOGS_DIR / f"{script_name}_{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"),
                  logging.StreamHandler(sys.stdout)],
    )
    return log_path


# ============================================================
# DATA LOADING
# ============================================================

def load_importance_curve(dataset: str, model: str, method: str) -> Dict:
    """Read one per-cell importance CSV and build the normalized cABC curve.

    Returns the sorted features with their normalized importance share and the
    cumulative coordinates (x = i / n, y = cumulative share). These coordinates
    are exactly what the cABC breakpoint rule argmax(y - x) operates on.
    """
    folder, suffix = METHOD_FILES[method]
    csv_path = OUTPUT_DIR / dataset / folder / f"{model}{suffix}"
    df = pd.read_csv(csv_path)
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)

    total = float(df["importance"].sum()) or 1.0
    n = len(df)
    feats: List[Dict] = []
    cum = 0.0
    for i, row in df.iterrows():
        share = float(row["importance"]) / total
        cum += share
        feats.append({
            "name": str(row["feature"]),
            "share": round(share, 6),
            "x": round((i + 1) / n, 6),
            "y": round(cum, 6),
        })
    return {"n": n, "features": feats}


def load_groups() -> Dict[str, Dict]:
    """Read authoritative cABC group assignments (one row per cell).

    Group membership and breakpoint sizes come straight from the pipeline
    output so it matches the manuscript exactly (including the |A| >= 3
    minimum-size constraint and tie handling baked into the pipeline).
    """
    df = pd.read_csv(AGREEMENT_DIR / "cabc_groups.csv")
    out: Dict[str, Dict] = {}
    for _, r in df.iterrows():
        year = str(r["year"])
        model = str(r["model"])
        method = "SHAP" if str(r["method"]).upper() == "SHAP" else "PI"
        key = f"{model}|{year}|{method}"
        out[key] = {
            "dataset": str(r["dataset"]),
            "year": year,
            "model": model,
            "method": method,
            "kA": int(r["|A|"]),
            "kB": int(r["|B|"]),
            "A": [s for s in str(r["A"]).split(",") if s],
            "B": [s for s in str(r["B"]).split(",") if s],
            "C": [s for s in str(r["C"]).split(",") if s],
            "break_ab": str(r["breakpoint_ab_feature"]),
            "break_bc": str(r["breakpoint_bc_feature"]),
        }
    return out


def _split(cell: str) -> List[str]:
    return [s for s in str(cell).split(",") if s]


def _read_csv_or_empty(path: Path) -> pd.DataFrame:
    """Read a CSV if present, else an empty frame (iterates to zero rows)."""
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_compare() -> Dict[str, List[Dict]]:
    """Read the three pairwise J_A summaries into a single comparison table."""
    compare: Dict[str, List[Dict]] = {"cross-method": [], "cross-model": [], "cross-year": []}

    wm = _read_csv_or_empty(AGREEMENT_DIR / "within_model_cabc.csv")
    for _, r in wm.iterrows():
        year, model = str(r["year"]), str(r["model"])
        compare["cross-method"].append({
            "label": f"{pretty_model(model)} · {year} · SHAP vs PI",
            "cellA": f"{model}|{year}|SHAP",
            "cellB": f"{model}|{year}|PI",
            "J_A": round(float(r["J_A"]), 4),
            "A1": _split(r["A_1"]), "A2": _split(r["A_2"]),
        })

    cm = _read_csv_or_empty(AGREEMENT_DIR / "cross_model_shap_cabc.csv")
    for _, r in cm.iterrows():
        year, ma, mb = str(r["year"]), str(r["model_A"]), str(r["model_B"])
        compare["cross-model"].append({
            "label": f"{year} · SHAP · {pretty_model(ma)} vs {pretty_model(mb)}",
            "cellA": f"{ma}|{year}|SHAP",
            "cellB": f"{mb}|{year}|SHAP",
            "J_A": round(float(r["J_A"]), 4),
            "A1": _split(r["A_1"]), "A2": _split(r["A_2"]),
        })

    ts = _read_csv_or_empty(AGREEMENT_DIR / "temporal_stability_cabc.csv")
    for _, r in ts.iterrows():
        model = str(r["model"])
        method = "SHAP" if str(r["method"]).upper() == "SHAP" else "PI"
        ya, yb = str(r["year_A"]), str(r["year_B"])
        compare["cross-year"].append({
            "label": f"{pretty_model(model)} · {method} · {ya} vs {yb}",
            "cellA": f"{model}|{ya}|{method}",
            "cellB": f"{model}|{yb}|{method}",
            "J_A": round(float(r["J_A"]), 4),
            "A1": _split(r["A_1"]), "A2": _split(r["A_2"]),
        })
    return compare


def discover_run_label() -> str:
    """Derive a run label from the output folder name plus the cABC build time,
    instead of hardcoding a run id."""
    label = OUTPUT_DIR.name
    groups_csv = AGREEMENT_DIR / "cabc_groups.csv"
    try:
        mtime = datetime.fromtimestamp(groups_csv.stat().st_mtime)
        return f"{label} ({mtime:%Y-%m-%d %H:%M})"
    except OSError:
        return label


def build_data() -> Dict:
    # cabc_groups.csv is the authoritative cell list: every (dataset, model,
    # method) it contains is discovered here, so adding/removing a cell in the
    # run flows through automatically with no edits to this script.
    groups = load_groups()

    cells: Dict[str, Dict] = {}
    for key, g in groups.items():
        cells[key] = load_importance_curve(g["dataset"], g["model"], g["method"])
        cells[key].update(g)

    years = sorted({g["year"] for g in groups.values()})
    present = {g["model"] for g in groups.values()}
    models = [m for m in PREFERRED_MODEL_ORDER if m in present] + \
             sorted(m for m in present if m not in PREFERRED_MODEL_ORDER)
    methods = [m for m in ("SHAP", "PI") if any(g["method"] == m for g in groups.values())]
    model_labels = {m: pretty_model(m) for m in models}

    core = None
    for c in cells.values():
        a = set(c.get("A", []))
        core = a if core is None else (core & a)

    return {
        "models": models,
        "modelLabels": model_labels,
        "years": years,
        "methods": methods,
        "cells": cells,
        "compare": load_compare(),
        "coreA": sorted(core) if core else [],
        "run": discover_run_label(),
        "nCells": len(cells),
    }


# ============================================================
# HTML RENDERING (self-contained, opens in any browser)
# ============================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cABC - diabetes-xai-agreement</title>
<style>
  :root{--ink:#1f2933;--muted:#6b7280;--line:#e3e6ea;--blue:#185fa5;--blueface:#e6f1fb;
        --teal:#0f6e56;--tealface:#e1f5ee;--amber:#854f0b;--amberface:#faeeda;--gray:#5f5e5a;--grayface:#f1efe8;}
  *{box-sizing:border-box;}
  body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:var(--ink);
       max-width:880px;margin:24px auto;padding:0 18px;line-height:1.55;}
  h1{font-size:20px;font-weight:600;margin:0 0 2px;}
  .sub{color:var(--muted);font-size:13px;margin:0 0 18px;}
  .controls{display:flex;flex-wrap:wrap;gap:10px 14px;align-items:flex-end;margin-bottom:16px;}
  .ctl{display:flex;flex-direction:column;gap:4px;}
  .ctl label{font-size:12px;color:var(--muted);}
  select,button{font-size:14px;padding:7px 10px;border:1px solid var(--line);border-radius:8px;background:#fff;color:var(--ink);}
  button{cursor:pointer;}
  button.active{background:var(--blueface);border-color:var(--blue);color:var(--blue);}
  .card{border:1px solid var(--line);border-radius:12px;padding:14px 12px;margin-bottom:14px;}
  svg{width:100%;max-width:520px;height:auto;display:block;margin:0 auto;}
  .chips{display:flex;flex-wrap:wrap;gap:8px;font-size:13px;margin-top:8px;}
  .chip{padding:4px 10px;border-radius:8px;}
  .cA{background:var(--tealface);color:var(--teal);}
  .cB{background:var(--amberface);color:var(--amber);}
  .cC{background:var(--grayface);color:var(--gray);}
  .meta{font-size:13px;color:var(--muted);margin-top:6px;}
  .ja{font-size:15px;font-weight:600;}
  .hidden{display:none;}
  .legend{font-size:12px;color:var(--muted);margin-top:4px;}
</style>
</head>
<body>
<h1>cABC &mdash; SHAP vs PI agreement</h1>
<p class="sub">Real values from run <code>__RUN__</code> (diabetes-xai-agreement). Core Group A = {__CORE__} is invariant across all __NCELLS__ cells.</p>

<div class="controls">
  <button id="mode-single" class="active">Single cell</button>
  <button id="mode-compare">Compare two cells</button>
</div>

<div id="panel-single">
  <div class="controls">
    <div class="ctl"><label>Model</label><select id="s-model"></select></div>
    <div class="ctl"><label>Year</label><select id="s-year"></select></div>
    <div class="ctl"><label>Method</label><select id="s-method"></select></div>
  </div>
  <div class="card"><svg id="s-plot" viewBox="0 0 420 260" role="img"></svg>
    <div class="legend">Blue = normalized cumulative importance &middot; grey diagonal = equilibrium &middot; green = largest y&minus;x (A-break)</div></div>
  <div id="s-chips" class="chips"></div>
  <div id="s-meta" class="meta"></div>
</div>

<div id="panel-compare" class="hidden">
  <div class="controls">
    <div class="ctl"><label>Comparison type</label><select id="c-type"></select></div>
    <div class="ctl" style="flex:1;min-width:260px;"><label>Pair</label><select id="c-pair" style="width:100%;"></select></div>
  </div>
  <div class="card">
    <div class="ja" id="c-ja"></div>
    <svg id="c-plot" viewBox="0 0 420 260" role="img"></svg>
    <div class="legend">Two curves = the two compared cells &middot; filled dot = Group A boundary of each</div>
  </div>
  <div id="c-chips" class="chips"></div>
</div>

<script>
const DATA = __DATA__;
__CORE_JS__
</script>
</body>
</html>"""


# Shared client-side logic (used by both the standalone HTML and inline review).
CORE_JS = r"""
const X0=52,X1=404,Y0=216,Y1=20;
const PX=x=>X0+(X1-X0)*x, PY=y=>Y0+(Y1-Y0)*y;

function curveSVG(cell, color, markFn){
  let pts=[[0,0]].concat(cell.features.map(f=>[f.x,f.y]));
  let poly=pts.map(p=>PX(p[0])+","+PY(p[1])).join(" ");
  let s='<polyline points="'+poly+'" fill="none" stroke="'+color+'" stroke-width="2.5"/>';
  if(markFn){ s+=markFn(cell,color); }
  return s;
}
function axes(){
  let s='<line x1="'+PX(0)+'" y1="'+PY(0)+'" x2="'+PX(1)+'" y2="'+PY(1)+'" stroke="#b4b2a9" stroke-width="1" stroke-dasharray="4 3"/>';
  s+='<line x1="'+X0+'" y1="'+Y0+'" x2="'+X1+'" y2="'+Y0+'" stroke="#d3d1c7"/>';
  s+='<line x1="'+X0+'" y1="'+Y0+'" x2="'+X0+'" y2="'+Y1+'" stroke="#d3d1c7"/>';
  s+='<text x="'+((X0+X1)/2)+'" y="246" text-anchor="middle" font-size="11" fill="#6b7280">fraction of features  x = i/n</text>';
  s+='<text transform="translate(16,'+((Y0+Y1)/2)+') rotate(-90)" text-anchor="middle" font-size="11" fill="#6b7280">cumulative importance  y</text>';
  return s;
}
function markA(cell,color){
  if(!cell.kA) return "";
  let f=cell.features[cell.kA-1], gx=PX(f.x);
  let s='<line x1="'+gx+'" y1="'+PY(f.x)+'" x2="'+gx+'" y2="'+PY(f.y)+'" stroke="#1d9e75" stroke-width="6" stroke-linecap="round" opacity="0.5"/>';
  s+='<circle cx="'+gx+'" cy="'+PY(f.y)+'" r="5" fill="#0f6e56"/>';
  s+='<text x="'+gx+'" y="'+(PY(f.y)-9)+'" text-anchor="middle" font-size="11" font-weight="600" fill="#0f6e56">A (|A|='+cell.kA+')</text>';
  return s;
}
function markDot(cell,color){
  if(!cell.kA) return "";
  let f=cell.features[cell.kA-1];
  return '<circle cx="'+PX(f.x)+'" cy="'+PY(f.y)+'" r="4.5" fill="'+color+'"/>';
}
function chips(cell){
  const g=(arr,cls,lab)=>'<span class="chip '+cls+'"><b>'+lab+'</b> '+(arr&&arr.length?arr.join(", "):"-")+'</span>';
  return g(cell.A,"cA","Group A")+g(cell.B,"cB","Group B")+g(cell.C,"cC","Group C");
}

function cellKey(){ return sModel.value+"|"+sYear.value+"|"+sMethod.value; }
function drawSingle(){
  let cell=DATA.cells[cellKey()];
  document.getElementById("s-plot").innerHTML=axes()+curveSVG(cell,"#185fa5",markA);
  document.getElementById("s-chips").innerHTML=chips(cell);
  document.getElementById("s-meta").innerHTML="A|B boundary after <b>"+cell.break_ab+"</b> &middot; B|C after <b>"+cell.break_bc+"</b> &middot; n="+cell.n;
}

function fillSel(sel,arr,labels){ sel.innerHTML=arr.map(v=>'<option value="'+v+'">'+(labels?labels[v]:v)+'</option>').join(""); }
const sModel=document.getElementById("s-model"), sYear=document.getElementById("s-year"), sMethod=document.getElementById("s-method");
fillSel(sModel,DATA.models,DATA.modelLabels); fillSel(sYear,DATA.years); fillSel(sMethod,DATA.methods);
sYear.value="2021";
[sModel,sYear,sMethod].forEach(el=>el.addEventListener("change",drawSingle));
drawSingle();

const cType=document.getElementById("c-type"), cPair=document.getElementById("c-pair");
const TYPES={"cross-method":"Cross-method (vary method)","cross-model":"Cross-model (vary model)","cross-year":"Cross-year (vary year)"};
fillSel(cType,Object.keys(TYPES),TYPES);
function fillPairs(){
  let list=DATA.compare[cType.value];
  cPair.innerHTML=list.map((p,i)=>'<option value="'+i+'">'+p.label+' (J_A='+p.J_A.toFixed(4)+')</option>').join("");
  drawCompare();
}
function drawCompare(){
  let p=DATA.compare[cType.value][parseInt(cPair.value,10)];
  let cA=DATA.cells[p.cellA], cB=DATA.cells[p.cellB];
  document.getElementById("c-plot").innerHTML=axes()
     +curveSVG(cA,"#185fa5",markDot)+curveSVG(cB,"#d85a30",markDot);
  let inter=p.A1.filter(x=>p.A2.includes(x));
  document.getElementById("c-ja").innerHTML="J_A = "+p.J_A.toFixed(4)
     +' &nbsp; <span style="color:#0f6e56">A overlap: '+(inter.join(", ")||"-")+'</span>';
  document.getElementById("c-chips").innerHTML=
     '<span class="chip" style="background:#e6f1fb;color:#185fa5"><b>A (cell 1)</b> '+p.A1.join(", ")+'</span>'+
     '<span class="chip" style="background:#faece7;color:#993c1d"><b>A (cell 2)</b> '+p.A2.join(", ")+'</span>';
}
cType.addEventListener("change",fillPairs);
cPair.addEventListener("change",drawCompare);
fillPairs();

const mS=document.getElementById("mode-single"), mC=document.getElementById("mode-compare");
const pS=document.getElementById("panel-single"), pC=document.getElementById("panel-compare");
mS.addEventListener("click",()=>{mS.classList.add("active");mC.classList.remove("active");pS.classList.remove("hidden");pC.classList.add("hidden");});
mC.addEventListener("click",()=>{mC.classList.add("active");mS.classList.remove("active");pC.classList.remove("hidden");pS.classList.add("hidden");});
"""


def render_html(data: Dict) -> str:
    html = HTML_TEMPLATE
    html = html.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    html = html.replace("__CORE_JS__", CORE_JS)
    html = html.replace("__RUN__", data["run"])
    html = html.replace("__CORE__", ", ".join(data["coreA"]))
    html = html.replace("__NCELLS__", f'{data["nCells"]}/{data["nCells"]}')
    return html


def main() -> None:
    log_path = setup_run_logger()
    logging.info("Building cABC from %s", AGREEMENT_DIR)

    data = build_data()
    logging.info("Loaded %d cells; Core Group A = %s", len(data["cells"]), data["coreA"])

    AGREEMENT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = AGREEMENT_DIR / "cabc_data.json"
    html_path = AGREEMENT_DIR / "cabc.html"
    json_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(data), encoding="utf-8")

    logging.info("Wrote %s", json_path)
    logging.info("Wrote %s", html_path)
    logging.info("Log: %s", log_path)


if __name__ == "__main__":
    main()
