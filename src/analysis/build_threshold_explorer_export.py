"""Export an interactive accuracy-vs-AUC threshold explorer (self-contained HTML).

Drag the decision threshold and watch accuracy, sensitivity, and specificity
move while the ROC curve and AUC stay fixed, for any (model, cohort) cell. This
is the visual counterpart of the model-comparison table: under the natural ~14%
class imbalance, accuracy at a single threshold is misleading (predicting the
majority class scores high accuracy yet misses most cases), whereas AUC measures
threshold-free ranking quality, the property used to pick the model that ranks
high-risk patients best.

The main pipeline does not store per-instance test predictions, so on first run
the three classifiers are reproduced on each cohort with the same split,
preprocessing, and configurations as the main pipeline, and the resulting score
histograms and ROC curves are cached under outputs/xai_agreement/. Later runs
read the cache (standard library only, no retraining).

Output:
    outputs/xai_agreement/threshold_explorer.html
    outputs/xai_agreement/threshold_explorer_data.json   (cache)

Right-click run in PyCharm (repo root auto-detected from this file's location,
so the script may live at repo root or under src/analysis/) or:
    python run_threshold_explorer_export.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

YEARS = ["2015", "2021", "2023"]
MODELS = [
    ("xgboost", "XGBoost"),
    ("random_forest", "Random Forest"),
    ("logistic_regression", "Logistic Regression"),
]
N_BINS = 100  # score resolution; matches the 0.01 step of the threshold slider


def _find_repo_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "outputs").is_dir() and (d / "src").is_dir():
            return d
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
OUT_DIR = REPO_ROOT / "outputs" / "xai_agreement"
OUT_HTML = OUT_DIR / "threshold_explorer.html"
CACHE = OUT_DIR / "threshold_explorer_data.json"


def trunc4(x: float) -> float:
    return int(x * 10000) / 10000


def _cell_payload(y_true, y_score, np, roc_curve, roc_auc_score) -> dict:
    pos = np.zeros(N_BINS, dtype=int)
    neg = np.zeros(N_BINS, dtype=int)
    idx = np.clip((y_score * N_BINS).astype(int), 0, N_BINS - 1)
    np.add.at(pos, idx[y_true == 1], 1)
    np.add.at(neg, idx[y_true == 0], 1)

    fpr, tpr, _ = roc_curve(y_true, y_score)
    step = max(1, len(fpr) // 160)
    roc = [[round(float(fpr[k]), 4), round(float(tpr[k]), 4)]
           for k in range(0, len(fpr), step)]
    last = [round(float(fpr[-1]), 4), round(float(tpr[-1]), 4)]
    if roc[-1] != last:
        roc.append(last)

    return {
        "pos": pos.tolist(),
        "neg": neg.tolist(),
        "nPos": int(pos.sum()),
        "nNeg": int(neg.sum()),
        "roc": roc,
        "auc": trunc4(float(roc_auc_score(y_true, y_score))),
    }


def _compute_payload() -> dict:
    # The repo mixes bare ('datasets...') and 'src.'-prefixed imports, so put
    # both the repo root and src on the path to let either style resolve.
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))

    import numpy as np
    from sklearn.base import clone
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.pipeline import Pipeline

    from datasets.loader import load_dataset_by_name
    from models.baselines import get_baseline_estimators
    from models.proposal import get_proposed_estimators
    from pipelines.step02_preprocessing import run_step02_preprocessing

    estimators = {}
    estimators.update(get_baseline_estimators())
    estimators.update(get_proposed_estimators())

    cells = {}
    for year in YEARS:
        frame, cfg = load_dataset_by_name(f"cdc_brfss_diabetes_{year}")
        data = run_step02_preprocessing(frame, cfg)
        y_test = np.asarray(data["y_test"]).astype(int)
        for model_key, _ in MODELS:
            pipeline = Pipeline(steps=[
                ("preprocessor", clone(data["preprocessor"])),
                ("model", estimators[model_key]),
            ])
            pipeline.fit(data["X_train"], data["y_train"])
            y_score = pipeline.predict_proba(data["X_test"])[:, 1]
            cells[f"{year}|{model_key}"] = _cell_payload(
                y_test, y_score, np, roc_curve, roc_auc_score)
            print(f"[threshold] {year} {model_key}: "
                  f"AUC = {cells[f'{year}|{model_key}']['auc']:.4f}")
    return {"bins": N_BINS, "cells": cells}


def get_payload(force: bool = False) -> dict:
    if CACHE.exists() and not force:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    payload = _compute_payload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return payload


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Accuracy vs AUC &mdash; threshold explorer</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{--paper:#F4EFE4;--card:#FBF8F1;--ink:#1B1A17;--muted:#736E61;--line:#E0D8C7;
  --neg:#736E61;--pos:#B0562A;--roc:#0F6E63;--gold:#C8A24B;
  --shadow:0 1px 2px rgba(27,26,23,.05),0 8px 28px -12px rgba(27,26,23,.18);}
*{box-sizing:border-box}html,body{margin:0}
body{background:radial-gradient(1200px 600px at 85% -10%,#FBF8F1 0%,transparent 60%),var(--paper);
  color:var(--ink);font-family:"Archivo",system-ui,sans-serif;-webkit-font-smoothing:antialiased;
  line-height:1.45;padding:26px 18px 60px;}
.wrap{max-width:1000px;margin:0 auto}
.kicker{font:600 11px/1 "Archivo";letter-spacing:.22em;text-transform:uppercase;color:var(--roc);margin-bottom:10px}
h1{font-family:"Fraunces";font-weight:600;font-size:clamp(26px,4.4vw,40px);line-height:1.02;margin:0 0 8px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:14.5px;max-width:70ch}.sub b{color:var(--ink);font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.controls{padding:16px;margin:22px 0 18px;display:flex;flex-wrap:wrap;gap:14px 20px;align-items:flex-end}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font:600 10.5px "Archivo";letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
select{font:500 14px "Archivo";color:var(--ink);background:var(--card);border:1px solid var(--line);
  border-radius:9px;padding:8px 30px 8px 11px;cursor:pointer;appearance:none;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23736E61' stroke-width='1.5' fill='none'/%3E%3C/svg%3E");
  background-repeat:no-repeat;background-position:right 11px center}
select:focus{outline:none;border-color:var(--roc)}
.thr{flex:1;min-width:240px}
.thr .top{display:flex;justify-content:space-between;align-items:baseline}
.thr .tv{font:700 16px "JetBrains Mono";color:var(--ink)}
input[type=range]{width:100%;accent-color:var(--roc);cursor:pointer}
.pbtn{font:600 12px "Archivo";color:var(--muted);background:var(--card);border:1px solid var(--line);
  border-radius:9px;padding:8px 12px;cursor:pointer}
.pbtn:hover{color:var(--ink);border-color:var(--muted)}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(110px,1fr));gap:12px;margin:18px 0}
.mc{position:relative;padding:13px 14px;border:1px solid var(--line);border-radius:12px;background:#FFFDF8}
.mc.fixed{background:#0F6E630D;border-color:#0F6E6333}
.mc .v{font-family:"Fraunces";font-weight:600;font-size:28px;line-height:1}
.mc .l{font:600 10.5px "Archivo";letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin-top:6px}
.mc.fixed .v,.mc.fixed .l{color:var(--roc)}
.mc.tpr{background:#B0562A0D;border-color:#B0562A33}.mc.tpr .v,.mc.tpr .l{color:var(--pos)}
.mc.fpr{background:#7368610D;border-color:#73686133}.mc.fpr .v,.mc.fpr .l{color:var(--neg)}
.live{font:600 13px "JetBrains Mono";margin:4px 0 8px;color:var(--muted)}
.mc[data-tip]{cursor:help}
.mc[data-tip]:hover::after{content:attr(data-tip);position:absolute;left:0;top:calc(100% + 8px);z-index:20;width:240px;max-width:80vw;background:var(--ink);color:var(--paper);font:500 12px/1.55 "Archivo";text-transform:none;letter-spacing:0;padding:10px 12px;border-radius:10px;box-shadow:0 10px 30px -8px rgba(27,26,23,.4);white-space:normal;pointer-events:none}
.plots{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.plot{padding:14px}
.plot h3{margin:0 0 2px;font:700 12px "JetBrains Mono"}
.plot .cap{font-size:11px;color:var(--muted);margin-bottom:8px}
.legend{display:flex;gap:16px;font-size:11.5px;color:var(--muted);margin:2px 0 8px}
.legend span{display:flex;align-items:center;gap:6px}
.legend i{width:11px;height:11px;border-radius:2px;display:inline-block}
.note{margin-top:16px;font-size:12.5px;color:var(--muted);line-height:1.55}.note b{color:var(--ink);font-weight:600}
@media(max-width:640px){.plots{grid-template-columns:1fr}.cards{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="kicker">Diabetes risk prediction &middot; BRFSS 2015 / 2021 / 2023</div>
  <h1>Accuracy vs AUC</h1>
  <p class="sub">Drag the <b>decision threshold</b> and watch accuracy, sensitivity and specificity move,
  while the <b>ROC curve and AUC stay fixed</b>. At ~14% prevalence, a high accuracy can hide a model that
  misses most cases &mdash; AUC measures threshold-free ranking quality, which is what selects the model
  that ranks high-risk patients best.</p>
</header>

<div class="panel controls">
  <div class="field"><label>Model</label><select id="model"></select></div>
  <div class="field"><label>Cohort</label><select id="year"></select></div>
  <div class="field thr">
    <div class="top"><label>Decision threshold</label><span class="tv" id="tv">0.50</span></div>
    <input type="range" id="thr" min="0" max="1" step="0.01" value="0.5">
  </div>
  <button class="pbtn" id="allneg">Predict all negative</button>
</div>

<div class="cards">
  <div class="mc" data-tip="Fraction of all patients classified correctly: (TP+TN)/N. Misleading under class imbalance - always predicting the majority class already scores high."><div class="v" id="mAcc">&mdash;</div><div class="l">Accuracy</div></div>
  <div class="mc" data-tip="Share of the whole population flagged positive at this threshold: (TP+FP)/N. 100% at threshold 0, falling as the threshold rises. Below the 14.2% true prevalence means the model under-calls disease."><div class="v" id="mPpr">&mdash;</div><div class="l">Predicted positive</div></div>
  <div class="mc tpr" data-tip="Sensitivity / recall: of patients who truly have diabetes, the fraction flagged positive - TP/(TP+FN). This is the ROC y-axis."><div class="v" id="mSens">&mdash;</div><div class="l">TPR (sensitivity)</div></div>
  <div class="mc fpr" data-tip="Of patients who truly do not have diabetes, the fraction wrongly flagged positive - FP/(FP+TN). Equals 1 - specificity; this is the ROC x-axis."><div class="v" id="mFpr">&mdash;</div><div class="l">FPR (1 - specificity)</div></div>
  <div class="mc" data-tip="Of patients who truly do not have diabetes, the fraction correctly cleared - TN/(TN+FP). Equals 1 - FPR."><div class="v" id="mSpec">&mdash;</div><div class="l">Specificity (TNR)</div></div>
  <div class="mc fixed" data-tip="Area under the ROC curve: probability a random positive ranks above a random negative - threshold-free ranking quality. Fixed as the threshold moves; changes only with model or cohort."><div class="v" id="mAuc">&mdash;</div><div class="l">AUC &middot; fixed</div></div>
</div>

<p style="margin:-4px 0 18px;font-size:12px;color:var(--muted);line-height:1.5"><b style="color:var(--ink)">Predicted positive</b> (share flagged, (TP+FP)/N) and <b style="color:var(--ink)">Accuracy</b> (share correct, (TP+TN)/N) measure different things. At threshold 0 they read 100% vs 14.2% &mdash; and 14.2% there is not a coincidence: flag everyone and accuracy must equal the prevalence.

<div class="plots">
  <div class="panel plot">
    <h3>Predicted probability by class</h3>
    <div class="cap">Left of the line = predicted &ldquo;no diabetes&rdquo; &middot; right = predicted &ldquo;diabetes&rdquo;</div>
    <div class="legend"><span><i style="background:var(--neg)"></i>No diabetes</span><span><i style="background:var(--pos)"></i>Diabetes</span><span><i style="background:var(--ink);opacity:.2"></i>Overlap</span></div>
    <div class="live" id="distLive">&mdash;</div>
    <svg id="dist" viewBox="0 0 460 250" width="100%"></svg>
    <div class="cap" style="margin-top:6px">Shaded band = overlap of the two classes (confusable mass). AUC measures how far apart the two distributions sit; the histograms do not move when you drag the line, so AUC is fixed &mdash; only the four areas change.</div>
  </div>
  <div class="panel plot">
    <h3>ROC curve</h3>
    <div class="cap">Dot = operating point at the chosen threshold</div>
    <div class="live" id="rocLive">&mdash;</div>
    <svg id="roc" viewBox="0 0 460 250" width="100%"></svg>
  </div>
</div>

<p class="note panel" style="padding:14px 16px" id="note"></p>
</div>

<script>
const DATA=__DATA__;const B=DATA.bins;
const MODELS=[["xgboost","XGBoost"],["random_forest","Random Forest"],["logistic_regression","Logistic Regression"]];
const YEARS=["2015","2021","2023"];
const $=id=>document.getElementById(id);
const opt=(s,a)=>{s.innerHTML=a.map(([v,t])=>`<option value="${v}">${t}</option>`).join("");};
opt($("model"),MODELS);opt($("year"),YEARS.map(y=>[y,y]));
$("model").value="xgboost";$("year").value="2021";
const C=getComputedStyle(document.documentElement);
const col=n=>C.getPropertyValue(n).trim();

function cell(){return DATA.cells[`${$("year").value}|${$("model").value}`];}
function metrics(c,t){
  const k=Math.round(t*B);let tp=0,fp=0;
  for(let i=k;i<B;i++){tp+=c.pos[i];fp+=c.neg[i];}
  const fn=c.nPos-tp,tn=c.nNeg-fp,tot=c.nPos+c.nNeg;
  return{acc:(tp+tn)/tot,sens:tp/c.nPos,spec:tn/c.nNeg,fpr:fp/c.nNeg,ppr:(tp+fp)/tot};
}
const pct=x=>(x*100).toFixed(1)+"%";

function drawDist(c,t){
  const W=460,H=250,L=12,R=12,T=12,Bm=28,pw=W-L-R,base=H-Bm;
  const dPos=c.pos.map(v=>v/c.nPos),dNeg=c.neg.map(v=>v/c.nNeg);
  const mx=Math.max(...dPos,...dNeg)||1,bw=pw/B;
  let s=`<line x1="${L}" y1="${base}" x2="${W-R}" y2="${base}" stroke="${col('--line')}" stroke-width="1"/>`;
  [0,0.25,0.5,0.75,1].forEach(g=>{const x=L+g*pw;
    s+=`<text x="${x}" y="${base+16}" text-anchor="middle" font-size="10" fill="${col('--muted')}" font-family="JetBrains Mono">${g.toFixed(2)}</text>`;});
  const tx=L+t*pw;
  s+=`<rect x="${tx}" y="${T}" width="${(W-R)-tx}" height="${base-T}" fill="${col('--roc')}" opacity="0.06"/>`;
  for(let i=0;i<B;i++){const x=L+i*bw;
    const hN=dNeg[i]/mx*(base-T),hP=dPos[i]/mx*(base-T);
    if(hN>0)s+=`<rect x="${x}" y="${base-hN}" width="${bw}" height="${hN}" fill="${col('--neg')}" opacity="0.5"/>`;
    if(hP>0)s+=`<rect x="${x}" y="${base-hP}" width="${bw}" height="${hP}" fill="${col('--pos')}" opacity="0.6"/>`;
  }
  for(let i=0;i<B;i++){const ox=L+i*bw,ov=Math.min(dNeg[i],dPos[i])/mx*(base-T);if(ov>0)s+=`<rect x="${ox}" y="${base-ov}" width="${bw}" height="${ov}" fill="${col('--ink')}" opacity="0.2"/>`;}
  s+=`<line x1="${tx}" y1="${T-2}" x2="${tx}" y2="${base}" stroke="${col('--ink')}" stroke-width="2"/>`;
  if((W-R)-tx>64)s+=`<text x="${tx+5}" y="${T+11}" font-size="9.5" fill="${col('--roc')}" font-family="Archivo">predicted: diabetes &#8594;</text>`;
  if(tx-L>64)s+=`<text x="${tx-5}" y="${T+11}" text-anchor="end" font-size="9.5" fill="${col('--muted')}" font-family="Archivo">&#8592; predicted: no diabetes</text>`;
  $("dist").innerHTML=s;
}
function drawRoc(c,m){
  const W=460,H=250,L=36,R=14,T=12,Bm=30,pw=W-L-R,ph=H-T-Bm,base=T+ph;
  const X=v=>L+v*pw,Y=v=>T+(1-v)*ph;
  const pts=c.roc.map(p=>`${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join(" ");
  const area=`${X(0)},${base} ${pts} ${X(1)},${base}`;
  let s=`<polygon points="${area}" fill="${col('--roc')}" opacity="0.12"/>`;
  s+=`<line x1="${L}" y1="${base}" x2="${X(1)}" y2="${Y(1)}" stroke="${col('--muted')}" stroke-width="1" stroke-dasharray="4 3"/>`;
  s+=`<polyline points="${pts}" fill="none" stroke="${col('--roc')}" stroke-width="2"/>`;
  s+=`<line x1="${L}" y1="${T}" x2="${L}" y2="${base}" stroke="${col('--line')}" stroke-width="1"/>`;
  s+=`<line x1="${L}" y1="${base}" x2="${X(1)}" y2="${base}" stroke="${col('--line')}" stroke-width="1"/>`;
  [0,0.5,1].forEach(g=>{
    s+=`<text x="${X(g)}" y="${base+15}" text-anchor="middle" font-size="10" fill="${col('--muted')}" font-family="JetBrains Mono">${g}</text>`;
    s+=`<text x="${L-6}" y="${Y(g)+3}" text-anchor="end" font-size="10" fill="${col('--muted')}" font-family="JetBrains Mono">${g}</text>`;});
  s+=`<text x="${X(0.5)}" y="${base+27}" text-anchor="middle" font-size="10.5" fill="${col('--muted')}">FPR (1 - specificity)</text>`;
  s+=`<text transform="translate(12,${T+ph/2}) rotate(-90)" text-anchor="middle" font-size="10.5" fill="${col('--muted')}">TPR (sensitivity)</text>`;
  s+=`<circle cx="${X(m.fpr)}" cy="${Y(m.sens)}" r="6" fill="${col('--pos')}" stroke="${col('--card')}" stroke-width="2"/>`;
  $("roc").innerHTML=s;
}

function render(){
  const c=cell(),t=+$("thr").value,m=metrics(c,t);
  $("tv").textContent=t.toFixed(2);
  $("mAcc").textContent=pct(m.acc);$("mPpr").textContent=(m.ppr*100).toFixed(2)+"%";$("mSens").textContent=pct(m.sens);
  $("mFpr").textContent=pct(m.fpr);$("mSpec").textContent=pct(m.spec);$("mAuc").textContent=c.auc.toFixed(4);
  $("distLive").innerHTML=`<b>Right</b> of line (predicted +): <span style="color:${col('--pos')}">TPR=${m.sens.toFixed(3)}</span>, <span style="color:${col('--neg')}">FPR=${m.fpr.toFixed(3)}</span> &middot; <b>Left</b> (predicted &minus;): <span style="color:${col('--pos')}">FNR=${(1-m.sens).toFixed(3)}</span>, <span style="color:${col('--neg')}">TNR=${m.spec.toFixed(3)}</span>`;
  $("rocLive").innerHTML=`Operating point (FPR, TPR) = <span style="color:${col('--pos')}">(${m.fpr.toFixed(3)}, ${m.sens.toFixed(3)})</span>`;
  drawDist(c,t);drawRoc(c,m);
  let msg;
  if(m.sens<0.3)msg=`At this high threshold the model calls almost everyone <b>negative</b>: accuracy ~${pct(m.acc)} looks strong, but sensitivity is only ~${pct(m.sens)} &mdash; most cases are missed. Accuracy is inflated by the ~86% negative majority.`;
  else if(m.sens>0.85)msg=`At this low threshold ~${pct(m.sens)} of cases are caught, but specificity drops to ~${pct(m.spec)} (many false alarms); accuracy ~${pct(m.acc)}.`;
  else msg=`Sensitivity ~${pct(m.sens)}, specificity ~${pct(m.spec)}, accuracy ~${pct(m.acc)}. Each threshold is one point on the ROC curve.`;
  $("note").innerHTML=msg+` However the threshold moves, <b>AUC stays ${c.auc.toFixed(4)}</b> &mdash; it is the whole curve, not a single operating point.`;
}
["model","year","thr"].forEach(id=>$(id).addEventListener("input",render));
$("allneg").addEventListener("click",()=>{$("thr").value=1;render();});
render();
</script>
</body>
</html>"""


def render_html(payload: dict) -> str:
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA__", data)


def main() -> None:
    payload = get_payload()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    html = render_html(payload)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[threshold] repo root : {REPO_ROOT}")
    print(f"[threshold] cells     : {len(payload['cells'])} "
          f"(expected {len(YEARS) * len(MODELS)})")
    print(f"[threshold] wrote     : {OUT_HTML.relative_to(REPO_ROOT)} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
