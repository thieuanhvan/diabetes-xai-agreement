"""Export an interactive Feature Agreement Explorer (self-contained HTML).

One standalone HTML that consolidates every feature-importance agreement
metric computed by the XAI-agreement run into a single visual tool. For a
chosen comparison axis and context it shows the two top-K importance lists
side by side (shared features highlighted) and a full metric panel:

    Set overlap : cABC J_A, Jaccard@70/80/90 (mass), top-5, top-10, Overlap@K, Jaccard@K
    Rank        : RBO, Spearman rho (temporal only)
    Vector      : cosine

Three comparison axes:
    cross-method  SHAP vs PI, same model x year
    cross-model   two models, same method, same year
    temporal      two years, same model x method

This is the visual counterpart of the agreement CSV tables: instead of
reading columns across several files, the reader picks a context and sees
all metrics for that cell at once, with the top-K lists making the set
overlaps concrete.

Inputs (relative to repo root, auto-detected):
    outputs/cdc_brfss_diabetes_{year}/shap/{model}_shap_feature_importance.csv
    outputs/cdc_brfss_diabetes_{year}/fi/{model}_feature_importance.csv
    outputs/xai_agreement/within_model_agreement.csv
    outputs/xai_agreement/cross_model_shap_agreement.csv
    outputs/xai_agreement/temporal_stability.csv
    outputs/xai_agreement/within_model_cabc.csv
    outputs/xai_agreement/cross_model_shap_cabc.csv
    outputs/xai_agreement/temporal_stability_cabc.csv
    outputs/xai_agreement/spearman_summary.csv

Output:
    outputs/xai_agreement/agreement_explorer.html

No model retraining, no network access, stdlib only.

Right-click run in PyCharm (repo root auto-detected from this file's
location, so the script may live at repo root or under src/analysis/) or:
    python run_agreement_explorer_export.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

YEARS = ["2015", "2021", "2023"]
MODELS = ["xgboost", "random_forest", "logistic_regression"]
METHOD_KEYS = ["SHAP", "FI"]  # second method stored under fi/; tables label it "PI"


def _find_repo_root(start: Path) -> Path:
    """Walk up from `start` until a directory containing outputs/xai_agreement."""
    for d in [start, *start.parents]:
        if (d / "outputs" / "xai_agreement").is_dir():
            return d
    # Fallback: assume the script sits at repo root.
    return start


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
OUTPUTS_DIR = REPO_ROOT / "outputs"
AGREEMENT_DIR = OUTPUTS_DIR / "xai_agreement"
OUT_HTML = AGREEMENT_DIR / "agreement_explorer.html"


def _read_ranking(path: Path) -> list[str]:
    rows = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            rows.append((r["feature"], float(r["importance"])))
    rows.sort(key=lambda x: -x[1])
    return [f for f, _ in rows]


def _read_rows(path: Path, keep: list[str]) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for r in csv.DictReader(fh):
            out.append({k: r.get(k, "") for k in keep})
    return out


def build_data() -> dict:
    rankings: dict[str, list[str]] = {}
    for year in YEARS:
        cohort = OUTPUTS_DIR / f"cdc_brfss_diabetes_{year}"
        for model in MODELS:
            rankings[f"{year}|{model}|SHAP"] = _read_ranking(
                cohort / "shap" / f"{model}_shap_feature_importance.csv")
            rankings[f"{year}|{model}|FI"] = _read_ranking(
                cohort / "fi" / f"{model}_feature_importance.csv")

    agreement = {
        "cross_method": _read_rows(
            AGREEMENT_DIR / "within_model_agreement.csv",
            ["year", "model", "J@70", "J@80", "J@90", "RBO", "cosine",
             "top5_overlap", "top10_overlap"]),
        "cross_model": _read_rows(
            AGREEMENT_DIR / "cross_model_shap_agreement.csv",
            ["year", "model_A", "model_B", "method", "J@70", "J@80", "J@90",
             "RBO", "cosine"]),
        "temporal": _read_rows(
            AGREEMENT_DIR / "temporal_stability.csv",
            ["model", "method", "year_A", "year_B", "J@70", "J@80", "J@90",
             "RBO", "cosine"]),
    }
    cabc = {
        "cross_method": _read_rows(
            AGREEMENT_DIR / "within_model_cabc.csv", ["year", "model", "J_A"]),
        "cross_model": _read_rows(
            AGREEMENT_DIR / "cross_model_shap_cabc.csv",
            ["year", "model_A", "model_B", "method", "J_A"]),
        "temporal": _read_rows(
            AGREEMENT_DIR / "temporal_stability_cabc.csv",
            ["model", "method", "year_A", "year_B", "J_A"]),
    }
    spearman = _read_rows(
        AGREEMENT_DIR / "spearman_summary.csv",
        ["model", "year_1", "year_2", "spearman_shap", "spearman_fi"])
    return {"rankings": rankings, "agreement": agreement, "cabc": cabc,
            "spearman": spearman}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature Agreement Explorer &mdash; XAI on BRFSS</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600&family=Archivo:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
:root{
  --paper:#F4EFE4; --card:#FBF8F1; --ink:#1B1A17; --muted:#736E61; --line:#E0D8C7;
  --shap:#0F6E63; --shap-bg:#0F6E6314; --pi:#B0562A; --pi-bg:#B0562A14;
  --gold:#C8A24B; --shadow:0 1px 2px rgba(27,26,23,.05),0 8px 28px -12px rgba(27,26,23,.18);
}
*{box-sizing:border-box}
html,body{margin:0}
body{
  background:radial-gradient(1200px 600px at 85% -10%, #FBF8F1 0%, transparent 60%), var(--paper);
  color:var(--ink); font-family:"Archivo",system-ui,sans-serif;
  -webkit-font-smoothing:antialiased; line-height:1.45; padding:26px 18px 60px;
}
.wrap{max-width:1000px;margin:0 auto}
header{margin-bottom:20px}
.kicker{font:600 11px/1 "Archivo";letter-spacing:.22em;text-transform:uppercase;color:var(--shap);margin-bottom:10px}
h1{font-family:"Fraunces";font-weight:600;font-size:clamp(26px,4.4vw,40px);line-height:1.02;margin:0 0 8px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:14.5px;max-width:66ch}
.sub b{color:var(--ink);font-weight:600}
.panel{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow)}
.controls{padding:16px 16px 6px;margin:22px 0 18px;display:flex;flex-wrap:wrap;gap:14px 20px;align-items:flex-end}
.modes{display:flex;gap:6px;background:#EFE8D8;border:1px solid var(--line);border-radius:11px;padding:4px}
.modes button{font:600 12.5px "Archivo";color:var(--muted);background:none;border:0;padding:8px 13px;border-radius:8px;cursor:pointer;transition:.15s}
.modes button.on{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.06)}
.field{display:flex;flex-direction:column;gap:5px}
.field label{font:600 10.5px "Archivo";letter-spacing:.12em;text-transform:uppercase;color:var(--muted)}
select{font:500 14px "Archivo";color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:9px;padding:8px 30px 8px 11px;cursor:pointer;
  appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M1 1l4 4 4-4' stroke='%23736E61' stroke-width='1.5' fill='none'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 11px center}
select:focus{outline:none;border-color:var(--shap)}
.kbtns{display:flex;gap:5px}
.kbtns button{font:600 13px "JetBrains Mono";width:38px;height:36px;border:1px solid var(--line);background:var(--card);color:var(--muted);border-radius:9px;cursor:pointer;transition:.15s}
.kbtns button.on{background:var(--ink);color:var(--paper);border-color:var(--ink)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.col{padding:14px 14px 16px}
.col h3{margin:0 0 2px;font-family:"JetBrains Mono";font-weight:700;font-size:13px;letter-spacing:.02em}
.col .ctx{font-size:11.5px;color:var(--muted);margin-bottom:12px;font-weight:500}
.colA h3{color:var(--shap)} .colB h3{color:var(--pi)}
.row{display:flex;align-items:center;gap:10px;padding:7px 9px;border-radius:9px;margin-bottom:5px;border:1px solid transparent;transition:.18s}
.rank{font:700 12px "JetBrains Mono";color:var(--muted);width:20px;text-align:right;flex:none}
.feat{font:500 13.5px "JetBrains Mono";letter-spacing:-.01em}
.row.miss{opacity:.42}
.colA .row.hit{background:var(--shap-bg);border-color:#0F6E6333}
.colB .row.hit{background:var(--pi-bg);border-color:#B0562A33}
.row.hit .feat{font-weight:700}
.dot{width:7px;height:7px;border-radius:50%;flex:none;margin-left:auto;background:transparent}
.row.hit .dot{background:var(--gold);box-shadow:0 0 0 3px #C8A24B22}
.metricsPanel{margin-top:16px;padding:18px}
.mpTitle{font:600 10.5px "Archivo";letter-spacing:.14em;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
.headline{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:18px}
.big{padding:14px 16px;border:1px solid var(--line);border-radius:13px;background:#FFFDF8}
.big .v{font-family:"Fraunces";font-weight:600;font-size:34px;line-height:1}
.big .l{font:600 11px "Archivo";letter-spacing:.05em;text-transform:uppercase;color:var(--muted);margin-top:6px}
.big .l b{color:var(--ink)}
.grp{font:700 10px "Archivo";letter-spacing:.13em;text-transform:uppercase;color:var(--shap);margin:6px 0 9px}
.chips{display:grid;grid-template-columns:repeat(auto-fit,minmax(96px,1fr));gap:9px;margin-bottom:14px}
.chip{padding:11px 12px;border:1px solid var(--line);border-radius:11px;background:#FFFDF8}
.chip .cv{font:700 19px "JetBrains Mono";line-height:1}
.chip .cl{font:600 10px "Archivo";letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin-top:5px}
.chip.na .cv{color:var(--line)} .chip.na{background:transparent}
.note{margin-top:16px;font-size:12.5px;color:var(--muted);line-height:1.55}
.note b{color:var(--ink);font-weight:600}
@media(max-width:620px){ .cols{grid-template-columns:1fr} h1{font-size:26px} }
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="kicker">XAI Feature Agreement &middot; BRFSS 2015 / 2021 / 2023 &middot; run 12</div>
  <h1>Feature Agreement Explorer</h1>
  <p class="sub">Do two methods (or two models, or two years) <b>agree on the most important features</b>? Pick a context and K. The lists show each side's top-K with <b>shared features in gold</b>; the panel below reports <b>every agreement metric</b> for that comparison &mdash; set overlap (cABC J<sub>A</sub>, mass Jaccard J@70/80/90, top-5/10), rank (RBO, Spearman&nbsp;&rho;) and vector (cosine).</p>
</header>

<div class="panel controls">
  <div class="field"><label>Comparison mode</label>
    <div class="modes" id="modes">
      <button data-m="cross_method" class="on">Cross-method</button>
      <button data-m="cross_model">Cross-model</button>
      <button data-m="temporal">Temporal</button>
    </div>
  </div>
  <div class="field" id="f_year"><label>Year</label><select id="year"></select></div>
  <div class="field" id="f_model"><label>Model</label><select id="model"></select></div>
  <div class="field" id="f_method"><label>Method</label><select id="method"></select></div>
  <div class="field" id="f_modelB"><label>Model B</label><select id="modelB"></select></div>
  <div class="field" id="f_yearB"><label>Year B</label><select id="yearB"></select></div>
  <div class="field"><label>K</label><div class="kbtns" id="kbtns">
    <button data-k="3">3</button><button data-k="5" class="on">5</button><button data-k="7">7</button><button data-k="10">10</button>
  </div></div>
</div>

<div class="cols">
  <div class="panel col colA"><h3 id="hA">A</h3><div class="ctx" id="cA"></div><div id="listA"></div></div>
  <div class="panel col colB"><h3 id="hB">B</h3><div class="ctx" id="cB"></div><div id="listB"></div></div>
</div>

<div class="panel metricsPanel">
  <div class="mpTitle" id="mpTitle">Agreement metrics</div>
  <div class="headline">
    <div class="big"><div class="v" id="mOv">&mdash;</div><div class="l">Overlap@K = <b id="mOvF">&middot;</b></div></div>
    <div class="big"><div class="v" id="mJa">&mdash;</div><div class="l">Jaccard@K = <b id="mJaF">&middot;</b></div></div>
  </div>
  <div class="grp">Set overlap</div>
  <div class="chips" id="setChips"></div>
  <div class="grp">Rank &amp; vector similarity</div>
  <div class="chips" id="rankChips"></div>
  <p class="note" id="note"></p>
</div>
</div>

<script>
const DATA = __DATA__;
const MODELS=[["xgboost","XGBoost"],["random_forest","Random Forest"],["logistic_regression","Logistic Regression"]];
const YEARS=["2015","2021","2023"];
const METHODS=[["SHAP","SHAP"],["FI","PI"]];
const MLAB=Object.fromEntries(MODELS), METHLAB={SHAP:"SHAP",FI:"PI"};
let mode="cross_method", K=5;
const $=id=>document.getElementById(id);
const opts=(sel,arr)=>{sel.innerHTML=arr.map(([v,t])=>`<option value="${v}">${t}</option>`).join("");};
opts($("year"),YEARS.map(y=>[y,y])); opts($("yearB"),YEARS.map(y=>[y,y]));
opts($("model"),MODELS); opts($("modelB"),MODELS); opts($("method"),METHODS);
$("year").value="2021"; $("model").value="xgboost"; $("method").value="SHAP";
$("modelB").value="random_forest"; $("yearB").value="2023";
const rk=(y,m,meth)=>DATA.rankings[`${y}|${m}|${meth}`]||[];
const fmt=v=>(v===undefined||v===null||v==="")?null:(+v).toFixed(v.toString().includes(".")?Math.min(4,(v.toString().split(".")[1]||"").length):2);

function showFields(){
  const vis=(id,on)=>{$(id).style.display=on?"flex":"none";};
  vis("f_year",true); vis("f_model",true);
  vis("f_method",mode!=="cross_method");
  document.querySelector('#f_year label').textContent=(mode==="temporal")?"Year A":"Year";
  vis("f_modelB",mode==="cross_model"); vis("f_yearB",mode==="temporal");
}

function getCtx(){
  const y=$("year").value,m=$("model").value,meth=$("method").value;
  if(mode==="cross_method")
    return {A:rk(y,m,"SHAP"),B:rk(y,m,"FI"),hA:"SHAP",hB:"PI",
            cA:`${y} &middot; ${MLAB[m]}`,cB:`${y} &middot; ${MLAB[m]}`};
  if(mode==="cross_model"){const mb=$("modelB").value;
    return {A:rk(y,m,meth),B:rk(y,mb,meth),hA:MLAB[m],hB:MLAB[mb],
            cA:`${METHLAB[meth]} &middot; ${y}`,cB:`${METHLAB[meth]} &middot; ${y}`};}
  const y2=$("yearB").value;
  return {A:rk(y,m,meth),B:rk(y2,m,meth),hA:`${y}`,hB:`${y2}`,
          cA:`${METHLAB[meth]} &middot; ${MLAB[m]}`,cB:`${METHLAB[meth]} &middot; ${MLAB[m]}`};
}

function lookup(){
  const y=$("year").value,m=$("model").value,meth=$("method").value;
  let ag,cb,sp=null;
  if(mode==="cross_method"){
    ag=DATA.agreement.cross_method.find(r=>r.year==y&&r.model==m);
    cb=DATA.cabc.cross_method.find(r=>r.year==y&&r.model==m);
  } else if(mode==="cross_model"){
    const mb=$("modelB").value;
    const pair=(r)=>(r.model_A==m&&r.model_B==mb)||(r.model_A==mb&&r.model_B==m);
    ag=DATA.agreement.cross_model.find(r=>r.year==y&&pair(r)&&r.method=="SHAP");
    cb=DATA.cabc.cross_model.find(r=>r.year==y&&pair(r)&&r.method=="SHAP");
  } else {
    const y2=$("yearB").value;
    const pr=(r)=>(r.year_A==y&&r.year_B==y2)||(r.year_A==y2&&r.year_B==y);
    ag=DATA.agreement.temporal.find(r=>r.model==m&&r.method==meth&&pr(r));
    cb=DATA.cabc.temporal.find(r=>r.model==m&&r.method==meth&&pr(r));
    const sr=DATA.spearman.find(r=>r.model==m&&((r.year_1==y&&r.year_2==y2)||(r.year_1==y2&&r.year_2==y)));
    if(sr) sp=(meth==="SHAP")?sr.spearman_shap:sr.spearman_fi;
  }
  return {ag,cb,sp};
}

function chip(label,val){
  const na=(val===null||val===undefined);
  return `<div class="chip ${na?'na':''}"><div class="cv">${na?'&mdash;':val}</div><div class="cl">${label}</div></div>`;
}

function render(){
  showFields();
  const c=getCtx(), A=c.A.slice(0,K), B=c.B.slice(0,K);
  const sA=new Set(A), sB=new Set(B);
  const inter=A.filter(f=>sB.has(f)), uni=new Set([...A,...B]);
  const ov=K?inter.length/K:0, ja=uni.size?inter.length/uni.size:0;
  const mkRow=(f,i,other)=>`<div class="row ${other.has(f)?'hit':'miss'}"><span class="rank">${i+1}</span><span class="feat">${f}</span><span class="dot"></span></div>`;
  $("hA").innerHTML=c.hA; $("cA").innerHTML=c.cA; $("hB").innerHTML=c.hB; $("cB").innerHTML=c.cB;
  $("listA").innerHTML=A.map((f,i)=>mkRow(f,i,sB)).join("");
  $("listB").innerHTML=B.map((f,i)=>mkRow(f,i,sA)).join("");
  $("mOv").textContent=(ov*100).toFixed(0)+"%"; $("mOvF").textContent=`${inter.length} / ${K}`;
  $("mJa").textContent=ja.toFixed(2); $("mJaF").textContent=`${inter.length} / ${uni.size}`;

  // live top-5/top-10 from rankings (works on every axis)
  const ovAt=(k)=>{const a=new Set(c.A.slice(0,k)),b=new Set(c.B.slice(0,k));let n=0;a.forEach(f=>{if(b.has(f))n++;});return (n/k).toFixed(2);};
  const {ag,cb,sp}=lookup();
  const jA = cb? fmt(cb.J_A):null;
  const j80= ag? fmt(ag["J@80"]):null, j70=ag?fmt(ag["J@70"]):null, j90=ag?fmt(ag["J@90"]):null;
  const rbo= ag? fmt(ag.RBO):null, cos=ag?fmt(ag.cosine):null;
  $("setChips").innerHTML =
      chip("cABC J<sub>A</sub>",jA)+chip("J@70 mass",j70)+chip("J@80 mass",j80)+chip("J@90 mass",j90)
      +chip("top-5",ovAt(5))+chip("top-10",ovAt(10));
  $("rankChips").innerHTML =
      chip("RBO",rbo)+chip("Spearman &rho;",sp!==null?fmt(sp):null)+chip("cosine",cos);

  const shared=inter.join(", ")||"none";
  const modeTxt={cross_method:"SHAP vs PI on the same model&times;year",
                 cross_model:"two models, same method &amp; year",
                 temporal:"two years, same model&times;method"}[mode];
  $("mpTitle").innerHTML=`Agreement metrics &mdash; ${modeTxt}`;
  $("note").innerHTML=`<b>${inter.length}</b> shared feature(s) in the top-${K}: ${shared}. &nbsp;`+
    `<b>Overlap@K</b>/<b>top-5</b>/<b>top-10</b> use a fixed-size cut (|&cap;|/K); `+
    `<b>J@70/80/90</b> use the variable-size sets that capture that share of importance mass; `+
    `<b>cABC J<sub>A</sub></b> uses the cABC breakpoint set. RBO and Spearman&nbsp;&rho; compare ranks; cosine compares the importance vectors. `+
    `Spearman&nbsp;&rho; is recorded for the temporal axis only; cross-model agreement is stored for SHAP pairs only.`;
}

$("modes").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;
  mode=b.dataset.m;[...$("modes").children].forEach(x=>x.classList.toggle("on",x===b));render();});
$("kbtns").addEventListener("click",e=>{const b=e.target.closest("button");if(!b)return;
  K=+b.dataset.k;[...$("kbtns").children].forEach(x=>x.classList.toggle("on",x===b));render();});
["year","model","method","modelB","yearB"].forEach(id=>$(id).addEventListener("change",render));
render();
</script>
</body>
</html>"""


def render_html(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__DATA__", payload)


def main() -> None:
    data = build_data()
    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    html = render_html(data)
    OUT_HTML.write_text(html, encoding="utf-8")
    print(f"[explorer] repo root    : {REPO_ROOT}")
    print(f"[explorer] rankings      : {len(data['rankings'])} "
          f"(expected {len(YEARS) * len(MODELS) * len(METHOD_KEYS)})")
    print(f"[explorer] cabc rows     : "
          f"cm={len(data['cabc']['cross_method'])}, "
          f"xm={len(data['cabc']['cross_model'])}, "
          f"tp={len(data['cabc']['temporal'])}")
    print(f"[explorer] spearman rows : {len(data['spearman'])}")
    print(f"[explorer] wrote         : {OUT_HTML.relative_to(REPO_ROOT)} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
