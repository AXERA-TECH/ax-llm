#!/usr/bin/env python3
"""
LocateAnything WebUI — a small dependency-free (stdlib only) web front-end for an
ax-llm `serve` instance running the LocateAnything-3B grounding/detection model.

Run:
    AXLLM_SERVE_URL=http://127.0.0.1:8010 \
    AXLLM_IMAGE_DIR=/path/to/sample_images \
    python3 locateanything_webui.py --port 7861
then open http://localhost:7861

The backend serves the UI, lists sample thumbnails from AXLLM_IMAGE_DIR, and proxies
detection to the ax-llm OpenAI-compatible /v1/chat/completions endpoint with stream=true,
parsing complete <box>..</box> / <ref>..</ref> tokens and re-emitting them to the browser
as clean SSE events (status / box / done) for real-time incremental drawing.
"""
import os, sys, json, re, argparse, mimetypes, urllib.request, urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SERVE_URL = os.environ.get("AXLLM_SERVE_URL", "http://127.0.0.1:8010").rstrip("/")
MODEL     = os.environ.get("AXLLM_MODEL", "AXERA-TECH/LocateAnything-3B")
IMAGE_DIR = os.environ.get("AXLLM_IMAGE_DIR", "")
IMG_EXT   = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# ----------------------------------------------------------------------------- token parsing
REF_RE   = re.compile(r"<ref>(.*?)</ref>")
BOX_RE   = re.compile(r"<box>((?:<\d+>)+)</box>")
COORD_RE = re.compile(r"<(\d+)>")

def parse_stream_into_events(buf, state):
    """Consume as many complete <ref>/<box> elements from the front of `buf` as possible,
    yielding ('box', label, coords) events; returns leftover buffer."""
    events = []
    while True:
        rm = REF_RE.search(buf)
        bm = BOX_RE.search(buf)
        cands = []
        if rm: cands.append((rm.start(), "ref", rm))
        if bm: cands.append((bm.start(), "box", bm))
        if not cands:
            break
        cands.sort(key=lambda c: c[0])
        _, kind, m = cands[0]
        if kind == "ref":
            state["label"] = m.group(1)
            buf = buf[m.end():]
        else:
            coords = [int(x) for x in COORD_RE.findall(m.group(1))]
            if len(coords) in (2, 4):
                events.append((state.get("label", ""), coords))
            buf = buf[m.end():]
    return buf, events

# ----------------------------------------------------------------------------- HTML/CSS/JS
PAGE = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>LocateAnything</title>
<style>
:root{
  --bg:#0b0f16; --panel:#131a24; --panel2:#0f151d; --line:#233041; --line2:#1a2430;
  --txt:#e6edf3; --muted:#8aa0b4; --accent:#22d3ee; --accent2:#2dd4bf;
  --ok:#22c55e; --warn:#f59e0b; --bad:#ef4444;
}
*{box-sizing:border-box}
html,body{height:100%;margin:0;overflow:hidden;background:var(--bg);color:var(--txt);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,"PingFang SC","Microsoft YaHei",sans-serif;}
::-webkit-scrollbar{width:8px;height:8px}
::-webkit-scrollbar-thumb{background:#2a3a4d;border-radius:8px}
::-webkit-scrollbar-track{background:transparent}
.app{height:100vh;display:grid;grid-template-rows:132px 1fr;gap:12px;padding:12px}

/* ---------- carousel ---------- */
.carousel{position:relative;overflow:hidden;border:1px solid var(--line);border-radius:14px;
  background:linear-gradient(180deg,#0f151d,#0c1119);}
.carousel::before,.carousel::after{content:"";position:absolute;top:0;bottom:0;width:60px;z-index:3;pointer-events:none}
.carousel::before{left:0;background:linear-gradient(90deg,#0c1119,transparent)}
.carousel::after{right:0;background:linear-gradient(270deg,#0c1119,transparent)}
.track{position:absolute;top:0;left:0;height:100%;display:flex;gap:12px;padding:12px;will-change:transform}
.thumb{position:relative;height:100%;width:176px;flex:0 0 auto;border-radius:10px;overflow:hidden;
  border:1px solid var(--line);cursor:pointer;background:#0a0e14;transition:transform .15s,border-color .15s,box-shadow .15s}
.thumb img{width:100%;height:100%;object-fit:cover;display:block}
.thumb:hover{transform:translateY(-2px);border-color:var(--accent);box-shadow:0 6px 20px rgba(34,211,238,.18)}
.thumb.active{border-color:var(--accent2);box-shadow:0 0 0 2px rgba(45,212,191,.35)}
.carousel .tag{position:absolute;left:12px;top:8px;z-index:4;font-size:11px;color:var(--muted);
  letter-spacing:.12em;text-transform:uppercase}

/* ---------- main ---------- */
.main{display:grid;grid-template-columns:340px 1fr;gap:12px;min-height:0}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;min-height:0}

/* ---------- config ---------- */
.config{display:flex;flex-direction:column;padding:16px;gap:14px;overflow:hidden}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16px;letter-spacing:.02em}
.brand .dot{width:10px;height:10px;border-radius:50%;background:var(--accent2);box-shadow:0 0 12px var(--accent2)}
.brand small{color:var(--muted);font-weight:500;font-size:11px;margin-left:auto}
section{display:flex;flex-direction:column;gap:8px}
label{font-size:11px;color:var(--muted);letter-spacing:.1em;text-transform:uppercase}
select,input[type=text]{background:var(--panel2);border:1px solid var(--line);color:var(--txt);
  border-radius:9px;padding:9px 11px;font-size:13px;outline:none;width:100%}
select:focus,input[type=text]:focus{border-color:var(--accent)}
.chips{display:flex;flex-wrap:wrap;gap:6px;max-height:96px;overflow-y:auto;padding-right:2px}
.chip{display:inline-flex;align-items:center;gap:7px;background:#0f1822;border:1px solid #2a3a4a;
  color:#cfe;border-radius:999px;padding:5px 9px 5px 9px;font-size:12px}
.chip .cdot{width:9px;height:9px;border-radius:50%;flex:0 0 auto;box-shadow:0 0 7px currentColor}
.chip b{font-weight:600}
.chip x{cursor:pointer;color:#88a;font-style:normal;line-height:1;font-size:14px;margin-left:2px}
.chip x:hover{color:var(--bad)}
.add-row{display:flex;gap:6px}
.add-row button{flex:0 0 38px;border:1px solid var(--line);background:var(--panel2);color:var(--txt);
  border-radius:9px;font-size:18px;cursor:pointer}
.add-row button:hover{border-color:var(--accent);color:var(--accent)}
.range-row{display:flex;align-items:center;gap:10px}
input[type=range]{flex:1;accent-color:var(--accent)}
.range-row b{font-variant-numeric:tabular-nums;font-size:12px;color:var(--accent);min-width:34px;text-align:right}
.uploadbtn{border:1px dashed #35526080;background:#0f1a22;color:#a9d7e2;border-radius:10px;padding:11px;
  cursor:pointer;font-size:13px;text-align:center;transition:.15s}
.uploadbtn:hover{border-color:var(--accent);color:var(--accent);background:#10222b}
.spacer{flex:1}
.status{display:flex;align-items:center;gap:10px;padding:10px 12px;background:var(--panel2);
  border:1px solid var(--line);border-radius:10px;font-size:13px}
.light{width:12px;height:12px;border-radius:50%;background:#3a4a5c;transition:.2s;flex:0 0 auto}
.light.warn{background:var(--warn);box-shadow:0 0 12px var(--warn);animation:pulse 1s infinite}
.light.run{background:var(--accent);box-shadow:0 0 12px var(--accent);animation:pulse .8s infinite}
.light.ok{background:var(--ok);box-shadow:0 0 12px var(--ok)}
.light.bad{background:var(--bad);box-shadow:0 0 12px var(--bad)}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
#statusText{color:var(--muted)}
.btns{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.btns button{border-radius:10px;padding:12px;font-size:14px;font-weight:600;cursor:pointer;border:1px solid transparent;transition:.15s}
.btns button:disabled{opacity:.4;cursor:not-allowed}
.primary{background:linear-gradient(180deg,#22d3ee,#0ea5b7);color:#04222a;border:none}
.primary:not(:disabled):hover{box-shadow:0 6px 18px rgba(34,211,238,.35)}
.danger{background:transparent;border:1px solid #6b2530;color:#ff8a95}
.danger:not(:disabled):hover{background:#2a1418;border-color:var(--bad)}

/* ---------- display ---------- */
.display{position:relative;overflow:hidden;background:
  radial-gradient(1200px 500px at 50% -10%,#12202b,transparent),var(--panel2)}
#canvas{position:absolute;inset:0;width:100%;height:100%;display:block}
.hint{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;color:var(--muted);
  font-size:14px;pointer-events:none}
.countbadge{position:absolute;right:12px;top:12px;z-index:5;background:#0c1620cc;border:1px solid var(--line);
  border-radius:999px;padding:6px 12px;font-size:12px;color:#bdeef5;backdrop-filter:blur(6px)}
.countbadge b{color:var(--accent);font-variant-numeric:tabular-nums}
/* scanning overlay */
.scan{position:absolute;inset:0;z-index:4;pointer-events:none;overflow:hidden}
.scan .grid{position:absolute;inset:0;opacity:.10;
  background-image:linear-gradient(#22d3ee 1px,transparent 1px),linear-gradient(90deg,#22d3ee 1px,transparent 1px);
  background-size:38px 38px;mask-image:linear-gradient(180deg,transparent,#000 30%,#000 70%,transparent)}
.scan .bar{position:absolute;left:0;right:0;height:26%;
  background:linear-gradient(180deg,transparent,rgba(34,211,238,.10) 45%,rgba(34,211,238,.55) 50%,rgba(34,211,238,.10) 55%,transparent);
  box-shadow:0 0 26px rgba(34,211,238,.35);animation:sweep 1.6s cubic-bezier(.5,0,.5,1) infinite}
.scan .edge{position:absolute;left:0;right:0;height:2px;background:var(--accent);filter:blur(.4px);
  box-shadow:0 0 14px var(--accent);animation:sweepEdge 1.6s cubic-bezier(.5,0,.5,1) infinite}
@keyframes sweep{0%{top:-26%}100%{top:100%}}
@keyframes sweepEdge{0%{top:0}100%{top:100%}}
.scan .txt{position:absolute;left:14px;bottom:12px;font-size:12px;color:#8fd6e4;letter-spacing:.14em;text-transform:uppercase}
.scan .txt::after{content:"";animation:dots 1.2s steps(4,end) infinite}
@keyframes dots{0%{content:""}25%{content:"·"}50%{content:"··"}75%{content:"···"}100%{content:""}}
/* ---------- modal ---------- */
.modal{position:fixed;inset:0;z-index:60;display:flex;align-items:center;justify-content:center;
  background:rgba(4,8,12,.55);backdrop-filter:blur(5px);animation:mfade .14s ease}
.modal[hidden]{display:none}
@keyframes mfade{from{opacity:0}to{opacity:1}}
.modal-card{width:380px;max-width:88vw;background:linear-gradient(180deg,#18212d,#131a24);border:1px solid var(--line);
  border-radius:16px;padding:22px;box-shadow:0 26px 70px rgba(0,0,0,.55);animation:mpop .18s cubic-bezier(.2,.8,.3,1.25)}
@keyframes mpop{from{transform:translateY(10px) scale(.96);opacity:0}to{transform:none;opacity:1}}
.modal-h{font-size:16px;font-weight:700;margin-bottom:10px}
.modal-b{font-size:13px;color:var(--muted);margin-bottom:18px}
.mtags{display:flex;flex-wrap:wrap;gap:7px;margin-top:10px}
.mtags .mt{display:inline-flex;align-items:center;border-radius:999px;padding:4px 12px;font-size:12px;
  font-weight:600;border:1px solid currentColor;background:#0f1822}
.modal-f{display:flex;gap:10px;justify-content:flex-end}
.modal-f button{border-radius:10px;padding:9px 16px;font-size:13px;font-weight:600;cursor:pointer;border:1px solid transparent}
.ghost{background:transparent;border:1px solid var(--line);color:var(--txt)}
.ghost:hover{border-color:var(--muted)}
</style></head>
<body>
<div class="app">
  <div class="carousel" id="carousel"><span class="tag">Samples</span><div class="track" id="track"></div></div>
  <div class="main">
    <div class="panel config">
      <div class="brand"><span class="dot"></span>LocateAnything<small id="srvtag"></small></div>

      <section>
        <label>Task</label>
        <select id="task">
          <option value="detection">Object detection</option>
          <option value="grounding">Phrase grounding</option>
        </select>
      </section>

      <section id="catsSection">
        <label>Categories</label>
        <div class="chips" id="chips"></div>
        <div class="add-row"><input id="catInput" type="text" placeholder="add category, e.g. person"><button id="addCat" title="add">+</button></div>
      </section>

      <section id="phraseSection" hidden>
        <label>Phrase</label>
        <input id="phrase" type="text" placeholder="e.g. the man in the red shirt">
      </section>

      <section>
        <label>Max targets</label>
        <div class="range-row"><input type="range" id="maxtok" min="16" max="256" step="8" value="64"><b id="maxtokv">64</b></div>
      </section>

      <div class="uploadbtn" id="uploadBtn">⬆  Upload image</div>
      <input type="file" id="fileInput" accept="image/*" hidden>

      <div class="spacer"></div>

      <div class="status"><span class="light" id="light"></span><span id="statusText">Idle</span></div>
      <div class="btns">
        <button class="primary" id="detectBtn">Detect</button>
        <button class="danger" id="stopBtn" disabled>Stop</button>
      </div>
    </div>

    <div class="panel display" id="display">
      <canvas id="canvas"></canvas>
      <div class="countbadge" id="countbadge" hidden><b id="count">0</b> boxes</div>
      <div class="scan" id="scan" hidden><div class="grid"></div><div class="bar"></div><div class="edge"></div><div class="txt">encoding image</div></div>
      <div class="hint" id="hint">Pick a sample above or upload an image, then press Detect</div>
    </div>
  </div>
</div>

<div class="modal" id="modal" hidden>
  <div class="modal-card">
    <div class="modal-h">Use this image’s categories?</div>
    <div class="modal-b">This image is tagged with:<div id="modalTags" class="mtags"></div></div>
    <div class="modal-f"><button class="ghost" id="modalKeep">Keep current</button><button class="primary" id="modalSwitch">Switch categories</button></div>
  </div>
</div>
<script>
const $=id=>document.getElementById(id);
let thumbs=[], current=null, boxes=[], detecting=false, abortCtrl=null, rafId=null;
let cats=["person"];
const colors={};      // label -> bright color, persistent; chips and boxes share it
const imageTags={};   // image src -> its category tags (seeded from server, updated as user edits)
const imagePhrases={};// image src -> its phrase-grounding sentence
function colorFor(lbl){ if(!(lbl in colors)){ const h=Math.floor(Math.random()*360); colors[lbl]="hsl("+h+",90%,62%)"; } return colors[lbl]; }

/* ---------------- carousel (seamless loop) ---------------- */
const THUMB_W=176, GAP=12, STEP=THUMB_W+GAP;
let scrollX=0, paused=false, unitWidth=0;
async function loadThumbs(){
  try{ thumbs=await (await fetch("/api/thumbs")).json(); }catch(e){ thumbs=[]; }
  thumbs.forEach(t=>{ if(t.tags && t.tags.length) imageTags[t.src]=t.tags.slice(); if(t.phrase) imagePhrases[t.src]=t.phrase; });
  buildTrack();
  if(thumbs.length && !current) selectSrc(thumbs[0].src, null, false);
}
function thumbEl(t){
  const d=document.createElement("div"); d.className="thumb"; d.dataset.src=t.src;
  if(current && t.src===current.src) d.classList.add("active");
  const im=document.createElement("img"); im.src=t.src; im.alt=""; d.appendChild(im);
  d.onclick=()=>selectSrc(t.src, d, true); return d;
}
function buildTrack(){
  const track=$("track"); track.innerHTML="";
  if(!thumbs.length){ unitWidth=0; return; }
  unitWidth = thumbs.length*STEP;
  const cw = $("carousel").clientWidth||1000;
  const copies = Math.max(2, Math.ceil((cw+unitWidth)/unitWidth)+1);  // always fill viewport+one unit -> no gap
  for(let k=0;k<copies;k++) for(const t of thumbs) track.appendChild(thumbEl(t));
}
function markActive(){ document.querySelectorAll(".thumb").forEach(n=>n.classList.toggle("active", current&&n.dataset.src===current.src)); }
function tick(){
  if(unitWidth>0){ if(!paused) scrollX+=0.4; scrollX=((scrollX%unitWidth)+unitWidth)%unitWidth;
    $("track").style.transform="translateX("+(-scrollX)+"px)"; }
  requestAnimationFrame(tick);
}
const carousel=$("carousel");
carousel.addEventListener("mouseenter",()=>paused=true);
carousel.addEventListener("mouseleave",()=>paused=false);
carousel.addEventListener("wheel",e=>{ e.preventDefault(); scrollX+=e.deltaY; },{passive:false});
new ResizeObserver(()=>buildTrack()).observe(carousel);

/* ---------------- image handling ---------------- */
function selectSrc(src, node, userInitiated){
  if(detecting) abortDetection("Stopped");     // switching image while detecting -> reset
  const img=new Image();
  img.onload=()=>{ current={img, src, w:img.naturalWidth, h:img.naturalHeight, b64:toB64(img,src)};
    boxes=[]; $("hint").style.display="none"; $("countbadge").hidden=true; setLight("idle","Ready"); resize(); drawStatic(); markActive();
    $("phrase").value = imagePhrases[src] || "";   // load this image's phrase (used in Phrase-grounding mode)
    const tags=imageTags[src];
    if(tags && tags.length){
      // Always load this image's categories too (so both category + phrase are ready and the
      // task selector just picks which to use). In detection mode a user click still asks via
      // the modal before replacing the current categories; otherwise load silently.
      if(userInitiated && $("task").value==="detection") maybeAskSwitch(tags);
      else { cats=tags.slice(); renderChips(); }
    }
  };
  img.onerror=()=>setLight("error","Image load failed");
  img.src=src;
}
function toB64(img,src){
  if(src.startsWith("data:")) return src.split(",")[1];
  const c=document.createElement("canvas"); c.width=img.naturalWidth; c.height=img.naturalHeight;
  c.getContext("2d").drawImage(img,0,0);
  try{ return c.toDataURL("image/jpeg",0.92).split(",")[1]; }catch(e){ return null; }
}
$("uploadBtn").onclick=()=>$("fileInput").click();
$("fileInput").onchange=e=>{ const f=e.target.files[0]; if(!f) return; const rd=new FileReader();
  rd.onload=()=>{ const src=rd.result; thumbs.unshift({src,name:f.name}); buildTrack(); selectSrc(src,null,false); }; rd.readAsDataURL(f); };

/* ---------------- canvas ---------------- */
const cv=$("canvas"), ctx=cv.getContext("2d");
function resize(){ const d=$("display"); const dpr=window.devicePixelRatio||1;
  cv.width=Math.max(1,Math.round(d.clientWidth*dpr)); cv.height=Math.max(1,Math.round(d.clientHeight*dpr));
  if(!detecting) drawStatic(); }
new ResizeObserver(resize).observe($("display"));
function fitRect(){ const iw=current.w, ih=current.h; const s=Math.min(cv.width/iw, cv.height/ih);
  const dw=iw*s, dh=ih*s; return {ox:(cv.width-dw)/2, oy:(cv.height-dh)/2, dw, dh, s}; }
function drawBox(r,b,alpha){
  const c=b.coords, col=colorFor(b.label);
  ctx.save(); ctx.globalAlpha=alpha;
  const scaleIn=0.9+0.1*alpha;
  if(b.coords.length===2){ // point
    const px=r.ox+c[0]/1000*r.dw, py=r.oy+c[1]/1000*r.dh, rad=8*(window.devicePixelRatio||1);
    ctx.fillStyle=col; ctx.beginPath(); ctx.arc(px,py,rad*scaleIn,0,7); ctx.fill();
    ctx.globalAlpha=alpha*0.35; ctx.beginPath(); ctx.arc(px,py,rad*2.2*scaleIn,0,7); ctx.fill();
    ctx.restore(); return;
  }
  let x1=r.ox+c[0]/1000*r.dw, y1=r.oy+c[1]/1000*r.dh, x2=r.ox+c[2]/1000*r.dw, y2=r.oy+c[3]/1000*r.dh;
  const cx=(x1+x2)/2, cy=(y1+y2)/2;
  x1=cx+(x1-cx)*scaleIn; x2=cx+(x2-cx)*scaleIn; y1=cy+(y1-cy)*scaleIn; y2=cy+(y2-cy)*scaleIn;
  const lw=Math.max(2,r.s*(window.devicePixelRatio||1)*0.9);
  ctx.lineWidth=lw; ctx.strokeStyle=col; ctx.shadowColor=col; ctx.shadowBlur=10*alpha;
  ctx.strokeRect(x1,y1,x2-x1,y2-y1);
  ctx.shadowBlur=0;
  if(b.label){ ctx.font=(13*(window.devicePixelRatio||1))+"px system-ui"; const tw=ctx.measureText(b.label).width;
    const pad=6*(window.devicePixelRatio||1), h=20*(window.devicePixelRatio||1);
    ctx.fillStyle=col; ctx.globalAlpha=alpha; ctx.fillRect(x1, Math.max(r.oy,y1-h), tw+pad*2, h);
    ctx.fillStyle="#04222a"; ctx.textBaseline="middle"; ctx.fillText(b.label, x1+pad, Math.max(r.oy,y1-h)+h/2); }
  ctx.restore();
}
function drawStatic(){ ctx.clearRect(0,0,cv.width,cv.height); if(!current) return; const r=fitRect();
  ctx.drawImage(current.img,r.ox,r.oy,r.dw,r.dh); for(const b of boxes) drawBox(r,b,1); }
function renderLoop(){ ctx.clearRect(0,0,cv.width,cv.height); const r=fitRect();
  ctx.drawImage(current.img,r.ox,r.oy,r.dw,r.dh); const now=performance.now();
  for(const b of boxes){ const a=Math.min(1,(now-b.t0)/240); drawBox(r,b,a); }
  if(detecting) rafId=requestAnimationFrame(renderLoop); }

/* ---------------- categories ---------------- */
function saveCurrentTags(){ if(current) imageTags[current.src]=cats.slice(); }
function renderChips(){ const box=$("chips"); box.innerHTML="";
  cats.forEach((c,i)=>{ const col=colorFor(c); const s=document.createElement("span"); s.className="chip";
    s.innerHTML='<i class="cdot"></i><b></b><x>×</x>'; s.querySelector("b").textContent=c;
    s.style.borderColor=col; s.querySelector("b").style.color=col;
    const dot=s.querySelector(".cdot"); dot.style.background=col; dot.style.color=col;
    s.querySelector("x").onclick=()=>{ cats.splice(i,1); saveCurrentTags(); renderChips(); }; box.appendChild(s); }); }
function addCat(){ const v=$("catInput").value.trim(); if(v && !cats.includes(v)){ colorFor(v); cats.push(v); saveCurrentTags(); renderChips(); } $("catInput").value=""; }
$("addCat").onclick=addCat;
$("catInput").addEventListener("keydown",e=>{ if(e.key==="Enter") addCat(); });
$("task").onchange=()=>{ const t=$("task").value;
  $("catsSection").hidden = t!=="detection"; $("phraseSection").hidden = t!=="grounding"; };
$("maxtok").oninput=()=>$("maxtokv").textContent=$("maxtok").value;

function buildPrompt(){ const t=$("task").value;
  if(t==="ocr") return "Detect all the text in box format.";
  if(t==="grounding"){ const p=$("phrase").value.trim()||"object"; return "Locate all the instances that match the following description: "+p+"."; }
  const tgt = cats.length? cats.join(", ") : "object";
  return "Locate all the instances that matches the following description:"+tgt; }
function buildRequest(){
  const mt=Math.min(1024,(+$("maxtok").value)*6+40), t=$("task").value;
  const r={image:current.b64, max_tokens:mt};
  if(t==="detection" && cats.length) r.categories=cats.slice();   // one query per category (own label/color)
  else r.prompt=buildPrompt();
  return r; }

/* ---------------- status light ---------------- */
function setLight(state,txt){ const l=$("light"); l.className="light"+(state==="idle"?"":" "+({encoding:"warn",detecting:"run",done:"ok",error:"bad"}[state]||""));
  if(txt!==undefined) $("statusText").textContent=txt; }

/* ---------------- detect ---------------- */
async function detect(){
  if(!current || detecting) return;
  if(!current.b64){ setLight("error","Cannot read image data (CORS?)"); return; }
  detecting=true; boxes=[];
  $("detectBtn").disabled=true; $("stopBtn").disabled=false;
  $("count").textContent="0"; $("countbadge").hidden=false;
  $("scan").hidden=false; setLight("encoding","Encoding image…");
  renderLoop();
  abortCtrl=new AbortController();
  try{
    const resp=await fetch("/api/detect",{method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(buildRequest()),signal:abortCtrl.signal});
    if(!resp.ok){ throw new Error("serve HTTP "+resp.status); }
    const reader=resp.body.getReader(), dec=new TextDecoder(); let buf="";
    while(true){ const {done,value}=await reader.read(); if(done) break;
      buf+=dec.decode(value,{stream:true}); let i;
      while((i=buf.indexOf("\n"))>=0){ const line=buf.slice(0,i).trim(); buf=buf.slice(i+1);
        if(!line.startsWith("data:")) continue; let ev; try{ ev=JSON.parse(line.slice(5).trim()); }catch(e){ continue; }
        if(ev.type==="status"){ if(ev.phase==="generating"){ $("scan").hidden=true; setLight("detecting","Detecting…"); } }
        else if(ev.type==="box"){ boxes.push({label:ev.label||"",coords:ev.box,t0:performance.now()}); $("count").textContent=boxes.length; }
        else if(ev.type==="done"){ finishDetect("done"); return; }     // finish the moment the model is done
        else if(ev.type==="error"){ finishDetect("error"); return; }
      } }
  }catch(e){ if(e.name==="AbortError") return; finishDetect("error"); return; }
  finishDetect("done");   // stream closed without an explicit done event
}
function finishDetect(kind){
  if(!detecting) return;
  detecting=false;
  if(abortCtrl){ try{ abortCtrl.abort(); }catch(e){} abortCtrl=null; }
  if(rafId){ cancelAnimationFrame(rafId); rafId=null; }
  $("detectBtn").disabled=false; $("stopBtn").disabled=true; $("scan").hidden=true;
  drawStatic();
  if(kind==="done") setLight("done","Done · "+boxes.length+" detections");
  else if(kind==="error") setLight("error","Detection failed");
}
function abortDetection(msg){
  if(!detecting) return;
  detecting=false;
  if(abortCtrl){ try{ abortCtrl.abort(); }catch(e){} abortCtrl=null; }
  if(rafId){ cancelAnimationFrame(rafId); rafId=null; }
  $("detectBtn").disabled=false; $("stopBtn").disabled=true; $("scan").hidden=true;
  setLight("idle", msg||"Stopped");
}
function stop(){ const n=boxes.length; abortDetection("Stopped · "+n+" detections"); drawStatic(); }
$("detectBtn").onclick=detect;
$("stopBtn").onclick=stop;

/* ---------------- switch-category modal (custom, not window.confirm) ---------------- */
let pendingSwitch=null;
function maybeAskSwitch(tags){
  if($("task").value!=="detection") return;
  const same = tags.length===cats.length && tags.every(t=>cats.includes(t));
  if(same) return;
  const box=$("modalTags"); box.innerHTML="";
  tags.forEach(t=>{ const s=document.createElement("span"); s.className="mt"; s.textContent=t;
    const col=colorFor(t); s.style.color=col; s.style.borderColor=col; box.appendChild(s); });
  pendingSwitch=()=>{ cats=tags.slice(); saveCurrentTags(); renderChips(); };
  $("modal").hidden=false;
}
function closeModal(){ $("modal").hidden=true; pendingSwitch=null; }
$("modalSwitch").onclick=()=>{ const p=pendingSwitch; closeModal(); if(p) p(); };
$("modalKeep").onclick=closeModal;
$("modal").addEventListener("click",e=>{ if(e.target===$("modal")) closeModal(); });
document.addEventListener("keydown",e=>{ if(e.key==="Escape" && !$("modal").hidden) closeModal(); });

/* ---------------- init ---------------- */
$("srvtag").textContent="";
renderChips(); loadThumbs(); tick();
</script>
</body></html>
"""

# ----------------------------------------------------------------------------- server
def list_images(d):
    out = []
    if d and os.path.isdir(d):
        for n in sorted(os.listdir(d)):
            if n.lower().endswith(IMG_EXT):
                out.append(n)
    return out[:48]

def load_tags(d):
    """Per-image presets from <image_dir>/tags.json. Each value is either a bare category
    list (legacy) or {"tags": [...], "phrase": "..."}. Returns {name: {"tags", "phrase"}}."""
    if not d:
        return {}
    try:
        with open(os.path.join(d, "tags.json"), encoding="utf-8") as f:
            m = json.load(f)
    except Exception:
        return {}
    out = {}
    for k, v in m.items():
        if isinstance(v, list):
            out[k] = {"tags": v, "phrase": ""}
        elif isinstance(v, dict):
            t = v.get("tags"); p = v.get("phrase")
            out[k] = {"tags": t if isinstance(t, list) else [],
                      "phrase": p if isinstance(p, str) else ""}
    return out

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, ctype, body, extra=None):
        self.send_response(code); self.send_header("Content-Type", ctype)
        if isinstance(body, str): body = body.encode("utf-8")
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/" or path == "/index.html":
            return self._send(200, "text/html; charset=utf-8", PAGE, {"Cache-Control": "no-store"})
        if path == "/api/thumbs":
            names = list_images(IMAGE_DIR)
            tags = load_tags(IMAGE_DIR)
            data = [{"src": "/thumb/" + n, "name": n,
                     "tags": tags.get(n, {}).get("tags", []),
                     "phrase": tags.get(n, {}).get("phrase", "")} for n in names]
            return self._send(200, "application/json", json.dumps(data))
        if path.startswith("/thumb/"):
            name = os.path.basename(path[len("/thumb/"):])
            fp = os.path.join(IMAGE_DIR, name)
            if IMAGE_DIR and os.path.isfile(fp) and name.lower().endswith(IMG_EXT):
                ctype = mimetypes.guess_type(fp)[0] or "image/jpeg"
                with open(fp, "rb") as f: return self._send(200, ctype, f.read(), {"Cache-Control": "max-age=3600"})
            return self._send(404, "text/plain", "not found")
        return self._send(404, "text/plain", "not found")

    def do_POST(self):
        if self.path.split("?")[0] != "/api/detect":
            return self._send(404, "text/plain", "not found")
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception as e:
            return self._send(400, "text/plain", "bad request: %s" % e)
        image_b64 = req.get("image", "")
        max_tokens = int(req.get("max_tokens", 512))
        categories = req.get("categories")
        # One detection per category (object_detection merges multiple categories under a
        # single <ref>; querying one category at a time keeps per-category labels + colors).
        if isinstance(categories, list) and categories:
            queries = [(c, "Locate all the instances that matches the following description:" + c) for c in categories]
        else:
            queries = [(None, req.get("prompt", ""))]
        # open SSE to client
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        self.close_connection = True
        def emit(obj):
            self.wfile.write(("data: " + json.dumps(obj) + "\n\n").encode("utf-8")); self.wfile.flush()
        emit({"type": "status", "phase": "encoding"})
        started = False; count = 0
        try:
            for forced_label, prompt in queries:
                body = {"model": MODEL, "stream": True, "temperature": 0, "max_tokens": max_tokens,
                        "messages": [{"role": "user", "content": [
                            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image_b64}},
                            {"type": "text", "text": prompt}]}]}
                up = urllib.request.Request(SERVE_URL + "/v1/chat/completions",
                                            data=json.dumps(body).encode("utf-8"),
                                            headers={"Content-Type": "application/json"})
                resp = urllib.request.urlopen(up, timeout=600)
                state = {"label": ""}; buf = ""
                for raw in resp:
                    line = raw.decode("utf-8", "ignore").strip()
                    if not line.startswith("data:"): continue
                    d = line[5:].strip()
                    if d == "[DONE]": break
                    try: o = json.loads(d)
                    except Exception: continue
                    piece = (o.get("choices", [{}])[0].get("delta", {}) or {}).get("content") or ""
                    if not piece: continue
                    if not started: emit({"type": "status", "phase": "generating"}); started = True
                    buf += piece
                    buf, evs = parse_stream_into_events(buf, state)
                    for lab, coords in evs:
                        count += 1
                        emit({"type": "box", "label": (forced_label if forced_label is not None else lab), "box": coords, "index": count})
            emit({"type": "done", "count": count})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:
            try: emit({"type": "error", "message": str(e)})
            except Exception: pass

class Server(ThreadingHTTPServer):
    daemon_threads = True

def main():
    global SERVE_URL, IMAGE_DIR, MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--serve-url", default=SERVE_URL)
    ap.add_argument("--image-dir", default=IMAGE_DIR)
    ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    SERVE_URL, IMAGE_DIR, MODEL = a.serve_url.rstrip("/"), a.image_dir, a.model
    print("LocateAnything WebUI")
    print("  serve : %s  (model=%s)" % (SERVE_URL, MODEL))
    print("  images: %s  (%d found)" % (IMAGE_DIR or "(none)", len(list_images(IMAGE_DIR))))
    print("  open  : http://localhost:%d" % a.port)
    Server((a.host, a.port), Handler).serve_forever()

if __name__ == "__main__":
    main()
