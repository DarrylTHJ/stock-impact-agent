"""Dependency-free interactive causal graph for Streamlit."""

from __future__ import annotations

import json

import streamlit.components.v1 as components


def render_interactive_graph(graph_data: dict, height: int = 700) -> None:
    payload = json.dumps(graph_data, ensure_ascii=False).replace("<", "\\u003c")
    html = r"""
<!doctype html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;font-family:Inter,ui-sans-serif,system-ui;color:#243145;background:#fff}
.bar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;padding:8px 10px;border:1px solid #dbe3ee;border-bottom:0;border-radius:12px 12px 0 0;background:#f8fafc;font-size:12px}
.bar label{display:flex;gap:5px;align-items:center}.hint{margin-left:auto;color:#718096}
.wrap{position:relative;height:590px;border:1px solid #dbe3ee;border-radius:0 0 12px 12px;overflow:hidden;background:radial-gradient(circle at center,#fff,#f3f7fc)}
canvas{width:100%;height:100%;display:block;cursor:grab}canvas.dragging{cursor:grabbing}
.detail{position:absolute;left:12px;right:12px;bottom:12px;padding:10px 12px;border:1px solid #cfdae8;border-radius:9px;background:rgba(255,255,255,.96);box-shadow:0 4px 15px #1e293b18;font-size:12px;display:none;max-height:105px;overflow:auto}.detail b{color:#172033}.legend{color:#64748b}
</style></head><body>
<div class="bar">
 <label><input id="chen" type="checkbox" checked> Chen</label>
 <label><input id="hlib" type="checkbox" checked> HLIB</label>
 <label><input id="inference" type="checkbox" checked> Applicable rules</label>
 <label><input id="context" type="checkbox"> Market context</label>
 <span class="legend">Solid = Direct · Dashed = Inference · Dotted = Context</span>
 <span class="hint">Drag · scroll to zoom · click for details</span>
</div>
<div class="wrap"><canvas></canvas><div class="detail"></div></div>
<script>
const DATA=__GRAPH_DATA__;
const canvas=document.querySelector('canvas'),ctx=canvas.getContext('2d'),wrap=document.querySelector('.wrap'),detail=document.querySelector('.detail');
const chen=document.getElementById('chen'),hlib=document.getElementById('hlib'),inference=document.getElementById('inference'),context=document.getElementById('context');
const sourceColor={Chen:'#4361d9','HLIB Research':'#8b5bc7'};
const directionColor={positive:'#18a567',negative:'#df4b4b',mixed:'#d28a19',neutral:null};
let W=0,H=0,dpr=1,zoom=1,panX=0,panY=0,drag=null,panning=false,lastX=0,lastY=0,positioned=false;
const nodes=DATA.nodes.map(n=>({...n,x:0,y:0}));
const byId=Object.fromEntries(nodes.map(n=>[n.id,n]));
function resize(){const oldW=W||wrap.clientWidth,oldH=H||wrap.clientHeight;dpr=devicePixelRatio||1;W=wrap.clientWidth;H=wrap.clientHeight;canvas.width=W*dpr;canvas.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);if(!positioned){nodes.forEach(n=>{n.x=(n.x_ratio??.5)*W;n.y=(n.y_ratio??.5)*H});positioned=true}else{nodes.forEach(n=>{n.x*=W/oldW;n.y*=H/oldH})}}
new ResizeObserver(resize).observe(wrap);resize();
function visibleEdge(e){if(e.source==='Chen'&&!chen.checked)return false;if(e.source==='HLIB Research'&&!hlib.checked)return false;if(e.support==='source_grounded_inference'&&!inference.checked)return false;if(e.support==='context'&&!context.checked)return false;return true}
function visibleNodes(){const ids=new Set(['event']);DATA.edges.filter(visibleEdge).forEach(e=>{ids.add(e.from);ids.add(e.to)});return ids}
function step(){const edges=DATA.edges.filter(visibleEdge),ids=visibleNodes();draw(edges,ids);requestAnimationFrame(step)}
function roundRect(x,y,w,h,r){ctx.beginPath();ctx.roundRect(x,y,w,h,r)}
function nodeSize(n){if(n.type==='event')return [150,66];if(n.type==='sector')return [105,56];if(n.type==='company')return [100,46];if(n.type==='context')return [120,44];return [130,52]}
function wrapText(text,max=24){const words=text.split(/\s+/),lines=[];let line='';for(const w of words){if((line+' '+w).trim().length>max&&line){lines.push(line);line=w}else line=(line+' '+w).trim()}if(line)lines.push(line);if(lines.length>3)return [...lines.slice(0,2),lines.slice(2).join(' ').slice(0,max-1)+'…'];return lines}
function draw(edges,ids){ctx.clearRect(0,0,W,H);ctx.save();ctx.translate(panX,panY);ctx.scale(zoom,zoom);ctx.font='700 12px Inter,system-ui';ctx.fillStyle='#4361d9';ctx.textAlign='center';ctx.fillText('CHEN',W*.19,28);ctx.fillStyle='#8b5bc7';ctx.fillText('HLIB RESEARCH',W*.81,28);
 edges.forEach(e=>{const a=byId[e.from],b=byId[e.to],color=sourceColor[e.source]||'#94a3b8';ctx.strokeStyle=color;ctx.lineWidth=2;ctx.setLineDash(e.support==='context'?[2,6]:e.support==='source_grounded_inference'?[8,5]:[]);ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();ctx.setLineDash([]);const ang=Math.atan2(b.y-a.y,b.x-a.x),tx=b.x-Math.cos(ang)*34,ty=b.y-Math.sin(ang)*34;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(tx,ty);ctx.lineTo(tx-Math.cos(ang-.5)*9,ty-Math.sin(ang-.5)*9);ctx.lineTo(tx-Math.cos(ang+.5)*9,ty-Math.sin(ang+.5)*9);ctx.fill()});
 nodes.filter(n=>ids.has(n.id)).forEach(n=>{const [w,h]=nodeSize(n),x=n.x-w/2,y=n.y-h/2;let fill='#fff',stroke=n.source?sourceColor[n.source]:'#a8b5c7';if(n.type==='event'){fill='#eaf0ff';stroke='#4568e5'}if(n.type==='source'){fill=n.source==='Chen'?'#eef2ff':'#f5efff'}if(n.type==='sector'||n.type==='company'){fill=n.direction==='positive'?'#eaf8f1':n.direction==='negative'?'#fff0f0':'#fff7e6';stroke=directionColor[n.direction]||stroke}if(n.type==='context'){fill='#f1f4f8';stroke='#9aa8b8'}roundRect(x,y,w,h,11);ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle=stroke;ctx.lineWidth=n.type==='event'||n.type==='source'?2.3:1.6;ctx.stroke();ctx.fillStyle='#243145';ctx.font=(n.type==='event'||n.type==='source'?'600 12px':'500 11px')+' Inter,system-ui';ctx.textAlign='center';ctx.textBaseline='middle';const lines=wrapText(n.label,n.type==='event'?27:23),lh=14;lines.forEach((line,i)=>ctx.fillText(line,n.x,n.y+(i-(lines.length-1)/2)*lh,w-12))});ctx.restore()}
function point(evt){const r=canvas.getBoundingClientRect();return{x:(evt.clientX-r.left-panX)/zoom,y:(evt.clientY-r.top-panY)/zoom}}
function hit(p){const ids=visibleNodes();return [...nodes].reverse().find(n=>{if(!ids.has(n.id))return false;const [w,h]=nodeSize(n);return Math.abs(p.x-n.x)<w/2&&Math.abs(p.y-n.y)<h/2})}
canvas.onpointerdown=e=>{const p=point(e),n=hit(p);lastX=e.clientX;lastY=e.clientY;if(n){drag=n;detail.style.display='none'}else panning=true;canvas.classList.add('dragging');canvas.setPointerCapture(e.pointerId)};
canvas.onpointermove=e=>{if(drag){const p=point(e);drag.x=p.x;drag.y=p.y;drag.vx=drag.vy=0}else if(panning){panX+=e.clientX-lastX;panY+=e.clientY-lastY;lastX=e.clientX;lastY=e.clientY}};
canvas.onpointerup=e=>{const p=point(e),n=hit(p);if(drag&&n===drag){detail.innerHTML='<b>'+escapeHtml(n.label)+'</b><br>'+escapeHtml(n.detail||'');detail.style.display='block'}drag=null;panning=false;canvas.classList.remove('dragging')};
canvas.onwheel=e=>{e.preventDefault();const old=zoom;zoom=Math.max(.45,Math.min(2.2,zoom*(e.deltaY<0?1.1:.9)));const r=canvas.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;panX=mx-(mx-panX)*zoom/old;panY=my-(my-panY)*zoom/old},{passive:false};
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
[chen,hlib,inference,context].forEach(x=>x.onchange=()=>{detail.style.display='none'});requestAnimationFrame(step);
</script></body></html>
""".replace("__GRAPH_DATA__", payload)
    components.html(html, height=height, scrolling=False)
