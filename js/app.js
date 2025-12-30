const selfieInput = document.getElementById("selfieInput");
const selfiePreview = document.getElementById("selfiePreview");
const outCanvas = document.getElementById("outCanvas");

const sidelenEl = document.getElementById("sidelen");
const itersEl = document.getElementById("iters");
const proximityEl = document.getElementById("proximity");
const edgeAlphaEl = document.getElementById("edgeAlpha");
const framesEl = document.getElementById("frames");
const fpsEl = document.getElementById("fps");
const annealEl = document.getElementById("anneal");

const runBtn = document.getElementById("runBtn");
const stopBtn = document.getElementById("stopBtn");
const downloadPngBtn = document.getElementById("downloadPngBtn");
const recordBtn = document.getElementById("recordBtn");
const downloadVideoLink = document.getElementById("downloadVideoLink");

const progressEl = document.getElementById("progress");
const statusEl = document.getElementById("status");

let selfieImage = null;
let stockImage = null;

function setStatus(s){ statusEl.textContent = s; }
function setProgress(v){ progressEl.value = v; }

async function loadImageFromFile(file){
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => { URL.revokeObjectURL(url); resolve(img); };
    img.onerror = reject;
    img.src = url;
  });
}

async function loadDefaultStock(){
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = "assets/target.jpg";
  await img.decode();
  return img;
}

function drawToCanvasSquare(img, canvas, sidelen){
  const ctx = canvas.getContext("2d");
  canvas.width = sidelen;
  canvas.height = sidelen;

  const w = img.naturalWidth || img.width;
  const h = img.naturalHeight || img.height;
  const s = Math.min(w,h);
  const sx = ((w - s)/2)|0;
  const sy = ((h - s)/2)|0;

  ctx.clearRect(0,0,sidelen,sidelen);
  ctx.drawImage(img, sx, sy, s, s, 0, 0, sidelen, sidelen);
}

function getRGBA(canvas){
  const ctx = canvas.getContext("2d");
  const imgData = ctx.getImageData(0,0,canvas.width,canvas.height);
  return imgData.data; // Uint8ClampedArray
}

/* ---------------- Simulation (seed motion) ---------------- */
function simStep(pos, vel, dst, sidelen, params){
  const N = pos.length/2;
  const H=sidelen, W=sidelen;

  // attraction
  for(let i=0;i<N;i++){
    const dy = dst[2*i+0] - pos[2*i+0];
    const dx = dst[2*i+1] - pos[2*i+1];
    const dist = Math.hypot(dy,dx) + 1e-6;
    const ax = params.k_dst * dx * dist / sidelen;
    const ay = params.k_dst * dy * dist / sidelen;
    vel[2*i+0] += ay * params.dt;
    vel[2*i+1] += ax * params.dt;
  }

  // neighbor buckets (grid)
  const buckets = Array.from({length: N}, () => []); // N cells (H*W)
  const cellY = new Int32Array(N);
  const cellX = new Int32Array(N);
  for(let i=0;i<N;i++){
    const y = Math.max(0, Math.min(H-1, pos[2*i+0]|0));
    const x = Math.max(0, Math.min(W-1, pos[2*i+1]|0));
    cellY[i]=y; cellX[i]=x;
    buckets[y*W+x].push(i);
  }

  for(let i=0;i<N;i++){
    let vSumY=0, vSumX=0, wSum=0;

    const cy = cellY[i], cx = cellX[i];
    for(let ny=Math.max(0,cy-1); ny<=Math.min(H-1,cy+1); ny++){
      for(let nx=Math.max(0,cx-1); nx<=Math.min(W-1,cx+1); nx++){
        const cell = buckets[ny*W+nx];
        for(let kk=0; kk<cell.length; kk++){
          const j = cell[kk];
          if(j===i) continue;

          const dpy = pos[2*i+0] - pos[2*j+0];
          const dpx = pos[2*i+1] - pos[2*j+1];
          const d2 = dpy*dpy + dpx*dpx + 1e-6;
          const d = Math.sqrt(d2);

          if(d < params.repel_radius){
            const wj = (params.repel_radius - d) / params.repel_radius;
            vel[2*i+0] += (dpy/d) * (params.repel_strength * wj);
            vel[2*i+1] += (dpx/d) * (params.repel_strength * wj);
          }

          const wv = 1.0 / (1.0 + d2);
          vSumY += vel[2*j+0] * wv;
          vSumX += vel[2*j+1] * wv;
          wSum += wv;
        }
      }
    }

    if(wSum > 0){
      const vAvgY = vSumY / wSum;
      const vAvgX = vSumX / wSum;
      vel[2*i+0] += (vAvgY - vel[2*i+0]) * params.align_strength;
      vel[2*i+1] += (vAvgX - vel[2*i+1]) * params.align_strength;
    }
  }

  // damping + clamp speed + integrate
  for(let i=0;i<N;i++){
    vel[2*i+0] *= params.damp;
    vel[2*i+1] *= params.damp;

    const sp = Math.hypot(vel[2*i+0], vel[2*i+1]) + 1e-8;
    const scale = Math.min(1.0, params.max_v / sp);
    vel[2*i+0] *= scale;
    vel[2*i+1] *= scale;

    pos[2*i+0] += vel[2*i+0] * params.dt;
    pos[2*i+1] += vel[2*i+1] * params.dt;

    pos[2*i+0] = Math.max(0, Math.min(sidelen-1, pos[2*i+0]));
    pos[2*i+1] = Math.max(0, Math.min(sidelen-1, pos[2*i+1]));
  }
}

/* ---------------- JFA Voronoi rendering (CPU) ---------------- */
function renderVoronoiJFA(pos, seedRgb, sidelen, ctx2d){
  const H=sidelen, W=sidelen, N=H*W;
  const bestId = new Int32Array(N);
  const bestD2 = new Float32Array(N);
  bestId.fill(-1);
  bestD2.fill(1e30);

  // Stamp seeds (rounded to nearest pixel)
  for(let s=0; s<N; s++){
    const y = Math.max(0, Math.min(H-1, Math.round(pos[2*s+0])));
    const x = Math.max(0, Math.min(W-1, Math.round(pos[2*s+1])));
    const idx = y*W + x;
    bestId[idx] = s;
    bestD2[idx] = 0;
  }

  // step init: greatest power of 2 <= max(H,W)/2 (standard JFA schedule)
  let step = 1;
  while(step < Math.max(H,W)) step <<= 1;
  step >>= 1;

  // iterate
  const offsets = []; // 9 offsets each step
  while(step >= 1){
    offsets.length = 0;
    for(const dy of [-step, 0, step]){
      for(const dx of [-step, 0, step]){
        offsets.push([dy,dx]);
      }
    }

    for(let y=0;y<H;y++){
      for(let x=0;x<W;x++){
        const idx = y*W+x;
        let curId = bestId[idx];
        let curD2 = bestD2[idx];

        for(let k=0;k<offsets.length;k++){
          const dy = offsets[k][0], dx = offsets[k][1];
          const ny = y + dy, nx = x + dx;
          if(ny<0 || ny>=H || nx<0 || nx>=W) continue;
          const nidx = ny*W + nx;
          const candId = bestId[nidx];
          if(candId < 0) continue;

          const sy = pos[2*candId+0];
          const sx = pos[2*candId+1];
          const ddy = (y - sy);
          const ddx = (x - sx);
          const d2 = ddy*ddy + ddx*ddx;
          if(d2 < curD2){
            curD2 = d2;
            curId = candId;
          }
        }

        bestId[idx] = curId;
        bestD2[idx] = curD2;
      }
    }

    step >>= 1;
  }

  // paint image
  const imgData = ctx2d.createImageData(W,H);
  const out = imgData.data;
  for(let i=0;i<N;i++){
    const id = bestId[i];
    const r = (id>=0) ? seedRgb[3*id+0] : 0;
    const g = (id>=0) ? seedRgb[3*id+1] : 0;
    const b = (id>=0) ? seedRgb[3*id+2] : 0;
    out[4*i+0]=r;
    out[4*i+1]=g;
    out[4*i+2]=b;
    out[4*i+3]=255;
  }
  ctx2d.putImageData(imgData, 0, 0);
}

function makeSimParams(){
  return {
    k_dst: 0.020,
    damp: 0.97,
    max_v: 2.0,
    repel_radius: 0.95,
    repel_strength: 0.06,
    align_strength: 0.03,
    dt: 1.0
  };
}

/* ---------------- Recording ---------------- */
let recorder = null;
let recordedChunks = [];
function startRecording(canvas, fps){
  recordedChunks = [];
  const stream = canvas.captureStream(fps);
  recorder = new MediaRecorder(stream, {mimeType: "video/webm;codecs=vp9"});
  recorder.ondataavailable = (e) => { if(e.data.size>0) recordedChunks.push(e.data); };
  recorder.onstop = () => {
    const blob = new Blob(recordedChunks, {type:"video/webm"});
    const url = URL.createObjectURL(blob);
    downloadVideoLink.href = url;
    downloadVideoLink.style.display = "inline-flex";
    downloadVideoLink.textContent = "Download WebM";
  };
  recorder.start();
}

function stopRecording(){
  if(recorder && recorder.state !== "inactive"){
    recorder.stop();
  }
}

/* ---------------- Pipeline orchestration ---------------- */
let worker = new Worker(new URL("./worker.js", import.meta.url), {type:"module"});
let running = false;
let stopRequested = false;

async function ensureImages(){
  if(!stockImage) stockImage = await loadDefaultStock(); // hidden target
  if(!selfieImage) throw new Error("Upload a selfie first.");
}


function drawSelfiePreview(){
  drawToCanvasSquare(selfieImage, selfiePreview, 256);
}

selfieInput.addEventListener("change", async (e)=>{
  const f = e.target.files?.[0];
  if(!f) return;
  selfieImage = await loadImageFromFile(f);
  if(!stockImage) stockImage = await loadDefaultStock();
  drawSelfiePreview();
});



runBtn.addEventListener("click", async ()=>{
  try{
    await ensureImages();
    stopRequested = false;
    running = true;

    runBtn.disabled = true;
    stopBtn.disabled = false;
    downloadPngBtn.disabled = true;
    recordBtn.disabled = true;
    downloadVideoLink.style.display = "none";

    const sidelen = +sidelenEl.value;
    const iters = +itersEl.value;
    const proximity = +proximityEl.value;
    const edgeAlpha = +edgeAlphaEl.value;
    const frames = +framesEl.value;
    const fps = +fpsEl.value;
    const anneal = !!annealEl.checked;

    setStatus("Preparing images…");
    setProgress(0);

    // Make working canvases
    const tmpSelfie = document.createElement("canvas");
    const tmpStock  = document.createElement("canvas");
    drawToCanvasSquare(selfieImage, tmpSelfie, sidelen);
    drawToCanvasSquare(stockImage,  tmpStock,  sidelen);

    const srcRGBA = getRGBA(tmpSelfie);
    const tgtRGBA = getRGBA(tmpStock);

    // Resize output canvas for nice display
    outCanvas.width = 512;
    outCanvas.height = 512;
    const outCtx = outCanvas.getContext("2d");
    outCtx.imageSmoothingEnabled = false;

    // Offscreen render canvas at sidelen
    const renderCanvas = document.createElement("canvas");
    renderCanvas.width = sidelen;
    renderCanvas.height = sidelen;
    const rctx = renderCanvas.getContext("2d", {willReadFrequently:true});
    rctx.imageSmoothingEnabled = false;

    // Kick worker
    setStatus("Computing assignment (this is the heavy step)…");
    worker.postMessage({
      type:"run",
      sidelen, iters, proximity, edgeAlpha, anneal,
      srcRGBA, tgtRGBA
    });

    let dstOfSrc = null;
    let seedRgb = null;

    const waitResult = new Promise((resolve, reject)=>{
      worker.onmessage = (e)=>{
        const msg = e.data;
        if(msg.type === "progress"){
          const stage = msg.stage;
          if(stage === "assignment"){
            const frac = msg.it / Math.max(1, msg.iters);
            setProgress(0.05 + 0.85*frac);
            setStatus(`Assignment: ${msg.it}/${msg.iters} (radius=${msg.radius}, accepted=${msg.accepted})`);
          } else if(stage === "lab"){
            setProgress(0.02); setStatus("Converting to Lab…");
          } else if(stage === "edges"){
            setProgress(0.03); setStatus("Computing edge importance…");
          } else if(stage === "init"){
            setProgress(0.05); setStatus("Building initial permutation…");
          } else if(stage === "invert"){
            setProgress(0.92); setStatus("Inverting permutation…");
          }
        }
        if(msg.type === "done"){
          dstOfSrc = new Float32Array(msg.dstOfSrc);
          seedRgb = new Uint8ClampedArray(msg.seedRgb);
          resolve();
        }
      };
      worker.onerror = reject;
    });

    await waitResult;
    setProgress(0.94);
    setStatus("Assignment done. Simulating + rendering…");

    // Seed positions: start on grid
    const N = sidelen*sidelen;
    const pos = new Float32Array(N*2);
    const vel = new Float32Array(N*2);
    for(let i=0;i<N;i++){
      pos[2*i+0] = (i/sidelen)|0;
      pos[2*i+1] = (i%sidelen);
      vel[2*i+0] = 0;
      vel[2*i+1] = 0;
    }

    const simParams = makeSimParams();

    // Enable record button now
    recordBtn.disabled = false;
    recordBtn.onclick = () => {
      downloadVideoLink.style.display = "none";
      startRecording(outCanvas, fps);
      recordBtn.disabled = true;
      setStatus("Recording… (will stop automatically at the end)");
    };

    // Animation loop
    const total = frames;
    for(let f=0; f<total; f++){
      if(stopRequested){ break; }

      renderVoronoiJFA(pos, seedRgb, sidelen, rctx);

      // scale to output canvas
      outCtx.clearRect(0,0,outCanvas.width,outCanvas.height);
      outCtx.drawImage(renderCanvas, 0,0,outCanvas.width,outCanvas.height);

      simStep(pos, vel, dstOfSrc, sidelen, simParams);

      setProgress(0.94 + 0.06*(f/(total-1)));
      setStatus(`Rendering frame ${f+1}/${total}…`);

      // let UI breathe
      await new Promise(r => requestAnimationFrame(r));
    }

    if(recorder && recorder.state !== "inactive"){
      stopRecording();
    }

    setProgress(1);
    setStatus(stopRequested ? "Stopped." : "Done.");
    running = false;
    runBtn.disabled = false;
    stopBtn.disabled = true;
    downloadPngBtn.disabled = false;

  } catch(err){
    console.error(err);
    setStatus(String(err?.message || err));
    setProgress(0);
    running = false;
    runBtn.disabled = false;
    stopBtn.disabled = true;
  }
});

stopBtn.addEventListener("click", ()=>{
  stopRequested = true;
  stopBtn.disabled = true;
  runBtn.disabled = false;
  setStatus("Stopping…");
});

downloadPngBtn.addEventListener("click", ()=>{
  outCanvas.toBlob((blob)=>{
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mosaic.png";
    a.click();
    URL.revokeObjectURL(url);
  }, "image/png");
});

// init
(async ()=>{
  stockImage = await loadDefaultStock();
  setStatus("Idle. Upload a selfie, then press ▶ Play.");
})();
