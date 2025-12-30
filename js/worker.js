/* Worker: compute permutation (dest->src) + dst_of_src for the simulation.
   No DOM access here.

   Message in:
   {
     type: "run",
     sidelen, iters, proximity, edgeAlpha, anneal,
     srcRGBA: Uint8ClampedArray (sidelen*sidelen*4),
     tgtRGBA: Uint8ClampedArray (sidelen*sidelen*4)
   }

   Message out:
   - progress updates
   - result: { dstOfSrc: Float32Array (N*2), seedRgb: Uint8ClampedArray (N*3) }
*/

function clamp01(x){ return Math.min(1, Math.max(0, x)); }

function rgbToLab(rgb){ // rgb in [0,1]
  // sRGB -> linear
  const a = 0.055;
  function toLin(u){
    return (u <= 0.04045) ? (u/12.92) : Math.pow((u + a)/(1 + a), 2.4);
  }
  const r = toLin(clamp01(rgb[0]));
  const g = toLin(clamp01(rgb[1]));
  const b = toLin(clamp01(rgb[2]));

  // linear RGB -> XYZ (D65)
  const X = 0.4124564*r + 0.3575761*g + 0.1804375*b;
  const Y = 0.2126729*r + 0.7151522*g + 0.0721750*b;
  const Z = 0.0193339*r + 0.1191920*g + 0.9503041*b;

  // normalize
  const Xn=0.95047, Yn=1.0, Zn=1.08883;
  let x = X/Xn, y=Y/Yn, z=Z/Zn;

  const eps = 216/24389;
  const kappa = 24389/27;
  function f(t){
    return (t > eps) ? Math.cbrt(t) : (kappa*t + 16)/116;
  }
  const fx=f(x), fy=f(y), fz=f(z);
  const L = 116*fy - 16;
  const A = 500*(fx - fy);
  const B = 200*(fy - fz);
  return [L,A,B];
}

function rgbaToLabAndRgb3(rgba){
  const N = rgba.length/4;
  const lab = new Float32Array(N*3);
  const rgb3 = new Uint8ClampedArray(N*3);
  for(let i=0;i<N;i++){
    const r = rgba[4*i+0]/255;
    const g = rgba[4*i+1]/255;
    const b = rgba[4*i+2]/255;
    const L = rgbToLab([r,g,b]);
    lab[3*i+0]=L[0];
    lab[3*i+1]=L[1];
    lab[3*i+2]=L[2];
    rgb3[3*i+0]=rgba[4*i+0];
    rgb3[3*i+1]=rgba[4*i+1];
    rgb3[3*i+2]=rgba[4*i+2];
  }
  return {lab, rgb3};
}

function importanceFromEdges(tgtRGBA, sidelen, alpha){
  const H=sidelen, W=sidelen;
  const gray = new Float32Array(H*W);
  for(let i=0;i<H*W;i++){
    const r=tgtRGBA[4*i+0]/255, g=tgtRGBA[4*i+1]/255, b=tgtRGBA[4*i+2]/255;
    gray[i] = 0.2126*r + 0.7152*g + 0.0722*b;
  }

  // Sobel kernels
  const Kx = [1,0,-1, 2,0,-2, 1,0,-1];
  const Ky = [1,2,1,  0,0,0, -1,-2,-1];

  function at(y,x){
    y = Math.min(H-1, Math.max(0,y));
    x = Math.min(W-1, Math.max(0,x));
    return gray[y*W+x];
  }

  const mag = new Float32Array(H*W);
  let maxv=1e-6;
  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      let gx=0, gy=0;
      let k=0;
      for(let dy=-1;dy<=1;dy++){
        for(let dx=-1;dx<=1;dx++){
          const v = at(y+dy,x+dx);
          gx += Kx[k]*v;
          gy += Ky[k]*v;
          k++;
        }
      }
      const m = Math.sqrt(gx*gx + gy*gy);
      mag[y*W+x]=m;
      if(m>maxv) maxv=m;
    }
  }

  const w = new Float32Array(H*W);
  for(let i=0;i<H*W;i++){
    const norm = mag[i]/maxv;
    w[i] = 1.0 + alpha*norm;
  }
  return w;
}

function initialPermSort(srcLab, tgtLab, N){
  const sKey = new Float32Array(N);
  const tKey = new Float32Array(N);
  for(let i=0;i<N;i++){
    const sL=srcLab[3*i+0], sA=srcLab[3*i+1], sB=srcLab[3*i+2];
    const tL=tgtLab[3*i+0], tA=tgtLab[3*i+1], tB=tgtLab[3*i+2];
    sKey[i]=0.70*sL + 0.15*sA + 0.15*sB;
    tKey[i]=0.70*tL + 0.15*tA + 0.15*tB;
  }

  const sOrder = Array.from({length:N}, (_,i)=>i).sort((i,j)=>sKey[i]-sKey[j]);
  const tOrder = Array.from({length:N}, (_,i)=>i).sort((i,j)=>tKey[i]-tKey[j]);

  const perm = new Int32Array(N);
  for(let k=0;k<N;k++){
    perm[tOrder[k]] = sOrder[k];
  }
  return perm;
}

function costAt(dst, src, tgtLab, srcLab, w, W, proximity){
  const dL = tgtLab[3*dst+0]-srcLab[3*src+0];
  const dA = tgtLab[3*dst+1]-srcLab[3*src+1];
  const dB = tgtLab[3*dst+2]-srcLab[3*src+2];
  const color = (dL*dL + dA*dA + dB*dB) * w[dst];

  const dy = ((dst/W)|0) - ((src/W)|0);
  const dx = (dst%W) - (src%W);
  const dist2 = dy*dy + dx*dx;
  let prox = proximity * dist2;
  prox = prox*prox; // heavy penalty for long moves
  return color + prox;
}

function refinePermSwaps(perm, srcLab, tgtLab, w, sidelen, iters, proximity, anneal){
  const H=sidelen, W=sidelen, N=H*W;
  const curCost = new Float64Array(N);
  for(let j=0;j<N;j++){
    curCost[j] = costAt(j, perm[j], tgtLab, srcLab, w, W, proximity);
  }

  // radius schedule
  let radius = Math.max(4, (H/2)|0);
  const stageIters = Math.max(10000, (iters / (Math.log2(radius)+1))|0);

  const T0=1.0, T1=0.02;

  function randInt(n){ return (Math.random()*n)|0; }
  function randRange(a,b){ return a + Math.floor(Math.random()*(b-a+1)); }

  let accepted=0;
  for(let it=0; it<iters; it++){
    if(it>0 && (it % stageIters)===0 && radius>1){
      radius = Math.max(1, (radius/2)|0);
    }
    const a = randInt(N);
    const ay = (a/W)|0, ax=a%W;
    const by = Math.min(H-1, Math.max(0, ay + randRange(-radius, radius)));
    const bx = Math.min(W-1, Math.max(0, ax + randRange(-radius, radius)));
    const b = by*W + bx;
    if(a===b) continue;

    const srcA = perm[a], srcB = perm[b];
    const newCostA = costAt(a, srcB, tgtLab, srcLab, w, W, proximity);
    const newCostB = costAt(b, srcA, tgtLab, srcLab, w, W, proximity);
    const oldCost = curCost[a] + curCost[b];
    const newCost = newCostA + newCostB;
    const delta = newCost - oldCost;

    let accept = (delta < 0);
    if(!accept && anneal){
      const t = it / Math.max(1, iters-1);
      const T = (1-t)*T0 + t*T1;
      if(delta > 0){
        const p = Math.exp(-delta / Math.max(1e-9, T));
        if(Math.random() < p) accept = true;
      }
    }

    if(accept){
      perm[a]=srcB; perm[b]=srcA;
      curCost[a]=newCostA; curCost[b]=newCostB;
      accepted++;
    }

    if((it % 5000)===0){
      postMessage({type:"progress", stage:"assignment", it, iters, radius, accepted});
    }
  }

  return perm;
}

function dstOfSrcFromPerm(perm, sidelen){
  const H=sidelen, W=sidelen, N=H*W;
  const inv = new Int32Array(N);
  for(let dst=0; dst<N; dst++){
    inv[perm[dst]] = dst;
  }
  const out = new Float32Array(N*2);
  for(let src=0; src<N; src++){
    const dst = inv[src];
    out[2*src+0] = (dst/W)|0;
    out[2*src+1] = (dst%W);
  }
  return out;
}

self.onmessage = (e) => {
  const msg = e.data;
  if(msg.type !== "run") return;

  const sidelen = msg.sidelen|0;
  const iters = msg.iters|0;
  const proximity = +msg.proximity;
  const edgeAlpha = +msg.edgeAlpha;
  const anneal = !!msg.anneal;

  const srcRGBA = msg.srcRGBA;
  const tgtRGBA = msg.tgtRGBA;

  const N = sidelen*sidelen;

  postMessage({type:"progress", stage:"lab", it:0, iters:1});
  const src = rgbaToLabAndRgb3(srcRGBA);
  const tgt = rgbaToLabAndRgb3(tgtRGBA);

  postMessage({type:"progress", stage:"edges", it:0, iters:1});
  const w = importanceFromEdges(tgtRGBA, sidelen, edgeAlpha);

  postMessage({type:"progress", stage:"init", it:0, iters:1});
  let perm = initialPermSort(src.lab, tgt.lab, N);

  postMessage({type:"progress", stage:"assignment", it:0, iters, radius: Math.max(4,(sidelen/2)|0), accepted:0});
  perm = refinePermSwaps(perm, src.lab, tgt.lab, w, sidelen, iters, proximity, anneal);

  postMessage({type:"progress", stage:"invert", it:0, iters:1});
  const dstOfSrc = dstOfSrcFromPerm(perm, sidelen);

  postMessage({type:"done", dstOfSrc, seedRgb: src.rgb3}, [dstOfSrc.buffer, src.rgb3.buffer]);
};
