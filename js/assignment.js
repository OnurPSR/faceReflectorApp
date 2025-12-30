import { mulberry32, clamp } from "./util.js";

/**
 * Initial permutation by sorting a 1D key in Lab.
 * perm[dst] = src
 */
export function initialPermSort(srcLab, tgtLab){
  const N = srcLab.length / 3;
  const sKey = new Float32Array(N);
  const tKey = new Float32Array(N);

  for(let i=0;i<N;i++){
    const sL=srcLab[3*i+0], sa=srcLab[3*i+1], sb=srcLab[3*i+2];
    const tL=tgtLab[3*i+0], ta=tgtLab[3*i+1], tb=tgtLab[3*i+2];
    sKey[i] = 0.70*sL + 0.15*sa + 0.15*sb;
    tKey[i] = 0.70*tL + 0.15*ta + 0.15*tb;
  }

  const sIdx = Array.from({length:N}, (_,i)=>i);
  const tIdx = Array.from({length:N}, (_,i)=>i);

  sIdx.sort((i,j)=>sKey[i]-sKey[j]);
  tIdx.sort((i,j)=>tKey[i]-tKey[j]);

  const perm = new Int32Array(N);
  for(let k=0;k<N;k++){
    perm[tIdx[k]] = sIdx[k];
  }
  return perm;
}

function costAt(dst, src, tgtLab, srcLab, w, dstY, dstX, srcY, srcX, proximity){
  const dL = tgtLab[3*dst+0] - srcLab[3*src+0];
  const da = tgtLab[3*dst+1] - srcLab[3*src+1];
  const db = tgtLab[3*dst+2] - srcLab[3*src+2];
  const color = (dL*dL + da*da + db*db) * w[dst];

  const dy = dstY[dst] - srcY[src];
  const dx = dstX[dst] - srcX[src];
  const dist2 = dy*dy + dx*dx;
  let prox = proximity * dist2;
  prox = prox * prox; // heavy penalty on large moves
  return color + prox;
}

/**
 * Swap-based local search (hill climb + optional annealing).
 * perm[dst] = src
 */
export function refinePermSwaps({
  perm,
  srcLab,
  tgtLab,
  w,
  proximity = 0.025,
  iters = 160000,
  sidelen,
  seed = 0,
  anneal = true
}){
  const H = sidelen, W = sidelen;
  const N = H*W;

  const dstY = new Int32Array(N);
  const dstX = new Int32Array(N);
  const srcY = new Int32Array(N);
  const srcX = new Int32Array(N);

  for(let i=0;i<N;i++){
    const y = (i / W) | 0;
    const x = i - y*W;
    dstY[i]=y; dstX[i]=x;
    srcY[i]=y; srcX[i]=x; // source indices live on original grid
  }

  const curCost = new Float64Array(N);
  for(let j=0;j<N;j++){
    curCost[j] = costAt(j, perm[j], tgtLab, srcLab, w, dstY, dstX, srcY, srcX, proximity);
  }

  const rng = mulberry32(seed);
  let radius = Math.max(4, (H/2)|0);

  const stageIters = Math.max(10000, Math.floor(iters / (Math.log2(radius)+1)));

  const T0 = 1.0, T1 = 0.02;

  for(let it=0; it<iters; it++){
    if(it>0 && (it % stageIters === 0) && radius > 1){
      radius = Math.max(1, (radius/2)|0);
    }

    const a = (rng()*N) | 0;
    const ay = dstY[a], ax = dstX[a];
    const by = clamp(ay + (((rng()*(2*radius+1))|0) - radius), 0, H-1);
    const bx = clamp(ax + (((rng()*(2*radius+1))|0) - radius), 0, W-1);
    const b = by*W + bx;
    if(a===b) continue;

    const srcA = perm[a];
    const srcB = perm[b];

    const newA = costAt(a, srcB, tgtLab, srcLab, w, dstY, dstX, srcY, srcX, proximity);
    const newB = costAt(b, srcA, tgtLab, srcLab, w, dstY, dstX, srcY, srcX, proximity);

    const old = curCost[a] + curCost[b];
    const neu = newA + newB;
    const delta = neu - old;

    let accept = delta < 0;
    if(!accept && anneal){
      const t = it / Math.max(1, iters-1);
      const T = (1-t)*T0 + t*T1;
      if(delta > 0){
        const p = Math.exp(-delta / Math.max(1e-9, T));
        if(rng() < p) accept = true;
      }
    }

    if(accept){
      perm[a] = srcB;
      perm[b] = srcA;
      curCost[a] = newA;
      curCost[b] = newB;
    }
  }

  return perm;
}

/**
 * perm[dst]=src -> dst_of_src[src] = (y,x) float
 */
export function destForEachSrcFromPerm(perm, sidelen){
  const H = sidelen, W = sidelen;
  const N = H*W;
  const inv = new Int32Array(N);
  for(let dst=0; dst<N; dst++){
    inv[perm[dst]] = dst;
  }
  const dstOfSrc = new Float32Array(N * 2);
  for(let src=0; src<N; src++){
    const dst = inv[src];
    const y = (dst / W) | 0;
    const x = dst - y*W;
    dstOfSrc[2*src+0] = y;
    dstOfSrc[2*src+1] = x;
  }
  return dstOfSrc;
}
