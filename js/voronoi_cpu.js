/**
 * Jump Flood Algorithm (CPU) to assign each pixel the id of the nearest seed.
 * seedPos: Float32Array length N*2 in (y,x) coords
 * returns Int32Array ids length H*W
 */
export function jfaVoronoiCPU(seedPos, sidelen){
  const H = sidelen, W = sidelen;
  const N = (seedPos.length/2)|0;
  const HW = H*W;

  const bestId = new Int32Array(HW);
  const bestD2 = new Float32Array(HW);
  bestId.fill(-1);
  bestD2.fill(Infinity);

  // stamp initial seeds at rounded positions
  for(let i=0;i<N;i++){
    const y = Math.max(0, Math.min(H-1, Math.round(seedPos[2*i+0])));
    const x = Math.max(0, Math.min(W-1, Math.round(seedPos[2*i+1])));
    const idx = y*W + x;
    bestId[idx] = i;
    const dy = y - seedPos[2*i+0];
    const dx = x - seedPos[2*i+1];
    bestD2[idx] = dy*dy + dx*dx;
  }

  // largest power of 2 <= max(H,W)
  let step = 1;
  while(step < Math.max(H,W)) step <<= 1;
  step >>= 1;

  // helper
  function consider(y, x, candId, outIdx){
    const cy = seedPos[2*candId+0];
    const cx = seedPos[2*candId+1];
    const dy = y - cy;
    const dx = x - cx;
    const d2 = dy*dy + dx*dx;
    if(d2 < bestD2[outIdx]){
      bestD2[outIdx] = d2;
      bestId[outIdx] = candId;
    }
  }

  while(step >= 1){
    for(let y=0;y<H;y++){
      for(let x=0;x<W;x++){
        const idx = y*W + x;

        // try 9 offsets
        for(let oy=-step; oy<=step; oy+=step){
          for(let ox=-step; ox<=step; ox+=step){
            const ny = y - oy;
            const nx = x - ox;
            if(ny<0 || ny>=H || nx<0 || nx>=W) continue;
            const nidx = ny*W + nx;
            const cand = bestId[nidx];
            if(cand >= 0) consider(y, x, cand, idx);
          }
        }
      }
    }
    step >>= 1;
  }

  return bestId;
}

/**
 * Render a frame given seed positions and seed colors (from selfie pixels).
 * seedRgb: Float32Array length N*3 in [0,1]
 * returns Uint8ClampedArray length H*W*4 (RGBA)
 */
export function renderFrameCPU(seedPos, seedRgb, sidelen){
  const H = sidelen, W = sidelen;
  const HW = H*W;
  const ids = jfaVoronoiCPU(seedPos, sidelen);

  const out = new Uint8ClampedArray(HW*4);
  for(let i=0;i<HW;i++){
    const id = ids[i];
    if(id < 0){
      out[4*i+0] = 0; out[4*i+1] = 0; out[4*i+2] = 0; out[4*i+3] = 255;
      continue;
    }
    out[4*i+0] = Math.max(0, Math.min(255, Math.round(seedRgb[3*id+0]*255)));
    out[4*i+1] = Math.max(0, Math.min(255, Math.round(seedRgb[3*id+1]*255)));
    out[4*i+2] = Math.max(0, Math.min(255, Math.round(seedRgb[3*id+2]*255)));
    out[4*i+3] = 255;
  }
  return out;
}
