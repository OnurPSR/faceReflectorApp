import { clamp } from "./util.js";

export class SimParams{
  constructor(){
    this.kDst = 0.020;
    this.damp = 0.97;
    this.maxV = 2.0;
    this.repelRadius = 0.95;
    this.repelStrength = 0.06;
    this.alignStrength = 0.03;
    this.dt = 1.0;
  }
}

/**
 * One sim step (pos/vel/dst are Float32Arrays length N*2 in (y,x) coords)
 * Uses grid buckets (linked list) for neighbor queries.
 */
export function simStep(pos, vel, dst, sidelen, p){
  const H = sidelen, W = sidelen;
  const N = (pos.length / 2) | 0;

  // Acceleration
  const acc = new Float32Array(N*2);

  // attraction toward destination: acc += k * d * ||d|| / sidelen
  for(let i=0;i<N;i++){
    const dy = dst[2*i+0] - pos[2*i+0];
    const dx = dst[2*i+1] - pos[2*i+1];
    const dist = Math.sqrt(dy*dy + dx*dx) + 1e-6;
    const s = (p.kDst * dist) / sidelen;
    acc[2*i+0] = dy * s;
    acc[2*i+1] = dx * s;
  }

  // Spatial hashing buckets (linked list)
  const heads = new Int32Array(H*W);
  heads.fill(-1);
  const next = new Int32Array(N);

  const cellY = new Int32Array(N);
  const cellX = new Int32Array(N);

  for(let i=0;i<N;i++){
    const y = clamp(pos[2*i+0] | 0, 0, H-1);
    const x = clamp(pos[2*i+1] | 0, 0, W-1);
    cellY[i]=y; cellX[i]=x;
    const c = y*W + x;
    next[i] = heads[c];
    heads[c] = i;
  }

  // neighbors in 3x3 cells
  for(let i=0;i<N;i++){
    const cy = cellY[i], cx = cellX[i];
    let vSumY=0, vSumX=0, wSum=0;

    for(let ny = Math.max(0, cy-1); ny <= Math.min(H-1, cy+1); ny++){
      for(let nx = Math.max(0, cx-1); nx <= Math.min(W-1, cx+1); nx++){
        let j = heads[ny*W + nx];
        while(j !== -1){
          if(j !== i){
            const dpy = pos[2*i+0] - pos[2*j+0];
            const dpx = pos[2*i+1] - pos[2*j+1];
            const d2 = dpy*dpy + dpx*dpx + 1e-6;
            const d = Math.sqrt(d2);

            // repulsion
            if(d < p.repelRadius){
              const wj = (p.repelRadius - d) / p.repelRadius;
              acc[2*i+0] += (dpy / d) * (p.repelStrength * wj);
              acc[2*i+1] += (dpx / d) * (p.repelStrength * wj);
            }

            // velocity alignment (soft)
            const wv = 1.0 / (1.0 + d2);
            vSumY += vel[2*j+0] * wv;
            vSumX += vel[2*j+1] * wv;
            wSum  += wv;
          }
          j = next[j];
        }
      }
    }

    if(wSum > 0){
      const vAvgY = vSumY / wSum;
      const vAvgX = vSumX / wSum;
      acc[2*i+0] += (vAvgY - vel[2*i+0]) * p.alignStrength;
      acc[2*i+1] += (vAvgX - vel[2*i+1]) * p.alignStrength;
    }
  }

  // integrate
  for(let i=0;i<N;i++){
    vel[2*i+0] = (vel[2*i+0] + acc[2*i+0]*p.dt) * p.damp;
    vel[2*i+1] = (vel[2*i+1] + acc[2*i+1]*p.dt) * p.damp;

    const sp = Math.sqrt(vel[2*i+0]*vel[2*i+0] + vel[2*i+1]*vel[2*i+1]) + 1e-8;
    if(sp > p.maxV){
      const s = p.maxV / sp;
      vel[2*i+0] *= s;
      vel[2*i+1] *= s;
    }

    pos[2*i+0] = clamp(pos[2*i+0] + vel[2*i+0]*p.dt, 0, H-1);
    pos[2*i+1] = clamp(pos[2*i+1] + vel[2*i+1]*p.dt, 0, W-1);
  }
}
