import { rgb01ToLab } from "./color.js";
import { importanceFromEdges } from "./edges.js";
import { initialPermSort, refinePermSwaps, destForEachSrcFromPerm } from "./assignment.js";
import { SimParams, simStep } from "./sim.js";
import { renderFrameCPU } from "./voronoi_cpu.js";

/**
 * Full pipeline:
 * - srcRgb, tgtRgb: Float32Array RGB in [0,1], length N*3
 * - returns async generator yielding {frameRGBA, progress}
 */
export async function* runPipeline({
  srcRgb,
  tgtRgb,
  sidelen,
  frames,
  iters,
  proximity,
  edgeAlpha,
  fps,
  abortSignal
}){
  const H=sidelen, W=sidelen;
  const N = H*W;

  // Seed colors: fixed from source pixels
  const seedRgb = srcRgb.slice(); // Float32Array copy

  // Seed initial positions: grid (y,x)
  const pos = new Float32Array(N*2);
  const vel = new Float32Array(N*2);
  for(let i=0;i<N;i++){
    const y = (i / W) | 0;
    const x = i - y*W;
    pos[2*i+0] = y;
    pos[2*i+1] = x;
  }

  // --- Assignment stage ---
  yield { phase: "lab", progress: 0.02 };

  const srcLab = rgb01ToLab(srcRgb);
  const tgtLab = rgb01ToLab(tgtRgb);

  yield { phase: "edges", progress: 0.06 };

  const w = importanceFromEdges(tgtRgb, sidelen, edgeAlpha);

  yield { phase: "init_perm", progress: 0.10 };

  let perm = initialPermSort(srcLab, tgtLab);

  yield { phase: "refine_perm", progress: 0.14 };

  // Refine swaps (expensive) – chunk progress updates
  const chunks = 8;
  const itPer = Math.max(1, Math.floor(iters / chunks));
  for(let c=0;c<chunks;c++){
    if(abortSignal?.aborted) return;
    perm = refinePermSwaps({
      perm,
      srcLab,
      tgtLab,
      w,
      proximity,
      iters: (c===chunks-1) ? (iters - itPer*(chunks-1)) : itPer,
      sidelen,
      seed: 1234 + c,
      anneal: true
    });
    yield { phase: "refine_perm", progress: 0.14 + 0.26*(c+1)/chunks };
    // Give UI a breath
    await new Promise(r => setTimeout(r, 0));
  }

  const dstOfSrc = destForEachSrcFromPerm(perm, sidelen);

  // --- Sim + render ---
  const sim = new SimParams();
  for(let t=0;t<frames;t++){
    if(abortSignal?.aborted) return;

    const frameRGBA = renderFrameCPU(pos, seedRgb, sidelen);
    const prog = 0.40 + 0.60*(t+1)/frames;
    yield { phase: "render", frameRGBA, sidelen, t, frames, progress: prog };

    simStep(pos, vel, dstOfSrc, sidelen, sim);

    // yield to UI
    if((t % 2) === 0) await new Promise(r => setTimeout(r, 0));
  }
}
