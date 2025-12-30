export function clamp(v, lo, hi){ return Math.max(lo, Math.min(hi, v)); }

export function mulberry32(seed){
  let t = seed >>> 0;
  return function(){
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

export function sleep(ms){ return new Promise(r => setTimeout(r, ms)); }

export function percent(x){ return `${(100*x).toFixed(1)}%`; }
