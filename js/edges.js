/**
 * Sobel edge magnitude -> importance weight: w = 1 + alpha*mag_norm
 * Input rgb: Float32Array length N*3 in [0,1]
 * Output w: Float32Array length N
 */
export function importanceFromEdges(rgb, sidelen, alpha){
  const H = sidelen, W = sidelen;
  const N = H*W;
  const gray = new Float32Array(N);

  // luminance
  for(let i=0;i<N;i++){
    gray[i] = 0.2126*rgb[3*i+0] + 0.7152*rgb[3*i+1] + 0.0722*rgb[3*i+2];
  }

  // Sobel kernels
  const Kx = [ 1,0,-1,  2,0,-2,  1,0,-1 ];
  const Ky = [ 1,2, 1,  0,0, 0, -1,-2,-1 ];

  // pad access helper (edge clamp)
  function at(y,x){
    y = (y<0)?0:(y>=H?H-1:y);
    x = (x<0)?0:(x>=W?W-1:x);
    return gray[y*W + x];
  }

  const mag = new Float32Array(N);
  let maxv = 0.0;

  for(let y=0;y<H;y++){
    for(let x=0;x<W;x++){
      let gx=0, gy=0;
      let k=0;
      for(let dy=-1;dy<=1;dy++){
        for(let dx=-1;dx<=1;dx++){
          const v = at(y+dy, x+dx);
          gx += Kx[k]*v;
          gy += Ky[k]*v;
          k++;
        }
      }
      const m = Math.sqrt(gx*gx + gy*gy);
      mag[y*W + x] = m;
      if(m>maxv) maxv = m;
    }
  }

  const w = new Float32Array(N);
  const inv = 1.0 / (maxv + 1e-6);
  for(let i=0;i<N;i++){
    const mn = mag[i] * inv;
    w[i] = 1.0 + alpha * mn;
  }
  return w;
}
