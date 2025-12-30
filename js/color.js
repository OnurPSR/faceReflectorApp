/**
 * sRGB [0,1] -> Lab (approx D65)
 * Input: Float32Array rgb length N*3
 * Output: Float32Array lab length N*3 (L in [0,100])
 */
export function rgb01ToLab(rgb){
  const N = rgb.length / 3;
  const lab = new Float32Array(N * 3);

  // sRGB -> linear
  const a = 0.055;
  // sRGB->XYZ matrix (D65)
  const m00=0.4124564, m01=0.3575761, m02=0.1804375;
  const m10=0.2126729, m11=0.7151522, m12=0.0721750;
  const m20=0.0193339, m21=0.1191920, m22=0.9503041;

  const Xn=0.95047, Yn=1.00000, Zn=1.08883;
  const eps = 216/24389;      // (6/29)^3
  const kappa = 24389/27;

  function f(t){
    return (t > eps) ? Math.cbrt(t) : (kappa*t + 16)/116;
  }

  for(let i=0;i<N;i++){
    let r = rgb[3*i+0], g = rgb[3*i+1], b = rgb[3*i+2];

    // clamp
    r = r<0?0:(r>1?1:r);
    g = g<0?0:(g>1?1:g);
    b = b<0?0:(b>1?1:b);

    // gamma remove
    r = (r <= 0.04045) ? (r/12.92) : Math.pow((r+a)/(1+a), 2.4);
    g = (g <= 0.04045) ? (g/12.92) : Math.pow((g+a)/(1+a), 2.4);
    b = (b <= 0.04045) ? (b/12.92) : Math.pow((b+a)/(1+a), 2.4);

    // linear rgb -> xyz
    const X = r*m00 + g*m01 + b*m02;
    const Y = r*m10 + g*m11 + b*m12;
    const Z = r*m20 + g*m21 + b*m22;

    const x = X / Xn;
    const y = Y / Yn;
    const z = Z / Zn;

    const fx = f(x), fy = f(y), fz = f(z);

    const L = 116*fy - 16;
    const A = 500*(fx - fy);
    const B = 200*(fy - fz);

    lab[3*i+0] = L;
    lab[3*i+1] = A;
    lab[3*i+2] = B;
  }
  return lab;
}
