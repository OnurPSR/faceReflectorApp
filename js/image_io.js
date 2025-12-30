import { clamp } from "./util.js";

export async function loadImageFromFile(file){
  const url = URL.createObjectURL(file);
  try{
    const img = await loadImageFromUrl(url);
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export function loadImageFromUrl(url){
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = (e) => reject(e);
    img.crossOrigin = "anonymous";
    img.src = url;
  });
}

export function canvasCropResizeToRGB01(img, sidelen){
  // Center square crop + resize to sidelen×sidelen, return Float32Array RGB in [0,1] (length N*3)
  const s = Math.min(img.naturalWidth || img.width, img.naturalHeight || img.height);
  const sx = Math.floor(((img.naturalWidth || img.width) - s) / 2);
  const sy = Math.floor(((img.naturalHeight || img.height) - s) / 2);

  const c = document.createElement("canvas");
  c.width = sidelen; c.height = sidelen;
  const ctx = c.getContext("2d", { willReadFrequently: true });

  ctx.drawImage(img, sx, sy, s, s, 0, 0, sidelen, sidelen);
  const im = ctx.getImageData(0, 0, sidelen, sidelen).data;

  const N = sidelen * sidelen;
  const rgb = new Float32Array(N * 3);
  for(let i=0;i<N;i++){
    rgb[3*i+0] = im[4*i+0] / 255;
    rgb[3*i+1] = im[4*i+1] / 255;
    rgb[3*i+2] = im[4*i+2] / 255;
  }
  return rgb;
}

export function rgb01ToImageData(rgb, sidelen){
  const N = sidelen * sidelen;
  const out = new Uint8ClampedArray(N * 4);
  for(let i=0;i<N;i++){
    out[4*i+0] = clamp(Math.round(rgb[3*i+0] * 255), 0, 255);
    out[4*i+1] = clamp(Math.round(rgb[3*i+1] * 255), 0, 255);
    out[4*i+2] = clamp(Math.round(rgb[3*i+2] * 255), 0, 255);
    out[4*i+3] = 255;
  }
  return new ImageData(out, sidelen, sidelen);
}
