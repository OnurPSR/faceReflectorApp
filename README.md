# Pixel Flow Mosaic (Selfie → Stock) — GitHub Pages

This is a **pure static** web app designed to run on **GitHub Pages** (no backend).
It implements the same high-level pipeline as your Python modular version:

1) preprocess selfie + target to a square `sidelen×sidelen`  
2) compute a **permutation** `perm[dst]=src` minimizing:
   - Lab color distance (weighted by target edges)
   - + spatial proximity penalty (discourages huge moves)
3) invert to `dst_of_src[src]=(y,x)`  
4) simulate seeds moving toward their destinations  
5) render each frame with **Jump Flood Algorithm (JFA)** Voronoi mosaic

All computation happens locally in the browser.

---

## Files

- `index.html` — UI
- `js/worker.js` — heavy assignment stage (runs in a Web Worker)
- `js/app.js` — simulation + JFA render + recording
- `assets/stock.jpg` — hidden default target (replace with your stock image)

---

## Run locally

Because this uses ES modules + a worker, you need a local server:

```bash
python -m http.server 8000
# open http://localhost:8000
```

---

## Deploy to GitHub Pages

1) Create a repo and push this folder to `main`.
2) In GitHub: **Settings → Pages**
3) Source: **Deploy from a branch**
4) Branch: `main` and folder `/ (root)`
5) Save → your site will appear at `https://<username>.github.io/<repo>/`

---

## Tips

- Start with `sidelen=64` and `iters=20000–60000`.
- Increase `iters` to improve resemblance (slower).
- `edgeAlpha` boosts structure (eyes/mouth edges, contours).

## Replacing the target (stock) image

The target image is **not shown in the UI**. The app always uses:

- `assets/stock.jpg`

Replace that file in the repo with your own stock image (keep the same filename), commit, and push.
