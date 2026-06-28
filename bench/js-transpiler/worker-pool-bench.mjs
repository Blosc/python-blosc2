// Throwaway: how fast can the transpiled Newton kernel go with real JS multithreading?
// Pure Node (worker_threads + SharedArrayBuffer), no Pyodide, no blosc2 -- isolates the
// compute ceiling and per-dispatch overhead a Web Worker pool would hit in a browser.
// The kernel is the same scalar loop dsl_js emits; hand-written here to avoid Pyodide.
//
//   node bench/js-transpiler/worker-pool-bench.mjs
import { Worker, isMainThread, workerData } from "node:worker_threads";
import os from "node:os";
import { performance } from "node:perf_hooks";

const WIDTH = 320, HEIGHT = 213, MAXITER = 48, NFRAMES = 24, SPANX = 3.4;
const ASPECT = HEIGHT / WIDTH;
const N = WIDTH * HEIGHT;

// Striped rows (rowStart, rowStep): worker i does rows i, i+nw, ... so the per-pixel
// early-exit work spreads evenly across workers instead of clumping in contiguous bands.
function newtonBand(A, B, OUT, rowStart, rowStep, H, W, maxIter, relax) {
  for (let row = rowStart; row < H; row += rowStep) {
    for (let col = 0; col < W; col++) {
      const i = row * W + col;
      let za = A[i], zb = B[i], it = maxIter;
      for (let k = 0; k < maxIter; k++) {
        const a2 = za * za, b2 = zb * zb;
        const fr = za * a2 - 3 * za * b2 - 1, fi = 3 * a2 * zb - zb * b2;
        const dr = 3 * (a2 - b2), di = 6 * za * zb, den = dr * dr + di * di + 1e-12;
        const qr = relax * (fr * dr + fi * di) / den, qi = relax * (fi * dr - fr * di) / den;
        za -= qr; zb -= qi;
        if (qr * qr + qi * qi < 1e-6) { it = k; break; }
      }
      const d0 = (za - 1) * (za - 1) + zb * zb;
      const d1 = (za + 0.5) * (za + 0.5) + (zb - 0.8660254) * (zb - 0.8660254);
      const d2 = (za + 0.5) * (za + 0.5) + (zb + 0.8660254) * (zb + 0.8660254);
      let root = 0, md = d0;
      if (d1 < md) { md = d1; root = 1; }
      if (d2 < md) { root = 2; }
      OUT[i] = root + 0.9 * (it / maxIter);
    }
  }
}

// ctrl: Int32[ gen, done ].  params: Float64[ relax, maxIter ].
if (!isMainThread) {
  const { ctrlSab, paramsSab, aSab, bSab, outSab, rowStart, rowStep, W } = workerData;
  const ctrl = new Int32Array(ctrlSab), params = new Float64Array(paramsSab);
  const A = new Float64Array(aSab), B = new Float64Array(bSab), OUT = new Float64Array(outSab);
  let gen = 0;
  for (;;) {
    Atomics.wait(ctrl, 0, gen);          // block until main bumps the generation
    gen = Atomics.load(ctrl, 0);
    if (gen < 0) break;                  // shutdown
    newtonBand(A, B, OUT, rowStart, rowStep, HEIGHT, W, params[1] | 0, params[0]);
    Atomics.add(ctrl, 1, 1);             // signal this band done
    Atomics.notify(ctrl, 1);
  }
} else {
  main();
}

function buildGrid() {
  const aSab = new SharedArrayBuffer(N * 8), bSab = new SharedArrayBuffer(N * 8),
        outSab = new SharedArrayBuffer(N * 8);
  const A = new Float64Array(aSab), B = new Float64Array(bSab);
  const x0 = -SPANX / 2, dx = SPANX / (WIDTH - 1);
  const y0 = -SPANX * ASPECT / 2, dy = SPANX * ASPECT / (HEIGHT - 1);
  for (let r = 0; r < HEIGHT; r++)
    for (let c = 0; c < WIDTH; c++) { A[r * WIDTH + c] = x0 + dx * c; B[r * WIDTH + c] = y0 + dy * r; }
  return { aSab, bSab, outSab };
}

function timeBest(fn, runs) {
  for (let w = 0; w < 2; w++) fn();      // warm V8 / workers
  let best = Infinity;
  for (let r = 0; r < runs; r++) { const t = performance.now(); fn(); best = Math.min(best, performance.now() - t); }
  return best;
}

async function benchPool(nw, sabs, relaxes) {
  const ctrlSab = new SharedArrayBuffer(8), paramsSab = new SharedArrayBuffer(16);
  const ctrl = new Int32Array(ctrlSab), params = new Float64Array(paramsSab);
  params[1] = MAXITER;
  const workers = [];
  for (let i = 0; i < nw; i++) {
    workers.push(new Worker(new URL(import.meta.url), {
      workerData: { ...sabs, ctrlSab, paramsSab, rowStart: i, rowStep: nw, W: WIDTH },
    }));
  }
  const frame = (relax) => {
    Atomics.store(ctrl, 1, 0);
    params[0] = relax;
    Atomics.add(ctrl, 0, 1);
    Atomics.notify(ctrl, 0, nw);
    let d;                               // barrier: wait until all bands reported done
    while ((d = Atomics.load(ctrl, 1)) < nw) Atomics.wait(ctrl, 1, d);
  };
  const sweep = () => { for (const rx of relaxes) frame(rx); };
  const best = timeBest(sweep, 5);
  Atomics.store(ctrl, 0, -1); Atomics.notify(ctrl, 0, nw);   // shutdown
  await Promise.all(workers.map((w) => w.terminate()));
  return best;
}

async function main() {
  const cores = os.cpus().length;
  const sabs = buildGrid();
  const OUT = new Float64Array(sabs.outSab);
  const relaxes = Array.from({ length: NFRAMES }, (_, i) => 1.0 + (1.85 - 1.0) * i / (NFRAMES - 1));

  // Single-thread baseline on the main thread (no worker overhead at all).
  const A = new Float64Array(sabs.aSab), B = new Float64Array(sabs.bSab);
  const tSingle = timeBest(() => { for (const rx of relaxes) newtonBand(A, B, OUT, 0, 1, HEIGHT, WIDTH, MAXITER, rx); }, 5);
  const ref = Float64Array.from(OUT);    // last frame (relax=1.85), for correctness check

  console.log(`Newton ${WIDTH}x${HEIGHT}, max_iter=${MAXITER}, ${NFRAMES}-frame sweep | cores=${cores}`);
  const per = (ms) => `${ms.toFixed(0)} ms total (${(ms / NFRAMES).toFixed(2)} ms/frame)`;
  console.log(`\nsingle-thread (main): ${per(tSingle)}`);

  const counts = [...new Set([1, 2, 4, cores])].filter((n) => n >= 1 && n <= cores * 2).sort((a, b) => a - b);
  for (const nw of counts) {
    const t = await benchPool(nw, sabs, relaxes);
    let maxdiff = 0;
    for (let i = 0; i < N; i++) maxdiff = Math.max(maxdiff, Math.abs(OUT[i] - ref[i]));
    const sp = tSingle / t;
    console.log(`pool x${String(nw).padStart(2)} : ${per(t)}  | speedup ${sp.toFixed(2)}x` +
                `  eff ${(100 * sp / nw).toFixed(0)}%  | maxdiff ${maxdiff.toExponential(1)}`);
  }
}
