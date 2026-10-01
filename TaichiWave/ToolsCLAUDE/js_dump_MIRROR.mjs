// Runs the web app's REAL solver (through js_harness_MIRROR.mjs, with its documented PATCHES
// unless ZRL_JS_NOPATCH=1) for each config in a JSON list and writes what the Taichi port must
// reproduce: mask, per-cell Co^2, the seeded field, the field after K steps, the source cells,
// and record()'s histories (ENERGY, PROBE, PROBE_B, MODEP) with S.amax.
//   node js_dump_MIRROR.mjs <configs.json> <outdir> [path to wave_membrane_MIRROR.html]
import fs from 'fs';
import path from 'path';
const [cfgPath, outDir, app] = process.argv.slice(2);
process.argv[2] = app;                           // the harness reads the app path from argv[2]
                                                 // (undefined -> its default, the canonical app)
const { T, change, PATCHED } = await import('./js_harness_MIRROR.mjs');
const cfgs = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
fs.mkdirSync(outDir, { recursive: true });
const S = T.S;
const put = (name, arr) => fs.writeFileSync(path.join(outDir, name), Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength));
for (const c of cfgs) {
  // fields straight into S (angles already in radians), then the handlers that also
  // rebuild module state the S object does not carry (HARM)
  for (const [k, v] of Object.entries(c.p)) if (k !== 'N') S[k] = v;
  change('wave', c.p.wave || 'sine');
  if (T.N !== c.p.N) T.alloc(c.p.N);
  T.reseed();
  put(`${c.name}_mask.bin`, T.MASK);
  put(`${c.name}_cc.bin`, T.CC);
  put(`${c.name}_u0.bin`, Float64Array.from(T.U1));
  const t0 = process.hrtime.bigint();
  for (let s = 0; s < c.steps; s++) T.stepOnce();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  put(`${c.name}_uK.bin`, Float64Array.from(T.U1));
  // record()'s histories: the arrays the ledger reads (capped at 4000 / 600 entries, newest kept)
  put(`${c.name}_hE.bin`, Float64Array.from(T.ENERGY));
  put(`${c.name}_hA.bin`, Float64Array.from(T.PROBE));
  put(`${c.name}_hB.bin`, Float64Array.from(T.PROBE_B));
  put(`${c.name}_hM.bin`, Float64Array.from(T.MODEP));
  const meta = { N: T.N, domR: T.domR(), innerR: T.innerR(), vtx: T.VTX.map(v => [v.i, v.j]),
    clipFrac: S.clipFrac, amax: S.amax, step: S.step, t: S.t, ms_per_step: ms / Math.max(1, c.steps),
    pA: T.pA(), pB: T.pB(), antinode: T.modeAntinode(), patches: PATCHED,
    nE: T.ENERGY.length, nA: T.PROBE.length, nB: T.PROBE_B.length, nM: T.MODEP.length };
  fs.writeFileSync(path.join(outDir, `${c.name}.json`), JSON.stringify(meta));
  console.log(`${c.name}: N=${T.N} steps=${c.steps} ${meta.ms_per_step.toFixed(3)} ms/step`);
}
