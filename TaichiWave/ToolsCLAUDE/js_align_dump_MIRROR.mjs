// Runs the web app's REAL alignment code (through js_harness_MIRROR.mjs) for each config in a
// JSON list and writes, one JSON line per config, everything the port's alignment_MIRROR.py must
// reproduce: the pure functions, alEdges / alMeasure / alRep, the panel NOTE and snap list (HTML),
// the two ledger rows, and the outcome of alStep(+1), alStep(-1), alOptimal and alFlavor.
//   node js_align_dump_MIRROR.mjs <configs.json> <out.jsonl> [path to wave_membrane_MIRROR.html]
//
// The harness exports only some names; the rest (alEdgeReport, alRefresh, alFlavorPick, ...) are
// reached through the sandbox's own Function constructor: a function built by it runs in the
// sandbox's global scope, which holds the app script's top-level const/let/function bindings.
//
// alStep / alOptimal / alFlavor move the rotation slider, whose handler calls reseed(). Their
// outcome (S.rot, S.lattice, S.Co, the flash text) depends only on geometry (shapePoly,
// clipRegion, alList -- no MASK read), so reseed is stubbed while they run: 4 ops x 2 classes
// x up to 2 reseeds per config would cost ~2 s per config at N = 257 (measured ~150 ms/reseed).
// The Co clamp of the lattice control calls buildDomain directly; the next config reseeds.
import fs from 'fs';
const [cfgPath, outPath, app] = process.argv.slice(2);
process.argv[2] = app;                           // the harness reads the app path from argv[2]
const { T, $ } = await import('./js_harness_MIRROR.mjs');
const G = code => new (T.reseed.constructor)(code)();
const F = G(`return { alEdgeReport, alRep, alNothingLeft, alFlavorPick, alBestOf, alRefresh, alDirs,
  alRotDeg, alLat, alMeasure, alEdges, alList, alCount, alSymPeriod, alCanon, alStep, alOptimal, alFlavor,
  fitsInside, shapePoly, clipRegion, clipConvex, alClass, alNearDir, alLatticeDir, alAngles, alGcdF,
  alEdgeDirs0, alBest, alVectors, polyArea, reseed }`);
G(`flashOvl = function(m){ (globalThis.__FL = globalThis.__FL || []).push(String(m)); }`);
G(`globalThis.__realReseedRef = reseed`);
const S = T.S;
const fin = (k, v) => (typeof v === 'number' && !Number.isFinite(v)) ? String(v) : v;
const cfgs = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
const out = fs.openSync(outPath, 'w');

// ---- pure functions on their own ------------------------------------------------------------
if (cfgs.pure) {
  const P = cfgs.pure, res = { pure: true, cls: [], near: [], angles: [], gcd: [], canon: [], ldir: [] };
  for (const lat of ['sq', 'tri']) for (const phi of P.phis) {
    res.cls.push(F.alClass(phi, lat)); res.ldir.push(F.alLatticeDir(phi, lat, 24));
    for (const len of P.lens) res.near.push(F.alNearDir(phi, lat, len));
  }
  for (const [dirs, lat, lo, hi] of P.angles) res.angles.push(F.alAngles(dirs, lat, lo, hi));
  for (const [a, b] of P.gcd) res.gcd.push(F.alGcdF(a, b));
  for (const [r, g] of P.canon) res.canon.push(F.alCanon(r, g));
  fs.writeSync(out, JSON.stringify(res, fin) + '\n');
}

let t0 = Date.now(), done = 0;
for (const c of cfgs.configs) {
  for (const [k, v] of Object.entries(c.p)) if (k !== 'N') S[k] = v;
  if (T.N !== c.p.N) T.alloc(c.p.N);
  S.edgeCol = !!c.edgeCol; S.alCls = 'row';
  T.reseed();
  const led = [...$('led').innerHTML.matchAll(/<tr><td>([\s\S]*?)<\/td><td class="s">([\s\S]*?)<\/td><td>([\s\S]*?)<\/td><td class="r">([\s\S]*?)<\/td><\/tr>/g)]
    .filter(r => r[1].startsWith('edge alignment')).map(r => [r[1], r[2], r[3], r[4]]);
  const lat = F.alLat(), d = F.alDirs(), rot = F.alRotDeg(), g = F.alSymPeriod(S.shape, lat);
  const r = { name: c.name, rotDeg: rot, lat, dirs: d, count: F.alCount(d, rot, lat), symPeriod: g,
    edges: F.alEdges(), meas: F.alMeasure().edges, rep: F.alRep().edges, nothingLeft: F.alNothingLeft(),
    fits: F.fitsInside(F.shapePoly()), ledger: led, cls: {} };
  for (const cls of ['row', 'smooth']) {
    S.alCls = cls; F.alRefresh();
    const L = F.alList();
    const q = { list: L, canon: L.map(e => F.alCanon(e.rot, g)), bestSq: F.alBestOf('sq'), bestTri: F.alBestOf('tri'),
      pick: F.alFlavorPick(), note: $('alNote').innerHTML, snap: $('alSnap').innerHTML, ops: {} };
    G(`reseed = function(){}`);
    for (const op of ['next', 'prev', 'opt', 'flav']) {
      const save = { rot: S.rot, lattice: S.lattice, Co: S.Co };
      G(`globalThis.__FL = []`);
      if (op === 'next') F.alStep(1); else if (op === 'prev') F.alStep(-1);
      else if (op === 'opt') F.alOptimal(); else F.alFlavor();
      q.ops[op] = { rot: S.rot, lattice: S.lattice, Co: S.Co, flash: G(`return globalThis.__FL`).slice() };
      Object.assign(S, save);
    }
    G(`reseed = globalThis.__realReseedRef`);
    // the lattice control's Co clamp (alFlavor off an unstable Co) rebuilt the mask for the OTHER
    // lattice through buildDomain; rebuild it for the restored state before the next class reads it
    if (Object.values(q.ops).some(o => o.Co !== S.Co)) T.reseed();
    q.snapCur = F.alRotDeg();
    r.cls[cls] = q;
  }
  fs.writeSync(out, JSON.stringify(r, fin) + '\n');
  if (++done % 100 === 0) console.log(`${done}/${cfgs.configs.length}  ${((Date.now() - t0) / done).toFixed(0)} ms/config`);
}
fs.closeSync(out);
console.log(`done: ${done} configs, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
