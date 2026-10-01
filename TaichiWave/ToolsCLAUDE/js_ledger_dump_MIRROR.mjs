// Runs the web app's REAL ledger (buildLedger, through js_harness_MIRROR.mjs WITH its documented
// PATCHES unless ZRL_JS_NOPATCH=1) for each config in a JSON list and writes, one JSON line per
// (config, checkpoint), the ledger rows exactly as the page shows them -- the four cells of every
// <tr> that buildLedger writes into #led, still as HTML -- plus the module state the rows read
// (step, S.amax, S.drops, CLEAR, CONE_WALL, SRC, the history lengths), so a mismatch can be traced.
//   node js_ledger_dump_MIRROR.mjs <configs.json> <out.jsonl> [path to wave_membrane_MIRROR.html]
// configs.json: [{name, p: {S keys}, checkpoints: [0, 50, 300, 1200], drops: [steps], seed_out: path|null}]
//   the ledger is built after `checkpoint` steps; a drop listed at step s runs dropPulse() after s
//   steps (before the ledger if s is also a checkpoint), the way the P key does in the page.
// Module-level names the harness does not export (dropPulse, CLEAR, CONE_WALL, SRC) are reached
// through the sandbox's own Function constructor, whose functions see the script's top-level
// bindings (the trick js_align_dump_MIRROR.mjs uses).
import fs from 'fs';
const [cfgPath, outPath, app] = process.argv.slice(2);
process.argv[2] = app;                           // the harness reads the app path from argv[2]
const { T, $, change, PATCHED } = await import('./js_harness_MIRROR.mjs');
const G = code => new (T.reseed.constructor)(code)();
const F = G(`return { buildLedger, dropPulse }`);
const S = T.S;
const fin = (k, v) => (typeof v === 'number' && !Number.isFinite(v)) ? String(v) : v;
const cfgs = JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
const out = fs.openSync(outPath, 'w');
const ROW = /<tr><td>([\s\S]*?)<\/td><td class="s">([\s\S]*?)<\/td><td>([\s\S]*?)<\/td><td class="r">([\s\S]*?)<\/td><\/tr>/g;
let t0 = Date.now(), nLed = 0;
for (const c of cfgs) {
  const tc = Date.now();
  for (const [k, v] of Object.entries(c.p)) if (k !== 'N') S[k] = v;
  change('wave', c.p.wave || 'sine');             // rebuilds HARM from S.wave / S.Co / S.freq
  if (T.N !== c.p.N) T.alloc(c.p.N);
  T.reseed();
  // the seeded field, for a port run started from the web app's own seed (c.seed_out)
  if (c.seed_out) { const u = Float64Array.from(T.U1); fs.writeFileSync(c.seed_out, Buffer.from(u.buffer)); }
  const drops = new Set(c.drops || []), done = new Set();
  const cps = [...c.checkpoints].sort((a, b) => a - b);
  let step = 0;
  const dropIfDue = () => { if (drops.has(step) && !done.has(step)) { F.dropPulse(); done.add(step); } };
  for (const cp of cps) {
    while (step < cp) { dropIfDue(); T.stepOnce(); step++; }
    dropIfDue();
    F.buildLedger();
    const rows = [...$('led').innerHTML.matchAll(ROW)].map(r => [r[1], r[2], r[3], r[4]]);
    const dbg = G(`return { step: S.step, amax: S.amax, drops: S.drops, CLEAR, CONE_WALL, SRC: [SRC.i, SRC.j],
      nE: ENERGY.length, nA: PROBE.length, nM: MODEP.length, E_last: ENERGY.length ? ENERGY[ENERGY.length-1] : null }`);
    fs.writeSync(out, JSON.stringify({ name: c.name, checkpoint: cp, rows, dbg, patches: PATCHED }, fin) + '\n');
    nLed++;
  }
  console.log(`${c.name}: N=${T.N} checkpoints ${cps.join(',')}  ${((Date.now() - tc) / 1000).toFixed(1)} s`);
}
fs.closeSync(out);
console.log(`done: ${cfgs.length} configs, ${nLed} ledgers, ${((Date.now() - t0) / 1000).toFixed(1)} s`);
