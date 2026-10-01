// Times the web app's stepOnce() the way the BROWSER runs it: the app's module script is
// loaded as a real ES module (module-scope variables), not inside a node:vm context -- in a
// vm context every top-level variable access goes through an interceptor, which measured
// ~8x slower (120 ns/cell vs ~15) and would flatter any port compared against it.
//   node js_bench_MIRROR.mjs [N ...]      -> JSON lines {lat, N, ms_step, ns_cell}
import fs from 'fs'; import os from 'os'; import path from 'path'; import { pathToFileURL } from 'url';
const APP = 'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/FlavorRenderersCLAUDE/wave_membrane_MIRROR.html';
const html = fs.readFileSync(APP, 'utf8');
let src = html.match(/<script type="module">([\s\S]*?)<\/script>/)[1].replace(/^\s*import .*$/mg, '');
src += `\nexport const __T = { S, alloc, reseed, stepOnce, get N(){ return N; } };\n`;

// ---- the same DOM/THREE stubs as js_harness_MIRROR.mjs, installed on the real global ----
const selects = {}, ranges = {}, els = new Map();
for (const sm of html.matchAll(/<select id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g))
  selects[sm[1]] = [...sm[2].matchAll(/<option value="([^"]*)"[^>]*>([\s\S]*?)<\/option>/g)].map(o => ({ value: o[1], textContent: o[2], disabled: false }));
for (const rm of html.matchAll(/<input type="range" id="([^"]+)" min="([^"]+)" max="([^"]+)" step="([^"]+)" value="([^"]+)">/g))
  ranges[rm[1]] = { min: rm[2], max: rm[3], step: rm[4], value: rm[5] };
function mkEl(id) {
  const L = {};
  const el = { id, innerHTML: '', textContent: '', checked: false, style: {}, dataset: {}, tagName: 'DIV',
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    addEventListener(t, f) { (L[t] = L[t] || []).push(f); }, dispatchEvent(ev) { for (const f of (L[ev.type] || [])) f(ev); return true; },
    click() {}, getContext() { return new Proxy({}, { get: () => () => {} }); },
    clientWidth: 800, clientHeight: 560, width: 800, height: 560,
    querySelectorAll() { return []; }, querySelector() { return null; }, closest() { return null; },
    appendChild() {}, insertBefore() {}, parentNode: { querySelector() { return null; }, insertBefore() {} },
    nextElementSibling: { textContent: '' }, requestFullscreen() { return Promise.reject(new Error('no')); } };
  if (selects[id]) { el.tagName = 'SELECT'; el.options = selects[id].map(o => ({ ...o })); el.selectedIndex = 0;
    Object.defineProperty(el, 'value', { get() { return el.options[el.selectedIndex]?.value ?? ''; },
      set(v) { const i = el.options.findIndex(o => o.value === String(v)); el.selectedIndex = i < 0 ? 0 : i; } }); }
  else if (ranges[id]) { el.tagName = 'INPUT'; Object.assign(el, ranges[id]); }
  else el.value = '';
  return el;
}
const $el = id => { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); };
class Stub { constructor() { this.position = { set() {} }; this.up = { set() {} }; this.target = { set() {} };
  this.attributes = {}; this.geometry = { dispose() {}, attributes: {} }; this.material = { dispose() {} }; }
  setPixelRatio() {} getPixelRatio() { return 1; } setSize() {} render() {} add() {} remove() {} dispose() {}
  updateProjectionMatrix() {} update() {} setIndex() {} setAttribute(n, a) { this.attributes[n] = a; } }
class BufferAttribute { constructor(array, n) { this.array = array; this.itemSize = n; } }
class Obj extends Stub { constructor(g, m) { super(); if (g) this.geometry = g; if (m) this.material = m; } }
Object.assign(globalThis, {
  document: { getElementById: $el, querySelector: () => $el('__stage'), querySelectorAll: () => [],
    createElement: () => mkEl('__new'), addEventListener() {}, body: { classList: { toggle() {}, contains() { return false; } } },
    fullscreenElement: null },
  THREE: { WebGLRenderer: Stub, Scene: Stub, Color: Stub, PerspectiveCamera: Stub, BufferGeometry: Stub, BufferAttribute,
    Mesh: Obj, LineSegments: Obj, MeshBasicMaterial: Stub, LineBasicMaterial: Stub, DoubleSide: 2 },
  OrbitControls: Stub, requestAnimationFrame: () => 0, devicePixelRatio: 1, addEventListener() {},
});
globalThis.window = globalThis;

const tmp = path.join(os.tmpdir(), `zrl_membrane_${process.pid}.mjs`);
fs.writeFileSync(tmp, src);
const { __T: T } = await import(pathToFileURL(tmp).href);
fs.unlinkSync(tmp);

const sizes = process.argv.slice(2).map(Number);
for (const lat of ['sq', 'tri']) for (const N of (sizes.length ? sizes : [161, 513, 1025, 2049])) {
  Object.assign(T.S, { lattice: lat, shape: lat === 'tri' ? 'n6' : 'square', src: 'pulse', sigma: 4, rad: 1, rot: 0 });
  T.alloc(N); T.reseed();
  const reps = Math.max(10, Math.round(3e7 / (N * N)));
  for (let s = 0; s < 3; s++) T.stepOnce();                       // JIT warm-up
  const t0 = process.hrtime.bigint();
  for (let s = 0; s < reps; s++) T.stepOnce();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6 / reps;
  console.log(JSON.stringify({ lat, N, ms_step: ms, ns_cell: ms * 1e6 / (N * N) }));
}
