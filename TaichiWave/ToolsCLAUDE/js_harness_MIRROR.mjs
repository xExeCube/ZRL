// Headless harness: runs the membrane app's REAL module script in a vm with DOM/THREE stubs.
import fs from 'fs'; import vm from 'vm';
const APP = process.argv[2] || 'F:/FDrive-Storage/Claude Local/FormConstantsCLAUDE/FlavorRenderersCLAUDE/wave_membrane_MIRROR.html';
const html = fs.readFileSync(APP, 'utf8');
const m = html.match(/<script type="module">([\s\S]*?)<\/script>/);
let src = m[1].replace(/^\s*import .*$/mg, '');

// ---- documented port deviations, applied to the web app's module source before it runs ------
// The Taichi port (membrane_core_MIRROR.py) is compared against "the web app + these patches",
// so each deviation is written down once, here, as the exact change it makes to the JS. The web
// app itself is frozen (read-only); a search string that is not found exactly once throws, so a
// patch can never silently stop applying. ZRL_JS_NOPATCH=1 runs the unpatched web app.
export const PATCHES = [
  {
    id: 'neumann-corner',
    date: '29/09/2026',
    rationale: 'The Neumann ring is copied in ONE sequential loop, so corner (0,0) is written at ' +
      'i = 0 from (1,0) BEFORE i = 1 refreshes it: it holds the rotating buffer old contents (the ' +
      'level three steps back), while the other three corners get this step values. Through the 9-point ' +
      'stencil diagonal that changed the field by 8.6% of its peak near the corner (400 steps). ' +
      'Rule: every corner equals its diagonal interior neighbour of THIS step (the other three ' +
      'already did), the ghost-cell rule for which the 9-point operator is symmetric.',
    search: 'U2[idx(0,i)]=U2[idx(1,i)]; U2[idx(n-1,i)]=U2[idx(n-2,i)]; }',
    replace: 'U2[idx(0,i)]=U2[idx(1,i)]; U2[idx(n-1,i)]=U2[idx(n-2,i)]; }' + String.fromCharCode(10) +
      '    U2[idx(0,0)]=U2[idx(1,1)]; U2[idx(0,n-1)]=U2[idx(1,n-2)]; ' +
      'U2[idx(n-1,0)]=U2[idx(n-2,1)]; U2[idx(n-1,n-1)]=U2[idx(n-2,n-2)];   /* PATCH neumann-corner */',
  },
];
export const PATCHED = process.env.ZRL_JS_NOPATCH === '1' ? [] : PATCHES.map(p => p.id);
if (process.env.ZRL_JS_NOPATCH !== '1') {
  for (const p of PATCHES) {
    const at = src.indexOf(p.search);
    if (at < 0) throw new Error(`js_harness: patch '${p.id}': search string not found in the web app`);
    if (src.indexOf(p.search, at + 1) >= 0) throw new Error(`js_harness: patch '${p.id}': search string is not unique`);
    src = src.slice(0, at) + p.replace + src.slice(at + p.search.length);
  }
}
src += `
globalThis.__T={S:(typeof S!=='undefined'?S:undefined),reseed:(typeof reseed!=='undefined'?reseed:undefined),stepOnce:(typeof stepOnce!=='undefined'?stepOnce:undefined),buildLedger:(typeof buildLedger!=='undefined'?buildLedger:undefined),alAngles:(typeof alAngles!=='undefined'?alAngles:undefined),alEdgeDirs0:(typeof alEdgeDirs0!=='undefined'?alEdgeDirs0:undefined),alCount:(typeof alCount!=='undefined'?alCount:undefined),alClass:(typeof alClass!=='undefined'?alClass:undefined),alBest:(typeof alBest!=='undefined'?alBest:undefined),alListFor:(typeof alListFor!=='undefined'?alListFor:undefined),alStep:(typeof alStep!=='undefined'?alStep:undefined),alOptimal:(typeof alOptimal!=='undefined'?alOptimal:undefined),alFlavor:(typeof alFlavor!=='undefined'?alFlavor:undefined),alFiltered:(typeof alFiltered!=='undefined'?alFiltered:undefined),setRot:(typeof setRot!=='undefined'?setRot:undefined),alSymPeriod:(typeof alSymPeriod!=='undefined'?alSymPeriod:undefined),alCanon:(typeof alCanon!=='undefined'?alCanon:undefined),pulseClipFraction:(typeof pulseClipFraction!=='undefined'?pulseClipFraction:undefined),cellXY:(typeof cellXY!=='undefined'?cellXY:undefined),domR:(typeof domR!=='undefined'?domR:undefined),innerR:(typeof innerR!=='undefined'?innerR:undefined),shapePoly:(typeof shapePoly!=='undefined'?shapePoly:undefined),fitsInside:(typeof fitsInside!=='undefined'?fitsInside:undefined),alNearDir:(typeof alNearDir!=='undefined'?alNearDir:undefined),alLatticeDir:(typeof alLatticeDir!=='undefined'?alLatticeDir:undefined),alloc:(typeof alloc!=='undefined'?alloc:undefined),alList:(typeof alList!=='undefined'?alList:undefined),alSetRot:(typeof alSetRot!=='undefined'?alSetRot:undefined),alMeasure:(typeof alMeasure!=='undefined'?alMeasure:undefined),alEdges:(typeof alEdges!=='undefined'?alEdges:undefined),
  get AL_REP(){return typeof AL_REP!=='undefined'?AL_REP:undefined}, get MASK(){return MASK}, get N(){return N}, get VTX(){return VTX},
  get CC(){return CC}, get U1(){return U1},
  get ENERGY(){return ENERGY}, get PROBE(){return PROBE}, get PROBE_B(){return PROBE_B}, get MODEP(){return MODEP},
  pA:(typeof pA!=='undefined'?pA:undefined), pB:(typeof pB!=='undefined'?pB:undefined), modeAntinode:(typeof modeAntinode!=='undefined'?modeAntinode:undefined)};`;

// ---- DOM stubs, with every <select>'s options and every range input's attributes parsed from the HTML ----
const els = new Map();
const selects = {};
for (const sm of html.matchAll(/<select id="([^"]+)"[^>]*>([\s\S]*?)<\/select>/g)) {
  selects[sm[1]] = [...sm[2].matchAll(/<option value="([^"]*)"[^>]*>([\s\S]*?)<\/option>/g)].map(o => ({ value: o[1], textContent: o[2].replace(/<[^>]+>/g, ''), disabled: false }));
}
const ranges = {};
for (const rm of html.matchAll(/<input type="range" id="([^"]+)" min="([^"]+)" max="([^"]+)" step="([^"]+)" value="([^"]+)">/g))
  ranges[rm[1]] = { min: rm[2], max: rm[3], step: rm[4], value: rm[5] };
function mkEl(id) {
  const L = {};
  const el = {
    id, innerHTML: '', textContent: '', checked: false, style: {}, dataset: {}, tagName: 'DIV',
    classList: { toggle() {}, add() {}, remove() {}, contains() { return false; } },
    addEventListener(t, f) { (L[t] = L[t] || []).push(f); },
    dispatchEvent(ev) { for (const f of (L[ev.type] || [])) f(ev); return true; },
    click() { if (el.onclick) el.onclick(); for (const f of (L.click || [])) f({}); },
    getContext() { return new Proxy({}, { get: () => () => {} }); },
    clientWidth: 800, clientHeight: 560, width: 800, height: 560,
    querySelectorAll() { return []; }, querySelector() { return null; }, closest() { return null; },
    appendChild() {}, insertBefore() {}, parentNode: null, nextElementSibling: { textContent: '' },
    requestFullscreen() { return Promise.reject(new Error('no')); },
  };
  if (selects[id]) {
    el.tagName = 'SELECT'; el.options = selects[id].map(o => ({ ...o })); el.selectedIndex = 0;
    Object.defineProperty(el, 'value', { get() { return el.options[el.selectedIndex] ? el.options[el.selectedIndex].value : ''; },
      set(v) { const i = el.options.findIndex(o => o.value === String(v)); el.selectedIndex = i < 0 ? 0 : i; } });
    // alSnap's options are written through innerHTML at runtime: re-parse them
    let ih = '';
    Object.defineProperty(el, 'innerHTML', { get() { return ih; }, set(v) { ih = v;
      el.options = [...v.matchAll(/<option value="([^"]*)"[^>]*>([\s\S]*?)<\/option>/g)].map(o => ({ value: o[1], textContent: o[2], disabled: false }));
      el.selectedIndex = 0; } });
  } else if (ranges[id]) {
    el.tagName = 'INPUT'; Object.assign(el, ranges[id]);
    let val = ranges[id].value;
    Object.defineProperty(el, 'value', { get() { return val; }, set(v) {
      // a range input clamps and snaps to its step, like the browser does
      const mn = +el.min, mx = +el.max, st = +el.step; let x = Math.min(mx, Math.max(mn, +v));
      x = mn + Math.round((x - mn) / st) * st; val = String(+x.toFixed(10)); } });
    el.parentNode = { querySelector() { return null; }, insertBefore() {} };
  } else el.value = '';
  return el;
}
const $el = id => { if (!els.has(id)) els.set(id, mkEl(id)); return els.get(id); };
const docL = {};
const document = {
  getElementById: $el, querySelector: () => $el('__stage'), querySelectorAll: () => [],
  createElement: () => mkEl('__new'), addEventListener(t, f) { (docL[t] = docL[t] || []).push(f); },
  body: { classList: { toggle() {}, contains() { return false; } } }, fullscreenElement: null,
};
const winL = {};
class Stub { constructor(...a) { this.args = a; this.position = { set() {} }; this.up = { set() {} }; this.target = { set() {} };
  this.attributes = {}; this.geometry = { dispose() {}, attributes: {} }; this.material = { dispose() {} }; }
  setPixelRatio() {} getPixelRatio() { return 1; } setSize() {} render() {} add() {} remove() {} dispose() {}
  updateProjectionMatrix() {} update() {} setIndex() {} setAttribute(n, a) { this.attributes[n] = a; } }
class BufferAttribute { constructor(array, n) { this.array = array; this.itemSize = n; this.needsUpdate = false; } }
class Geo extends Stub {}
class Obj extends Stub { constructor(g, m) { super(); this.geometry = g || this.geometry; this.material = m || this.material; } }
const THREE = { WebGLRenderer: Stub, Scene: Stub, Color: Stub, PerspectiveCamera: Stub, BufferGeometry: Geo,
  BufferAttribute, Mesh: Obj, LineSegments: Obj, MeshBasicMaterial: Stub, LineBasicMaterial: Stub, DoubleSide: 2 };
const ctx = {
  THREE, OrbitControls: Stub, document, window: null, console, Math, JSON, Promise, Set, Map, Array, Float32Array,
  Float64Array, Uint8Array, Int32Array, Number, String, Object, Error, isFinite, parseFloat, parseInt,
  setTimeout: () => 0, clearTimeout: () => {}, requestAnimationFrame: () => 0, devicePixelRatio: 1,
  performance: { now: () => 0 }, Event: class { constructor(t) { this.type = t; } },
  addEventListener(t, f) { (winL[t] = winL[t] || []).push(f); }, URL, Blob: class {},
};
ctx.window = ctx; ctx.globalThis = ctx;
vm.createContext(ctx);
vm.runInContext(src, ctx, { filename: 'membrane.js' });
export const T = ctx.__T, $ = $el;
export const key = (k, shift) => { for (const f of (winL.keydown || [])) f({ key: k, shiftKey: !!shift, ctrlKey: false,
  metaKey: false, altKey: false, target: { tagName: 'BODY' }, preventDefault() {} }); };
export const change = (id, v) => { const e = $el(id); if (e.tagName === 'SELECT' || e.tagName === 'INPUT') e.value = v; else e.checked = v;
  e.dispatchEvent(new ctx.Event(e.tagName === 'INPUT' && ranges[id] ? 'input' : 'change')); };
export const check = (id, on) => { const e = $el(id); e.checked = on; e.dispatchEvent(new ctx.Event('change')); };
