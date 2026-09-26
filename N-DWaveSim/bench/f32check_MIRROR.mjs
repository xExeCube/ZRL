// strict float32 arithmetic (Math.fround after every operation), N = 101, 200 steps
const fr = Math.fround, n = 101, c = (n - 1) / 2, R = 0.45 * n;
for (const st of [5, 9, 6]) {
  let U0 = new Float32Array(n * n), U1 = new Float32Array(n * n), U2 = new Float32Array(n * n); const M = new Uint8Array(n * n);
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) { const k = j * n + i, x = i - c, y = j - c; M[k] = x * x + y * y <= R * R; U1[k] = U0[k] = M[k] ? Math.exp(-(x * x + y * y) / 18) : 0; }
  const w23 = fr(2 / 3), w16 = fr(1 / 6), w103 = fr(10 / 3), CC = fr(0.25);
  for (let s = 0; s < 200; s++) {
    for (let j = 1; j < n - 1; j++) for (let i = 1; i < n - 1; i++) { const k = j * n + i;
      if (!M[k]) { U2[k] = 0; continue; }
      const u = U1; const e = fr(fr(fr(u[k - 1] + u[k + 1]) + u[k - n]) + u[k + n]); let lap;
      if (st === 5) lap = fr(e - fr(4 * u[k]));
      else if (st === 9) { const g = fr(fr(fr(u[k - n - 1] + u[k - n + 1]) + u[k + n - 1]) + u[k + n + 1]);
        lap = fr(fr(fr(w23 * e) + fr(w16 * g)) - fr(w103 * u[k])); }
      else { const s6 = fr(fr(e + u[k + 1 - n]) + u[k - 1 + n]); lap = fr(w23 * fr(s6 - fr(6 * u[k]))); }
      U2[k] = fr(fr(fr(2 * u[k]) - U0[k]) + fr(CC * lap)); }
    const t = U0; U0 = U1; U1 = U2; U2 = t; }
  let q = 0; for (const v of U1) q += v * v;
  const ref = { 5: 1.540773101133e+01, 9: 1.532643659774e+01, 6: 1.856106542595e+01 }[st];
  console.log(`stencil ${st}: strict float32 sum u^2 = ${q.toPrecision(10)}, rel. dev from float64 ${(Math.abs(q - ref) / ref).toExponential(2)}`);
}
