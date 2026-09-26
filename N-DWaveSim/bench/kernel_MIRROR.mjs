// The same update in JS (V8 = Chrome's engine), exactly as the app writes it.
const [,, N_, ST_, SECS_, F32] = process.argv;
const n = +N_, st = +ST_, secs = +SECS_, T = F32 === 'f32' ? Float32Array : Float64Array;
let U0 = new T(n*n), U1 = new T(n*n), U2 = new T(n*n); const CC = new T(n*n), M = new Uint8Array(n*n);
const c = (n-1)/2, R = 0.45*n, d = 0;
for (let j=0;j<n;j++) for (let i=0;i<n;i++){ const k=j*n+i, x=i-c, y=j-c;
  M[k] = x*x+y*y <= R*R ? 1 : 0; CC[k] = 0.25; U1[k] = U0[k] = M[k] ? Math.exp(-(x*x+y*y)/18) : 0; }
function step(){
  for (let j=1;j<n-1;j++){ const row=j*n;
    for (let i=1;i<n-1;i++){ const k=row+i;
      if (!M[k]) { U2[k]=0; continue; }
      let lap;
      if (st===5) lap=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n]-4*U1[k];
      else if (st===9){ const e=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n], g=U1[k-n-1]+U1[k-n+1]+U1[k+n-1]+U1[k+n+1];
        lap=(2/3)*e+(1/6)*g-(10/3)*U1[k]; }
      else { const s=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n]+U1[k+1-n]+U1[k-1+n]; lap=(2/3)*(s-6*U1[k]); }
      U2[k]=(2*U1[k]-U0[k]+CC[k]*lap)-d*(U1[k]-U0[k]);
    } }
  const t=U0; U0=U1; U1=U2; U2=t;
}
if (secs < 0) { for (let s=0; s<-secs; s++) step(); let q=0; for (const v of U1) q+=v*v; console.log(q.toExponential(12)); process.exit(0); }
for (let w=0; w<3; w++) step();                     // JIT warm-up
let steps=0; const t0=performance.now(); let t;
do { step(); steps++; t=(performance.now()-t0)/1000; } while (t<secs);
console.log((steps*(n-2)*(n-2)/t).toExponential(4));
