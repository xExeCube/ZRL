// The membrane app's update in C# (.NET 8 RyuJIT): masked, per-cell Co^2, damping.
// Two variants: "safe" (plain arrays, the idiomatic port) and "unsafe" (raw pointers,
// no bounds checks -- closest to what Unity's Burst compiler would emit).
// Usage: kernel N stencil seconds variant   (secs < 0: run |secs| steps, print sum u^2)
using System; using System.Diagnostics;
unsafe class P {
  static void Main(string[] a){
    int n=int.Parse(a[0]), st=int.Parse(a[1]); double secs=double.Parse(a[2]); bool uns=a.Length>3&&(a[3]=="unsafe"||a[3]=="par"); bool par=a.Length>3&&a[3]=="par";
    int nn=n*n; var U0=new double[nn]; var U1=new double[nn]; var U2=new double[nn]; var CC=new double[nn]; var M=new byte[nn];
    double c=(n-1)/2.0, R=0.45*n;
    for(int j=0;j<n;j++) for(int i=0;i<n;i++){ int k=j*n+i; double x=i-c,y=j-c; M[k]=(byte)(x*x+y*y<=R*R?1:0); CC[k]=0.25;
      U1[k]=U0[k]=M[k]!=0?Math.Exp(-(x*x+y*y)/18.0):0; }
    void Step(){
      if(uns){ fixed(double* u0p=U0,u1p=U1,u2p=U2,ccp=CC) fixed(byte* mp=M){
        double* u0=u0p; double* u1=u1p; double* u2=u2p; double* cc=ccp; byte* m=mp;
        void Row(int j){ int row=j*n; for(int i=1;i<n-1;i++){ int k=row+i;
          if(m[k]==0){ u2[k]=0; continue; }
          double lap;
          if(st==5) lap=u1[k-1]+u1[k+1]+u1[k-n]+u1[k+n]-4*u1[k];
          else if(st==9){ double e=u1[k-1]+u1[k+1]+u1[k-n]+u1[k+n], g=u1[k-n-1]+u1[k-n+1]+u1[k+n-1]+u1[k+n+1]; lap=(2.0/3)*e+(1.0/6)*g-(10.0/3)*u1[k]; }
          else { double s=u1[k-1]+u1[k+1]+u1[k-n]+u1[k+n]+u1[k+1-n]+u1[k-1+n]; lap=(2.0/3)*(s-6*u1[k]); }
          u2[k]=2*u1[k]-u0[k]+cc[k]*lap; } }
        if(par) System.Threading.Tasks.Parallel.For(1, n-1, Row); else for(int j=1;j<n-1;j++) Row(j); } }
      else {
        for(int j=1;j<n-1;j++){ int row=j*n; for(int i=1;i<n-1;i++){ int k=row+i;
          if(M[k]==0){ U2[k]=0; continue; }
          double lap;
          if(st==5) lap=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n]-4*U1[k];
          else if(st==9){ double e=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n], g=U1[k-n-1]+U1[k-n+1]+U1[k+n-1]+U1[k+n+1]; lap=(2.0/3)*e+(1.0/6)*g-(10.0/3)*U1[k]; }
          else { double s=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n]+U1[k+1-n]+U1[k-1+n]; lap=(2.0/3)*(s-6*U1[k]); }
          U2[k]=2*U1[k]-U0[k]+CC[k]*lap; } } }
      var t=U0; U0=U1; U1=U2; U2=t;
    }
    if(secs<0){ for(int s=0;s<(int)(-secs);s++) Step(); double q=0; foreach(var v in U1) q+=v*v; Console.WriteLine(q.ToString("E12")); return; }
    for(int w=0;w<30;w++) Step();                       // let tiered JIT reach the optimised tier
    var sw=Stopwatch.StartNew(); long steps=0;
    do { Step(); steps++; } while(sw.Elapsed.TotalSeconds<secs);
    Console.WriteLine((steps*(double)(n-2)*(n-2)/sw.Elapsed.TotalSeconds).ToString("E4"));
  }
}
