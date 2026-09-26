/* (real)-cast constants: a bare 2.0/3 is a DOUBLE literal, which silently promoted the float
   build of the 9-point and triangular stencils to double arithmetic (fixed 26/09).
   The membrane app's interior update, verbatim math: masked, per-cell Co^2 (CC), damping.
   stencil 5 = square 5-point, 9 = square 9-point isotropic, 6 = triangular 6-neighbour.
   Usage: ./kernel N stencil seconds   (threads via OMP_NUM_THREADS) -> cell-updates/s */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
typedef double real;
static double now(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec+1e-9*t.tv_nsec; }
int main(int argc, char**argv){
  int n=atoi(argv[1]), st=atoi(argv[2]); double secs=atof(argv[3]);
  size_t nn=(size_t)n*n;
  real *U0=calloc(nn,sizeof(real)), *U1=calloc(nn,sizeof(real)), *U2=calloc(nn,sizeof(real)), *CC=malloc(nn*sizeof(real));
  unsigned char *M=malloc(nn); double c=(n-1)/2.0, R=0.45*n; const real d=0;
  for(int j=0;j<n;j++) for(int i=0;i<n;i++){ size_t k=(size_t)j*n+i; double x=i-c,y=j-c;
    M[k]=(x*x+y*y<=R*R); CC[k]=0.25; U1[k]=U0[k]=M[k]?__builtin_exp(-(x*x+y*y)/18.0):0; }
  long steps=0; double t0=now(), t; long fixed = secs<0 ? (long)(-secs) : 0;
  do {
    #pragma omp parallel for schedule(static)
    for(int j=1;j<n-1;j++){ size_t row=(size_t)j*n;
      for(int i=1;i<n-1;i++){ size_t k=row+i;
        if(!M[k]){ U2[k]=0; continue; }
        real lap;
        if(st==5) lap=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n]-4*U1[k];
        else if(st==9){ real e=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n], g=U1[k-n-1]+U1[k-n+1]+U1[k+n-1]+U1[k+n+1];
          lap=(real)(2.0/3)*e+(real)(1.0/6)*g-(real)(10.0/3)*U1[k]; }
        else { real s=U1[k-1]+U1[k+1]+U1[k-n]+U1[k+n]+U1[k+1-n]+U1[k-1+n]; lap=(real)(2.0/3)*(s-6*U1[k]); }
        U2[k]=(2*U1[k]-U0[k]+CC[k]*lap)-d*(U1[k]-U0[k]);
      } }
    real*tmp=U0; U0=U1; U1=U2; U2=tmp; steps++; t=now()-t0;
  } while(fixed ? steps<fixed : t<secs);
  if(fixed){ double s=0; for(size_t k=0;k<nn;k++) s+=(double)U1[k]*U1[k]; printf("%.12e\n", s); return 0; }
  printf("%.4e\n", steps*(double)(n-2)*(n-2)/t);
  return 0;
}
