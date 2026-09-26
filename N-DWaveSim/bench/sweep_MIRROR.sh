#!/bin/bash
# Build first:  gcc -O3 -march=native -fopenmp -o kernel_MIRROR kernel_MIRROR.c -lm
#               sed "s/typedef double real;/typedef float real;/" kernel_MIRROR.c > k32.c && gcc -O3 -march=native -fopenmp -o kernel32_MIRROR k32.c -lm
# Needs node, numpy, numba.  Run:  ./sweep_MIRROR.sh > sweep_MIRROR.csv && python3 analyze_MIRROR.py
# cell-updates per second, per implementation, stencil and N
echo "impl,stencil,N,cups"
for st in 5 6 9; do for N in 161 321 641 1281 2561 4097; do
  echo "C-f64-1T,$st,$N,$(OMP_NUM_THREADS=1 ./kernel_MIRROR $N $st 1.5)"
  echo "C-f64-4T,$st,$N,$(OMP_NUM_THREADS=4 ./kernel_MIRROR $N $st 1.5)"
  echo "C-f32-1T,$st,$N,$(OMP_NUM_THREADS=1 ./kernel32_MIRROR $N $st 1.5)"
  echo "C-f32-4T,$st,$N,$(OMP_NUM_THREADS=4 ./kernel32_MIRROR $N $st 1.5)"
  echo "JS-f64,$st,$N,$(node kernel_MIRROR.mjs $N $st 1.5)"
  echo "JS-f32,$st,$N,$(node kernel_MIRROR.mjs $N $st 1.5 f32)"
  echo "numpy,$st,$N,$(python3 kernel_MIRROR.py $N $st 1.5 numpy)"
  echo "numba-1T,$st,$N,$(NUMBA_NUM_THREADS=1 python3 kernel_MIRROR.py $N $st 1.5 numba)"
  echo "numba-4T,$st,$N,$(NUMBA_NUM_THREADS=4 python3 kernel_MIRROR.py $N $st 1.5 numba_par)"
done; done
