// The same update in idiomatic C++ (std::vector, lambdas), g++ -O3 -march=native -fopenmp.
#include <vector>
#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <omp.h>
int main(int argc, char** argv){
  const int n = std::atoi(argv[1]), st = std::atoi(argv[2]); const double secs = std::atof(argv[3]);
  const size_t nn = size_t(n) * n;
  std::vector<double> U0(nn), U1(nn), U2(nn), CC(nn, 0.25); std::vector<unsigned char> M(nn);
  const double c = (n - 1) / 2.0, R = 0.45 * n;
  for (int j = 0; j < n; j++) for (int i = 0; i < n; i++) { size_t k = size_t(j) * n + i; double x = i - c, y = j - c;
    M[k] = x*x + y*y <= R*R; U1[k] = U0[k] = M[k] ? std::exp(-(x*x + y*y) / 18.0) : 0; }
  auto step = [&]{
    double *u0 = U0.data(), *u1 = U1.data(), *u2 = U2.data(); const double* cc = CC.data(); const unsigned char* m = M.data();
    #pragma omp parallel for schedule(static)
    for (int j = 1; j < n - 1; j++) { const size_t row = size_t(j) * n;
      for (int i = 1; i < n - 1; i++) { const size_t k = row + i;
        if (!m[k]) { u2[k] = 0; continue; }
        double lap;
        if (st == 5) lap = u1[k-1] + u1[k+1] + u1[k-n] + u1[k+n] - 4*u1[k];
        else if (st == 9) { double e = u1[k-1] + u1[k+1] + u1[k-n] + u1[k+n], g = u1[k-n-1] + u1[k-n+1] + u1[k+n-1] + u1[k+n+1];
          lap = (2.0/3)*e + (1.0/6)*g - (10.0/3)*u1[k]; }
        else { double s = u1[k-1] + u1[k+1] + u1[k-n] + u1[k+n] + u1[k+1-n] + u1[k-1+n]; lap = (2.0/3)*(s - 6*u1[k]); }
        u2[k] = 2*u1[k] - u0[k] + cc[k]*lap; } }
    std::swap(U0, U1); std::swap(U1, U2);
  };
  if (secs < 0) { for (int s = 0; s < int(-secs); s++) step(); double q = 0; for (double v : U1) q += v*v; std::printf("%.12e\n", q); return 0; }
  auto t0 = std::chrono::steady_clock::now(); long steps = 0; double t;
  do { step(); steps++; t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); } while (t < secs);
  std::printf("%.4e\n", steps * double(n - 2) * (n - 2) / t);
}
