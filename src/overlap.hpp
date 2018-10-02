#ifndef OVERLAP
#define OVERLAP
#include "qnn.hpp"
#include "hubbard.hpp"
#include "creater.hpp"

struct overlap{
  const int batch_size;
  Complex*** psi;

  overlap(int b_size): batch_size(b_size){
    psi = alloc<Complex>(batch_size, 2, 2);
  }
  ~overlap(){
    free(psi);
  }
  template<class Net, class Model>
  void calOverlap(Net* nn, Model* model, Complex* R, Complex &Rbar);
};

template<class Net, class Model>
void overlap::calOverlap(Net* nn, Model* model, Complex* R, Complex &Rbar){
  for(int smp=0; smp<2; ++smp){
    model[smp].makeSamples(nn[smp]);;
  }
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<batch_size; ++n){
    for(int smp=0; smp<2; ++smp){
      for(int nw=0; nw<2; ++nw){
	psi[n][smp][nw] = model[smp].makeWF(n, nn[nw]);
      }
    }
  }
  Complex rxaa = 0;
  double re = 0.0, im = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:re, im)
#endif
  for(int n=0; n<batch_size; ++n){
    Complex c = psi[n][0][1]/psi[n][0][0];
    re += c.real();
    im += c.imag();
  }
  rxaa = -Complex(re, im)/(double)batch_size;
  re = 0.0, im = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:re, im)
#endif
  for(int n=0; n<batch_size; ++n){
    R[n] = rxaa*psi[n][1][0]/psi[n][1][1];
    re += R[n].real();
    im += R[n].imag();
  }
  Rbar = Complex(re, im)/(double)batch_size;  
}

#endif
