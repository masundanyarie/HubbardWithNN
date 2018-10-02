#ifndef EVOLUTIONNETWORK
#define EVOLUTIONNETWORK
#include "qnn.hpp"
#include "solver.hpp"

const bool IMAG_TIME = false;
const bool REAL_TIME = true;
struct EvolutionNetwork : network_base{
  param_t **Mat, *F, *Wdot;
  int Nstock;
  param_t** WdotStock;
  param_t* W0;
  param_t** RK;
  solver<param_t> sv;
  bool timeEV;
  double dt;
  double lambda0, b, lambda_min, tolerance;
  EvolutionNetwork(double h, int b_size, bool ev=IMAG_TIME):network_base(b_size){
    timeEV = ev;
    Nstock = 5;
    dt = h;
    lambda0 = 100, b = 0.99;
    lambda_min = 1.0e-4; //1.0e-4
    tolerance = 1.0e-4;  //1.0e-4
  }
  ~EvolutionNetwork(){
    free(RK); free(W0); free(WdotStock);
    free(Mat); free(Wdot); free(F);
  }
  EvolutionNetwork& train();
  EvolutionNetwork& pred();
  void setupNetwork();
  void calGradient(const Complex* E, Complex Eave);
  double wdotNorm();
  void backupWdot();
  void restoreWdot();
  void outWdot(const char* fname);
  void loadWdot(const char* fname);
  double errorWdot(const Complex* E, Complex Eave);
  void rungeKutta1(int work);
  void rungeKutta2(int work, const Complex*E, Complex Eave);
  void euler(const Complex*E,  Complex Eave);
  void update1ThFor();
  void update4ThFor();
  void ABM4pre(const Complex* E, Complex Eave);
  double ABM4cor(const Complex* E, Complex Eave);
  double ABM5cor(const Complex* E, Complex Eave);
  void ABMsav(const Complex* E, Complex Eave);
};
EvolutionNetwork& EvolutionNetwork::train(){
  network_base::train();
  return *this;
}
EvolutionNetwork& EvolutionNetwork::pred(){
  network_base::pred();
  return *this;
}
void EvolutionNetwork::setupNetwork(){
  network_base::setupNetwork();
  sv.plan(LParamTotal);
  Mat = alloc<param_t>(LParamTotal, LParamTotal);
  Wdot = alloc<param_t>(LParamTotal);
  F = alloc<param_t>(LParamTotal);
  WdotStock = alloc<param_t>(5, LParamTotal);
  W0 = alloc<param_t>(LParamTotal);
  RK = alloc<param_t>(Nstock, LParamTotal);
  for(int k=0; k<LParamTotal; ++k){
    Wdot[k] = W0[k] = F[k] = 0.0;
    for(int n=0; n<4; ++n){
      RK[n][k] = WdotStock[n][k] = 0.0;
    }
  }
}
void EvolutionNetwork::ABM4pre(const Complex* E, Complex Eave){
  calGradient(E, Eave);
  backupWdot();
  update4ThFor();
}
double EvolutionNetwork::ABM4cor(const Complex* E, Complex Eave){
  calGradient(E, Eave);
  double sum = 0.0;
  for(int k=0; k<LParamTotal; ++k){
    param_t w_old = *W[k];
    *W[k] = W0[k] + dt/24.0*(9.00*Wdot[k]
			     +19.*WdotStock[0][k]
			     -5.0*WdotStock[1][k]
			     +1.0*WdotStock[2][k]);
    sum += abs((*W[k] - w_old));
  }
  return sum;
}
double EvolutionNetwork::ABM5cor(const Complex* E, Complex Eave){
  calGradient(E, Eave);

  double sum = 0.0;
  for(int k=0; k<LParamTotal; ++k){
    param_t w_old = *W[k];
    *W[k] = W0[k] + dt/720.0*(251.*Wdot[k]
			     +646.*WdotStock[0][k]
			     -264.*WdotStock[1][k]
			     +106.*WdotStock[2][k]
			     -19.0*WdotStock[3][k]);
    sum += abs((*W[k] - w_old));
  }
  return sum;
}
void EvolutionNetwork::ABMsav(const Complex* E, Complex Eave){
  calGradient(E, Eave);
  backupWdot();
}

#ifdef USE_DOUBLE_PARAMETER
void EvolutionNetwork::calGradient(const Complex* E, Complex Eave){
  forwardPropagation();
  backwardPropagation();
  double lambda = max(lambda0*pow(b, it), lambda_min);  
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<LParamTotal; ++k){
    Complex sum = 0;
    for(int n=0; n<batch_size; ++n){
      sum += *O[n][k];
    }
    Oave[k] = sum/(double)batch_size;
  }
  
  int total = (LParamTotal-1)*(LParamTotal+2)/2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<=total; ++k){
    int i=0, j=k, len=LParamTotal, s=k;    
    for(;;len--, j = ++i + s){
      s -= len;
      if(s<0)break;
    }
    double sum = 0.0;
    for(int n=0; n<batch_size; ++n){
      sum += real(*O[n][i]*conj(*O[n][j] - Oave[j]));
    }
    Mat[i][j] = sum/(double)batch_size;
    if(i==j && !timeEV){
      Mat[i][j] *= 1.0 + lambda;
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<LParamTotal; ++i){
    for(int j=0; j<i; ++j){
      Mat[i][j] = Mat[j][i];
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for
#endif  
  for(int k=0; k<LParamTotal; ++k){   
    double sum = 0.0;
    for(int n=0; n<batch_size; ++n){
      sum += (timeEV)? imag(*O[n][k]*conj(E[n] - Eave)):real(*O[n][k]*conj(E[n] - Eave));
    }
    F[k] = -sum/(double)batch_size;
  }
  
  if(timeEV) sv.pi(Mat, Wdot, F, tolerance);
  else       sv.lu(Mat, Wdot, F);
}
#endif
#ifdef USE_COMPLEX_PARAMETER
void EvolutionNetwork::calGradient(const Complex* E, Complex Eave){
  forwardPropagation();
  backwardPropagation();
  double lambda = max(lambda0*pow(b, it), lambda_min);  
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<LParamTotal; ++k){
    Complex sum = 0;
    for(int n=0; n<batch_size; ++n){
      sum += *O[n][k];
    }
    Oave[k] = sum/(double)batch_size;
  }
  
  int total = (LParamTotal-1)*(LParamTotal+2)/2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<=total; ++k){
    int i=0, j=k, len=LParamTotal, s=k;    
    for(;;len--, j = ++i + s){
      s -= len;
      if(s<0)break;
    }
    Complex sum = 0.0;
    for(int n=0; n<batch_size; ++n){
      sum += conj(*O[n][i])*(*O[n][j] - Oave[j]);
    }
    Mat[i][j] = sum/(double)batch_size;
    if(i==j && !timeEV){
      Mat[i][j] *= 1.0 + lambda;
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<LParamTotal; ++i){
    for(int j=0; j<i; ++j){
      Mat[i][j] = conj(Mat[j][i]);
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for
#endif  
  for(int k=0; k<LParamTotal; ++k){   
    Complex sum = 0.0;
    for(int n=0; n<batch_size; ++n){
      sum += (timeEV)? AI*conj(*O[n][k])*(E[n] - Eave):conj(*O[n][k])*(E[n] - Eave);
    }
    F[k] = -sum/(double)batch_size;
  }  
  if(timeEV) sv.pi(Mat, Wdot, F, tolerance);
  else       sv.lu(Mat, Wdot, F);
}
#endif
double EvolutionNetwork::wdotNorm(){
  double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
  for(int k=0; k<LParamTotal; ++k){
    sum += norm(Wdot[k]);
  }
  return sum;
}
void EvolutionNetwork::backupWdot(){
  for(int k=0; k<LParamTotal; ++k){
    W0[k] = *W[k];
    for(int i=Nstock-1; i>0; --i){
      WdotStock[i][k] = WdotStock[i-1][k];
    }
    WdotStock[0][k] = Wdot[k];
  }
}
void EvolutionNetwork::restoreWdot(){
  for(int k=0; k<LParamTotal; ++k){
    Wdot[k] = WdotStock[3][k];
  }
}
void EvolutionNetwork::outWdot(const char* fname){
  ofstream ofs(fname);
  for(int k=0; k<LParamTotal; ++k){
    ofs << Wdot[k] << endl;;
  }
}
void EvolutionNetwork::loadWdot(const char* fname){
  ifstream ifs(fname);
  for(int k=0; k<LParamTotal; ++k){
    ifs >> Wdot[k];
  }
}
double EvolutionNetwork::errorWdot(const Complex*E, Complex Eave){
  for(int k=0; k<LParamTotal; ++k){
    double re=0, im=0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:re, im)
#endif
    for(int n=0; n<batch_size; ++n){
      re += O[n][k]->real();
      im += O[n][k]->imag();
    }
    Complex sum(re, im);
    Oave[k] = sum/(double)batch_size;
  }
  
  double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for  reduction(+:sum)
#endif
  for(int n=0; n<batch_size; ++n){
    Complex OW = 0.0;
    for(int k=0; k<LParamTotal; ++k){
      OW += (*O[n][k] - Oave[k])*Wdot[k];
    }
    sum += norm(OW + AI*(E[n] - Eave));
  }
  sum /= (double)batch_size;
  return sum;
}
void EvolutionNetwork::rungeKutta1(int work){
  switch (work){
  case 0:
    for(int k=0; k<LParamTotal; ++k){
      W0[k] = *W[k];
    }
    break; 
  case 1:
    for(int k=0; k<LParamTotal; ++k){
      *W[k] = W0[k] + dt*RK[0][k]/2.0;
    }
    break;
  case 2:
    for(int k=0; k<LParamTotal; ++k){
      *W[k] = W0[k] + dt*RK[1][k]/2.0;
    }
    break;
  case 3:
    for(int k=0; k<LParamTotal; ++k){
      *W[k] = W0[k] + dt*RK[2][k];
    }
    break;
  case 4:
    for(int k=0; k<LParamTotal; ++k){
      *W[k] = W0[k] + dt*(RK[0][k] + 2.0*RK[1][k] + 2.0*RK[2][k] + RK[3][k])/6.0;
    }
    it++;
    break;
  }
}
void EvolutionNetwork::rungeKutta2(int work, const Complex*E, Complex Eave){
  if(work<4){
    calGradient(E, Eave);
    for(int k=0;k<LParamTotal; ++k){
      RK[work][k] = Wdot[k];
    }
  }
}
void EvolutionNetwork::euler(const Complex*E,  Complex Eave){
  calGradient(E, Eave);
  update1ThFor();
}
void EvolutionNetwork::update1ThFor(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<LParamTotal; ++k){
    *W[k] += dt*Wdot[k];
  }
  it++;
}
void EvolutionNetwork::update4ThFor(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<LParamTotal; ++k){
    *W[k] += dt/24.0*(55.0*WdotStock[0][k]
		      -59.*WdotStock[1][k]
		      +37.*WdotStock[2][k]
		      -9.0*WdotStock[3][k]);
  }
  it++;
}
#endif
