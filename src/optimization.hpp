#ifndef OPTIMIZATION
#define OPTIMIZATION
#include "qnn.hpp"

interface optimization_network : network_base{
  param_t* dLdw;
  optimization_network(int b_size);
  virtual ~optimization_network();
  virtual void allocOptimizer() = 0;
  virtual void initOptimizer() = 0;
  virtual void update() = 0;
  optimization_network& train();
  optimization_network& pred();
  void setupNetwork();
  void calGradient(const Complex* E, Complex Eave);
  void run(const Complex* E, Complex Eave);
};

optimization_network::optimization_network(int b_size):network_base(b_size){}
optimization_network::~optimization_network(){
  free(dLdw);
}
optimization_network& optimization_network::train(){
  network_base::train();
  return *this;
}
optimization_network& optimization_network::pred(){
  network_base::pred();
  return *this;
}
void optimization_network::setupNetwork(){
  network_base::setupNetwork();
  dLdw = alloc<param_t>(LParamTotal);
  allocOptimizer();
  initOptimizer();
}
#ifdef USE_DOUBLE_PARAMETER
void optimization_network::calGradient(const Complex* E, Complex Eave){
  forwardPropagation();
  backwardPropagation();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<LParamTotal; ++k){
    double sum = 0.0;
    for(int n=0; n<batch_size; ++n){
      sum += 2.0*real(*O[n][k]*conj(E[n] - Eave));
    }
    dLdw[k] = sum/(double)batch_size;
  }
}
#endif
#ifdef USE_COMPLEX_PARAMETER
void optimization_network::calGradient(const Complex* E, Complex Eave){
  forwardPropagation();
  backwardPropagation();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int k=0; k<LParamTotal; ++k){
    Complex sum = 0.0;
    for(int n=0; n<batch_size; ++n){
       sum += conj(*O[n][k])*(E[n] - Eave);
    }
    dLdw[k] = sum/(double)batch_size;
  }
}
#endif
void optimization_network::run(const Complex* E, Complex Eave){
  calGradient(E, Eave);
  update();
  it++;
}

namespace Network{
  
  struct SGD : virtual optimization_network{
    double eta, lambda;
    SGD(int b_size, double e=1.0e-3, double decay=0.0)
      :optimization_network(b_size){
      eta = e, lambda = decay;
    }
    virtual ~SGD(){}
    virtual void allocOptimizer(){}
    virtual void initOptimizer(){}
    virtual void update();
  };
  void SGD::update(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int k = 0; k<LParamTotal; ++k){
      param_t g = (isReg[k])? dLdw[k] + lambda* *W[k]: dLdw[k];
      *W[k] += -eta*g;
    }
  }
  
  struct AdaGrad : virtual optimization_network{
    param_t *AG;
    double eta, epsilon, lambda;
    AdaGrad(int b_size, double e=0.2, double eps = 1.0e-8, double decay=0.0)
      :optimization_network(b_size){
      eta=e, epsilon=eps, lambda=decay;
    }
    virtual ~AdaGrad(){
      free(AG);
    }
    virtual void allocOptimizer();
    virtual void initOptimizer();
    virtual void update();
  };
  void AdaGrad::allocOptimizer(){
    AG = alloc<param_t>(LParamTotal);
  }
  void AdaGrad::initOptimizer(){
    for(int k = 0; k<LParamTotal; ++k){
      AG[k] = 0.0;
    }
  }
  void AdaGrad::update(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int k = 0; k<LParamTotal; ++k){
      param_t g = (isReg[k])? dLdw[k] + lambda* *W[k]: dLdw[k];
      AG[k] += g*g;
      *W[k] += -eta*g/(sqrt(AG[k]+1.0E-30) + epsilon);
    }
  }
  
  struct Adam : virtual optimization_network{
    param_t **AD;
    double alpha, beta1, beta2, b1, b2, epsilon, lambda;
    Adam(int b_size, double al=0.02, double be1=0.9, double be2=0.999,
	 double eps=1.0e-8, double decay=0.0)
      :optimization_network(b_size){
      alpha=al, beta1=be1, beta2=be2, epsilon=eps, lambda=decay;
    }
    virtual ~Adam(){
      free(AD);
    }
    virtual void allocOptimizer();
    virtual void initOptimizer();
    virtual void update();
  };
  void Adam::allocOptimizer(){
    AD = alloc<param_t>(2, LParamTotal);
  }
  void Adam::initOptimizer(){
    b1 = b2 = 1.0;
    for(int id = 0; id<LParamTotal; ++id){
      AD[0][id] = AD[1][id] = 0.0;
    }
  }
  void Adam::update(){
    b1 *= beta1;
    b2 *= beta2;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int k = 0; k<LParamTotal; ++k){
      param_t g = (isReg[k])? dLdw[k] + lambda* *W[k]: dLdw[k];
      AD[0][k] = beta1*AD[0][k] + (1.0-beta1)*g;
      AD[1][k] = beta2*AD[1][k] + (1.0-beta2)*g*g;
      param_t m = AD[0][k]/(1.0 - b1);
      param_t v = AD[1][k]/(1.0 - b2);
      *W[k] += -alpha*m/(sqrt(v)+epsilon);
    }
  }
}
#endif
