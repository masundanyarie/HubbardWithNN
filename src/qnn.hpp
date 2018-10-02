#ifndef QNN
#define QNN

#include <vector>
#include "input_layer.hpp"
#include "output_layer.hpp"
#include "full_conection.hpp"
#include "convolution.hpp"
#include "pooling.hpp"
#include "sum_layer.hpp"
#include "gutzwiller.hpp"

struct network_base{
  const int batch_size;
  int LParamTotal, it;
  param_t **W;
  odata_t ***O, *Oave;
  bool *isReg;
  propagator *InputLayer;
  vector<propagator*> Lprop;
  vector<layer_base*> Lpara;
  output_layer *OutputLayer;
  
  network_base(int b_size);
  virtual ~network_base();
  
  network_base &operator=(network_base &nn);
  layer_base &operator()(int l);
  layer_base &operator()(int l)const;
  
  virtual odata_t output(int n);
  virtual void setupNetwork();
  virtual network_base& train();
  virtual network_base& pred();
  void globalInitializer();
  void forwardPropagation();
  void backwardPropagation();
  void debugmode();
  void save(const char* fname)const;
  void load(const char* fname);
  void checkDiff(bool fineCheck = false);
  double weightNorm();
};
network_base::network_base(int b_size):batch_size(b_size){
  it = 0, LParamTotal = 0;
}
network_base::~network_base(){
  for(size_t l=0; l<Lprop.size(); ++l){
    delete Lprop[l];
  }
  free<param_t*>(W);
  free<odata_t*>(O);
  free<odata_t>(Oave); 
  free(isReg);
}

network_base& network_base::operator=(network_base &nn){
  for(int k=0; k<LParamTotal; ++k){
    *this->W[k] = *nn.W[k];
  }
  return *this;
}
layer_base& network_base::operator()(int l){
  return *Lpara[l];
}
layer_base& network_base::operator()(int l)const{
  return *Lpara[l];
}
odata_t network_base::output(int n){
  for(size_t l=0; l<Lprop.size(); ++l){
    Lprop[l]->fPropLayerBatch(n);
  }
  return OutputLayer->phi[n][0];
}
network_base& network_base::train(){
  for(size_t l=0; l<Lprop.size(); ++l){
    Lprop[l]->setMaskForTrain();
  }
  return *this;
}
network_base& network_base::pred(){
  for(size_t l=0; l<Lprop.size(); ++l){
    Lprop[l]->setMaskForPred();
  }
  return *this;
}
void network_base::setupNetwork(){
  InputLayer = Lprop[0];
  for(size_t l=0; l<Lprop.size(); ++l){
    if(l!=0)Lprop[l]->in = Lprop[l-1];
    Lprop[l]->setProperty(batch_size);
    Lprop[l]->allocUnits();
    Lprop[l]->initUnits();
  }
  
  for(size_t l=0; l<Lpara.size(); ++l){
    Lpara[l]->allocLParams();
    Lpara[l]->initLParams();
    LParamTotal += Lpara[l]->Nt;
  }
  
  W = alloc<param_t*>(LParamTotal);
  O = alloc<odata_t*>(batch_size, LParamTotal);
  Oave = alloc<odata_t>(LParamTotal);
  isReg = alloc<bool>(LParamTotal);
  
  for(int k=0;k<LParamTotal;){
    for(size_t l=0; l<Lpara.size(); ++l){
      for(int id=0; id<Lpara[l]->Nw; ++id){
	W[k] = &(Lpara[l]->w[id]);
	isReg[k] = Lpara[l]->isreg;
	for(int n=0; n<batch_size; ++n){
	  O[n][k] = &(Lpara[l]->Ow[n*Lpara[l]->Nw + id]);
	}
	k++;
      }
      for(int id=0; id<Lpara[l]->Nb; ++id){
	W[k] = &(Lpara[l]->b[id]);
	isReg[k] = false;
	for(int n=0; n<batch_size; ++n){
	  O[n][k] = &(Lpara[l]->Ob[n*Lpara[l]->Nb + id]);
	}
	k++;
      }
    }
  }  
}
void network_base::globalInitializer(){
  for(size_t l=0; l<Lprop.size(); ++l){
    Lprop[l]->initUnits();
  }
  for(size_t l=0; l<Lpara.size(); ++l){
    Lpara[l]->initLParams();
  }  
}
void network_base::forwardPropagation(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<batch_size; ++n){
    for(size_t l=0; l<Lprop.size(); ++l){
      Lprop[l]->fPropLayerBatch(n);
    }
  }
}
void network_base::debugmode(){
  int cnt = 0;
  while(cnt==0){
    cout << "Debug-Mode" << endl;  
    for(int jw=0; jw<InputLayer->unit_width; ++jw){
      for(int jh=0; jh<InputLayer->unit_height; ++jh){
	cout << "Input[" << jw << ',' << jh << "] : ";
	cin >> InputLayer->unit[0][0][jw][jh];
      }
    }
    for(size_t l=0; l<Lprop.size(); ++l){
      Lprop[l]->showUnit(0);
    }
    output(0);
    cout << "fin ? (continue:0, finish:1)" << endl;
    cin >> cnt;
  }
}
void network_base::backwardPropagation(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<batch_size; ++n){
    for(int l=Lprop.size()-1; l>0; --l){
      Lprop[l]->bPropLayerBatch(n);
    }
  }  
}
void network_base::save(const char* fname)const{
  for(size_t l=0; l<Lpara.size(); ++l){
    stringstream ss;
    ss << fname << format("_L%02d") %l;
    Lpara[l]->outLParams(ss.str().c_str());
  }
}
void network_base::load(const char* fname){
  for(size_t l=0; l<Lpara.size(); ++l){
    stringstream ss;
    ss << fname << format("_L%02d") %l;
    Lpara[l]->loadLParams(ss.str().c_str());
  }
}
double network_base::weightNorm(){
  double sum = 0.0;
#ifdef _OPENMP
#pragma omp parallel for  reduction(+:sum)
#endif
  for(int k=0; k<LParamTotal; ++k){
    sum += norm(*W[k]);
  }
  return sum;
}
void network_base::checkDiff(bool fineCheck){
  forwardPropagation();
  backwardPropagation();
  int n = omp_rand()%batch_size;
  odata_t psi, psi1, psi2, cdiff, fdiff;
  param_t hh = 1.0E-8;
  psi = output(n);
  for(int l=Lpara.size()-1; l>=0; --l){
    int ptotal[] = {Lpara[l]->Nw, Lpara[l]->Nb};
    param_t* W[] = {Lpara[l]->w,  Lpara[l]->b};
    odata_t* O[] = {Lpara[l]->Ow, Lpara[l]->Ob};
    double Error = 0.0, d;
    for(int para_type=0; para_type<2; ++para_type){
      for(int id=0; id<ptotal[para_type]; ++id){
	W[para_type][id] += hh;
	psi2 = output(n);
	W[para_type][id] -= 2.0*hh;
	psi1 = output(n);
	W[para_type][id] += hh;
	
	fdiff = (psi2 - psi1)/(2.0*psi*hh);
	cdiff = O[para_type][n*ptotal[para_type] + id];
      	d = abs(cdiff - fdiff);
	Error += d;
	
	if(fineCheck && d > abs(hh*100.0)){
	  cout << format(" chain-difference: %8.5f %8.5f") %cdiff.real() %cdiff.imag() << endl;
	  cout << format("finite-difference: %8.5f %8.5f") %fdiff.real() %fdiff.imag() << endl;
	  cout << format("Error : %.5e at L=%d ID=%d") %d %l %id<< endl;
	}
      }
    }
    Error /= (Lpara[l]->Nt!=0)? Lpara[l]->Nt : 1;
    if(Error > abs(hh*100.0)){
      cout << format("Error : %.5e at L=%d ") %Error %l
	   << Lpara[l]->tag << endl;
    }
  }
}

#endif
