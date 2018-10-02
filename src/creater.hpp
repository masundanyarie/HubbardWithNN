#ifndef CREATER
#define CREATER
#include "qnn.hpp"

void print(const network_base& nn){
  for(size_t l=0; l<nn.Lprop.size(); ++l){
    nn.Lprop[l]->showProperty();
    cout << endl;
  }
  cout << " Total Parameter " << nn.LParamTotal << endl;
}
void syncInput(const network_base &A, network_base &B){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<A.batch_size; ++n){
    for(int q=0; q<A.InputLayer->unit_chan; ++q){
      for(int jw=0; jw<A.InputLayer->unit_width; ++jw){
	for(int jh=0; jh<A.InputLayer->unit_height; ++jh){
	  B.InputLayer->unit[n][q][jw][jh] = A.InputLayer->unit[n][q][jw][jh];
	}
      }
    }
  }
}

network_base& operator<<(network_base& nn, layer_base& l){
  nn.Lprop.push_back(&l);
  nn.Lpara.push_back(&l);
  return nn;
}
network_base& operator<<(network_base& nn, propagator& pg){
  nn.Lprop.push_back(&pg);
  return nn;
}

#ifdef INPUT_LAYER
template<class actfunc_t>
input_layer& inputLayer(int u_chan, int width, int height=1){
  input_layer* il = new input_layer(u_chan, width, height);
  il->f = actfunc_t::f;
  il->g = actfunc_t::g;
  return *il;
}
#endif

#ifdef OUTPUT_LAYER
void operator<<(network_base& nn, output_layer& ol){
  nn.Lprop.push_back(&ol);
  nn.OutputLayer = &ol;
  nn.setupNetwork();
}
template<class outlay_t>
outlay_t& outputLayer(int o_size){
  outlay_t* of = new outlay_t(o_size);
  return *of;
}
#endif

#ifdef SUM_LAYER
sum_layer& sumUnitLayer(){
  sum_layer* sl = new sum_layer();
  sl->f = af_eqal::f;
  sl->g = af_eqal::g;
  return *sl;
}
#endif

#ifdef POOLING
pooling_stride& poolingLayer(int p_width, int p_height=1, int s_width=1, int s_height=1){
  pooling_stride* pl = new pooling_stride(p_width, p_height, s_width, s_height);
  pl->f = af_eqal::f;
  pl->g = af_eqal::g;
  return *pl;
}
pooling_periodic& poolingPeriodicLayer(int p_width, int p_height=1){
  pooling_periodic* pl = new pooling_periodic(p_width, p_height);
  pl->f = af_eqal::f;
  pl->g = af_eqal::g;
  return *pl;
}
#endif

#ifdef FULL_CONECTION
template<class actfunc_t>
full_conection& fullConectionLayer(int width, int height=1){
  full_conection* fc = new full_conection(width, height);
  fc->f = actfunc_t::f;
  fc->g = actfunc_t::g;
  return *fc;
}
template<class actfunc_t>
full_conection_nobias& fullConectionNoBiasLayer(int width, int height=1){
  full_conection_nobias* fc = new full_conection_nobias(width, height);
  fc->f = actfunc_t::f;
  fc->g = actfunc_t::g;
  return *fc;
}
#endif

#ifdef CONVOLUTION
template<class actfunc_t>
convolution& convolutionLayer(int u_chan, int f_width, int f_height=1){
  convolution* cnn = new convolution(u_chan, f_width, f_height);
  cnn->f = actfunc_t::f;
  cnn->g = actfunc_t::g;
  return *cnn;
}
#endif

#ifdef GUTZWILLER
gutzwiller& gutzwillerLayer(int total_atoms, int max_atoms, int width, int height=1, 
			    const char* name= ""){
  gutzwiller* gw = new gutzwiller(total_atoms, max_atoms, width, height);
  gw->f = af_eqal::f;
  gw->g = af_eqal::g;
  gw->fname = name;
  return *gw;
}
network_base& operator<<(network_base& nn, gutzwiller& gw){
  gw.state = nn.Lprop[0];
  nn.Lprop.push_back(&gw); 
  nn.Lpara.push_back(&gw);
  return nn;
}
#endif

#endif
