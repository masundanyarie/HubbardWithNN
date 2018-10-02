#ifndef SUM_LAYER
#define SUM_LAYER

#include "layer.hpp"

struct sum_layer : propagator{
  sum_layer();
  virtual ~sum_layer(){};
  virtual void setProperty(int b_size);
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
};

sum_layer::sum_layer(){
  tag = "sum_layer";
  unit_chan = 1;
}

void sum_layer::setProperty(int b_size){
  propagator::setProperty(b_size);
  unit_width = in->unit_chan;
  unit_height = 1;
  fan_in = input_width*input_height;
  in->fan_out = 1;
}

void sum_layer::fPropLayerBatch(int n){
  for(int j=0; j<unit_width; ++j){
    param_t sum = 0.0;
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	sum += in->mval[n][j][iw][ih];
      }
    }
    unit[n][0][j][0] = sum;
    valu[n][0][j][0] = f(unit[n][0][j][0]);
    mval[n][0][j][0] = valu[n][0][j][0]*mask[0][j][0];
  }
}

void sum_layer::bPropLayerBatch(int n){
  for(int p=0; p<input_chan; ++p)
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	in->delta[n][p][iw][ih] = 
	  delta[n][0][p][0]*in->g(in->valu[n][p][iw][ih])*in->mask[p][iw][ih];
      }
    }
}

#endif
