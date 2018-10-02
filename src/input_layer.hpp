#ifndef INPUT_LAYER
#define INPUT_LAYER

#include "layer.hpp"

struct input_layer : propagator{
  input_layer(int u_chan, int width, int height){
    tag="input";
    unit_chan = u_chan;
    unit_width = width;
    unit_height = height;
  }
  virtual ~input_layer(){}
  virtual void setProperty(int b_size){
    in = NULL;
    batch_size = b_size;
    input_chan = 0;
    input_width = 0;
    input_height = 0;
    fan_in = fan_out = 0;
  }
  virtual void fPropLayerBatch(int n){
    for(int q=0; q<unit_chan; ++q){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  valu[n][q][jw][jh] = f(unit[n][q][jw][jh]);
	  mval[n][q][jw][jh] = valu[n][q][jw][jh]*mask[q][jw][jh];
	}
      }
    }
  }
  virtual void bPropLayerBatch(int n){
    cout << "Error: Called Input_laer::bPropLayer" << endl;
  }
};

#endif
