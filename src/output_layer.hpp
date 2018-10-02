#ifndef OUTPUT_LAYER
#define OUTPUT_LAYER

#include "layer.hpp"

interface output_layer : propagator{
  int output_size;
  odata_t** phi;
  output_layer(int o_size){
    tag="output";
    phi = NULL;
    output_size = o_size;
  }
  virtual ~output_layer(){}
  virtual void allocUnits(){
    phi = alloc<odata_t>(batch_size, output_size);
  }
  virtual void initUnits(){
    for(int n=0; n<batch_size; ++n)
      for(int j=0; j<output_size; ++j){
	phi[n][j] = 0.0;
      }
  }
};

#ifdef USE_DOUBLE_PARAMETER
struct of_exp : output_layer{
  of_exp(int o_size):output_layer(o_size){};
  void fPropLayerBatch(int n){
    odata_t ans = in->mval[n][0][0][0] + AI*in->mval[n][0][1][0];
    phi[n][0] = exp(ans);
  }
  void bPropLayerBatch(int n){
    for(int j=0; j<input_width; ++j){
      double re = (j==0)? 1.0 : 0.0;
      double im = (j==1)? 1.0 : 0.0;
      in->delta[n][0][j][0] = (re + AI*im)*in->g(in->valu[n][0][j][0])*in->mask[0][j][0];
    }
  }
  void setProperty(int b_size){
    output_layer::setProperty(b_size);
    if(input_width !=2){
      cerr << "Error : Input width in Output Layer." << endl;
    }
    if(input_height!=1){
      cerr << "Error : Input height in Output Layer." << endl;
    }
    if(input_chan  !=1){
      cerr << "Error : Input channel in Output Layer." << endl;
    }
  }
};
struct of_aib: output_layer{
  of_aib(int o_size):output_layer(o_size){};
  void fPropLayerBatch(int n){
    phi[n][0] = in->mval[n][0][0][0] + AI*in->mval[n][0][1][0];
  }
  void bPropLayerBatch(int n){
    for(int j=0; j<input_width; ++j){
      double re = (j==0)? 1.0 : 0.0;
      double im = (j==1)? 1.0 : 0.0;
      odata_t dphi = re + AI*im;
      if(norm(phi[n][0])>1.0e-8)
	in->delta[n][0][j][0] = dphi/phi[n][0]*in->g(in->valu[n][0][j][0])*in->mask[0][j][0];
      else
	in->delta[n][0][j][0] = 0;
    }
  }
  void setProperty(int b_size){
    output_layer::setProperty(b_size);
    if(input_width !=2){
      cerr << "Error : Input width in Output Layer." << endl;
    }
    if(input_height!=1){
      cerr << "Error : Input height in Output Layer." << endl;
    }
    if(input_chan  !=1){
      cerr << "Error : Input channel in Output Layer." << endl;
    }
  }
};
struct of_sum : output_layer{
  of_sum(int o_size):output_layer(o_size){};
  void fPropLayerBatch(int n){
    odata_t sum = 0, ans;
    for(int j=0; j < input_width/2; ++j){
      ans = in->mval[n][0][2*j][0] + AI*in->mval[n][0][2*j + 1][0];
      sum += exp(ans);
    }
    phi[n][0] = sum;
  }
  void bPropLayerBatch(int n){
    odata_t dphi;
    for(int j=0; j< input_width; ++j){
      if(j%2==0){
	dphi = exp(in->mval[n][0][j][0] + AI*in->mval[n][0][j+1][0]);
      }else{
	dphi = AI*exp(in->mval[n][0][j-1][0] + AI*in->mval[n][0][j][0]);
      }   
      in->delta[n][0][j][0] = dphi/phi[n][0]*in->g(in->valu[n][0][j][0])*in->mask[0][j][0];
     }
  }
  void setProperty(int b_size){
    output_layer::setProperty(b_size);
    if(input_width%2!=0){
      cerr << "Error : Input width in Output Layer." << endl;
    }
    if(input_height!=1){
      cerr << "Error : Input height in Output Layer." << endl;
    }
    if(input_chan  !=1){
      cerr << "Error : Input channel in Output Layer." << endl;
    }
  }
};
#endif

#ifdef USE_COMPLEX_PARAMETER
struct of_exp : output_layer{
  of_exp(int o_size):output_layer(o_size){};
  void fPropLayerBatch(int n){
    phi[n][0] = exp(in->mval[n][0][0][0]);
  }
  void bPropLayerBatch(int n){
    in->delta[n][0][0][0] = in->g(in->valu[n][0][0][0])*in->mask[0][0][0];
  }
  void setProperty(int b_size){
    output_layer::setProperty(b_size);
    if(input_width !=1){
      cerr << "Error : Input width in Output Layer." << endl;
    }
    if(input_height!=1){
      cerr << "Error : Input height in Output Layer." << endl;
    }
    if(input_chan  !=1){
      cerr << "Error : Input channel in Output Layer." << endl;
    }
  }
};
struct of_aib: output_layer{
  of_aib(int o_size):output_layer(o_size){};
  void fPropLayerBatch(int n){
    phi[n][0] = in->mval[n][0][0][0];
  }
  void bPropLayerBatch(int n){
    in->delta[n][0][0][0] = in->g(in->valu[n][0][0][0])/phi[n][0]*in->mask[0][0][0];
  }
  void setProperty(int b_size){
    output_layer::setProperty(b_size);
    if(input_width !=1){
      cerr << "Error : Input width in Output Layer." << endl;
    }
    if(input_height!=1){
      cerr << "Error : Input height in Output Layer." << endl;
    }
    if(input_chan  !=1){
      cerr << "Error : Input channel in Output Layer." << endl;
    }
  }
};
struct of_sum : output_layer{
  of_sum(int o_size):output_layer(o_size){};
  void fPropLayerBatch(int n){
    odata_t sum = 0, ans;
    for(int j=0; j < input_width; ++j){
      ans = in->mval[n][0][j][0];
      sum += exp(ans);
    }
    phi[n][0] = sum;
  }
  void bPropLayerBatch(int n){
    odata_t dphi;
    for(int j=0; j< input_width; ++j){
      dphi = exp(in->mval[n][0][j][0]);
      in->delta[n][0][j][0] = dphi/phi[n][0]*in->g(in->valu[n][0][j][0])*in->mask[0][j][0];
    }
  }
  void setProperty(int b_size){
    output_layer::setProperty(b_size);
    if(input_height!=1){
      cerr << "Error : Input height in Output Layer." << endl;
    }
    if(input_chan  !=1){
      cerr << "Error : Input channel in Output Layer." << endl;
    }
  }

};
#endif

#endif
