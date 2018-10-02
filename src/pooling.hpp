#ifndef POOLING
#define POOLING

#include "layer.hpp"
struct pooling_stride : propagator{
  int pool_width, pool_height;
  int stride_width, stride_height;
  pooling_stride(int p_width, int p_height, int s_width, int s_s_height);
  virtual ~pooling_stride(){};
  virtual void setProperty(int b_size);
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
  virtual void showProperty()const;
};

pooling_stride::pooling_stride(int p_width, int p_height, int s_width, int s_height){
  tag = "pooling";
  pool_width = p_width, pool_height = p_height;
  stride_width = s_width, stride_height = s_height;
}
void pooling_stride::setProperty(int b_size){
  propagator::setProperty(b_size);
  unit_chan = in->unit_chan;
  unit_width = in->unit_width/stride_width;
  unit_height = in->unit_height/stride_height;
  fan_in = in->fan_out = (pool_width*pool_height)/(stride_width*stride_height); 
  if(in->unit_width%(pool_width*stride_width)!=0||
     in->unit_height%(pool_height*stride_height)!=0){
    cerr << "Error : Size is wrong in " << tag << endl;
  }
}

void pooling_stride::fPropLayerBatch(int n){
  for(int q=0; q<unit_chan; ++q){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	param_t sum = 0.0;
	for(int iw=jw*stride_width; iw<jw*stride_width + pool_width; ++iw){
	  int iiw = (iw + input_width)%input_width;
	  for(int ih=jh*stride_height; ih<jh*stride_height + pool_height; ++ih){
	    int iih = (ih + input_height)%input_height;
	    //sum += in->valu[n][q][iiw][iih];
	    sum += in->mval[n][q][iiw][iih];
	  }
	}
	// valu[n][q][jw][jh] = unit[n][q][jw][jh]
	//   = sum/(double)(pool_width*pool_height)*mask[q][jw][jh];
	unit[n][q][jw][jh] = sum/(double)(pool_width*pool_height);
	valu[n][q][jw][jh] = f(unit[n][q][jw][jh]);
	mval[n][q][jw][jh] = valu[n][q][jw][jh]*mask[q][jw][jh];
      }
    }
  }
}

void pooling_stride::bPropLayerBatch(int n){
  for(int p=0; p<input_chan; ++p){
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	odata_t sum = 0.0;
	for(int jw=iw/stride_width; iw-pool_width<jw*stride_width; --jw){
	  int jjw = (jw + unit_width)%unit_width;
	  for(int jh=ih/stride_height; ih-pool_height<jh*stride_height; --jh){
	    int jjh = (jh + unit_height)%unit_height;
	    sum += delta[n][p][jjw][jjh];
	  }
	}
	in->delta[n][p][iw][ih] = 
	  sum/(double)(pool_width*pool_height)*in->g(in->valu[n][p][iw][ih])*in->mask[p][iw][ih];
      }
    }
  }
}

void pooling_stride::showProperty()const{
  propagator::showProperty();
  cout << " Pooling Width : " << pool_width << endl;
  cout << " Pooling Height: " << pool_height << endl;
  cout << " Stride Width  : " << stride_width << endl;
  cout << " Stride Height : " << stride_height << endl;
}

struct pooling_periodic: propagator{
  int pool_width, pool_height;
  pooling_periodic(int p_width, int p_height);
  virtual ~pooling_periodic(){};
  virtual void setProperty(int b_size);
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
  virtual void showProperty()const;
};

pooling_periodic::pooling_periodic(int p_width, int p_height){
  tag = "pooling periodic";
  pool_width = p_width;
  pool_height = p_height;
}

void pooling_periodic::setProperty(int b_size){
  propagator::setProperty(b_size);
  unit_chan = in->unit_chan;
  unit_width = in->unit_width/pool_width;
  unit_height = in->unit_height/pool_height;
  fan_in = pool_width*pool_height;
  in->fan_out = 1;
  if(in->unit_width%pool_width!=0||
     in->unit_height%pool_height!=0){
    cerr << "Error : Size is wrong in " << tag << endl;
  }
}
void pooling_periodic::fPropLayerBatch(int n){
  for(int q=0; q<unit_chan; ++q){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	param_t sum = 0.0;
	// for(int iw=0; iw<input_width; ++iw){
	//   for(int ih=0; ih<input_height; ++ih){
	for(int iw=0; iw<pool_width; ++iw){
	  for(int ih=0; ih<pool_height; ++ih){
	    //sum += in->valu[n][q][jw+unit_width*iw][jh+unit_height*ih];
	    sum += in->mval[n][q][jw+unit_width*iw][jh+unit_height*ih];
	  }
	}
	//valu[n][q][jw][jh] = unit[n][q][jw][jh] = sum/(double)(pool_width*pool_height);
	unit[n][q][jw][jh] = sum/(double)(pool_width*pool_height);
	valu[n][q][jw][jh] = f(unit[n][q][jw][jh]);
	mval[n][q][jw][jh] = valu[n][q][jw][jh]*mask[q][jw][jh];
      }
    }
  }
}

void pooling_periodic::bPropLayerBatch(int n){
  for(int p=0; p<input_chan; ++p){
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	in->delta[n][p][iw][ih]
	  = delta[n][p][iw%unit_width][ih%unit_height]/(double)(pool_width*pool_height)
	  *in->g(in->valu[n][p][iw][ih])*in->mask[p][iw][ih];
      }
    }
  }
}

void pooling_periodic::showProperty()const{
  propagator::showProperty();
  cout << " Pooling Width : " << pool_width << endl;
  cout << " Pooling Height: " << pool_height<< endl;
}


#endif
