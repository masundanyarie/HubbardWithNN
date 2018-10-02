#ifndef CONVOLUTION
#define CONVOLUTION

#include "layer.hpp"

struct convolution : layer_base{
  int filt_width, filt_height;
  param_t * * **filter, *bias;
  odata_t * * * **pfilter, * *pbias;
  
  convolution(int u_chan, int f_width, int f_height);
  virtual ~convolution();
  virtual void setMaskForTrain();
  virtual void setMaskForPred();
  virtual void setProperty(int _b_size);
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
  virtual void allocLParams();
  virtual void showLParams()const;
  virtual void showProperty()const;
  virtual bool outLParams(const char* fname)const;
  virtual bool loadLParams(const char* fname);
};

convolution::convolution(int u_chan, int f_width, int f_height){
  tag="convolution";
  unit_chan = u_chan;
  filt_width = f_width;
  filt_height = f_height;
}
convolution::~convolution(){
  free(filter);
  free(pfilter);
  free(bias);
  free(pbias);
}
void convolution::setMaskForTrain(){
  if(isDropout){
    for(int q=0; q<unit_chan; ++q){
      mask[q][0][0] = (omp_uniform()>dropRate)? 1.0 : 0.0;
      for(int jw=1; jw<unit_width; ++jw){
	for(int jh=1; jh<unit_height; ++jh){
	  mask[q][jw][jh] = mask[q][0][0];
	}
      }
    }
  }
}
void convolution::setMaskForPred(){
  if(isDropout){
    for(int q=0; q<unit_chan; ++q){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  mask[q][jw][jh] = 1.0 - dropRate;
	}
      }
    }
  }
}
void convolution::allocLParams(){
  Nw = unit_chan*input_chan*filt_width*filt_height;
  Nb = unit_chan;
  Nt = Nw + Nb;
  w = ***(filter = alloc<param_t>(unit_chan, input_chan, filt_width, filt_height));
  b = bias = alloc<param_t>(unit_chan);
  Ow = ****(pfilter = alloc<odata_t>(batch_size, unit_chan, input_chan, filt_width, filt_height));
  Ob = *(pbias = alloc<odata_t>(batch_size, unit_chan));
}

void convolution::showLParams()const{
  for(int q=0; q<unit_chan; ++q){
    for(int p=0; p<input_chan; ++p){
      for(int fw=0; fw<filt_width; ++fw){
	for(int fh=0; fh<filt_height; ++fh){
	  cout << filter[q][p][fw][fh] << ' ';
	}
	cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;
  for(int q=0; q<unit_chan; ++q){
    cout << bias[q] << ' ';
  }
  cout << endl;
}
void convolution::showProperty()const{
  propagator::showProperty();
  cout << " Filter Width  : " << filt_width << endl;
  cout << " Filter Height : " << filt_height << endl;
}

bool convolution::outLParams(const char* fname)const{
  stringstream ssw;
  ssw << fname << ".cnn";
  ofstream ofsw(ssw.str().c_str());
  if(!ofsw){
    cout << "cannot open " << ssw.str() <<endl;
    return false;
  }else{
    for(int q=0; q<unit_chan; ++q){
      for(int p=0; p<input_chan; ++p){
	for(int fw=0; fw<filt_width; ++fw){
	  for(int fh=0; fh<filt_height; ++fh){
	  ofsw << q << ' ' << p << ' ' 
	       << fw<< ' ' << fh<< ' ' <<filter[q][p][fw][fh] << endl;
	  }
	  ofsw << endl;
	}
	ofsw << endl;
      }
      ofsw << endl;
    }
  }
  
  stringstream ssb;
  ssb << fname << ".bi";
  ofstream ofsb(ssb.str().c_str());
  if(!ofsb){
    cout << "cannot open " << ssb.str() <<endl;
    return false;
  }else{
    for(int q=0; q<unit_chan; ++q){
      ofsb << q << ' ' << bias[q] << endl;
    }
  }
  return true;
}
bool convolution::loadLParams(const char* fname){
  stringstream ssw;
  ssw << fname << ".cnn";
  ifstream ifsw(ssw.str().c_str());
  if(!ifsw){
    cout << "cannot open " << ssw.str() <<endl;
    return false;
  }else{
    int dq, dp, dfw, dfh;
    for(int q=0; q<unit_chan; ++q){
      for(int p=0; p<input_chan; ++p){
	for(int fw=0; fw<filt_width; ++fw){
	  for(int fh=0; fh<filt_height; ++fh){
	    ifsw >> dq >> dp >> dfw >> dfh >> filter[q][p][fw][fh];
	    if(q!=dq || p!=dp || fw!=dfw|| fh!=dfh){
	      cout << "layer size is wrong : " << ssw.str() << endl;;
	      return false;
	    }
	  }
	}
      }
    }
  }
  
  stringstream ssb;
  ssb << fname << ".bi";
  ifstream ifsb(ssb.str().c_str());
  if(!ifsb){
    cout << "cannot open " << ssb.str() <<endl;
    return false;
  }else{
    int dq;
    for(int q=0; q<unit_chan; ++q){
      ifsb >> dq >> bias[q];
    }
  }
  return true;
}

void convolution::setProperty(int b_size){
  layer_base::setProperty(b_size);
  unit_width = in->unit_width;
  unit_height = in->unit_height;
  fan_in = input_chan*filt_width*filt_height;
  in->fan_out = unit_chan*filt_width*filt_height;
}
void convolution::fPropLayerBatch(int n){
  for(int q=0; q<unit_chan; ++q){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	param_t sum = 0.0;
	for(int p=0; p<input_chan; ++p){
	  for(int fw=0; fw<filt_width; ++fw){
	    for(int fh=0; fh<filt_height; ++fh){
	      int iw = (jw+fw)%input_width;
	      int ih = (jh+fh)%input_height;
	      sum += filter[q][p][fw][fh]*in->mval[n][p][iw][ih];
	    }
	  }
	}
	unit[n][q][jw][jh] = sum + bias[q];
	valu[n][q][jw][jh] = f(unit[n][q][jw][jh]);
	mval[n][q][jw][jh] = valu[n][q][jw][jh]*mask[q][jw][jh];
      }
    }
  }  
}
void convolution::bPropLayerBatch(int n){
  for(int p=0; p<input_chan; ++p){
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	odata_t sum = 0.0;
	for(int q=0; q<unit_chan; ++q){
	  for(int jw=iw; jw>iw-filt_width; --jw){
	    int jjw = (jw + unit_width)%unit_width;
	    for(int jh=ih; jh>ih-filt_height; --jh){
	      int jjh = (jh + unit_height)%unit_height;
	      sum += delta[n][q][jjw][jjh]*filter[q][p][iw-jw][ih-jh];
	    }
	  }
	}
	in->delta[n][p][iw][ih] = sum*in->g(in->valu[n][p][iw][ih])*in->mask[p][iw][ih];
      }
    }
  }
  for(int q=0; q<unit_chan; ++q)
    for(int p=0; p<input_chan; ++p)
      for(int fw=0; fw<filt_width; ++fw){
	for(int fh=0; fh<filt_height; ++fh){
	  odata_t sum = 0.0;
	  for(int jw=0; jw<unit_width; ++jw){
	    for(int jh=0; jh<unit_height; ++jh){
	      int iw = (jw+fw)%input_width;
	      int ih = (jh+fh)%input_height;
	      sum += delta[n][q][jw][jh]*in->mval[n][p][iw][ih];
	    }
	  }
	  pfilter[n][q][p][fw][fh] = sum;
	}
      }
  for(int q=0; q<unit_chan; ++q){
    odata_t sum = 0.0;
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	sum += delta[n][q][jw][jh];
      }
    }
    pbias[n][q] = sum;
  }
}

#endif
