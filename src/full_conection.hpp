#ifndef FULL_CONECTION
#define FULL_CONECTION

#include "layer.hpp"

struct full_conection : layer_base{
  param_t * ** **weight, **bias;
  odata_t * * ** **pweight, * **pbias;
  full_conection(int width, int height);
  virtual ~full_conection();
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
  virtual void allocLParams();
  virtual void showLParams()const;
  virtual bool outLParams(const char* fname)const;
  virtual bool loadLParams(const char* fname);
};

full_conection::full_conection(int width, int height){
  tag="full conection";
  unit_chan = 1;
  unit_width = width;
  unit_height = height;
}
full_conection::~full_conection(){
  free(weight);
  free(pweight);
  free(bias);
  free(pbias); 
}

void full_conection::allocLParams(){
  Nw = input_chan*unit_width*unit_height*input_width*input_height;
  Nb = unit_width*unit_height;
  Nt = Nw + Nb;
 
  w = ****(weight = alloc<param_t>(input_chan, unit_width, unit_height, input_width, input_height));
  b = *(bias = alloc<param_t>(unit_width, unit_height));
  
  Ow = *****(pweight = alloc<odata_t>(batch_size, input_chan,
				      unit_width, unit_height, input_width, input_height));
  Ob = **(pbias = alloc<odata_t>(batch_size, unit_width, unit_height));
}
void full_conection::showLParams()const{
  for(int p=0; p<input_chan; ++p){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	for(int iw=0; iw<input_width; ++iw){
	  for(int ih=0; ih<input_height; ++ih){
	    cout << weight[p][jw][jh][iw][ih] << ' ';
	  }
	  cout << endl;
	}
	cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      cout << bias[jw][jh] << ' ';
    }
    cout << endl;
  }
  cout << endl;
}
bool full_conection::outLParams(const char* fname)const{
  stringstream ssw;
  ssw << fname << ".fcn";
  ofstream ofsw(ssw.str().c_str());
  if(!ofsw){
    cout << "cannot open " << ssw.str() <<endl;
    return false;
  }else{
    for(int p=0; p<input_chan; ++p){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  for(int iw=0; iw<input_width; ++iw){
	    for(int ih=0; ih<input_height; ++ih){
	      ofsw << p << ' ' 
		   << jw << ' ' << jh << ' ' 
		   << iw << ' ' << ih << ' ' 
		   << weight[p][jw][jh][iw][ih] << endl;
	    }
	    ofsw << endl;
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
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	ofsb << jw << ' ' << jh << ' '<< bias[jw][jh] << endl;
      }
      ofsb << endl;
    }
  }
  return true;
}
bool full_conection::loadLParams(const char* fname){
  stringstream ssw;
  ssw << fname << ".fcn";
  ifstream ifsw(ssw.str().c_str());
  if(!ifsw){
    cout << "cannot open " << ssw.str() <<endl;
    return false;
  }else{
    int dp, djw, djh, diw, dih;
    for(int p=0; p<input_chan; ++p){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  for(int iw=0; iw<input_width; ++iw){
	    for(int ih=0; ih<input_height; ++ih){
	      ifsw >> dp >> djw >> djh >> diw >> dih >> weight[p][jw][jh][iw][ih];
	      if(p!=dp || jw!=djw || jh!=djh || iw!= diw || ih!= dih){
		cout << "layer size is wrong : " << ssw.str() << endl;;
		return false;
	      }
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
    int dj;
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	ifsb >> dj >> bias[jw][jh];
      }
    }
  }
  return true;
}
void full_conection::fPropLayerBatch(int n){
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      param_t sum = 0.0;
      for(int p=0; p<input_chan; ++p){
	for(int iw=0; iw<input_width; ++iw){
	  for(int ih=0; ih<input_height; ++ih){
	    sum += weight[p][jw][jh][iw][ih]*in->mval[n][p][iw][ih];
	  }
	}
      }
      unit[n][0][jw][jh] = sum + bias[jw][jh];
      valu[n][0][jw][jh] = f(unit[n][0][jw][jh]);
      mval[n][0][jw][jh] = valu[n][0][jw][jh]*mask[0][jw][jh];
    }
  }

}
void full_conection::bPropLayerBatch(int n){
  for(int p=0; p<input_chan; ++p)
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	odata_t sum = 0.0;
	for(int jw=0; jw<unit_width; ++jw){
	  for(int jh=0; jh<unit_height; ++jh){
	    sum += delta[n][0][jw][jh]*weight[p][jw][jh][iw][ih];
	  }
	}
	in->delta[n][p][iw][ih] = sum*in->g(in->valu[n][p][iw][ih])*in->mask[p][iw][ih];
      }
    }
  for(int p=0; p<input_chan; ++p)
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	for(int iw=0; iw<input_width; ++iw){
	  for(int ih=0; ih<input_height; ++ih){
	    pweight[n][p][jw][jh][iw][ih] = delta[n][0][jw][jh]*in->mval[n][p][iw][ih];
	  }
	}
      }
    }
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      pbias[n][jw][jh] = delta[n][0][jw][jh];
    }
  }
}

struct full_conection_nobias : layer_base{
  param_t * ** **weight;
  odata_t * * ** **pweight;
  full_conection_nobias(int width, int height);
  virtual ~full_conection_nobias();
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
  virtual void allocLParams();
  virtual void showLParams()const;
  virtual bool outLParams(const char* fname)const;
  virtual bool loadLParams(const char* fname);
};

full_conection_nobias::full_conection_nobias(int width, int height){
  tag="full conection";
  unit_chan = 1;
  unit_width = width;
  unit_height = height;
}
full_conection_nobias::~full_conection_nobias(){
  free(weight);
  free(pweight);
}

void full_conection_nobias::allocLParams(){
  Nw = input_chan*unit_width*unit_height*input_width*input_height;
  Nb = 0;
  Nt = Nw + Nb;
  w = ****(weight = alloc<param_t>(input_chan, unit_width, unit_height, input_width, input_height));
  Ow = *****(pweight = alloc<odata_t>(batch_size, input_chan,
				      unit_width, unit_height, input_width, input_height));
}
void full_conection_nobias::showLParams()const{
  for(int p=0; p<input_chan; ++p){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	for(int iw=0; iw<input_width; ++iw){
	  for(int ih=0; ih<input_height; ++ih){
	    cout << weight[p][jw][jh][iw][ih] << ' ';
	  }
	  cout << endl;
	}
	cout << endl;
      }
      cout << endl;
    }
    cout << endl;
  }
  cout << endl;
}
bool full_conection_nobias::outLParams(const char* fname)const{
  stringstream ssw;
  ssw << fname << ".fcn";
  ofstream ofsw(ssw.str().c_str());
  if(!ofsw){
    cout << "cannot open " << ssw.str() <<endl;
    return false;
  }else{
    for(int p=0; p<input_chan; ++p){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  for(int iw=0; iw<input_width; ++iw){
	    for(int ih=0; ih<input_height; ++ih){
	      ofsw << p << ' ' 
		   << jw << ' ' << jh << ' ' 
		   << iw << ' ' << ih << ' ' 
		   << weight[p][jw][jh][iw][ih] << endl;
	    }
	    ofsw << endl;
	  }
	  ofsw << endl;
	}
	ofsw << endl;
      }
      ofsw << endl;
    }
  }
  return true;
}
bool full_conection_nobias::loadLParams(const char* fname){
  stringstream ssw;
  ssw << fname << ".fcn";
  ifstream ifsw(ssw.str().c_str());
  if(!ifsw){
    cout << "cannot open " << ssw.str() <<endl;
    return false;
  }else{
    int dp, djw, djh, diw, dih;
    for(int p=0; p<input_chan; ++p){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  for(int iw=0; iw<input_width; ++iw){
	    for(int ih=0; ih<input_height; ++ih){
	      ifsw >> dp >> djw >> djh >> diw >> dih >> weight[p][jw][jh][iw][ih];
	      if(p!=dp || jw!=djw || jh!=djh || iw!= diw || ih!= dih){
		cout << "layer size is wrong : " << ssw.str() << endl;;
		return false;
	      }
	    }
	  }
	}
      }
    }
  }
  return true;
}
void full_conection_nobias::fPropLayerBatch(int n){
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      param_t sum = 0.0;
      for(int p=0; p<input_chan; ++p){
	for(int iw=0; iw<input_width; ++iw){
	  for(int ih=0; ih<input_height; ++ih){
	    sum += weight[p][jw][jh][iw][ih]*in->mval[n][p][iw][ih];
	  }
	}
      }
      unit[n][0][jw][jh] = sum;
      valu[n][0][jw][jh] = f(unit[n][0][jw][jh]);
      mval[n][0][jw][jh] = valu[n][0][jw][jh]*mask[0][jw][jh];
    }
  }
}
void full_conection_nobias::bPropLayerBatch(int n){
  for(int p=0; p<input_chan; ++p){
    for(int iw=0; iw<input_width; ++iw){
      for(int ih=0; ih<input_height; ++ih){
	odata_t sum = 0.0;
	for(int jw=0; jw<unit_width; ++jw){
	  for(int jh=0; jh<unit_height; ++jh){
	    sum += delta[n][0][jw][jh]*weight[p][jw][jh][iw][ih];
	  }
	}
	in->delta[n][p][iw][ih] = sum*in->g(in->valu[n][p][iw][ih])*in->mask[p][iw][ih];
      }
    }
  }
  for(int p=0; p<input_chan; ++p){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	for(int iw=0; iw<input_width; ++iw){
	  for(int ih=0; ih<input_height; ++ih){
	    pweight[n][p][jw][jh][iw][ih] = delta[n][0][jw][jh]*in->mval[n][p][iw][ih];
	  }
	}
      }
    }
  }
}

#endif
