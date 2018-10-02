#ifndef GUTZWILLER
#define GUTZWILLER

#include "layer.hpp"

struct gutzwiller : layer_base{
  string fname;
  int atoms, sitecap, site_width, site_height, max_per_site;
  propagator* state;
  param_t * **weight;
  odata_t * * **pweight;
  gutzwiller(int total_atoms, int max_per_site, int width, int height);
  virtual ~gutzwiller();
  void initLParams();
  virtual void setProperty(int b_size);
  virtual void fPropLayerBatch(int n);
  virtual void bPropLayerBatch(int n);
  virtual void allocLParams();
  virtual void showLParams()const;
  virtual bool outLParams(const char* fname)const;
  virtual bool loadLParams(const char* fname);
  virtual void showProperty()const;
  virtual gutzwiller& dropout(double rate);
};

gutzwiller::gutzwiller(int total_atoms, int m_p_site, int width, int height){
  tag="Gutzwiller";
  isreg = false;
  max_per_site = m_p_site;
  atoms = total_atoms, sitecap = m_p_site + 1;
  site_width = width, site_height = height;
}
gutzwiller::~gutzwiller(){
  free(weight); free(pweight);
}

void gutzwiller::initLParams(){
  if(fname.length() == 0){
    for(int n=0; n<sitecap; ++n){
      for(int jw=0; jw<unit_width; ++jw){
	weight[n][jw][0] = 0.0;
      }
    }
  }else{
    ifstream ifs(fname.c_str());
    if(!ifs){
      cout << "cannot open " << fname <<endl;
    }else{
      int dn, djw;
      for(int n=0; n<sitecap; ++n){
	for(int jw=0; jw<unit_width; ++jw){
	  ifs >> dn >> djw >> weight[n][jw][0];
	}
      }
    } 
  }
}
void gutzwiller::allocLParams(){
  Nw = sitecap*unit_width*unit_height;
  Nb = 0;
  Nt = Nw + Nb;
  w = **(weight = alloc<param_t>(sitecap, unit_width, unit_height));
  Ow = ***(pweight = alloc<odata_t>(batch_size, sitecap, unit_width, unit_height));
}
void gutzwiller::showLParams()const{
  for(int n=0; n<sitecap; ++n){
    for(int jw=0; jw<unit_width; ++jw){
      cout << weight[n][jw][0] << ' ';
    }
    cout << endl;
  }
}
bool gutzwiller::outLParams(const char* fname)const{
  stringstream ssw;
  ssw << fname << ".gw";
  ofstream ofsw(ssw.str().c_str());
  if(!ofsw){
    cout << "cannot open " << ssw.str() << endl;
    return false;
  }else{
    for(int n=0; n<sitecap; ++n){
      for(int jw=0; jw<unit_width; ++jw){
	ofsw << n << ' ' << jw << ' ' << weight[n][jw][0] << endl;
      }
      ofsw << endl;
    }
  }
  return true;
}
bool gutzwiller::loadLParams(const char* fname){
  stringstream ss;
  ss << fname << ".gw";
  ifstream ifsw(ss.str().c_str());
  if(!ifsw){
    cout << "cannot open " << ss.str() <<endl;
    return false;
  }else{
    int dn, djw, djh;
    for(int n=0; n<sitecap; ++n){
      for(int jw=0; jw<unit_width; ++jw){
	ifsw >> dn >> djw >> djh >> weight[n][jw][0];
      }
    }
  }
  return true;
}
void gutzwiller::setProperty(int b_size){
  layer_base::setProperty(b_size);
  unit_chan = input_chan;
  unit_width = in->unit_width;
  unit_height = in->unit_height;
  if(in->unit_height!=1||in->unit_height!=1){
    cerr << "Error : Size is wrong in " << tag << endl;
  }
}
void gutzwiller::fPropLayerBatch(int n){
  int n_i;
  for(int jw=0; jw<unit_width; ++jw){
    param_t sum = 0.0;
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	n_i = (int)abs(state->unit[n][0][iw][ih] + atoms/(double)site_width*site_height + 0.5);
	sum += (n_i<sitecap)? weight[n_i][jw][0] : weight[sitecap-1][jw][0];
      }
      unit[n][0][jw][0] = in->mval[n][0][jw][0] + sum;
      valu[n][0][jw][0] = f(unit[n][0][jw][0]);
      mval[n][0][jw][0] = valu[n][0][jw][0]*mask[0][jw][0];
    }
  }
}
void gutzwiller::bPropLayerBatch(int n){
  for(int iw=0; iw<input_width; ++iw){
    for(int ih=0; ih<input_height; ++ih){
      in->delta[n][0][iw][ih] = delta[n][0][iw][ih]*
      in->g(in->valu[n][0][iw][ih])*in->mask[0][iw][ih];
    }
  }
  for(int m=0; m<sitecap; ++m){
    for(int jw=0; jw<unit_width; ++jw){
      Complex sum = 0.0;
      for(int iw=0; iw<site_width; ++iw){
	for(int ih=0; ih<site_height; ++ih){
	  int n_i= (int)abs(state->unit[n][0][iw][ih] + atoms/(double)site_width*site_height + 0.5);
	  if(m == n_i || (m == max_per_site && n_i>max_per_site)){
	    sum += 1.0;
	  }
	}
      }
      pweight[n][m][jw][0] = sum*delta[n][0][jw][0];
    }
  }
}
void gutzwiller::showProperty()const{
  cout << " Layer Type    : " << tag << endl;
  cout << " Total atoms   : " << atoms << endl;
  cout << "  Site Width   : " << site_width << endl;
  cout << "  Site Height  : " << site_height << endl;
  cout << " Nmax per site : " << sitecap-1 << endl;
  if(isDropout){
    cout << "   Dropout     : " << dropRate  << endl;
  }
}

gutzwiller& gutzwiller::dropout(double rate){
  isDropout = true;
  dropRate = rate;
  return *this;
}
#endif
