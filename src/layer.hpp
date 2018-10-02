#ifndef LAYER_BASE
#define LAYER_BASE

#include "actfunc.hpp"

interface propagator{
  string tag;  
  propagator* in;
  Activation f,g;
  int batch_size;
  int input_height, input_width, input_chan;
  int unit_height, unit_width, unit_chan;
  int fan_in, fan_out;
  bool isDropout;
  double dropRate;
  double ***mask;  
  param_t ****unit, ****valu, ****mval;
  odata_t ****delta;
  
  propagator();
  virtual ~propagator();
  virtual void allocUnits();
  virtual void initUnits();
  virtual void setMaskForTrain();
  virtual void setMaskForPred();
  
  virtual propagator& dropout(double rate);
  virtual void setProperty(int b_size);
  virtual void fPropLayerBatch(int batch) = 0;
  virtual void bPropLayerBatch(int batch) = 0;
  
  virtual void showProperty()const;
  
  void showUnit(int n)const;
  void showValu(int n)const;
  void showDelta(int n)const;
  void showMask()const;
  
  void histUnit(const char* fname, int bins, int q, double h = 0.0)const;
  void histValu(const char* fname, int bins, int q, double h = 0.0)const;
  void histDelta(const char* fname, int bins, int q, double h = 0.0)const;
  
  param_t meanUnit(int q)const;
  param_t meanValu(int q)const;
  odata_t meanDelta(int q)const;
  param_t meanUnit(int q, int jw, int jh)const;
  param_t meanValu(int q, int jw, int jh)const;
  odata_t meanDelta(int q, int jw, int jh)const;

private:
  template<class T>
    void showUnitClass(T*** u)const;
  template<class T>
    void histUnitClass(T**** u, const char* fname, int bins, int q, double h = 0.0)const;
  template<class T>
    T meanUnitClass(T**** u, int q)const;
  template<class T>
    T meanUnitClass(T**** u, int q, int jw, int jh)const;
};

propagator::propagator(){
  in = NULL;
  isDropout = false;
  f = g = NULL;
  unit = valu = mval = NULL;
  mask = NULL;
  delta = NULL;
  dropRate= 0.0;
  fan_in = fan_out = 0;
}
propagator::~propagator(){
  free(unit);
  free(valu);
  free(mval);
  free(mask);
  free(delta);
}
void propagator::setMaskForTrain(){
  if(isDropout){
    for(int q=0; q<unit_chan; ++q)
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  mask[q][jw][jh] = (omp_uniform()>dropRate)? 1.0 : 0.0;
	}
      }
  }
}
void propagator::setMaskForPred(){
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
void propagator::setProperty(int b_size){
  batch_size = b_size;
  input_chan = in->unit_chan;
  input_height = in->unit_height;
  input_width = in->unit_width;
  fan_in = input_chan*input_width*input_height;
  in->fan_out = unit_chan*unit_width*unit_height;
}

void propagator::showProperty()const{
  cout << " Layer Type    : " << tag << endl;
  cout << " Input Channel : " << input_chan << endl;
  cout << " Input Width   : " << input_width << endl;
  cout << " Input Height  : " << input_height << endl;
  cout << "  Unit Channel : " << unit_chan  << endl;
  cout << "  Unit Width   : " << unit_width << endl;
  cout << "  Unit Height  : " << unit_height << endl;
  if(isDropout){
    cout << "   Dropout     : " << dropRate  << endl;
  }
}

template<class T>
void propagator::showUnitClass(T*** u)const{
  int djw = unit_width/10+1;
  int djh = unit_height/10+1;
  for(int q=0; q<unit_chan; ++q){
    T sum = 0;
    cout <<"CH" << q <<" :: ";
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	sum += u[q][jw][jh];
	if(jw%djw==0 && jh%djh==0)cout << format("%5.2f ") %u[q][jw][jh];
      }
    }
    cout << format(": %6.2f") %sum << endl;
  }
}
template<class T>
void propagator::histUnitClass(T**** u, const char* fname, int bins, int q, double h)const{
  double max, min, hh;
  int Count[unit_width][unit_height][bins+1];
  int Ave[bins+1];
  if(q>=unit_chan)return;
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      for(int k=0; k<=bins; ++k){
	Count[jw][jh][k] = 0;
      }
    }
  }
  for(int k=0; k<=bins; ++k){
    Ave[k] = 0;
  }
  if(h==0.0){
    max = u[0][q][0][0], min = u[0][q][0][0];
    for(int n=1; n<batch_size; ++n){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  if(u[n][q][jw][jh] > max) max = u[n][q][jw][jh];
	  if(u[n][q][jw][jh] < min) min = u[n][q][jw][jh];
	}
      }
    }
    hh = (max - min)/(double)bins;
  }else{
    max = bins/2.0*h, min = -bins/2.0*h, hh = h;
  }
  
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      for(int n=0; n<batch_size; ++n){
	int k = (int)((u[n][q][jw][jh] - min)/hh);
	if(0<=k&&k<=bins)Count[jw][jh][k]++;
      }
    }
  }
  for(int k=0; k<=bins; ++k){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	Ave[k] += Count[jw][jh][k];
      }
    }
    Ave[k] /= (double)unit_width*unit_height;
  }
  ofstream ofs(fname);
  ofs << "#" << tag << endl;
  ofs << "#" << "CH : " << q << endl;
  for(int k=0; k<=bins; ++k){
    ofs << (double)k*hh + min << ' ' << Ave[k] << ' ';
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	ofs << Count[jw][jh][k] << ' ';
      }
    }
    ofs << endl;
  }
}
template<>
void propagator::histUnitClass(Complex**** u,const char* fname, int bins, int q, double h)const{
  double max, min, hh;
  int Count[unit_width][unit_height][bins+1];
  int Ave[bins+1];
  if(q>=unit_chan)return;
  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      for(int k=0; k<=bins; ++k){
	Count[jw][jh][k] = 0;
      }
    }
  }
  for(int k=0; k<=bins; ++k){
    Ave[k] = 0;
  }
  if(h==0.0){
    max = abs(u[0][q][0][0]), min = abs(u[0][q][0][0]);
    for(int n=1; n<batch_size; ++n){
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  if(abs(u[n][q][jw][jh]) > max) max = abs(u[n][q][jw][jh]);
	  if(abs(u[n][q][jw][jh]) < min) min = abs(u[n][q][jw][jh]);
	}
      }
    }
    hh = (max - min)/(double)bins;
  }else{
    max = bins/2.0*h, min = -bins/2.0*h, hh = h;
  }

  for(int jw=0; jw<unit_width; ++jw){
    for(int jh=0; jh<unit_height; ++jh){
      for(int n=0; n<batch_size; ++n){
	int k = (int)((abs(u[n][q][jw][jh]) - min)/hh);
	if(0<=k&&k<=bins)Count[jw][jh][k]++;
      }
    }
  }
  for(int k=0; k<=bins; ++k){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	Ave[k] += Count[jw][jh][k];
      }
    }
    Ave[k] /= (double)unit_width*unit_height;
  }
  ofstream ofs(fname);
  ofs << "#" << tag << endl;
  ofs << "#" << "CH : " << q << endl;
  for(int k=0; k<=bins; ++k){
    ofs << (double)k*hh + min << ' ' << Ave[k] << ' ';
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	ofs << Count[jw][jh][k] << ' ';
      }
    }
    ofs << endl;
  }
}
template<class T> 
T propagator::meanUnitClass(T**** u, int q)const{
  T sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
  for(int n=0; n<batch_size; ++n){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	sum += u[n][q][jw][jh];
      }
    }
  }
  return sum/(double)(batch_size*unit_width*unit_height);
}
template<> 
Complex propagator::meanUnitClass(Complex**** u, int q)const{
  double re = 0, im = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:re, im)
#endif
  for(int n=0; n<batch_size; ++n){
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	re += u[n][q][jw][jh].real();
	im += u[n][q][jw][jh].imag();
      }
    }
  }
  return Complex(re, im)/(double)(batch_size*unit_width*unit_height);
}

template<class T> 
T propagator::meanUnitClass(T**** u, int q, int jw, int jh)const{
  T sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
  for(int n=0; n<batch_size; ++n){
    sum += u[n][q][jw][jh];
  }
  return sum/(double)(batch_size*unit_width*unit_height);
}
template<>
Complex propagator::meanUnitClass(Complex**** u, int q, int jw, int jh)const{
  double re = 0, im = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:re, im)
#endif
  for(int n=0; n<batch_size; ++n){
    re += u[n][q][jw][jh].real();
    im += u[n][q][jw][jh].imag();
  }
  return Complex(re, im)/(double)(batch_size*unit_width*unit_height);
}

void propagator::showUnit(int n)const{
  showUnitClass(unit[n]);
}
void propagator::showValu(int n)const{
  showUnitClass(valu[n]);
}
void propagator::showDelta(int n)const{
  showUnitClass(delta[n]);
}
void propagator::showMask()const{
  showUnitClass(mask);
}

void propagator::histUnit(const char* fname, int bins, int q, double h)const{
  histUnitClass(unit, fname, bins, q, h);
}
void propagator::histValu(const char* fname, int bins, int q, double h)const{
  histUnitClass(valu, fname, bins, q, h);
}
void propagator::histDelta(const char* fname, int bins, int q, double h)const{
  histUnitClass(delta, fname, bins, q, h);
}

param_t propagator::meanUnit(int q)const{
  return meanUnitClass(unit, q);
}
param_t propagator::meanValu(int q)const{
  return meanUnitClass(valu, q);
}
odata_t propagator::meanDelta(int q)const{
  return meanUnitClass(delta, q);
}

param_t propagator::meanUnit(int q, int jw, int jh)const{
  return meanUnitClass(unit, q, jw, jh);
}
param_t propagator::meanValu(int q, int jw, int jh)const{
  return meanUnitClass(valu, q, jw, jh);
}
odata_t propagator::meanDelta(int q, int jw, int jh)const{
  return meanUnitClass(delta, q, jw, jh);
}

propagator& propagator::dropout(double rate){
  isDropout = true;
  dropRate = rate;
  return *this;
}

void propagator::allocUnits(){
  unit = alloc<param_t>(batch_size,unit_chan, unit_width, unit_height);
  valu = alloc<param_t>(batch_size,unit_chan, unit_width, unit_height);
  mval = alloc<param_t>(batch_size,unit_chan, unit_width, unit_height);
  mask = alloc<double>(unit_chan, unit_width, unit_height);
  delta = alloc<odata_t>(batch_size, unit_chan, unit_width, unit_height);
}
void propagator::initUnits(){
  for(int n=0; n<batch_size; ++n)
    for(int q=0; q<unit_chan; ++q)
      for(int jw=0; jw<unit_width; ++jw){
	for(int jh=0; jh<unit_height; ++jh){
	  unit[n][q][jw][jh] = 0.0;
	  valu[n][q][jw][jh] = 0.0;
	  mval[n][q][jw][jh] = 0.0;
	  delta[n][q][jw][jh] = 0.0;
	}
      }
  for(int q=0; q<unit_chan; ++q)
    for(int jw=0; jw<unit_width; ++jw){
      for(int jh=0; jh<unit_height; ++jh){
	mask[q][jw][jh] = 1.0;
      }
    }
}

interface layer_base : propagator{
  param_t *w,  *b;
  odata_t *Ow, *Ob;
  bool isreg;
  int Nt, Nw, Nb;
  layer_base();
  virtual ~layer_base(){}
  
  virtual layer_base& dropout(double rate);
  
  virtual void allocLParams() = 0;
  virtual void initLParams();
  
  virtual void showLParams()const{}
  virtual bool outLParams(const char* fname)const{return true;}
  virtual bool loadLParams(const char* fname){return true;}
};
layer_base::layer_base(){
   w =  b = NULL;
  Ow = Ob = NULL;
  isreg = true;
  Nw = Nb = Nt = 0;
}
layer_base& layer_base::dropout(double rate){
  isDropout = true;
  dropRate = rate;
  return *this;
}
void layer_base::initLParams(){
  double sigma;
  sigma = 1.0/sqrt((double)fan_in); // LeCun
  //sigma = sqrt(2.0)/sqrt((double)(fan_in + fan_out)); // Glorot
  //sigma = sqrt(2.0)/sqrt((double)(fan_in)); // He
  for(int id=0; id<Nw; ++id){
    w[id] = nomal_dist(0.0, sigma);
    for(int n=0; n<batch_size; ++n){
      Ow[n*Nw + id] = 0.0;
    }
  }
  for(int id=0; id<Nb; ++id){
    b[id] = 0.0;
    for(int n=0; n<batch_size; ++n){
      Ob[n*Nb + id] = 0.0;
    }
  }
}
#endif
