#ifndef MODEL
#define MODEL
#include "qnn.hpp"

struct hubbard_model{
  
  int Tmps;
  int sample_size;
  int sample_seed, sample_rept, burn_in;
  
  int atoms, site_width, site_height, state_num;
  double J, V, U;
  double**** fockState, ***meanState, ***variState, ****tempState;
  double**  potential;
  hubbard_model(int b_size, int atm, int s_num, int width, int height,
		double j, double v, double u){
    sample_size = b_size;
    atoms = atm, state_num = s_num, site_width = width, site_height = height;
    J=j, V=v, U=u;
    fockState = alloc<double>(sample_size, state_num, site_width, site_height);
    tempState = alloc<double>(sample_size, state_num, site_width, site_height);
    meanState = alloc<double>(state_num, site_width, site_height);
    variState = alloc<double>(state_num, site_width, site_height);
    potential = alloc<double>(site_width, site_height);
  }
  virtual ~hubbard_model(){
    free(fockState);
    free(tempState);
    free(meanState);
    free(variState);
    free(potential);
  }  
  void input2Network(int n, network_base &nn, double*** state)const;
  
  void setSamplingParams(int tmps, int seeds, int burn_in);
  virtual Complex sampling(int sample, network_base &nn, Complex psi1) = 0;
  
  void backupSamples();
  void restoreSamples();
  
  void idling(network_base &nn, int N);
  void makeSamples(network_base &nn);
  void makeSamplesRand(network_base &nn);
  virtual Complex makeWF(int n, network_base &nn)const;
  
  virtual Complex Energy(int sample, network_base &nn) = 0;
  
  void calEnergy(network_base &nn, Complex* E, Complex& Eave);
  void calRatio(network_base &nn1, network_base &nn2, const Complex E[], Complex* R, double dt);
  void calStateMean();
  double calStateVariance();
  void setPotential();
  virtual void randState(int n) = 0;
  void initState();
  
  virtual void showParams()const = 0;
  void showState(int sample)const;
  void showState(double const*const*const* state)const;
};

void hubbard_model::input2Network(int n, network_base &nn, double*** state)const{
  double ave = (double)atoms/(site_width*site_height*state_num);
  for(int s=0; s<state_num; ++s){
    for(int jw=0; jw<site_width; ++jw){
      for(int jh=0; jh<site_height; ++jh){
	nn.InputLayer->unit[n][s][jw][jh] = state[s][jw][jh] - ave;
      }
    }
  }
}
void hubbard_model::idling(network_base &nn, int N){
#ifdef _OPENMP
#pragma omp parallel for num_threads(sample_seed)
#endif
  for(int i=0; i<sample_seed; ++i){
    int id = sample_seed*(sample_rept - 1) + i;
    Complex psi = makeWF(id, nn);
    for(int t=0; t<N; ++t){
      psi = sampling(id, nn, psi);
    }
  }
}
void hubbard_model::makeSamples(network_base &nn){
  idling(nn, burn_in);
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<sample_seed; ++i){
    for(int j=0; j<sample_rept; ++j){
      int id1 = sample_seed*((j - 1 + sample_rept)%sample_rept) + i;
      int id2 = sample_seed*j + i;
      for(int s=0; s<state_num; ++s){
	for(int iw=0; iw<site_width; ++iw){
	  for(int ih=0; ih<site_height; ++ih){
	    fockState[id2][s][iw][ih] = fockState[id1][s][iw][ih];
	  }
	}
      }
      Complex psi1 = makeWF(id2, nn);
      for(int t=0; t<Tmps; ++t){
	psi1 = sampling(id2, nn, psi1);
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    input2Network(n, nn, fockState[n]);
  }  
}
void hubbard_model::makeSamplesRand(network_base &nn){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    randState(n);
    input2Network(n, nn, fockState[n]);
  }  
}
void hubbard_model::setSamplingParams(int tmps, int seeds, int b_in){
  Tmps = tmps;
  sample_seed = seeds;
  sample_rept = sample_size/sample_seed;
  burn_in = b_in;
}

void hubbard_model::backupSamples(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    for(int s=0; s<state_num; ++s){
      for(int iw=0; iw<site_width; ++iw){
	for(int ih=0; ih<site_height; ++ih){
	  tempState[n][s][iw][ih] = fockState[n][s][iw][ih];
	}
      }
    }
  }
}
void hubbard_model::restoreSamples(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    for(int s=0; s<state_num; ++s){
      for(int iw=0; iw<site_width; ++iw){
	for(int ih=0; ih<site_height; ++ih){
	  fockState[n][s][iw][ih] = tempState[n][s][iw][ih];
	}
      }
    }
  }
}

Complex hubbard_model::makeWF(int n, network_base &nn)const{
  input2Network(n, nn, fockState[n]);
  return nn.output(n);
}

void hubbard_model::calEnergy(network_base &nn, Complex* E, Complex& Eave){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    E[n] = Energy(n, nn);
    input2Network(n, nn, fockState[n]);
  }
  Eave = mean(E, sample_size);
}
void hubbard_model::calRatio(network_base &nn1, network_base &nn2, const Complex E[],
			     Complex* R, double dt){
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    Complex psi1 = makeWF(n, nn1);
    Complex psi2 = makeWF(n, nn2);
    R[n] = psi2/psi1*exp(AI*dt*E[n]);
  }
}
void hubbard_model::calStateMean(){
  for(int s=0; s<state_num; ++s){
#ifdef _OPENMP
#pragma omp parallel for
#endif    
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	meanState[s][iw][ih] = 0.0;
	for(int n=0; n<sample_size; ++n){
	  meanState[s][iw][ih] += fockState[n][s][iw][ih];
	}
	meanState[s][iw][ih] /= (double)sample_size;
      }
    }
  }
}
double hubbard_model::calStateVariance(){
  calStateMean();
  for(int s=0; s<state_num; ++s){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	variState[s][iw][ih] = 0.0;
	for(int n=0; n<sample_size; ++n){
	  double del = meanState[s][iw][ih]- fockState[n][s][iw][ih];
	  variState[s][iw][ih] += del*del;
	}
	variState[s][iw][ih] /= (double)sample_size;
      }
    }
  }
  
  double totalVari=0.0;
  for(int s=0; s<state_num; ++s){
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	totalVari += variState[s][iw][ih];
      }
    }
  }
  totalVari /= (double)site_width*site_height*state_num;
  return totalVari;
}
void hubbard_model::setPotential(){
  for(int iw=0; iw<site_width; ++iw){
    double Vx = V*(iw - (site_width-1.0)/2.0)*(iw - (site_width-1.0)/2.0);
    for(int ih=0; ih<site_height; ++ih){
      double Vy = V*(ih - (site_height-1.0)/2.0)*(ih - (site_height-1.0)/2.0);
      potential[iw][ih] = Vx + Vy;
    }
  }
}
void hubbard_model::initState(){
  setPotential();
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int n=0; n<sample_size; ++n){
    for(int s=0; s<state_num; ++s){
      for(int iw=0; iw<site_width; ++iw){
	for(int ih=0; ih<site_height; ++ih){
	  fockState[n][s][iw][ih] = 0.0;
	}
      }
    }
  }
#ifdef _OPENMP
#pragma omp parallel for
#endif
  for(int i=0; i<sample_seed; ++i){
    int id = sample_seed*(sample_rept - 1) + i;
    randState(id);
  }
}

void hubbard_model::showState(int n)const{
  for(int s=0; s<state_num; ++s){
    double sum = 0;
    for(int ih=0; ih<site_height; ++ih){
      for(int iw=0; iw<site_width; ++iw){
	sum += fockState[n][s][iw][ih];
	cout << format("%4.2f ") %fockState[n][s][iw][ih];
      }
      cout << endl;
    }
    cout << "Total :" << sum<< endl;
  }
}
void hubbard_model::showState(double const*const*const* state)const{
  for(int s=0; s<state_num; ++s){
    double sum = 0.0;
    for(int ih=0; ih<site_height; ++ih){
      for(int iw=0; iw<site_width; ++iw){
	sum += state[s][iw][ih];
	cout << format("%4.2f ") %state[s][iw][ih];
      }
      cout << endl;
    }
    cout << "Total :" << sum << endl;
  }
}

struct bose_hubbard_model : hubbard_model{
  bose_hubbard_model(int b_size, int atm, int width, int height, double j, double v, double u)
    : hubbard_model(b_size, atm, 1, width, height, j, v, u){}
  virtual Complex Energy(int sample, network_base &nn);
  virtual Complex sampling(int sample, network_base &nn, Complex psi1);
  void randState(int n);
  void showParams()const;
};
Complex bose_hubbard_model::Energy(int n, network_base &nn){
  Complex E1 = 0.0; double E2 = 0.0;
  Complex psi1, psi2;
  
  psi1 = makeWF(n, nn); 
  for(int s=0; s<state_num; s++){
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	if(site_width>1){
	  int LR[] = {(iw+1+site_width)%site_width, (iw-1+site_width)%site_width};
	  for(int lr=0; lr<2; lr++){
	    int jw=LR[lr], jh=ih;
	    if(0<fockState[n][s][jw][jh]){
	      fockState[n][s][jw][jh] -= 1.0;
	      fockState[n][s][iw][ih] += 1.0;
	      psi2 = makeWF(n, nn);
	      fockState[n][s][jw][jh] += 1.0;
	      fockState[n][s][iw][ih] -= 1.0;
	      E1 += sqrt((fockState[n][s][iw][ih]+1.0)*fockState[n][s][jw][jh] + 1.0E-30)*psi2;
	    }
	  }
	}
	if(site_height>1){
	  int UD[] = {(ih+1+site_height)%site_height, (ih-1+site_height)%site_height};
	  for(int ud=0; ud<2; ud++){
	    int jw=iw, jh=UD[ud];
	    if(0<fockState[n][s][jw][jh]){
	      fockState[n][s][jw][jh] -= 1.0;
	      fockState[n][s][iw][ih] += 1.0;
	      psi2 = makeWF(n, nn);
	      fockState[n][s][jw][jh] += 1.0;
	      fockState[n][s][iw][ih] -= 1.0;
	      E1 += sqrt((fockState[n][s][iw][ih]+1.0)*fockState[n][s][jw][jh] + 1.0E-30)*psi2;
	    }
	  }
	}
      }
    }
  }
  E1 *= -J/psi1;
  
  for(int s=0; s<state_num; s++){
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	double nn = fockState[n][s][iw][ih];
	double vv = potential[iw][ih];
	E2 += vv*nn + U/2.0*nn*(nn - 1.0);
      }
    }
  }
  return E1 + E2;
}

Complex bose_hubbard_model::sampling(int n, network_base &nn, Complex psi1){
  Complex psi2;
  int outsiteX, outsiteY, insiteX, insiteY, s=omp_rand()%state_num;
  double p; bool isAdopt;
  outsiteX = omp_rand()%site_width;
  outsiteY = omp_rand()%site_height;
  if(fockState[n][s][outsiteX][outsiteY]>0){
    insiteX = omp_rand()%site_width;
    insiteY = omp_rand()%site_height;
    fockState[n][s][outsiteX][outsiteY] -= 1.0;
    fockState[n][s][insiteX][insiteY] += 1.0;
    psi2 = makeWF(n, nn);
    p = norm(psi2)/norm(psi1);
    isAdopt = (p>omp_uniform())? true : false;
    if(isAdopt){
      return psi2;
    }else{
      fockState[n][s][outsiteX][outsiteY] += 1.0;
      fockState[n][s][insiteX][insiteY] -= 1.0;
    }
  }
  return psi1;
}

void bose_hubbard_model::randState(int n){
  for(int iw=0; iw<site_width; ++iw){
    for(int ih=0; ih<site_height; ++ih){
      fockState[n][0][iw][ih] = 0.0;
    }
  }
  for(int atm=0; atm<atoms; ++atm){
    fockState[n][0][omp_rand()%site_width][omp_rand()%site_height] += 1.0;
  } 
}
void bose_hubbard_model::showParams()const{
  cout << "model : BoseHubbard" << endl;
  cout << format("sample_size = %d, Tmps = %d,  sample_seed = %d, sample_rept = %d")
    %sample_size %Tmps %sample_seed %sample_rept << endl;
  cout << format("atoms = %d, width = %d, height = %d,") %atoms %site_width %site_height << endl;
  cout << format("J = %.2f, V = %.2f, U = %.2f") %J %V %U << endl;
}

struct fermi_hubbard_model : hubbard_model{
  int Ns[2];
  fermi_hubbard_model(int b_size, int nup, int ndown, int width, int height,
		      double j, double v, double u)
    : hubbard_model(b_size, nup + ndown, 2, width, height, j, v, u)
  {
    Ns[0] = nup, Ns[1] = ndown;
  }
  virtual Complex Energy(int sample, network_base &nn);
  virtual Complex sampling(int sample, network_base &nn, Complex psi1);
  void randState(int n);
  void showParams()const;
};

Complex fermi_hubbard_model::Energy(int n, network_base &nn){
  Complex E1 = 0.0; double E2 = 0.0, sign;
  Complex psi1, psi2;
  int** sumrow = alloc<int>(state_num, site_height);
  int** sumcol = alloc<int>(state_num, site_width);
  psi1 = makeWF(n, nn);
  for(int s=0; s<state_num; s++){
    for(int iw=0; iw<site_width; ++iw){
      sumcol[s][iw] = 0;
    }
    for(int ih=0; ih<site_height; ++ih){
      sumrow[s][ih] = 0;
    }
  }
  for(int s=0; s<state_num; s++){
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	sumcol[s][iw] += (int)fockState[n][s][iw][ih];
	sumrow[s][ih] += (int)fockState[n][s][iw][ih];
      }
    }
  }
  
  for(int s=0; s<state_num; s++){
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	if(site_width>1){
	  int LR[] = {(iw+1+site_width)%site_width, (iw-1+site_width)%site_width};
	  for(int lr=0; lr<2; lr++){
	    int jw=LR[lr], jh = ih;
	    if(fockState[n][s][jw][jh]>0.5 && fockState[n][s][iw][ih]<0.5){
	      fockState[n][s][jw][jh] -= 1.0;
	      fockState[n][s][iw][ih] += 1.0;
	      psi2 = makeWF(n, nn);
	      fockState[n][s][jw][jh] += 1.0;
	      fockState[n][s][iw][ih] -= 1.0;
	      if((jw==0 && iw==site_width-1) || (iw==0 && jw==site_width-1)){
		sign = ((sumrow[s][ih] - 1)%2==0)? 1.0 :-1.0;
		//sign = ((atoms/2 - 1)%2==0)? 1.0 :-1.0;
	      }else{
		sign = 1.0;
	      }
	      E1 += sign*sqrt((fockState[n][s][iw][ih]+1.0)*fockState[n][s][jw][jh] + 1.0E-30)*psi2;
	    }
	  }
	}
	if(site_height>1){
	  int UD[] = {(ih+1+site_height)%site_height, (ih-1+site_height)%site_height};
	  for(int ud=0; ud<2; ud++){
	    int jw = iw, jh=UD[ud];
	    if(fockState[n][s][jw][jh]>0.5 && fockState[n][s][iw][ih]<0.5){
	      fockState[n][s][jw][jh] -= 1.0;
	      fockState[n][s][iw][ih] += 1.0;
	      psi2 = makeWF(n, nn);
	      fockState[n][s][jw][jh] += 1.0;
	      fockState[n][s][iw][ih] -= 1.0;
	      if((jh==0 && ih==site_height-1) || (ih==0 && jh==site_height-1)){
		sign = ((sumcol[s][iw] - 1)%2==0)? 1.0 :-1.0;
	      }else{
		sign = 1.0;
	      }
	      E1 += sign*sqrt((fockState[n][s][iw][ih]+1.0)*fockState[n][s][jw][jh] + 1.0E-30)*psi2;
	    }
	  }
	}
      }
    }
  }
  E1 *= -J/psi1;
  
  for(int iw=0; iw<site_width; ++iw){
    for(int ih=0; ih<site_height; ++ih){
      double np = fockState[n][0][iw][ih];
      double nm = fockState[n][1][iw][ih];
      double vv = potential[iw][ih];
      E2 += vv*(np + nm) + U*np*nm;
    }
  }
  free(sumrow), free(sumcol);
  return E1 + E2;
}
Complex fermi_hubbard_model::sampling(int n, network_base &nn, Complex psi1){
  Complex psi2;
  int outsiteX, outsiteY, insiteX, insiteY, s=omp_rand()%state_num;
  double p; bool isAdopt;
  
  outsiteX = omp_rand()%site_width;
  outsiteY = omp_rand()%site_height;
  insiteX = omp_rand()%site_width;
  insiteY = omp_rand()%site_height;
  
  if(fockState[n][s][outsiteX][outsiteY]>0.5 && fockState[n][s][insiteX][outsiteY]<0.5){
    fockState[n][s][outsiteX][outsiteY] -= 1.0;
    fockState[n][s][insiteX][insiteY]   += 1.0;
    psi2 = makeWF(n, nn);
    p = norm(psi2)/norm(psi1);
    isAdopt = (p>omp_uniform())? true : false;
    if(isAdopt){
      return psi2;
    }else{
      fockState[n][s][outsiteX][outsiteY] += 1.0;
      fockState[n][s][insiteX][insiteY]   -= 1.0;
    }
  }
  return psi1;
}

void fermi_hubbard_model::randState(int n){
  for(int s=0; s<state_num; ++s){
    for(int iw=0; iw<site_width; ++iw){
      for(int ih=0; ih<site_height; ++ih){
	fockState[n][s][iw][ih] = 0.0;
      }
    }
  }
  for(int s=0; s<state_num; ++s){
    for(int atm=0; atm<Ns[s]; ++atm){
      int insiteX = omp_rand()%site_width;
      int insiteY = omp_rand()%site_height;
      while(fockState[n][s][insiteX][insiteY] != 0.0){
	insiteX = omp_rand()%site_width;
	insiteY = omp_rand()%site_height;
      }
      fockState[n][s][insiteX][insiteY] += 1.0;
    }
  }
}
void fermi_hubbard_model::showParams()const{
  cout << "model : FermiHubbard" << endl;
  cout << format("sample_size = %d, Tmps = %d,  sample_seed = %d, sample_rept = %d")
    %sample_size %Tmps %sample_seed %sample_rept << endl;
  cout << format("Nup = %d, Ndown = %d, width = %d, height = %d,") 
    %Ns[0] %Ns[1] %site_width %site_height << endl;
  cout << format("J = %.2f, V = %.2f, U = %.2f") %J %V %U << endl;
}

#endif
