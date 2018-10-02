#include "src/hubbard.hpp"
#include "src/overlap.hpp"
#include "src/creater.hpp"
#include "src/evolution.hpp"
#include "src/optimization.hpp"

void outMean(int t, const network_base& nn){
  for(size_t l=0; l<nn.Lpara.size(); ++l){
    ofstream ofv;  stringstream valu;
    ofstream ofd;  stringstream delta;
    valu  << "valu"  << format("L%02d") %l << ".plt";
    delta << "delta" << format("L%02d") %l << ".plt";
	  
    if(t==0){
      ofv.open( valu.str().c_str());
      ofd.open(delta.str().c_str());
    }else{
      ofv.open( valu.str().c_str(), ios::out|ios::app);
      ofd.open(delta.str().c_str(), ios::out|ios::app);
    }
    ofv << "#" << nn(l).tag << endl;
    ofd << "#" << nn(l).tag << endl;
    ofv << t << ' ';
    ofd << t << ' ';
    if(nn(l).tag == "full conection"){
      for(int jw=0; jw<nn(l).unit_width; ++jw){
	for(int jh=0; jh<nn(l).unit_height; ++jh){
	  ofv << nn(l).meanValu(0, jw, jh) << ' ';
	  ofd << abs(nn(l).meanDelta(0, jw, jh)) << ' ';
	}
      }
    }else if(nn(l).tag == "convolution"){
      for(int q=0; q<nn(l).unit_chan; ++q){
	ofv << nn(l).meanValu(q) << ' ';
	ofd << abs(nn(l).meanDelta(q)) << ' ';
      }
    }
    ofv << endl;
    ofd << endl;
  }
}

void outHist(int t, const network_base& nn){
  for(size_t l=0; l<nn.Lpara.size(); ++l){
    for(int q=0; q<nn(l).unit_chan; ++q){
      stringstream valu;
      stringstream delta;
      valu << "valu" << format("L%02dCH%02d_%04d") %l %q %t << ".plt";
      nn(l).histValu(valu.str().c_str(), 50, q);//0.04
      delta << "delta" << format("L%02dCH%02d_%04d") %l %q %t << ".plt";
      nn(l).histDelta(delta.str().c_str(), 50, q);
    }
  }
}

void outNN(const char* hed, int t, const network_base& nn){
  stringstream nname;
  nname << hed << '_' << format("%04d") %t;
  nn.save(nname.str().c_str());
}

void BHevolution(bool evolution_type = false){
  double start = omp_timer(), end;
  ofstream ofse("Energy.plt");
  
  const double dt = 1.e-3;
  const int MOVE = (int)(0.1/dt);  
  const int 
    Totalstep   = (evolution_type==REAL_TIME)? (int)(2.0/dt) : (int)(4.0/dt), //4.0. 3.0 
    HealingTime = (evolution_type==REAL_TIME)? (int)(0.0/dt) : (int)(1.0/dt), //0.0, 2.0
    Tout = 50;
  const int Batches = (2048/THREADS_NUM)*THREADS_NUM;
  const int Tmps = 32; //32
  const int BurnIn = 128;
  const int Atoms = 12;
  const int XLength = 12;
  const int YLength =1;
  double J=1.0, V=0.0, Uinit=10.0, U=5.0;
  double Emoveave, Estack[MOVE], Var = 0.0;
  Complex* E = alloc<Complex>(Batches);
  Complex  Ebar = 0.0, Et;
  double err;
  EvolutionNetwork nn(dt, Batches, evolution_type); 
  bose_hubbard_model bh(Batches, Atoms, XLength, YLength, J, V, U);
  
  nn << inputLayer<af_eqal>(1, XLength, YLength)
     << convolutionLayer<af_tanh>(4, 6)
     << convolutionLayer<af_tanh>(4, 6)
     << convolutionLayer<af_tanh>(4, 6)
     << sumUnitLayer()
     << fullConectionNoBiasLayer<af_eqal>(2)
     << outputLayer<of_exp>(2);
  
  if(evolution_type==REAL_TIME)nn.load("BH");
  //nn.load("BH");
  
  bh.setSamplingParams(Tmps, THREADS_NUM, BurnIn);
  
  print(nn);
  bh.showParams();
  bh.initState();
  bh.idling(nn, 512);
  
  for(int it=0; it<MOVE; ++it){
    bh.U = (evolution_type==REAL_TIME)? U : Uinit;
    bh.makeSamples(nn);
    bh.calEnergy(nn, E, Ebar);
    Estack[it] = Ebar.real();
  }
  Et = Emoveave = mean(Estack, MOVE);
  cout << "<H> = "<< Emoveave << endl;
  
  for(int t=0; t<=Totalstep + HealingTime; ++t){
    if(t<=HealingTime&&HealingTime!=0){
      if(evolution_type==REAL_TIME){
	bh.U = Uinit - (Uinit - U)/HealingTime*t;
      }else{
	bh.U = Uinit/(HealingTime + 1.0e-30)*t;
      }
    }
    
    /* Euler */
    bh.idling(nn.pred(), 512);
    bh.makeSamples(nn.pred());
    bh.calEnergy(nn.pred(), E, Ebar);
    nn.train().euler(E, Ebar);
    
    Var = bh.calStateVariance();
    Estack[t%MOVE] = Ebar.real();
    Emoveave = mean(Estack, MOVE);
    
    if(std::isnan(Ebar.real()))break;
    ofse << format("%d %7.5f %7.5f %7.5f")
      %(t-HealingTime) %Estack[t%MOVE] %Emoveave %Var << endl;
    if(t%Tout == 0){
      cout << format("%4d %7.5f %7.5f %7.5f %7.5f %7.5f %7.5f, U=%.2f") 
	%(t-HealingTime) %Estack[t%MOVE] %Emoveave %Var 
	%abs(Ebar-Emoveave) %nn.wdotNorm() %nn.weightNorm() %bh.U << endl;
      
      bh.calStateMean();
      bh.showState(bh.meanState);
      nn.train().checkDiff(true);
      nn.pred().checkDiff(true);
      if(t%500 == 0 && t!=0){
	if(evolution_type==REAL_TIME){
	  //outHist(t, nn);
	  //outMean(t, nn);
	  stringstream nname;
	  nname << "BH_" << format("%04d") %t;
	  nn.save(nname.str().c_str());
	}else{
	  nn.save("BH");
	}
      }
    }
  }
  
  if(evolution_type==IMAG_TIME)nn.save("BH");
  
  end = omp_timer();
  cout << "time = " << end-start << "sec" << endl;
  free(E);
}

int main(){
  omp_srand(0);
  BHevolution(IMAG_TIME);
  BHevolution(REAL_TIME);
  return 0;
}
