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

void BHoptimization(){
  
  const int MOVE = 100;  
  const int Totalstep = 20000, HealingTime = 0, Tout = 50;
  const int Batches = (1024/THREADS_NUM)*THREADS_NUM;
  const int Tmps = 32; //32
  const int BurnIn = 128;
  const int Atoms = 8;
  const int XLength = 8;
  const int YLength = 1;
  double Emoveave, Estack[MOVE];
  Complex* E = alloc<Complex>(Batches);
  Complex  Ebar = 0.0, Et;
  double J=1.0, V=0.0, U=10.0;
  
  Network::Adam nn(Batches, 0.02, 0.9, 0.999, 1.0e-8, 1.0e-6);
  //Network::AdaGrad nn(Batches, 0.2, 1.0e-8, 0.0);
  
  bose_hubbard_model bh(Batches, Atoms, XLength, YLength, J, V, U);
  
  double start = omp_timer(), end;
  ofstream ofse("Energy.plt");
  
  nn << inputLayer<af_eqal>(1, XLength, YLength)
     << convolutionLayer<af_elu>(4, 6)
     << convolutionLayer<af_elu>(4, 6)
     << sumUnitLayer()
     << fullConectionNoBiasLayer<af_eqal>(2)
     << outputLayer<of_exp>(1);
  
  print(nn);
  //nn.load("BH");
  bh.setSamplingParams(Tmps, THREADS_NUM, BurnIn); 
  bh.initState();
  bh.makeSamples(nn);
  bh.showParams();
  
  for(int t=0; t<Totalstep + HealingTime; ++t){
    
    if(t<=HealingTime&&HealingTime!=0){
      bh.U = U/(HealingTime + 1.0e-30)*t;
    }
    
    bh.makeSamples(nn.pred());
    bh.calEnergy(nn.pred(), E, Ebar);
    nn.train().run(E, Ebar);
    
    Estack[t%MOVE] = Ebar.real();
    Emoveave = mean(Estack, min(t+1, MOVE));
    if(std::isnan(Ebar.real())){break;}
    ofse << format("%d %7.5f %7.5f") %(t-HealingTime) %Estack[t%MOVE] %Emoveave << endl;
    if(t%Tout == 0){
      cout << format("%4d %7.5f %7.5f %7.5f %7.5f, U=%.2f") 
	%(t-HealingTime) %Estack[t%MOVE] %Emoveave %abs(Ebar-Emoveave) %nn.weightNorm() %bh.U<< endl;
      //nn.pred().checkDiff(true);
      //nn.train().checkDiff(true);
      //bh.calStateMean();
      //bh.showState(bh.meanState);
    }
    if(t%(Tout*2) == 0 && t!=0)nn.save("BH");
  }
  
  nn.save("BH");  
  end = omp_timer();
  cout << "time = " << end-start << "sec" << endl;
  free(E);
}

int main(){
  omp_srand(0);
  BHoptimization();
  return 0;
}
