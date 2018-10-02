#ifndef __MYLIB
#define __MYLIB

#include <iostream>
#include <math.h>
#include <complex>
#include <climits>
#include <fstream>
#include <omp.h>
#include <boost/format.hpp>
using namespace std;
using namespace boost;
typedef complex<double> Complex;
const Complex AI(0.0, 1.0);

#ifdef _OPENMP
double omp_timer(){
  return (double)omp_get_wtime();
}
#else
#include <time.h>
double omp_timer(){
  return (double)clock()/CLOCKS_PER_SEC;
}
#endif

#ifdef _OPENMP
#include <random>
class omp_threads_rand{
protected:
  random_device* rd;
  mt19937* mt;
  mt19937 mt_seed;
  int thread_max;
  int main_seed;
  uint32_t max, min;
  
public:
  omp_threads_rand(){
    main_seed = 0;
    thread_max = omp_get_max_threads();
    rd = new random_device[thread_max];
    mt = new mt19937[thread_max];
    mt_seed.seed(main_seed);
    max = mt[0].max();
    min = mt[0].min();
    for(int i=0; i<thread_max; i++){
      mt[i].seed(mt_seed());
    }
  }
    
  ~omp_threads_rand(){
    delete[] rd;
    delete[] mt;
  }
  
  void srand(int seed){
    main_seed = seed;
    mt_seed.seed(main_seed);
    for(int i=0; i<thread_max; i++){
      mt[i].seed(mt_seed());
    }
  }
  
  void srand(){
    for(int i=0; i<thread_max; i++){
      mt[i].seed(rd[i]());
    }
  }  
};
class omp_threads_int_rand : public omp_threads_rand
{
public:
  inline unsigned operator()(){
    return mt[omp_get_thread_num()]();
  }
  uint32_t get_max(){
    return max;
  }
}omp_rand;
class omp_threads_double_rand : public omp_threads_rand
{
public:
  inline double operator()(){
    return (mt[omp_get_thread_num()]() + 1.0)/(max + 2.0);
  }
}omp_uniform;
#else
class omp_threads_rand{
public:
  omp_threads_rand(){}
  ~omp_threads_rand(){}
  
  void srand(int seed = 0){
    std::srand(seed);
  }
};
class omp_threads_int_rand : public omp_threads_rand 
{
public:
  inline unsigned operator()(){
    return rand();
  }
  int get_max(){
    return RAND_MAX;
  }
}omp_rand;
class omp_threads_double_rand : public omp_threads_rand
{
public:
  inline double operator()(){
    return ((double)rand() + 1.0)/((double)RAND_MAX + 2.0);
  }
}omp_uniform;
#endif
void omp_srand(int seed = 0){
  omp_rand.srand(seed);
  omp_uniform.srand(seed);
}

inline double nomal_dist(double mu, double sigma){
  double z = sqrt(-2.0*log(omp_uniform()))*sin(2.0*M_PI*omp_uniform());
  return mu + sigma*z;
}

template <class T>
static T* alloc(int ID0){
  T* mat0 = new T[ID0];
  return mat0;
}

template <class T>
static T** alloc(int ID0, int ID1){
  T** mat0 = new T*[ID0];
  T* mat1 = new T[ID0*ID1];
  for (int id0 = 0; id0 < ID0; id0++) {
    mat0[id0] = &mat1[id0*ID1];
  }
  return mat0;
}

template <class T>
static T** alloc(int ID0, int *ID1){
  int total_id1=0;
  for(int l=0; l<ID0; l++){
    total_id1 += ID1[l];
  }
  T** mat0 = new T*[ID0];
  T* mat1 = new T[total_id1];
  for (int id0 = 0; id0 < ID0; id0++) {
    mat0[id0] = &mat1[id0*ID1[id0]];
  }
  return mat0;
}

template <class T>
static T*** alloc(int ID0, int ID1, int ID2){
  T*** mat0 = new T**[ID0];
  T** mat1 = new T*[ID0*ID1];
  T* mat2 = new T[ID0*ID1*ID2];
  for (int i0 = 0; i0 < ID0; i0++){
    int id0=i0;
    mat0[i0] = &mat1[i0*ID1];
    for (int i1 = 0; i1 < ID1; i1++){
      int id1=id0*ID1+i1;
      mat1[id1] = &mat2[id1*ID2];
    }
  }
  return mat0;
}

template <class T>
static T**** alloc(int ID0, int ID1, int ID2, int ID3){
  T**** mat0 = new T***[ID0];
  T*** mat1 = new T**[ID0*ID1];
  T** mat2 = new T*[ID0*ID1*ID2];
  T* mat3 = new T[ID0*ID1*ID2*ID3];
  for (int i0 = 0; i0 < ID0; i0++){
    int id0=i0;
    mat0[id0] = &mat1[id0*ID1];
    for (int i1 = 0; i1 < ID1; i1++) {
      int id1=id0*ID1+i1;
      mat1[id1] = &mat2[id1*ID2];
      for (int i2 = 0; i2 < ID2; i2++) {
	int id2=id1*ID2+i2;
	mat2[id2] = &mat3[id2*ID3];
      }
    }
  }
  return mat0;
}

template <class T>
static T***** alloc(int ID0, int ID1, int ID2, int ID3, int ID4){
  T***** mat0 = new T****[ID0];
  T**** mat1 = new T***[ID0*ID1];
  T*** mat2 = new T**[ID0*ID1*ID2];
  T** mat3 = new T*[ID0*ID1*ID2*ID3];
  T* mat4 = new T[ID0*ID1*ID2*ID3*ID4];
  for (int i0 = 0; i0 < ID0; i0++){
    int id0=i0;
    mat0[id0] = &mat1[id0*ID1];
    for (int i1 = 0; i1 < ID1; i1++) {
      int id1=id0*ID1+i1;
      mat1[id1] = &mat2[id1*ID2];
      for (int i2 = 0; i2 < ID2; i2++) {
	int id2=id1*ID2+i2;
	mat2[id2] = &mat3[id2*ID3];
	for (int i3 = 0; i3 < ID3; i3++) {
	  int id3=id2*ID3+i3;
	  mat3[id3] = &mat4[id3*ID4];
	}
      }
    }
  }
  return mat0;
}

template <class T>
static T****** alloc(int ID0, int ID1, int ID2, int ID3, int ID4, int ID5){
  T****** mat0 = new T*****[ID0];
  T***** mat1 = new T****[ID0*ID1];
  T**** mat2 = new T***[ID0*ID1*ID2];
  T*** mat3 = new T**[ID0*ID1*ID2*ID3];
  T** mat4 = new T*[ID0*ID1*ID2*ID3*ID4];
  T* mat5 = new T[ID0*ID1*ID2*ID3*ID4*ID5];
  
  for (int i0 = 0; i0 < ID0; i0++){
    int id0=i0;
    mat0[id0] = &mat1[id0*ID1];
    for (int i1 = 0; i1 < ID1; i1++) {
      int id1=id0*ID1+i1;
      mat1[id1] = &mat2[id1*ID2];
      for (int i2 = 0; i2 < ID2; i2++) {
	int id2=id1*ID2+i2;
	mat2[id2] = &mat3[id2*ID3];
	for (int i3 = 0; i3 < ID3; i3++) {
	  int id3=id2*ID3+i3;
	  mat3[id3] = &mat4[id3*ID4];
	  for (int i4 = 0; i4 < ID4; i4++) {
	    int id4=id3*ID4+i4;
	    mat4[id4] = &mat5[id4*ID5];
	  }
	}
      }
    }
  }
  return mat0;
}

template <class T>
static void free(T****** mat){
  if(mat!=NULL){
    delete[] mat[0][0][0][0][0];
    delete[] mat[0][0][0][0];
    delete[] mat[0][0][0];
    delete[] mat[0][0];
    delete[] mat[0];
    delete[] mat;
  }
}

template <class T>
static void free(T***** mat){
  if(mat!=NULL){
    delete[] mat[0][0][0][0];
    delete[] mat[0][0][0];
    delete[] mat[0][0];
    delete[] mat[0];
    delete[] mat;
  }
}

template <class T>
static void free(T**** mat){
  if(mat!=NULL){
    delete[] mat[0][0][0];
    delete[] mat[0][0];
    delete[] mat[0];
    delete[] mat;
  }
}

template <class T>
static void free(T*** mat){
  if(mat!=NULL){
    delete[] mat[0][0];
    delete[] mat[0];
    delete[] mat;
  }
}

template <class T>
static void free(T** mat){
  if(mat!=NULL){
    delete[] mat[0];
    delete[] mat;
  }
}

template <class T>
static void free(T* mat){
  if(mat!=NULL){
    delete[] mat;
  }
}

void bp(const char* s="OK!"){
  cout << s << endl;
  getchar();
}

template<class T>
T mean(const T* A, int size){
  T sum = 0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:sum)
#endif
  for(int i=0; i<size; i++){
    sum += A[i];
  }
  return sum/(double)size;
}

Complex mean(const Complex* A, int size){
  double  re=0, im=0;
#ifdef _OPENMP
#pragma omp parallel for reduction(+:re,im)
#endif
  for(int i=0; i<size; i++){
    re += A[i].real();
    im += A[i].imag();
  }
  Complex sum(re, im);
  return sum/(double)size;
}

ostream& operator<<(ostream& s, Complex z){
  return s << z.real() << ' ' << z.imag();
}

istream& operator>>(istream& s, Complex &z){
  double re, im;
  s >> re >> im;
  z.real(re), z.imag(im);
  return s;
}

template<class T>
void print(const T* A, int size){
  for(int i=0; i<size; ++i){
    cout << A[i] << endl;
  }
}

template<class T>
bool fprint(const char* name, const T* A, int size){
  ofstream ofs(name);
  if(ofs){
    for(int i=0; i<size; ++i){
      ofs << A[i] << endl;
    }
    return true;
  }
  else return false;
}

#endif
