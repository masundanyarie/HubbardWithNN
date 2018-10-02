#ifndef DEF_HPP
#define DEF_HPP
#include "mylib.hpp"

#define USE_DOUBLE_PARAMETER
//#define USE_COMPLEX_PARAMETER
#define USE_MKL_SOLVER
//#define USE_EIGEN_SOLVER

#ifndef interface
#define interface struct
#endif

typedef std::complex<double> odata_t;

#ifdef USE_COMPLEX_PARAMETER
typedef std::complex<double> param_t;
#endif

#ifdef USE_DOUBLE_PARAMETER
typedef double  param_t;
#endif

#ifdef _OPENMP
const int THREADS_NUM = omp_get_max_threads();
#else
const int THREADS_NUM = 1;
#endif

#endif
