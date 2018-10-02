#ifndef SOLVER
#define SOLVER
#include "mylib.hpp"

#ifdef USE_MKL_SOLVER
#include <mkl.h>

template<typename T>
void print_mat(T** mat, int size){
  for(int i=0; i<size; ++i){
    for(int j=0; j<size; ++j){
      cout << mat[i][j] << ' ';
    }
    cout << endl;
  }
}

template<typename T>
void print_vec(T* vec, int size){
  for(int i=0; i<size; ++i){
    cout << vec[i] << ' ';
  }
  cout << endl;
}

template<typename T>
class solver{
public:
  int info;
  int M;
  char jobu, jobvt;
  double *S, *dS;
  T **U, **Vt, **invA;
  lapack_int *ipiv;
  solver(){
  }
  ~solver(){
    free(invA), free(S), free(dS),free(U), free(Vt);
    free(ipiv);
  }
    
  void init(){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<M; ++i){
      S[i] = dS[i] = ipiv[i] = 0;
      for(int j=0; j<M; ++j){
	U[i][j] = Vt[i][j] = invA[i][j] = 0;
      }
    }
  }
      
  void plan(int size){
    M = size;
    jobu='S', jobvt= 'S';
    S  = alloc<double>(M);
    dS = alloc<double>(M);
    U  = alloc<T>(M, M);
    Vt = alloc<T>(M, M);
    invA = alloc<T>(M, M);
    ipiv = alloc<lapack_int>(M);
  }
    
  double error(T const*const* a, T const*x, T const* b){
    double err = 0.0;
    for(int i=0; i<M; ++i){
      T sum = 0;
      for(int j=0; j<M; ++j){
	sum += a[i][j]*x[j];
      }
      err += norm(sum - b[i]);
    }
      
    double nrmb = 0.0;
    for(int i=0; i<M; ++i){
      nrmb += norm(b[i]);
    }
      
    return err/nrmb;
  }
    
  int pseudoinverse(double ** a, double tolerance){
    //A = u*s*vt
    info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, M, M, a[0], M, S, U[0], M, Vt[0], M, dS);
      
    //u' = (S^-1)*U
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j=0; j<M; ++j){
      double ss = (S[j]>tolerance)? 1.0/S[j]:0.0;
      for(int i=0; i<M; i++){
	U[i][j] *= ss;
      }
    }
    
    //inv(A) = Vt^T*u^T = V*S^(+)*U^T
    double alpha = 1.0, beta=0.0;
    cblas_dgemm(CblasRowMajor, CblasConjTrans, CblasConjTrans, M, M, M, 
		alpha, Vt[0], M, U[0], M,
		beta, invA[0], M);
    return info;
  }
    
  int pseudoinverse(Complex ** a, double tolerance){
    //A = u*s*vt
    info = LAPACKE_zgesvd(LAPACK_ROW_MAJOR, jobu, jobvt, M, M, 
			  (MKL_Complex16*)a[0], M, S, (MKL_Complex16*)U[0], M,
			  (MKL_Complex16*)Vt[0], M, dS);
      
    //u' = U*(S^-1)
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int j=0; j<M; ++j){
      double ss = (S[j]>tolerance)? 1.0/S[j]:0.0;
      for(int i=0; i<M; i++){
	U[i][j] *= ss;
      }
    }
         
    double alpha=1.0, beta=0.0;
    cblas_zgemm(CblasRowMajor, CblasConjTrans, CblasConjTrans, M, M, M,
		&alpha, (MKL_Complex16*)Vt[0], M, (MKL_Complex16*)U[0], M,
		&beta, (MKL_Complex16*)invA[0], M);
    return info;
  }
    
  int pi(T** A, T* x, T* b, double tolerance=1.0e-6){
    init();
    int flag = pseudoinverse(A, tolerance);
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<M; ++i){
      T sum = 0.0;
      for(int j=0; j<M; ++j){
	sum += invA[i][j]*b[j];
      }
      x[i] = sum;
    }
    return flag;
  }
  
  int lu(Complex ** A, Complex* x, Complex const* b){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<M; ++i) x[i] = b[i];
    info = LAPACKE_zhesv(LAPACK_ROW_MAJOR, 'U', M, 1,
			 (MKL_Complex16*)A[0], M, ipiv, 
			 (MKL_Complex16*)x, 1);
    return info;
  }
    
  int lusy(double ** A, double* x, double const* b){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<M; ++i) x[i] = b[i];
    info = LAPACKE_dsysv(LAPACK_ROW_MAJOR, 'U', M, 1, A[0], M, ipiv, x, 1);
    return info;
  }
  
  int lu(double ** A, double* x, double const* b){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<M; ++i) x[i] = b[i];
    info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, M, 1, A[0], M, ipiv, x, 1);
    return info;
  }
  
};
#endif

#ifdef USE_EIGEN_SOLVER

#include "Eigen/SVD"
#include "Eigen/LU"
#include "Eigen/QR"
#include "mylib.hpp"
using namespace Eigen;
template <typename T>
class solver{
private:
  int size;
  typedef Matrix<T,Dynamic,Dynamic> t_matrix;
  typedef Matrix<T,Dynamic,1> t_vector;
  typedef JacobiSVD<t_matrix> TSVD;
  t_matrix A;
  t_vector B, X;
  
  void map(T ** a, T* b){
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int i=0; i<size; ++i){
      B[i] = b[i];
      for(int j=0; j<size; ++j){
	A(i, j) = a[i][j];
      }
    }
  }
  void mapans(T* x)const{
#ifdef _OPENMP
#pragma omp parallel
#endif
    for(int i=0; i<size; ++i){
      x[i] = X[i];
    }    
  }
  
public:
  int plan(int s){
    size = s;
    A.resize(size, size);
    B.resize(size);
    X.resize(size);
    return 0;
  }
  
  bool exist(T ** a, T* b, double precision=1.0-9){
    map(a,b);
    X = A.lu().solve(B);
    return (A*X).isApprox(B, precision);
  }

  double error(T ** a, T* x, T* b){
    map(a,b); mapans(x);
    return (A*X-B).norm()/B.norm();
  }
  
  int lu(T ** a, T*x, T* b){
    map(a,b);
    X = A.lu().solve(B);
    mapans(x);
    return 0;
  }
  
  int pi(T ** a, T* x, T* b, double tolerance=1.e-6){    
    map(a,b);
    TSVD svd(A, ComputeFullU|ComputeFullV);
    typename TSVD::SingularValuesType sigma(svd.singularValues());
    typename TSVD::SingularValuesType sigma_inv(sigma.size());
    for(long i=0; i<sigma.size(); ++i){
      if(sigma(i) > tolerance)
	sigma_inv(i)= 1.0/sigma(i);
      else
	sigma_inv(i)= 0.0;
    }
    X = svd.matrixV()*sigma_inv.asDiagonal()*svd.matrixU().adjoint()*B;
    mapans(x);
    return 0;
  }
  
  int svd(T ** a, T*x, T* b){
    map(a,b);
    TSVD svd(A, ComputeFullU|ComputeFullV);
    X = svd.solve(B);
    mapans(x);
    return 0;
  }
  
  int od(T ** a, T*x, T* b){
    map(a,b);
    X = A.completeOrthogonalDecomposition().solve(B);
    mapans(x);
    return 0;
  }
  
};
#endif

#endif
