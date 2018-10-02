#ifndef ACTFUNCTION_CLASS
#define ACTFUNCTION_CLASS

#include "def.hpp"

typedef param_t (*Activation)(param_t);

interface actfunc_class{
  actfunc_class(){}
  virtual ~actfunc_class(){}
  static param_t f(param_t u);
  static param_t g(param_t z);
};

#ifdef USE_DOUBLE_PARAMETER

struct af_tanh : actfunc_class{
  static param_t f(param_t u){
    return tanh(u);
  }
  static param_t g(param_t z){
    return 1.0 - z*z;
  }
};

struct af_tanh2 : actfunc_class{
  static param_t f(param_t u){
    return tanh(u*0.5);
  }
  static param_t g(param_t z){
    return (1.0 - z*z)*0.5;
  }
};

struct af_sigmoid : actfunc_class{
  static param_t f(param_t u){
    return 1.0/(1.0+exp(-u));
  }
  static param_t g(param_t z){
    return z*(1.0 - z);
  }
};

struct af_relu : actfunc_class{
  static param_t f(param_t u){
    return (u>0.0)? u:0.0;
  }
  static param_t g(param_t z){
    return (z>0.0)? 1.0:0.0;
  }
};

struct af_srelu : actfunc_class{
  static param_t f(param_t u){
    return (u>-1.0)? u:-1.0;
  }
  static param_t g(param_t z){
    return (z>-1.0)? 1.0:0.0;
  }
};

struct af_lrelu : actfunc_class{
  static param_t f(param_t u){
    return (u>0.0)? u:0.25*u;
  }
  static param_t g(param_t z){
    return (z>0.0)? 1.0:0.25;
  }
};

struct af_elu : actfunc_class{
  static param_t f(param_t u){
    param_t a = 1.0;
    return (u>0.0)? u:a*(exp(u)-1.0);
  }
  static param_t g(param_t z){
    param_t a = 1.0;
    return (z>0.0)? 1.0:z+a;
  }
};

struct af_selu : actfunc_class{
  static param_t f(param_t u){
    param_t a = 1.6732632423543772848170429916717;
    param_t s = 1.0507009873554804934193349852946;
    return (u>0.0)? s*u:s*a*(exp(u)-1.0);
  }
  static param_t g(param_t z){
    param_t a = 1.6732632423543772848170429916717;
    param_t s = 1.0507009873554804934193349852946;
    return (z>0.0)? s:z+s*a;
  }
};

struct af_eqal : actfunc_class{
  static param_t f(param_t u){
    return u;
  }
  static param_t g(param_t z){
    return 1.0;
  }
};


#endif

#ifdef USE_COMPLEX_PARAMETER

struct af_tanh : actfunc_class{
  static param_t f(param_t u){
    return tanh(u);
  }
  static param_t g(param_t z){
    return 1.0 - z*z;
  }
};

struct af_tanh2 : actfunc_class{
  static param_t f(param_t u){
    return tanh(u*0.5);
  }
  static param_t g(param_t z){
    return (1.0 - z*z)*0.5;
  }
};

struct af_eqal : actfunc_class{
  static param_t f(param_t u){
    return u;
  }
  static param_t g(param_t z){
    return 1.0;
  }
};


#endif

#endif
