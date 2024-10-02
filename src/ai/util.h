#ifndef A3EM_AI_UTIL_H
#define A3EM_AI_UTIL_H

#include "./tensor.h"

template<typename T>
struct Complex {
    T real;
    T imag;
};
typedef Complex<f32> c32;

Tensor<c32, 1> rfft(const Tensor<f32, 1> &a);

#endif
