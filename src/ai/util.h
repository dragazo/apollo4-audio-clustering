#ifndef A3EM_AI_UTIL_H
#define A3EM_AI_UTIL_H

#include "./tensor.h"

#define PI 3.14159265358979323846f

template<typename T> Tensor<complicate_t<T>, 1> fft_impl(const Tensor<T, 1> &x, u32 N, f32 ang_scale, f32 res_scale) {
    Tensor<complicate_t<T>, 1> res { new complicate_t<T>[N], [](auto *v) { delete[] v; }, N };
    for (u32 k = 0; k < N; ++k) {
        complicate_t<T> sum = {0, 0};
        for (u32 n = 0; n < x.template dim<0>(); ++n) {
            f32 ang = ang_scale * k * n / x.template dim<0>();
            sum = sum + x(n) * complicate_t<T> { std::cos(ang), std::sin(ang) };
        }
        res(k) = res_scale * sum;
    }
    return res;
}

template<typename T> Tensor<complicate_t<T>, 1> fft(const Tensor<T, 1> &x) { return fft_impl(x, x.template dim<0>(), -2 * PI, 1); }
template<typename T> Tensor<complicate_t<T>, 1> ifft(const Tensor<T, 1> &x) { return fft_impl(x, x.template dim<0>(), 2 * PI, 1.0 / x.template dim<0>()); }

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0> Tensor<complicate_t<T>, 1> rfft(const Tensor<T, 1> &x) { return fft_impl(x, x.template dim<0>() / 2 + 1, -2 * PI, 1); }
template<typename T> Tensor<complicate_t<T>, 1> irfft(const Tensor<T, 1> &x) {
    Tensor<T, 1> extended { new T[2 * (x.template dim<0>() - 1)], [](auto *v) { delete[] v; }, 2 * (x.template dim<0>() - 1) };
    for (u32 i = 0; i < x.template dim<0>(); ++i) extended(i) = x(i);
    for (u32 i = x.template dim<0>(); i < extended.template dim<0>(); ++i) extended(i) = conj(x(extended.template dim<0>() - i));
    return ifft(extended);
}

#endif
