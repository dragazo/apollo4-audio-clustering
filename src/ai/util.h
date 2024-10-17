#ifndef A3EM_AI_UTIL_H
#define A3EM_AI_UTIL_H

#include <vector>

#include "./tensor.h"

#define PI 3.14159265358979323846

template<typename T> Tensor<complicate_t<T>, 1> fft_impl(const Tensor<T, 1> &x, u32 N, simplify_t<T> ang_scale, simplify_t<T> res_scale) {
    Tensor<complicate_t<T>, 1> res { new complicate_t<T>[N], [](auto *v) { delete[] v; }, N };
    for (u32 k = 0; k < N; ++k) {
        complicate_t<T> sum = {0, 0};
        for (u32 n = 0; n < x.template dim<0>(); ++n) {
            simplify_t<T> ang = ang_scale * k * n / x.template dim<0>();
            sum = sum + x(n) * complicate_t<T> { std::cos(ang), std::sin(ang) };
        }
        res(k) = res_scale * sum;
    }
    return res;
}

template<typename T> Tensor<complicate_t<T>, 1> fft(const Tensor<T, 1> &x) { return fft_impl(x, x.template dim<0>(), -2 * (simplify_t<T>)PI, 1); }
template<typename T> Tensor<complicate_t<T>, 1> ifft(const Tensor<T, 1> &x) { return fft_impl(x, x.template dim<0>(), 2 * (simplify_t<T>)PI, 1.0 / x.template dim<0>()); }

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0> Tensor<complicate_t<T>, 1> rfft(const Tensor<T, 1> &x) { return fft_impl(x, x.template dim<0>() / 2 + 1, -2 * PI, 1); }
template<typename T> Tensor<complicate_t<T>, 1> irfft(const Tensor<T, 1> &x) {
    Tensor<T, 1> extended { new T[2 * (x.template dim<0>() - 1)], [](auto *v) { delete[] v; }, 2 * (x.template dim<0>() - 1) };
    for (u32 i = 0; i < x.template dim<0>(); ++i) extended(i) = x(i);
    for (u32 i = x.template dim<0>(); i < extended.template dim<0>(); ++i) extended(i) = conj(x(extended.template dim<0>() - i));
    return ifft(extended);
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0>
void low_pass_filter(Tensor<T, 1> &audio, T sample_rate, T band_limit = 0) {
    if (band_limit == 0) band_limit = sample_rate / 2;
    u32 cutoff_index = (u32)std::round(band_limit * audio.template dim<0>() / sample_rate);
    auto F = rfft(audio);
    for (u32 i = cutoff_index + 1; i < audio.template dim<0>(); ++i) audio(i) = 0;
    auto f = irfft(F);
    for (u32 i = 0; i < audio.template dim<0>(); ++i) audio(i) = f(i).real;
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0>
void normalize_audio(Tensor<T, 1> &audio) {
    T s = 0;
    for (u32 i = 0; i < audio.template dim<0>(); ++i) s = std::max(s, std::abs(audio(i)));
    if (s != 0) audio /= s;
}

template<typename T>
std::vector<Tensor<T, 1>> overlapping_chunks(Tensor<T, 1> &audio, u32 size) {
    std::vector<Tensor<T, 1>> res;
    for (u32 i = 0; i * (size / 2) + size <= audio.template dim<0>(); ++i) {
        res.emplace_back(&audio(i * (size / 2)), nullptr, +size);
    }
    return res;
}

template<typename T>
void hann_window(Tensor<T, 1> &x) {
    for (u32 i = 0; i < x.template dim<0>(); ++i) {
        T t = std::cos((T)PI * ((T)i - (T)x.template dim<0>() / 2) / (T)x.template dim<0>());
        x(i) = t * t;
    }
}

#endif
