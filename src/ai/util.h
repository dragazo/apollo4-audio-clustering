#ifndef A3EM_AI_UTIL_H
#define A3EM_AI_UTIL_H

#include <vector>
#include <cmath>
#include <cstring>

#include "./tensor.h"

#define PI 3.14159265358979323846

void *operator new(std::size_t s) noexcept(noexcept(operator new(1))) {
    return std::malloc(s);
}
void *operator new[](std::size_t s) noexcept(noexcept(operator new[](1))) {
    return std::malloc(s);
}

void operator delete(void *p) noexcept {
    std::free(p);
}
void operator delete(void *p, std::size_t) noexcept {
    std::free(p);
}

void operator delete[](void *p) noexcept {
    std::free(p);
}
void operator delete[](void *p, std::size_t) noexcept {
    std::free(p);
}

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
void mul_hann_window(Tensor<T, 1> &x) {
    for (u32 i = 0; i < x.template dim<0>(); ++i) {
        T t = std::cos((T)PI * ((T)i - (T)x.template dim<0>() / 2) / (T)x.template dim<0>());
        x(i) *= t * t;
    }
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0>
Tensor<complicate_t<T>, 2> spectrogram(Tensor<T, 1> &audio, u32 fft_size, T sample_rate) {
    low_pass_filter(audio, sample_rate);
    normalize_audio(audio);

    const u32 chunks_cap = (audio.template dim<0>() - fft_size) / (fft_size / 2) + 2;
    Tensor<T, 1> chunks[chunks_cap];
    std::memset(&chunks, 0, sizeof(chunks));
    u32 chunks_len = 0;
    while (chunks_len * (fft_size / 2) + fft_size <= audio.template dim<0>()) {
        if (chunks_len >= chunks_cap) THROW(std::runtime_error("chunks overflow"));
        chunks[chunks_len] = Tensor<T, 1> { &audio(chunks_len * (fft_size / 2)), nullptr, +fft_size };
        ++chunks_len;
    }

    for (u32 i = 0; i < chunks_len; ++i) {
        chunks[i] = static_cast<Tensor<T, 1>&&>(chunks[i]).into_owned();
        mul_hann_window(chunks[i]);
    }

    Tensor<complicate_t<T>, 2> res { new complicate_t<T>[chunks_len * (fft_size / 2)], [](auto *v) { delete[] v; }, chunks_len, fft_size / 2 };
    for (u32 i = 0; i < chunks_len; ++i) {
        Tensor<complicate_t<T>, 1> F = fft(chunks[i]);
        for (u32 j = 0; j < fft_size / 2; ++j) res(i, j) = F(j);
    }
    return res;
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0> T freq_to_mel(T f) {
    return 1127 * std::log(1 + f / 700);
}
template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0> T mel_to_freq(T m) {
    return 700 * (std::exp(m / 1127) - 1);
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0>
Tensor<T, 2> dct(u32 in_filters, u32 out_filters) {
    T t1 = 1 / std::sqrt((T)in_filters);
    T t2 = std::sqrt(2 / (T)in_filters);
    T t3 = (T)PI / (2 * (T)in_filters);

    Tensor<T, 2> res { new T[in_filters * out_filters], [](auto *v) { delete[] v; }, out_filters, in_filters };
    for (u32 j = 0; j < in_filters; ++j) res(0, j) = t1;
    for (u32 i = 1; i < out_filters; ++i) {
        for (u32 j = 0; j < in_filters; ++j) res(i, j) = std::cos(i * (1 + 2 * j) * t3) * t2;
    }
    return res;
}

template<typename T> Tensor<T, 1> linspace(T a, T b, u32 num) {
    Tensor<T, 1> res { new T[num], [](auto *v) { delete[] v; }, num };
    T step = (b - a) / (num - 1);
    T val = a;
    for (u32 i = 0; i < res.template dim<0>(); ++i, val += step) res(i) = val;
    return res;
}

template<typename T> Tensor<T, 2> transpose(const Tensor<T, 2> &x) {
    Tensor<T, 2> res { new T[x.size()], [](auto *v) { delete[] v; }, x.template dim<1>(), x.template dim<0>() };
    for (u32 i = 0; i < x.template dim<0>(); ++i) {
        for (u32 j = 0; j < x.template dim<1>(); ++j) {
            res(j, i) = x(i, j);
        }
    }
    return res;
}

template<typename T> Tensor<T, 2> matmul(const Tensor<T, 2> &a, const Tensor<T, 2> &b) {
    if (a.template dim<1>() != b.template dim<0>()) THROW(std::runtime_error("matmul incompatible sizes"));

    Tensor<T, 2> res { new T[a.template dim<0>() * b.template dim<1>()], [](auto *v) { delete[] v; }, a.template dim<0>(), b.template dim<1>() };
    for (u32 i = 0; i < res.template dim<0>(); ++i) {
        for (u32 j = 0; j < res.template dim<1>(); ++j) {
            T acc = (T)0;
            for (u32 k = 0; k < a.template dim<1>(); ++k) acc += a(i, k) * b(k, j);
            res(i, j) = acc;
        }
    }
    return res;
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0>
Tensor<T, 2> mfcc_spectrogram(Tensor<T, 1> &signal, u32 fft_size, T sample_rate, u32 mel_filters, u32 dct_filters) {
    Tensor<T, 1> mel_freqs = linspace(freq_to_mel((T)0), freq_to_mel(sample_rate / (T)2), mel_filters + 2);
    for (u32 i = 0; i < mel_freqs.template dim<0>(); ++i) mel_freqs(i) = mel_to_freq(mel_freqs(i));

    Tensor<u32, 1> filter_points { new u32[mel_freqs.template dim<0>()], [](auto *v) { delete[] v; }, mel_freqs.template dim<0>() };
    for (u32 i = 0; i < filter_points.template dim<0>(); ++i) filter_points(i) = ((T)fft_size / sample_rate) * mel_freqs(i);

    u32 size = mel_filters * (fft_size / 2);
    T *p = new T[size];
    for (u32 i = 0; i < size; ++i) p[i] = (T)0;
    Tensor<T, 2> filters { p, [](auto *v) { delete[] v; }, mel_filters, fft_size / 2 };

    for (u32 n = 0; n < mel_filters; ++n) {
        T s = (T)2 / (mel_freqs(n + 2) - mel_freqs(n));
        Tensor<T, 1> temp = linspace((T)0, (T)1, filter_points(n + 1) - filter_points(n));
        for (u32 m = filter_points(n); m < filter_points(n + 1); ++m) filters(n, m) = s * temp(m - filter_points(n));
        temp = linspace((T)1, (T)0, filter_points(n + 2) - filter_points(n + 1));
        for (u32 m = filter_points(n + 1); m < filter_points(n + 2); ++m) filters(n, m) = s * temp(m - filter_points(n + 1));
    }

    Tensor<Complex<T>, 2> power_complex = spectrogram(signal, fft_size, sample_rate);
    Tensor<T, 2> power_trans { new T[power_complex.size()], [](auto *v) { delete[] v; }, power_complex.template dim<1>(), power_complex.template dim<0>() };
    for (u32 i = 0; i < power_trans.template dim<0>(); ++i) {
        for (u32 j = 0; j < power_trans.template dim<1>(); ++j) {
            power_trans(i, j) = sqr_mag(power_complex(j, i));
        }
    }

    Tensor<T, 2> filtered = matmul(filters, power_trans);
    for (u32 i = 0; i < filtered.template dim<0>(); ++i) {
        for (u32 j = 0; j < filtered.template dim<1>(); ++j) {
            if (filtered(i, j) > 0) {
                filtered(i, j) = 10 * std::log10(filtered(i, j));
            }
        }
    }

    return matmul(dct<T>(mel_filters, dct_filters), filtered);
}

template<typename T, std::enable_if_t<std::is_same<T, simplify_t<T>>::value, int> = 0>
Tensor<T, 2> mfcc_spectrogram_for_learning(Tensor<T, 1> &signal, T sample_rate) {
    u32 fft_size = (u32)(i32)((T)30 / (T)1000 * sample_rate);

    if ((i32)fft_size <= 0) THROW(std::runtime_error("mfcc_spectrogram_for_learning: input too small!"));

    Tensor<T, 2> s = mfcc_spectrogram(signal, fft_size, sample_rate, 32, 16);

    T std = s.std();

    s -= s.mean();
    if (std > (T)0) s /= std;
    s.maximum((T)(-1));
    s.minimum((T)(+1));

    return s;
}

#endif
