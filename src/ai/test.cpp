#include <iostream>
#include <cassert>
#include <cmath>

#include "./tensor.h"
#include "./util.h"
#include "./tf.h"

template<typename T>
void deleter(T *v) { delete[] v; }

#ifndef NO_EXCEPTIONS
#define TRY try
#define CATCH(body) catch (const std::exception &x) body
#else
#define TRY
#define CATCH(body) {}
#endif

int main() {
    std::cout << "starting tests...\n";

    TRY { // tensor

        Tensor<f32, 1> vec{new f32[52], deleter, 52};
        Tensor<f64, 2> tab{new f64[30], deleter, 6, 5};

        assert(vec.dim<0>() == 52);
        assert(tab.dim<0>() == 6);
        assert(tab.dim<1>() == 5);

        u32 v = 0;
        for (u32 i = 0; i < 6; ++i) {
            for (u32 j = 0; j < 5; ++j) {
                tab(i, j) = v++;
            }
        }
        for (u32 i = 0; i < 30; ++i) {
            assert(static_cast<f64*>(&tab(0, 0))[i] == i);
        }

        f32 data[16];
        Tensor<f32, 3> ref{data, nullptr, 2, 4, 4};
    } CATCH({
        std::cout << "!!!! tensor error: " << x.what() << '\n';
        throw;
    })

    TRY { // complex
        c32 a = c32 {5, 7} * c32 {-4, 1};
        assert(a.real == -27 && a.imag == -23);
        c32 b = c32 {6, 2} + c32 { 4, -8 };
        assert(b.real == 10 && b.imag == -6);
        c32 c = c32 {6, 2} * 2.0f;
        assert(c.real == 12 && c.imag == 4);
        c32 d = 3.0f * c32 {4, -2};
        assert(d.real == 12 && d.imag == -6);
    } CATCH({
        std::cout << "!!!! complex error: " << x.what() << '\n';
        throw;
    })

    TRY { // fft
        c32 sig_raw[] = { {1, 2}, {3, -2}, {-2, 0}, {1, 11}, {8, -5}, {4, 0}, {0, -3}, {0, 0}, {1, 4}, {-5, 3} };
        Tensor<c32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };
        assert(sig.dim<0>() == 10);

        Tensor<c32, 1> sig_fft = fft(sig);
        assert(sig_fft.dim<0>() == 10);
        assert(std::abs(sig_fft(0).real - 11.00000) < 0.01 && std::abs(sig_fft(0).imag - 10.000000) < 0.01);
        assert(std::abs(sig_fft(1).real - -9.16531) < 0.01 && std::abs(sig_fft(1).imag - -0.384416) < 0.01);
        assert(std::abs(sig_fft(2).real - -4.81585) < 0.01 && std::abs(sig_fft(2).imag - -9.947230) < 0.01);
        assert(std::abs(sig_fft(3).real - -9.06369) < 0.01 && std::abs(sig_fft(3).imag - -11.51050) < 0.01);
        assert(std::abs(sig_fft(4).real - 12.64840) < 0.01 && std::abs(sig_fft(4).imag - 8.4941500) < 0.01);
        assert(std::abs(sig_fft(5).real - 4.999980) < 0.01 && std::abs(sig_fft(5).imag - -14.00000) < 0.01);
        assert(std::abs(sig_fft(6).real - -12.3566) < 0.01 && std::abs(sig_fft(6).imag - 16.102600) < 0.01);
        assert(std::abs(sig_fft(7).real - 12.48010) < 0.01 && std::abs(sig_fft(7).imag - 21.274400) < 0.01);
        assert(std::abs(sig_fft(8).real - 18.52400) < 0.01 && std::abs(sig_fft(8).imag - -14.64950) < 0.01);
        assert(std::abs(sig_fft(9).real - -14.2511) < 0.01 && std::abs(sig_fft(9).imag - 14.620500) < 0.01);

        Tensor<c32, 1> sig_fft_ifft = ifft(sig_fft);
        assert(sig_fft_ifft.dim<0>() == 10);
        assert(std::abs(sig_fft_ifft(0).real - 1.00) < 0.01 && std::abs(sig_fft_ifft(0).imag - 2.00) < 0.01);
        assert(std::abs(sig_fft_ifft(1).real - 3.00) < 0.01 && std::abs(sig_fft_ifft(1).imag - -2.0) < 0.01);
        assert(std::abs(sig_fft_ifft(2).real - -2.0) < 0.01 && std::abs(sig_fft_ifft(2).imag - 0.00) < 0.01);
        assert(std::abs(sig_fft_ifft(3).real - 1.00) < 0.01 && std::abs(sig_fft_ifft(3).imag - 11.0) < 0.01);
        assert(std::abs(sig_fft_ifft(4).real - 8.00) < 0.01 && std::abs(sig_fft_ifft(4).imag - -5.0) < 0.01);
        assert(std::abs(sig_fft_ifft(5).real - 4.00) < 0.01 && std::abs(sig_fft_ifft(5).imag - 0.00) < 0.01);
        assert(std::abs(sig_fft_ifft(6).real - 0.00) < 0.01 && std::abs(sig_fft_ifft(6).imag - -3.0) < 0.01);
        assert(std::abs(sig_fft_ifft(7).real - 0.00) < 0.01 && std::abs(sig_fft_ifft(7).imag - 0.00) < 0.01);
        assert(std::abs(sig_fft_ifft(8).real - 1.00) < 0.01 && std::abs(sig_fft_ifft(8).imag - 4.00) < 0.01);
        assert(std::abs(sig_fft_ifft(9).real - -5.0) < 0.01 && std::abs(sig_fft_ifft(9).imag - 3.00) < 0.01);
    } CATCH({
        std::cout << "!!!! fft error: " << x.what() << '\n';
        throw;
    })

    TRY { // rfft
        f32 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1};
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };
        assert(sig.dim<0>() == 10);

        Tensor<c32, 1> sig_rfft = rfft(sig);
        assert(sig_rfft.dim<0>() == 6);
        assert(std::abs(sig_rfft(0).real - 35.0000000000000000) < 0.01 && std::abs(sig_rfft(0).imag - 0.0000000000000000) < 0.01);
        assert(std::abs(sig_rfft(1).real - -7.0000000000000000) < 0.01 && std::abs(sig_rfft(1).imag - 1.4530850560107220) < 0.01);
        assert(std::abs(sig_rfft(2).real - -4.4721359549995805) < 0.01 && std::abs(sig_rfft(2).imag - 5.4288245463451460) < 0.01);
        assert(std::abs(sig_rfft(3).real - -7.0000000000000000) < 0.01 && std::abs(sig_rfft(3).imag - -6.155367074350506) < 0.01);
        assert(std::abs(sig_rfft(4).real - 4.47213595499958000) < 0.01 && std::abs(sig_rfft(4).imag - -4.530768593185974) < 0.01);
        assert(std::abs(sig_rfft(5).real - 3.00000000000000000) < 0.01 && std::abs(sig_rfft(5).imag - 0.0000000000000000) < 0.01);

        Tensor<c32, 1> sig_rfft_irfft = irfft(sig_rfft);
        assert(sig_rfft_irfft.dim<0>() == 10);
        assert(std::abs(sig_rfft_irfft(0).real - 1) < 0.01 && std::abs(sig_rfft_irfft(0).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(1).real - 2) < 0.01 && std::abs(sig_rfft_irfft(1).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(2).real - 3) < 0.01 && std::abs(sig_rfft_irfft(2).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(3).real - 4) < 0.01 && std::abs(sig_rfft_irfft(3).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(4).real - 5) < 0.01 && std::abs(sig_rfft_irfft(4).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(5).real - 6) < 0.01 && std::abs(sig_rfft_irfft(5).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(6).real - 2) < 0.01 && std::abs(sig_rfft_irfft(6).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(7).real - 3) < 0.01 && std::abs(sig_rfft_irfft(7).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(8).real - 8) < 0.01 && std::abs(sig_rfft_irfft(8).imag - 0) < 0.01);
        assert(std::abs(sig_rfft_irfft(9).real - 1) < 0.01 && std::abs(sig_rfft_irfft(9).imag - 0) < 0.01);
    } CATCH({
        std::cout << "!!!! rfft error: " << x.what() << '\n';
        throw;
    })

    TRY { // low_pass_filter
        f32 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1};
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };
        assert(sig.dim<0>() == 10);

        low_pass_filter(sig, 6.0f);

        assert(sig.dim<0>() == 10);
        assert(std::abs(sig(0) - 1) < 0.01);
        assert(std::abs(sig(1) - 2) < 0.01);
        assert(std::abs(sig(2) - 3) < 0.01);
        assert(std::abs(sig(3) - 4) < 0.01);
        assert(std::abs(sig(4) - 5) < 0.01);
        assert(std::abs(sig(5) - 6) < 0.01);
        assert(std::abs(sig(6) - 2) < 0.01);
        assert(std::abs(sig(7) - 3) < 0.01);
        assert(std::abs(sig(8) - 8) < 0.01);
        assert(std::abs(sig(9) - 1) < 0.01);
    } CATCH({
        std::cout << "!!!! low_pass_filter error: " << x.what() << '\n';
        throw;
    })

    TRY { // normalize_audio
        f32 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1};
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };

        normalize_audio(sig);

        assert(sig.dim<0>() == 10);
        assert(std::abs(sig(0) - 0.125) < 0.01);
        assert(std::abs(sig(1) - 0.250) < 0.01);
        assert(std::abs(sig(2) - 0.375) < 0.01);
        assert(std::abs(sig(3) - 0.500) < 0.01);
        assert(std::abs(sig(4) - 0.625) < 0.01);
        assert(std::abs(sig(5) - 0.750) < 0.01);
        assert(std::abs(sig(6) - 0.250) < 0.01);
        assert(std::abs(sig(7) - 0.375) < 0.01);
        assert(std::abs(sig(8) - 1.000) < 0.01);
        assert(std::abs(sig(9) - 0.125) < 0.01);
    } CATCH({
        std::cout << "!!!! normalize_audio error: " << x.what() << '\n';
        throw;
    })

    TRY { // mul_hann_window
        f32 sig_raw[] = {1, 1, 1, 1, 2, 1, 1, 1, 1, 3};
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };

        mul_hann_window(sig);

        assert(sig.dim<0>() == 10);
        assert(std::abs(sig(0) - 0.000) < 0.01);
        assert(std::abs(sig(1) - 0.095) < 0.01);
        assert(std::abs(sig(2) - 0.345) < 0.01);
        assert(std::abs(sig(3) - 0.655) < 0.01);
        assert(std::abs(sig(4) - 1.810) < 0.01);
        assert(std::abs(sig(5) - 1.000) < 0.01);
        assert(std::abs(sig(6) - 0.905) < 0.01);
        assert(std::abs(sig(7) - 0.655) < 0.01);
        assert(std::abs(sig(8) - 0.345) < 0.01);
        assert(std::abs(sig(9) - 0.285) < 0.01);
    } CATCH({
        std::cout << "!!!! mul_hann_window error: " << x.what() << '\n';
        throw;
    })

    TRY { // spectrogram
        f32 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1};
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };

        Tensor<Complex<f32>, 2> spec = spectrogram(sig, 6, 12.0f);
        assert(spec.dim<0>() == 2 && spec.dim<1>() == 3);
        assert(std::abs(spec(0, 0).real -  1.5000) < 0.01 && std::abs(spec(0, 0).imag -  0.0000) < 0.01);
        assert(std::abs(spec(0, 1).real - -0.7500) < 0.01 && std::abs(spec(0, 1).imag -  0.2706) < 0.01);
        assert(std::abs(spec(0, 2).real -  0.0000) < 0.01 && std::abs(spec(0, 2).imag - -0.0541) < 0.01);
        assert(std::abs(spec(1, 0).real -  1.5000) < 0.01 && std::abs(spec(1, 0).imag -  0.0000) < 0.01);
        assert(std::abs(spec(1, 1).real - -0.4687) < 0.01 && std::abs(spec(1, 1).imag - -0.1623) < 0.01);
        assert(std::abs(spec(1, 2).real - -0.3750) < 0.01 && std::abs(spec(1, 2).imag -  0.3248) < 0.01);
    } CATCH({
        std::cout << "!!!! spectrogram error: " << x.what() << '\n';
        throw;
    })

    TRY { // freq_to_mel / mel_to_freq
        assert(std::abs(freq_to_mel(921.0f) - 946.3624) < 0.01);
        assert(std::abs(freq_to_mel(391.0f) - 500.1284) < 0.01);

        assert(std::abs(mel_to_freq(129.0f) - 84.8899) < 0.01);
        assert(std::abs(mel_to_freq(1236.0f) - 1396.0235) < 0.01);
    } CATCH({
        std::cout << "!!!! freq_to_mel / mel_to_freq error: " << x.what() << '\n';
        throw;
    })

    TRY { // dct
        Tensor<f32, 2> t = dct<f32>(4, 6);
        assert(t.dim<0>() == 6 && t.dim<1>() == 4);
        assert(std::abs(t(0, 0) -  0.5) < 0.01);
        assert(std::abs(t(0, 1) -  0.5) < 0.01);
        assert(std::abs(t(0, 2) -  0.5) < 0.01);
        assert(std::abs(t(0, 3) -  0.5) < 0.01);
        assert(std::abs(t(1, 0) -  0.6533) < 0.01);
        assert(std::abs(t(1, 1) -  0.2706) < 0.01);
        assert(std::abs(t(1, 2) - -0.2706) < 0.01);
        assert(std::abs(t(1, 3) - -0.6533) < 0.01);
        assert(std::abs(t(2, 0) -  0.5) < 0.01);
        assert(std::abs(t(2, 1) - -0.5) < 0.01);
        assert(std::abs(t(2, 2) - -0.5) < 0.01);
        assert(std::abs(t(2, 3) -  0.5) < 0.01);
        assert(std::abs(t(3, 0) -  0.2706) < 0.01);
        assert(std::abs(t(3, 1) - -0.6533) < 0.01);
        assert(std::abs(t(3, 2) -  0.6533) < 0.01);
        assert(std::abs(t(3, 3) - -0.2706) < 0.01);
        assert(std::abs(t(4, 0) -  0) < 0.01);
        assert(std::abs(t(4, 1) -  0) < 0.01);
        assert(std::abs(t(4, 2) -  0) < 0.01);
        assert(std::abs(t(4, 3) -  0) < 0.01);
        assert(std::abs(t(5, 0) - -0.2706) < 0.01);
        assert(std::abs(t(5, 1) -  0.6533) < 0.01);
        assert(std::abs(t(5, 2) - -0.6533) < 0.01);
        assert(std::abs(t(5, 3) -  0.2706) < 0.01);
    } CATCH({
        std::cout << "!!!! dct error: " << x.what() << '\n';
        throw;
    })

    TRY { // linspace
        Tensor<f32, 1> p = linspace(23.0f, 175.0f, 7);
        assert(p.dim<0>() == 7);
        assert(std::abs(p(0) -  23.0000) < 0.01);
        assert(std::abs(p(1) -  48.3333) < 0.01);
        assert(std::abs(p(2) -  73.6666) < 0.01);
        assert(std::abs(p(3) -  99.0000) < 0.01);
        assert(std::abs(p(4) - 124.3333) < 0.01);
        assert(std::abs(p(5) - 149.6666) < 0.01);
        assert(std::abs(p(6) - 175.0000) < 0.01);
    } CATCH({
        std::cout << "!!!! linspace error: " << x.what() << '\n';
        throw;
    })

    TRY { // transpose
        f32 sig_raw[] = {1, 2, 3, 4, 5, 6};
        Tensor<f32, 2> sig { sig_raw, nullptr, 3, 2 };

        Tensor<f32, 2> p = transpose(sig);
        assert(p.dim<0>() == 2 && p.dim<1>() == 3);
        assert(std::abs(p(0, 0) - 1) < 0.01);
        assert(std::abs(p(0, 1) - 3) < 0.01);
        assert(std::abs(p(0, 2) - 5) < 0.01);
        assert(std::abs(p(1, 0) - 2) < 0.01);
        assert(std::abs(p(1, 1) - 4) < 0.01);
        assert(std::abs(p(1, 2) - 6) < 0.01);
    } CATCH({
        std::cout << "!!!! transpose error: " << x.what() << '\n';
        throw;
    })

    TRY { // matmul
        f32 a_raw[] = { 1, 2, 3, 4, 5, 6 };
        Tensor<f32, 2> a { a_raw, nullptr, 2, 3 };

        f32 b_raw[] = { 7, 2, 1, 3, 3, 4, 7, 3, 1, 5, 1, 2 };
        Tensor<f32, 2> b { b_raw, nullptr, 3, 4 };

        Tensor<f32, 2> x = matmul(a, b);
        assert(x.dim<0>() == 2 && x.dim<1>() == 4);
        assert(std::abs(x(0, 0) - 16) < 0.01);
        assert(std::abs(x(0, 1) - 25) < 0.01);
        assert(std::abs(x(0, 2) - 18) < 0.01);
        assert(std::abs(x(0, 3) - 15) < 0.01);
        assert(std::abs(x(1, 0) - 49) < 0.01);
        assert(std::abs(x(1, 1) - 58) < 0.01);
        assert(std::abs(x(1, 2) - 45) < 0.01);
        assert(std::abs(x(1, 3) - 39) < 0.01);
    } CATCH({
        std::cout << "!!!! matmul error: " << x.what() << '\n';
        throw;
    })

    TRY { // mfcc spectrogram
        f64 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1, 7, 2, 5, 2, 6, 4, 7, 2, 4, 7, 1, 3, 6, 3, 1, 6};
        Tensor<f64, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };

        Tensor<f64, 2> x = mfcc_spectrogram(sig, 8, 16.0, 7, 3);
        assert(x.dim<0>() == 3 && x.dim<1>() == 5);
        assert(std::abs(x(0, 0) -  0.05911126) < 0.0001);
        assert(std::abs(x(0, 1) -  0.54501131) < 0.0001);
        assert(std::abs(x(0, 2) - -3.64432206) < 0.0001);
        assert(std::abs(x(0, 3) -  0.36432148) < 0.0001);
        assert(std::abs(x(0, 4) -  0.19648368) < 0.0001);
        assert(std::abs(x(1, 0) -  6.32131747) < 0.0001);
        assert(std::abs(x(1, 1) -  5.58810832) < 0.0001);
        assert(std::abs(x(1, 2) -  8.37222244) < 0.0001);
        assert(std::abs(x(1, 3) -  6.41954012) < 0.0001);
        assert(std::abs(x(1, 4) -  3.81325159) < 0.0001);
        assert(std::abs(x(2, 0) - -1.55129303) < 0.0001);
        assert(std::abs(x(2, 1) - -0.87868320) < 0.0001);
        assert(std::abs(x(2, 2) - -0.30648488) < 0.0001);
        assert(std::abs(x(2, 3) - -1.05776904) < 0.0001);
        assert(std::abs(x(2, 4) -  1.11236952) < 0.0001);
    } CATCH({
        std::cout << "!!!! mfcc spectrogram error: " << x.what() << '\n';
        throw;
    })

    TRY { // mfcc spectrogram for learning
        f64 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1, 7, 2, 5, 2, 6, 4, 7, 2, 4, 7, 1, 3, 6, 3, 1, 6};
        Tensor<f64, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };

        Tensor<f64, 2> x = mfcc_spectrogram_for_learning(sig, 200.0);
        assert(x.dim<0>() == 65 && x.dim<1>() == 7);
        assert(std::abs(x( 0, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x( 0, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x( 0, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x( 0, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x( 0, 4) - 0.43088771) < 0.0001);
        assert(std::abs(x( 0, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x( 0, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x( 1, 0) - 0.82208827) < 0.0001);
        assert(std::abs(x( 1, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x( 1, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x( 1, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x( 1, 4) - 0.73544054) < 0.0001);
        assert(std::abs(x( 1, 5) - 0.84287696) < 0.0001);
        assert(std::abs(x( 1, 6) - 0.99460552) < 0.0001);
        assert(std::abs(x( 2, 0) - 0.28403959) < 0.0001);
        assert(std::abs(x( 2, 1) - 0.85568659) < 0.0001);
        assert(std::abs(x( 2, 2) - 0.20345964) < 0.0001);
        assert(std::abs(x( 2, 3) - 0.68445204) < 0.0001);
        assert(std::abs(x( 2, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x( 2, 5) - 0.15790724) < 0.0001);
        assert(std::abs(x( 2, 6) - 0.71444008) < 0.0001);
        assert(std::abs(x( 3, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x( 3, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x( 3, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x( 3, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x( 3, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x( 3, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x( 3, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x( 4, 0) - 0.30838649) < 0.0001);
        assert(std::abs(x( 4, 1) - 0.92903310) < 0.0001);
        assert(std::abs(x( 4, 2) - 0.22089950) < 0.0001);
        assert(std::abs(x( 4, 3) - 0.74312092) < 0.0001);
        assert(std::abs(x( 4, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x( 4, 5) - 0.17144251) < 0.0001);
        assert(std::abs(x( 4, 6) - 0.77567944) < 0.0001);
        assert(std::abs(x( 5, 0) - 0.75359708) < 0.0001);
        assert(std::abs(x( 5, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x( 5, 2) - 0.94789890) < 0.0001);
        assert(std::abs(x( 5, 3) - 0.98780447) < 0.0001);
        assert(std::abs(x( 5, 4) - 0.67416830) < 0.0001);
        assert(std::abs(x( 5, 5) - 0.77265380) < 0.0001);
        assert(std::abs(x( 5, 6) - 0.91174130) < 0.0001);
        assert(std::abs(x( 6, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x( 6, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x( 6, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x( 6, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x( 6, 4) - 0.60865564) < 0.0001);
        assert(std::abs(x( 6, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x( 6, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x( 7, 0) - 0.88865943) < 0.0001);
        assert(std::abs(x( 7, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x( 7, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x( 7, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x( 7, 4) - 0.79499513) < 0.0001);
        assert(std::abs(x( 7, 5) - 0.91113155) < 0.0001);
        assert(std::abs(x( 7, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x( 8, 0) - 0.25902930) < 0.0001);
        assert(std::abs(x( 8, 1) - 0.78034157) < 0.0001);
        assert(std::abs(x( 8, 2) - 0.18554459) < 0.0001);
        assert(std::abs(x( 8, 3) - 0.62418459) < 0.0001);
        assert(std::abs(x( 8, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x( 8, 5) - 0.14400318) < 0.0001);
        assert(std::abs(x( 8, 6) - 0.65153212) < 0.0001);
        assert(std::abs(x( 9, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x( 9, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x( 9, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x( 9, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x( 9, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x( 9, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x( 9, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(10, 0) - 0.33201314) < 0.0001);
        assert(std::abs(x(10, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(10, 2) - 0.23782344) < 0.0001);
        assert(std::abs(x(10, 3) - 0.80005421) < 0.0001);
        assert(std::abs(x(10, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(10, 5) - 0.18457737) < 0.0001);
        assert(std::abs(x(10, 6) - 0.83510715) < 0.0001);
        assert(std::abs(x(11, 0) - 0.68334584) < 0.0001);
        assert(std::abs(x(11, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(11, 2) - 0.85953460) < 0.0001);
        assert(std::abs(x(11, 3) - 0.89572013) < 0.0001);
        assert(std::abs(x(11, 4) - 0.61132150) < 0.0001);
        assert(std::abs(x(11, 5) - 0.70062606) < 0.0001);
        assert(std::abs(x(11, 6) - 0.82674765) < 0.0001);
        assert(std::abs(x(12, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(12, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(12, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(12, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(12, 4) - 0.60652249) < 0.0001);
        assert(std::abs(x(12, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(12, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(13, 0) - 0.95315508) < 0.0001);
        assert(std::abs(x(13, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(13, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(13, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(13, 4) - 0.85269297) < 0.0001);
        assert(std::abs(x(13, 5) - 0.97725815) < 0.0001);
        assert(std::abs(x(13, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(14, 0) - 0.23341404) < 0.0001);
        assert(std::abs(x(14, 1) - 0.70317403) < 0.0001);
        assert(std::abs(x(14, 2) - 0.16719619) < 0.0001);
        assert(std::abs(x(14, 3) - 0.56245933) < 0.0001);
        assert(std::abs(x(14, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(14, 5) - 0.12976278) < 0.0001);
        assert(std::abs(x(14, 6) - 0.58710248) < 0.0001);
        assert(std::abs(x(15, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(15, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(15, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(15, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(15, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(15, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(15, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(16, 0) - 0.35486435) < 0.0001);
        assert(std::abs(x(16, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(16, 2) - 0.25419194) < 0.0001);
        assert(std::abs(x(16, 3) - 0.85511893) < 0.0001);
        assert(std::abs(x(16, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(16, 5) - 0.19728113) < 0.0001);
        assert(std::abs(x(16, 6) - 0.89258444) < 0.0001);
        assert(std::abs(x(17, 0) - 0.61149861) < 0.0001);
        assert(std::abs(x(17, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(17, 2) - 0.76916282) < 0.0001);
        assert(std::abs(x(17, 3) - 0.80154379) < 0.0001);
        assert(std::abs(x(17, 4) - 0.54704694) < 0.0001);
        assert(std::abs(x(17, 5) - 0.62696198) < 0.0001);
        assert(std::abs(x(17, 6) - 0.73982310) < 0.0001);
        assert(std::abs(x(18, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(18, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(18, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(18, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(18, 4) - 0.60297278) < 0.0001);
        assert(std::abs(x(18, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(18, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(19, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(19, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(19, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(19, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(19, 4) - 0.90839931) < 0.0001);
        assert(std::abs(x(19, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(19, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(20, 0) - 0.20725363) < 0.0001);
        assert(std::abs(x(20, 1) - 0.62436420) < 0.0001);
        assert(std::abs(x(20, 2) - 0.14845729) < 0.0001);
        assert(std::abs(x(20, 3) - 0.49942041) < 0.0001);
        assert(std::abs(x(20, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(20, 5) - 0.11521932) < 0.0001);
        assert(std::abs(x(20, 6) - 0.52130163) < 0.0001);
        assert(std::abs(x(21, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(21, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(21, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(21, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(21, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(21, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(21, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(22, 0) - 0.37688677) < 0.0001);
        assert(std::abs(x(22, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(22, 2) - 0.26996675) < 0.0001);
        assert(std::abs(x(22, 3) - 0.90818648) < 0.0001);
        assert(std::abs(x(22, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(22, 5) - 0.20952414) < 0.0001);
        assert(std::abs(x(22, 6) - 0.94797705) < 0.0001);
        assert(std::abs(x(23, 0) - 0.53822320) < 0.0001);
        assert(std::abs(x(23, 1) - 0.92863596) < 0.0001);
        assert(std::abs(x(23, 2) - 0.67699462) < 0.0001);
        assert(std::abs(x(23, 3) - 0.70549540) < 0.0001);
        assert(std::abs(x(23, 4) - 0.48149472) < 0.0001);
        assert(std::abs(x(23, 5) - 0.55183360) < 0.0001);
        assert(std::abs(x(23, 6) - 0.65117066) < 0.0001);
        assert(std::abs(x(24, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(24, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(24, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(24, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(24, 4) - 0.59801481) < 0.0001);
        assert(std::abs(x(24, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(24, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(25, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(25, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(25, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(25, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(25, 4) - 0.96198404) < 0.0001);
        assert(std::abs(x(25, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(25, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(26, 0) - 0.18060917) < 0.0001);
        assert(std::abs(x(26, 1) - 0.54409613) < 0.0001);
        assert(std::abs(x(26, 2) - 0.12937167) < 0.0001);
        assert(std::abs(x(26, 3) - 0.43521508) < 0.0001);
        assert(std::abs(x(26, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(26, 5) - 0.10040676) < 0.0001);
        assert(std::abs(x(26, 6) - 0.45428326) < 0.0001);
        assert(std::abs(x(27, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(27, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(27, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(27, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(27, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(27, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(27, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(28, 0) - 0.39802895) < 0.0001);
        assert(std::abs(x(28, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(28, 2) - 0.28511105) < 0.0001);
        assert(std::abs(x(28, 3) - 0.95913292) < 0.0001);
        assert(std::abs(x(28, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(28, 5) - 0.22127779) < 0.0001);
        assert(std::abs(x(28, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(29, 0) - 0.46369074) < 0.0001);
        assert(std::abs(x(29, 1) - 0.80003964) < 0.0001);
        assert(std::abs(x(29, 2) - 0.58324527) < 0.0001);
        assert(std::abs(x(29, 3) - 0.60779930) < 0.0001);
        assert(std::abs(x(29, 4) - 0.41481795) < 0.0001);
        assert(std::abs(x(29, 5) - 0.47541639) < 0.0001);
        assert(std::abs(x(29, 6) - 0.56099738) < 0.0001);
        assert(std::abs(x(30, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(30, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(30, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(30, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(30, 4) - 0.59166014) < 0.0001);
        assert(std::abs(x(30, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(30, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(31, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(31, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(31, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(31, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(31, 4) - 1.00000000) < 0.0001);
        assert(std::abs(x(31, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(31, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(32, 0) - 0.15354289) < 0.0001);
        assert(std::abs(x(32, 1) - 0.46255731) < 0.0001);
        assert(std::abs(x(32, 2) - 0.10998389) < 0.0001);
        assert(std::abs(x(32, 3) - 0.36999329) < 0.0001);
        assert(std::abs(x(32, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(32, 5) - 0.08535970) < 0.0001);
        assert(std::abs(x(32, 6) - 0.38620388) < 0.0001);
        assert(std::abs(x(33, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(33, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(33, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(33, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(33, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(33, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(33, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(34, 0) - 0.41824151) < 0.0001);
        assert(std::abs(x(34, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(34, 2) - 0.29958945) < 0.0001);
        assert(std::abs(x(34, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(34, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(34, 5) - 0.23251464) < 0.0001);
        assert(std::abs(x(34, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(35, 0) - 0.38807531) < 0.0001);
        assert(std::abs(x(35, 1) - 0.66957479) < 0.0001);
        assert(std::abs(x(35, 2) - 0.48813373) < 0.0001);
        assert(std::abs(x(35, 3) - 0.50868366) < 0.0001);
        assert(std::abs(x(35, 4) - 0.34717235) < 0.0001);
        assert(std::abs(x(35, 5) - 0.39788882) < 0.0001);
        assert(std::abs(x(35, 6) - 0.46951387) < 0.0001);
        assert(std::abs(x(36, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(36, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(36, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(36, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(36, 4) - 0.58392362) < 0.0001);
        assert(std::abs(x(36, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(36, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(37, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(37, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(37, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(37, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(37, 4) - 1.00000000) < 0.0001);
        assert(std::abs(x(37, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(37, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(38, 0) - 0.12611800) < 0.0001);
        assert(std::abs(x(38, 1) - 0.37993816) < 0.0001);
        assert(std::abs(x(38, 2) - 0.09033925) < 0.0001);
        assert(std::abs(x(38, 3) - 0.30390735) < 0.0001);
        assert(std::abs(x(38, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(38, 5) - 0.07011327) < 0.0001);
        assert(std::abs(x(38, 6) - 0.31722251) < 0.0001);
        assert(std::abs(x(39, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(39, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(39, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(39, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(39, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(39, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(39, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(40, 0) - 0.43747725) < 0.0001);
        assert(std::abs(x(40, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(40, 2) - 0.31336816) < 0.0001);
        assert(std::abs(x(40, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(40, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(40, 5) - 0.24320844) < 0.0001);
        assert(std::abs(x(40, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(41, 0) - 0.31155351) < 0.0001);
        assert(std::abs(x(41, 1) - 0.53754612) < 0.0001);
        assert(std::abs(x(41, 2) - 0.39188212) < 0.0001);
        assert(std::abs(x(41, 3) - 0.40837996) < 0.0001);
        assert(std::abs(x(41, 4) - 0.27871591) < 0.0001);
        assert(std::abs(x(41, 5) - 0.31943197) < 0.0001);
        assert(std::abs(x(41, 6) - 0.37693378) < 0.0001);
        assert(std::abs(x(42, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(42, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(42, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(42, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(42, 4) - 0.57482332) < 0.0001);
        assert(std::abs(x(42, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(42, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(43, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(43, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(43, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(43, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(43, 4) - 1.00000000) < 0.0001);
        assert(std::abs(x(43, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(43, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(44, 0) - 0.09839855) < 0.0001);
        assert(std::abs(x(44, 1) - 0.29643164) < 0.0001);
        assert(std::abs(x(44, 2) - 0.07048360) < 0.0001);
        assert(std::abs(x(44, 3) - 0.23711163) < 0.0001);
        assert(std::abs(x(44, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(44, 5) - 0.05470309) < 0.0001);
        assert(std::abs(x(44, 6) - 0.24750025) < 0.0001);
        assert(std::abs(x(45, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(45, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(45, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(45, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(45, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(45, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(45, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(46, 0) - 0.45569124) < 0.0001);
        assert(std::abs(x(46, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(46, 2) - 0.32641497) < 0.0001);
        assert(std::abs(x(46, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(46, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(46, 5) - 0.25333422) < 0.0001);
        assert(std::abs(x(46, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(47, 0) - 0.23430406) < 0.0001);
        assert(std::abs(x(47, 1) - 0.40426198) < 0.0001);
        assert(std::abs(x(47, 2) - 0.29471526) < 0.0001);
        assert(std::abs(x(47, 3) - 0.30712247) < 0.0001);
        assert(std::abs(x(47, 4) - 0.20960852) < 0.0001);
        assert(std::abs(x(47, 5) - 0.24022906) < 0.0001);
        assert(std::abs(x(47, 6) - 0.28347335) < 0.0001);
        assert(std::abs(x(48, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(48, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(48, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(48, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(48, 4) - 0.56438049) < 0.0001);
        assert(std::abs(x(48, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(48, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(49, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(49, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(49, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(49, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(49, 4) - 1.00000000) < 0.0001);
        assert(std::abs(x(49, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(49, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(50, 0) - 0.07044930) < 0.0001);
        assert(std::abs(x(50, 1) - 0.21223280) < 0.0001);
        assert(std::abs(x(50, 2) - 0.05046335) < 0.0001);
        assert(std::abs(x(50, 3) - 0.16976212) < 0.0001);
        assert(std::abs(x(50, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(50, 5) - 0.03916515) < 0.0001);
        assert(std::abs(x(50, 6) - 0.17719995) < 0.0001);
        assert(std::abs(x(51, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(51, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(51, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(51, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(51, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(51, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(51, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(52, 0) - 0.47284094) < 0.0001);
        assert(std::abs(x(52, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(52, 2) - 0.33869943) < 0.0001);
        assert(std::abs(x(52, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(52, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(52, 5) - 0.26286832) < 0.0001);
        assert(std::abs(x(52, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(53, 0) - 0.15650739) < 0.0001);
        assert(std::abs(x(53, 1) - 0.27003368) < 0.0001);
        assert(std::abs(x(53, 2) - 0.19686008) < 0.0001);
        assert(std::abs(x(53, 3) - 0.20514768) < 0.0001);
        assert(std::abs(x(53, 4) - 0.14001158) < 0.0001);
        assert(std::abs(x(53, 5) - 0.16046509) < 0.0001);
        assert(std::abs(x(53, 6) - 0.18935085) < 0.0001);
        assert(std::abs(x(54, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(54, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(54, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(54, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(54, 4) - 0.55261953) < 0.0001);
        assert(std::abs(x(54, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(54, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(55, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(55, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(55, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(55, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(55, 4) - 1.00000000) < 0.0001);
        assert(std::abs(x(55, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(55, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(56, 0) - 0.04233550) < 0.0001);
        assert(std::abs(x(56, 1) - 0.12753828) < 0.0001);
        assert(std::abs(x(56, 2) - 0.03032523) < 0.0001);
        assert(std::abs(x(56, 3) - 0.10201613) < 0.0001);
        assert(std::abs(x(56, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(56, 5) - 0.02353574) < 0.0001);
        assert(std::abs(x(56, 6) - 0.10648578) < 0.0001);
        assert(std::abs(x(57, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(57, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(57, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(57, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(57, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(57, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(57, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(58, 0) - 0.48888631) < 0.0001);
        assert(std::abs(x(58, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(58, 2) - 0.35019284) < 0.0001);
        assert(std::abs(x(58, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(58, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(58, 5) - 0.27178848) < 0.0001);
        assert(std::abs(x(58, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(59, 0) - 0.07834518) < 0.0001);
        assert(std::abs(x(59, 1) - 0.13517469) < 0.0001);
        assert(std::abs(x(59, 2) - 0.09854512) < 0.0001);
        assert(std::abs(x(59, 3) - 0.10269377) < 0.0001);
        assert(std::abs(x(59, 4) - 0.07008764) < 0.0001);
        assert(std::abs(x(59, 5) - 0.08032635) < 0.0001);
        assert(std::abs(x(59, 6) - 0.09478611) < 0.0001);
        assert(std::abs(x(60, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(60, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(60, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(60, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(60, 4) - 0.53956790) < 0.0001);
        assert(std::abs(x(60, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(60, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(61, 0) - 1.00000000) < 0.0001);
        assert(std::abs(x(61, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(61, 2) - 1.00000000) < 0.0001);
        assert(std::abs(x(61, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(61, 4) - 1.00000000) < 0.0001);
        assert(std::abs(x(61, 5) - 1.00000000) < 0.0001);
        assert(std::abs(x(61, 6) - 1.00000000) < 0.0001);
        assert(std::abs(x(62, 0) - 0.01412283) < 0.0001);
        assert(std::abs(x(62, 1) - 0.04254588) < 0.0001);
        assert(std::abs(x(62, 2) - 0.01011629) < 0.0001);
        assert(std::abs(x(62, 3) - 0.03403187) < 0.0001);
        assert(std::abs(x(62, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(62, 5) - 0.00785136) < 0.0001);
        assert(std::abs(x(62, 6) - 0.03552292) < 0.0001);
        assert(std::abs(x(63, 0) - 0.00000000) < 0.0001);
        assert(std::abs(x(63, 1) - 0.00000000) < 0.0001);
        assert(std::abs(x(63, 2) - 0.00000000) < 0.0001);
        assert(std::abs(x(63, 3) - 0.00000000) < 0.0001);
        assert(std::abs(x(63, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(63, 5) - 0.00000000) < 0.0001);
        assert(std::abs(x(63, 6) - 0.00000000) < 0.0001);
        assert(std::abs(x(64, 0) - 0.50378985) < 0.0001);
        assert(std::abs(x(64, 1) - 1.00000000) < 0.0001);
        assert(std::abs(x(64, 2) - 0.36086836) < 0.0001);
        assert(std::abs(x(64, 3) - 1.00000000) < 0.0001);
        assert(std::abs(x(64, 4) - 0.00000000) < 0.0001);
        assert(std::abs(x(64, 5) - 0.28007387) < 0.0001);
        assert(std::abs(x(64, 6) - 1.00000000) < 0.0001);
    } CATCH({
        std::cout << "!!!! mfcc spectrogram for learning error: " << x.what() << '\n';
        throw;
    })

    TRY { // inference
        f32 sig_raw[] = {
-0.0011146776378154755, 0.0042790696024894714, -0.008131816983222961, -0.020017728209495544, -0.016952985897660255, -0.018140768632292747, -0.032759666442871094, -0.033158864825963974, -0.03552606329321861, -0.03607349097728729, 
-0.037852607667446136, -0.045238982886075974, -0.05475057289004326, -0.046342022716999054, -0.04243842884898186, -0.05098174512386322, -0.05029767006635666, -0.05304955691099167, -0.052316442131996155, -0.04782833158969879, 
-0.04930320382118225, -0.04925704002380371, -0.03555312007665634, -0.027229420840740204, -0.023845188319683075, -0.026167597621679306, -0.017169147729873657, -0.01604258269071579, -0.012408807873725891, -0.011941693723201752, 
-0.012505557388067245, -0.005231164395809174, -0.006649836897850037, -0.007717922329902649, -0.012473344802856445, -0.012255609035491943, -0.014763034880161285, -0.02100856602191925, -0.027960196137428284, -0.022268515080213547, 
-0.02044537477195263, -0.020224474370479584, -0.022853847593069077, -0.01692270301282406, -0.012865878641605377, -0.0112812090665102, -0.010233333334326744, -0.012327775359153748, -0.006673172116279602, -0.0022673457860946655, 
-0.003045029938220978, -0.005691833794116974, 0.006727328523993492, 0.01286817155778408, 0.018671199679374695, 0.011753659695386887, 0.015073470771312714, 0.019869886338710785, 0.023131152614951134, 0.025490358471870422, 
0.025713548064231873, 0.034085310995578766, 0.03207699954509735, 0.03616304323077202, 0.03124743513762951, 0.02727394551038742, 0.03307810053229332, 0.035354919731616974, 0.03223871439695358, 0.03978327661752701, 
0.04308519512414932, 0.040861114859580994, 0.030916769057512283, 0.02703152224421501, 0.026259705424308777, 0.027142956852912903, 0.03178238868713379, 0.028402183204889297, 0.02543441206216812, 0.021249987185001373, 
0.015955638140439987, 0.0037562698125839233, 0.003214355558156967, 0.0016836896538734436, -0.0018625110387802124, -0.004097111523151398, -0.0009089978411793709, -0.0004855021834373474, -0.0009008646011352539, -0.005703270435333252, 
-0.010620906949043274, -0.012772839516401291, -0.016562525182962418, -0.013602513819932938, -0.01130477711558342, -0.015698283910751343, -0.015460909344255924, -0.013218555599451065, -0.017191123217344284, -0.009265486150979996, 
-0.00734698586165905, -0.0050867958925664425, -0.009535577148199081, -0.008282884955406189, -0.004808824509382248, 0.00029610469937324524, 0.0008846689015626907, -0.0024366453289985657, 0.0014964593574404716, -0.006062094122171402, 
-0.011139054782688618, -0.01472838968038559, -0.015150236897170544, -0.015335597097873688, -0.010159745812416077, -0.013843156397342682, -0.021274983882904053, -0.024405092000961304, -0.02321244776248932, -0.026390358805656433, 
-0.032743971794843674, -0.031003665179014206, -0.028913186863064766, -0.02552545815706253, -0.026737842708826065, -0.030920781195163727, -0.03071010857820511, -0.02560049295425415, -0.022674545645713806, -0.018800534307956696, 
-0.017933260649442673, -0.01634461060166359, -0.016091516241431236, -0.010961204767227173, -0.010154731571674347, -0.004944780841469765, 0.00022513419389724731, -0.0005023553967475891, 0.009418297559022903, 0.012564614415168762, 
0.014540795236825943, 0.010000674985349178, 0.012195044197142124, 0.008493753150105476, 0.006609200965613127, 0.010691527277231216, 0.013944609090685844, 0.0169247854501009, 0.010897967964410782, 0.011671427637338638, 
0.010622058063745499, 0.005605719983577728, 0.005035017617046833, 0.013241061940789223, 0.004587562754750252, 0.0007568774744868279, 0.003531835973262787, -0.0026777395978569984, 3.3892691135406494e-05, 0.002950318157672882, 
0.0034243203699588776, 0.009067105129361153, 0.012043392285704613, 0.01413218304514885, 0.008079783990979195, 0.0052179004997015, 0.012477684766054153, 0.010116945020854473, 0.01291266642510891, 0.0110022546723485, 
0.015830032527446747, 0.015300986357033253, 0.007208019495010376, 0.007628876715898514, 0.012986834160983562, 0.01738610491156578, 0.008297896012663841, 0.0024958979338407516, 0.003970641642808914, 0.005075471475720406, 
0.007007853128015995, 0.003049100749194622, -0.000758766196668148, 0.0030339080840349197, -0.0026239752769470215, -0.005561963655054569, -0.010846883058547974, -0.010174430906772614, -0.009081416763365269, -0.009406445547938347, 
-0.01571863517165184, -0.021278899163007736, -0.016941189765930176, -0.017755774781107903, -0.017133696004748344, -0.02138381078839302, -0.012328099459409714, -0.0130035150796175, -0.009786868467926979, -0.015925375744700432, 
-0.015390634536743164, -0.00858783908188343, -0.011868055909872055, -0.008208675310015678, -0.009468577802181244, -0.0031084176152944565, -0.0029226019978523254, -0.004510257393121719, -0.007246237248182297, -0.002063339576125145, 
0.005678892135620117, 0.014515332877635956, 0.01001027226448059, 0.0055953264236450195, 0.00740524847060442, 0.016403689980506897, 0.012073714286088943, 0.0027305148541927338, 0.00799337774515152, 0.005356276407837868, 
0.009007468819618225, 0.012097243219614029, 0.008238081820309162, 0.015111112967133522, 0.019225910305976868, 0.007293839007616043, 0.008257387205958366, 0.011061737313866615, 0.01436676923185587, 0.01301938109099865, 
0.012607153505086899, 0.012562878429889679, 0.015234138816595078, 0.012636866420507431, 0.009647822007536888, 0.011664777994155884, 0.01363370567560196, 0.012335065752267838, 0.013546518981456757, 0.015453426167368889, 
0.01689709722995758, 0.023075789213180542, 0.016824882477521896, 0.01732715778052807, 0.01582956686615944, 0.027138687670230865, 0.02905123680830002, 0.03043411299586296, 0.029413003474473953, 0.023799318820238113, 
0.02488509938120842, 0.012707188725471497, 0.015011336654424667, 0.01802445389330387, 0.014482177793979645, 0.014447529800236225, 0.014776887372136116, 0.00976850837469101, 0.016147181391716003, 0.01497950404882431, 
0.007751502096652985, 0.008249524980783463, 0.0069429390132427216, 0.006424260325729847, 0.0016046874225139618, -0.0004888307303190231, 0.0017269104719161987, 0.0015973672270774841, -0.002279140055179596, -0.0074465274810791016, 
-0.005303170531988144, -0.006444333121180534, -0.003326975740492344, -0.0027815811336040497, -0.02569304220378399, -0.018416160717606544, -0.014452997595071793, -0.02030113711953163, -0.014959082007408142, -0.016123708337545395, 
-0.025974245741963387, -0.027705185115337372, -0.004754912108182907, -0.01923522725701332, -0.019588712602853775, -0.007219996303319931, -0.02767610177397728, -0.024033132940530777, -0.020161230117082596, -0.02612505480647087, 
-0.026386186480522156, -0.02032679319381714, -0.014495853334665298, -0.011348400264978409, -0.011429778300225735, -0.018646178767085075, -0.010486483573913574, -0.0033476054668426514, -0.005459073930978775, -0.009350303560495377, 
-0.0035896189510822296, 0.007644381374120712, -0.0003217468038201332, -0.003964155912399292, -0.007950987666845322, -0.007803354412317276, 0.004361823201179504, 0.011920195072889328, 0.011005882173776627, 0.007212616503238678, 
-0.007795931771397591, -0.012539783492684364, 0.00645188894122839, 0.018208559602499008, 0.007632569409906864, 0.005458994768559933, 0.009938862174749374, -0.006125792860984802, 0.007714550942182541, 0.007805082947015762, 
-0.006678842008113861, 0.004766538739204407, 0.017958300188183784, 0.012057004496455193, 0.01840609312057495, 0.010183781385421753, 0.006299607455730438, 0.02481243386864662, 0.027577992528676987, 0.008388867601752281, 
0.02029058337211609, 0.038580700755119324, 0.010837018489837646, 0.01636602357029915, 0.01417875662446022, 0.010042384266853333, 0.015916718170046806, 0.017531566321849823, -0.0017986930906772614, 0.01226256974041462, 
0.008647128939628601, 7.914379239082336e-05, 0.01824384555220604, 0.007938776165246964, 0.017169807106256485, 0.01884620450437069, 0.012765780091285706, -0.002982604317367077, 0.013269077986478806, 0.02369186095893383, 
-0.001953856088221073, 0.0006205905228853226, -0.009453430771827698, -0.012435279786586761, 0.0038089901208877563, -0.00422760471701622, -0.01327421236783266, 0.0009522922337055206, 0.0014025336131453514, -0.013109570369124413, 
0.007173473946750164, -0.0008836705237627029, -0.0003612414002418518, 0.011117637157440186, -0.004747748374938965, -0.00588226318359375, -0.0028029978275299072, 0.0017965659499168396, -0.0014043785631656647, -0.0006181821227073669, 
-0.008178070187568665, -0.020197924226522446, -0.015197288244962692, -0.008500732481479645, -0.017728166654706, -0.011445725336670876, -0.0022622295655310154, -0.017487185075879097, -0.014203693717718124, -0.018051058053970337, 
-0.026417076587677002, -0.027021296322345734, -0.02484152838587761, -0.03159337118268013, -0.029900843277573586, -0.01368163712322712, -0.05326978862285614, -0.05223754048347473, -0.03411369025707245, -0.032883062958717346, 
-0.02945389598608017, -0.026684381067752838, -0.014080960303544998, -0.017261838540434837, -0.0016281455755233765, -0.022116616368293762, -0.011181708425283432, 0.008409930393099785, -0.009129859507083893, -0.004215791821479797, 
-0.00679556280374527, -0.024017538875341415, -0.006542284041643143, -0.015394076704978943, -0.018844854086637497, 0.014815445989370346, -0.011214062571525574, -0.0025536492466926575, 0.01287345215678215, 0.006417658179998398, 
0.021727606654167175, 0.002951212227344513, -0.015186269767582417, 0.006237173452973366, 0.01851724088191986, 0.029483910650014877, 0.020041368901729584, 0.008154995739459991, 0.01310804858803749, 0.016858894377946854, 
0.029440395534038544, -0.007521744817495346, 0.0005926713347434998, 0.01466621458530426, -0.0003657415509223938, 0.027298904955387115, 0.008203353732824326, 0.0010228510946035385, 0.0157662034034729, 0.02365899085998535, 
0.012340240180492401, 0.0178961381316185, 0.015141675248742104, 0.0018560737371444702, 0.020732931792736053, 0.004248756915330887, -0.0067339688539505005, 0.009836051613092422, 0.005139097571372986, -0.00021018832921981812, 
0.014814946800470352, -0.004343658685684204, -0.004413913935422897, 0.01547570712864399, 0.020417235791683197, 0.018451157957315445, 0.025218581780791283, 0.0022615380585193634, -0.008308383636176586, 0.007482776418328285, 
0.019553320482373238, 0.007192555814981461, -0.017730969935655594, -0.0050486549735069275, -0.007819551974534988, 0.0023369714617729187, -0.004814751446247101, -0.014714784920215607, -0.018448079004883766, -0.011338554322719574, 
-0.009902704507112503, -0.02553793042898178, -0.03153669089078903, -0.03346708416938782, -0.02377270720899105, -0.030508145689964294, -0.046563632786273956, -0.036463916301727295, -0.016191808506846428, -0.021285902708768845, 
-0.033137544989585876, -0.013608507812023163, -0.016124587506055832, -0.0064012035727500916, -0.0008961111307144165, -0.015017164871096611, 0.004163671284914017, -0.01816185750067234, 0.0003382489085197449, 0.016808312386274338, 
-0.013613829389214516, -0.013971254229545593, -0.018863938748836517, -0.037900425493717194, -0.010292034596204758, -0.005575060844421387, -0.02189009077847004, -0.020941738039255142, -0.022782690823078156, -0.03146332502365112, 
-0.005095824599266052, -0.008411416783928871, -0.02636798471212387, -0.01968894526362419, -0.024735629558563232, -0.033759452402591705, -0.03277631476521492, -0.0016140937805175781, -0.025273894891142845, -0.005629591643810272, 
-0.02567577362060547, -0.03665151819586754, -0.004263322800397873, -0.0014537610113620758, -0.015779653564095497, -0.009843788109719753, -0.012374671176075935, -0.04860356077551842, -0.030902205035090446, -0.01972021535038948, 
-0.0020618848502635956, 0.03943963348865509, 0.028222061693668365, 0.05198223516345024, 0.07487227767705917, 0.07538269460201263, 0.0680311843752861, 0.05096152424812317, 0.05990542471408844, 0.029405884444713593, 
0.07298353314399719, 0.0583333894610405, 0.0399554967880249, 0.07226958870887756, 0.06228508800268173, 0.025884918868541718, 0.06803129613399506, 0.12551744282245636, 0.1378432661294937, 0.16750559210777283, 
0.12997892498970032, 0.12500125169754028, 0.140016570687294, 0.10213032364845276, 0.05563052371144295, -0.01661037467420101, -0.056056033819913864, -0.04196607321500778, -0.04963769391179085, -0.046509869396686554, 
-0.05837858468294144, -0.05893966555595398, -0.07030829787254333, -0.09985940158367157, -0.13491539657115936, -0.15890341997146606, -0.16936857998371124, -0.1552949845790863, -0.11307781934738159, -0.07904688268899918, 
-0.03921294957399368, -0.01685260981321335, 0.024881906807422638, 0.02538428269326687, -0.004719849675893784, 0.045623380690813065, 0.08195053786039352, 0.12870359420776367, 0.1572451889514923, 0.14998266100883484, 
0.15964122116565704, 0.16023719310760498, 0.13843423128128052, 0.10053928941488266, 0.08040764927864075, 0.05315078794956207, 0.07496457546949387, 0.06554188579320908, 0.0443447083234787, 0.04134335368871689, 
0.04203095659613609, 0.004836244508624077, -0.003395107574760914, -0.0041262321174144745, -0.026705076918005943, -0.027398213744163513, -0.04143650829792023, -0.05979720503091812, -0.111249640583992, -0.11297635734081268, 
-0.15473970770835876, -0.18538054823875427, -0.16449424624443054, -0.1648152768611908, -0.16848613321781158, -0.10253788530826569, -0.0863625630736351, -0.0972815603017807, -0.04784432798624039, -0.08195459097623825, 
-0.07846330106258392, -0.04212737828493118, -0.051946789026260376, -0.03487378731369972, 0.002503900323063135, 0.047526370733976364, 0.08486706018447876, 0.10044534504413605, 0.11496232450008392, 0.08858492225408554, 
0.08892457187175751, 0.13524028658866882, 0.09779120981693268, 0.09312540292739868, 0.08789289742708206, 0.024738185107707977, 0.048574890941381454, 0.04291903227567673, 0.02757919579744339, 0.06406690180301666, 
0.02348841354250908, -0.01666240580379963, -0.05488155037164688, -0.08518729358911514, -0.0372297428548336, -0.007072377949953079, -0.01864830031991005, -0.017207542434334755, 0.011585928499698639, -0.0037630684673786163, 
0.03911428153514862, 0.0005511865019798279, -0.029111681506037712, -0.03653509542346001, -0.05912051349878311, -0.009462859481573105, -0.04252908006310463, 0.008331824094057083, 0.010633362457156181, -0.011765174567699432, 
-0.015822164714336395, -0.012303143739700317, -0.007723476737737656, -0.007983453571796417, 0.014180295169353485, 0.016626212745904922, 0.0381825789809227, 0.012201271019876003, 0.02318509668111801, 0.025336673483252525, 
0.025744043290615082, 0.0205194391310215, -0.01983209326863289, 0.0042631058022379875, 0.04492788761854172, 0.068677619099617, 0.0629851222038269, 0.06716708093881607, 0.06630463898181915, 0.047189127653837204, 
0.07043500244617462, 0.05661555379629135, 0.03631773591041565, 0.047460541129112244, 0.044202372431755066, 0.003780517727136612, -0.002491312101483345, -0.0010620318353176117, -0.03267914056777954, -0.024929488077759743, 
-0.05498047173023224, -0.0809887945652008, -0.06061478331685066, -0.05971711501479149, -0.09039926528930664, -0.07194659113883972, -0.05910675972700119, -0.09886986017227173, -0.10143382102251053, -0.0971260592341423, 
-0.09085939824581146, -0.09000936150550842, -0.06281536817550659, -0.05093827843666077, -0.08001523464918137, -0.051824167370796204, -0.06000968813896179, -0.08748973160982132, -0.05526355281472206, -0.03318656608462334, 
-0.06095639616250992, -0.05900631472468376, -0.008551369421184063, -0.019422736018896103, -0.003325115889310837, 0.02257460728287697, -0.016981007531285286, 0.014554055407643318, 0.04942650347948074, -0.003621235489845276, 
-0.010605547577142715, 0.02680136263370514, 0.031207038089632988, 0.03690056502819061, 0.01591292768716812, 0.022379055619239807, 0.026392027735710144, 0.004997319541871548, 0.047233276069164276, 0.05462762713432312, 
0.013028143905103207, 0.00020088069140911102, -0.027680300176143646, -0.011624125763773918, 0.006185438483953476, -0.02714603766798973, -0.017195917665958405, -0.017642220482230186, 0.01757502555847168, 0.022964192554354668, 
0.0003235079348087311, -0.008257957175374031, -0.00148741714656353, 0.010847722180187702, -0.011918829753994942, 0.023622311651706696, 0.03062387928366661, 0.03719592094421387, 0.03026585280895233, -0.0072065796703100204, 
0.04248277097940445, 0.03886399790644646, 0.012449238449335098, 0.010207576677203178, 0.004385867156088352, 0.00756518729031086, 0.004321367479860783, 0.07977500557899475, 0.06431496143341064, 0.0021807439625263214, 
0.004497501999139786, 0.03044925630092621, 0.034385353326797485, 0.022816380485892296, 0.08905883133411407, 0.05669839680194855, 0.02540956810116768, 0.03726842999458313, 0.03898167982697487, 0.011942770332098007, 
0.015622604638338089, 0.0781785249710083, 0.02917415089905262, 0.03374490141868591, 0.006739890202879906, 0.014951095916330814, 0.05529514327645302, -0.007502177730202675, -0.0021974537521600723, -0.029632773250341415, 
-0.036999259144067764, -0.0037911850959062576, -0.03089463710784912, -0.09910538047552109, -0.0828065350651741, -0.07645530998706818, -0.08905676007270813, -0.048166148364543915, -0.09404732286930084, -0.04811722785234451, 
-0.01696460321545601, -0.029247978702187538, 0.034650109708309174, 0.025805620476603508, 0.06511367857456207, 0.07713688164949417, 0.07796823233366013, 0.05976491421461105, 0.0034416625276207924, 0.0346088781952858, 
-0.010605357587337494, 0.023000996559858322, -0.019710345193743706, -0.032309696078300476, -0.03824234753847122, -0.1625952422618866, -0.11788373440504074, -0.07972133159637451, -0.09127912670373917, -0.11197374761104584, 
-0.011295866221189499, 0.039044082164764404, -0.03566928952932358, -0.024775052443146706, 0.002829229924827814, -0.009950374253094196, 0.016251156106591225, 0.013301841914653778, 0.008175564929842949, 0.005910065025091171, 
-0.05442307889461517, -0.021081997081637383, 0.012398691847920418, -0.06077655777335167, -0.07230690866708755, -0.03142600506544113, -0.012379101477563381, 0.0617682971060276, 0.005439991131424904, -0.03744722157716751, 
0.0900614783167839, 0.04214077442884445, 0.022000819444656372, 0.057001154869794846, 0.028707381337881088, 0.0900135189294815, 0.05836381018161774, 0.011086851358413696, 0.006310975179076195, 0.09597041457891464, 
0.08479409664869308, 0.03215275704860687, 0.021459240466356277, 0.00215205829590559, 0.0404215008020401, 0.058594681322574615, 0.11860030889511108, 0.037817828357219696, -0.02614312618970871, -0.03948364406824112, 
-0.020564042031764984, 0.014127155765891075, 0.011474167928099632, 0.019039954990148544, 0.04538267105817795, 0.008217554539442062, -0.024664070457220078, 0.008625337854027748, 0.015974977985024452, 0.044536978006362915, 
0.07321204245090485, 0.10711447149515152, 0.07146979868412018, 0.05478260666131973, 0.02608768455684185, -0.030394352972507477, -0.052376940846443176, -0.026974482461810112, -0.045259565114974976, -0.053050871938467026, 
-0.018231656402349472, -0.055406130850315094, -0.08888866752386093, -0.07997731864452362, -0.02251816913485527, 0.0025126617401838303, -0.0017780466005206108, -0.00973374955356121, -0.05038776248693466, -0.04454069212079048, 
-0.02929450199007988, -0.029194707050919533, -0.02796284854412079, -0.019036339595913887, -0.035529524087905884, -0.1418180763721466, -0.07308049499988556, -0.05477587878704071, -0.09220568835735321, 0.01303668599575758, 
0.06878672540187836, 0.004193536005914211, -0.03121103346347809, 0.023062318563461304, 0.023130977526307106, 0.035517219454050064, 0.04333258420228958, 0.011232700198888779, -0.0008977609686553478, 0.06809630990028381, 
0.01577341929078102, -0.0429585725069046, -0.04725301265716553, -0.07790958136320114, -0.07160447537899017, -0.028673438355326653, -0.05867493152618408, -0.0486452579498291, -0.01689668744802475, -0.06737136840820312, 
-0.021995171904563904, 0.03249119967222214, -0.08546515554189682, -0.09711968898773193, 0.052663810551166534, -0.02215743251144886, -0.07320321351289749, -0.08313968032598495, -0.043531157076358795, -0.020086856558918953, 
0.0777556300163269, 0.03127501532435417, -0.03305630385875702, 0.12320967018604279, 0.13252682983875275, 0.11384747177362442, 0.06915759295225143, 0.006302696652710438, 0.0009845942258834839, 0.09180203080177307, 
0.07754340767860413, -0.022800087928771973, -0.046698197722435, -0.03623519837856293, -0.007969561964273453, 0.005780642852187157, -0.04860691353678703, -0.11188242584466934, -0.04432258754968643, 0.04282011464238167, 
-0.04160911589860916, 0.019592255353927612, 0.05177490413188934, 0.05893222987651825, 0.07952946424484253, 0.06915125250816345, 0.08504904806613922, 0.05248214676976204, 0.11269950866699219, 0.11536335945129395, 
0.0845276415348053, 0.033946335315704346, 0.05203147232532501, 0.021993674337863922, 0.05155673623085022, 0.03237801045179367, -0.047969356179237366, 0.010361653752624989, 0.015764400362968445, 0.02355889603495598, 
-0.00621446967124939, 0.017552470788359642, 0.029291583225131035, 0.032368626445531845, 0.1060718223452568, 0.0042280214838683605, -0.03740524500608444, -0.005734991282224655, 0.013344766572117805, 0.05746756121516228, 
0.09620887041091919, 0.018585437908768654, -0.13941282033920288, -0.020988522097468376, -0.04542680084705353, -0.09326569736003876, 0.037731021642684937, 0.005977562628686428, -0.07872743904590607, -0.10077351331710815, 
-0.05285392329096794, -0.09078522026538849, -0.02892760932445526, 0.06576845794916153, 0.024096064269542694, -0.013488535769283772, -0.08746218681335449, -0.010943321511149406, -0.023202890530228615, -0.0254618301987648, 
-0.03560209274291992, -0.10788768529891968, 0.10084407031536102, -0.017399687319993973, -0.0739765614271164, -0.08328655362129211, -0.09930165112018585, 0.012309471145272255, -0.06337833404541016, -0.014211899600923061, 
-0.029872393235564232, 0.07507719099521637, 0.06221745163202286, 0.06368176639080048, 0.07837409526109695, -0.02475288137793541, 0.014389767311513424, 0.022313114255666733, -0.0723513513803482, -0.08609987795352936, 
-0.038221295922994614, -0.11804704368114471, -0.10237036645412445, 0.022670693695545197, 0.028111688792705536, 0.043143466114997864, 0.16570588946342468, 0.02922821044921875, 0.018006393685936928, 0.06770843267440796, 
0.05838775634765625, 0.09264145791530609, 0.08491243422031403, 0.028620745986700058, -0.019415773451328278, 0.00657446775585413, -0.06548410654067993, -0.0011116333771497011, -0.015382619574666023, -0.10536094754934311, 
-0.11117522418498993, -0.09561076760292053, -0.04790135473012924, -0.09630327671766281, -0.078518807888031, 0.015384767204523087, 0.0672001764178276, 0.08582527935504913, 0.08635710179805756, 0.1353064477443695, 
0.12487466633319855, 0.0560855008661747, 0.06357306241989136, 0.049978841096162796, 0.028771191835403442, 0.012061478570103645, 0.039169423282146454, 0.08847706019878387, 0.028854696080088615, 0.04099499434232712, 
-0.001290764193981886, -0.021770834922790527, 0.0228872187435627, -0.024497462436556816, -0.06944508850574493, -0.09250332415103912, -0.039040084928274155, 0.05488874018192291, 0.05090367794036865, -0.03586766868829727, 
0.05203410983085632, 0.1011374220252037, 0.06299680471420288, -0.0076629347167909145, 0.07271765172481537, 0.13471300899982452, 0.0887933075428009, 0.11236423254013062, 0.025848394259810448, -0.061560921370983124, 
-0.10036452859640121, 0.027239978313446045, 0.017360785976052284, -0.027422968298196793, -0.06855371594429016, -0.1533471643924713, -0.16705235838890076, -0.16505315899848938, -0.038482628762722015, -0.03365525230765343, 
-0.12420870363712311, -0.08381250500679016, -0.06376948952674866, -0.10973916947841644, -0.0036215810105204582, 0.05522257089614868, 0.01039422582834959, -0.018123764544725418, -0.017193598672747612, 0.015527908690273762, 
-0.06967228651046753, -0.14816978573799133, -0.2027696818113327, -0.24288567900657654, -0.10215230286121368, 0.02348681539297104, -0.01275250781327486, 0.026148557662963867, 0.05967343598604202, 0.10909150540828705, 
0.24524334073066711, 0.22555065155029297, 0.18840989470481873, 0.24388006329536438, 0.14992552995681763, -0.015992645174264908, 0.07799340039491653, 0.1136368066072464, -0.03726861625909805, -0.013163428753614426, 
-0.0769638791680336, -0.25195980072021484, -0.23811763525009155, -0.12838077545166016, -0.045855022966861725, -0.11937471479177475, -0.12671616673469543, -0.06327006220817566, -0.0019116406328976154, 0.10680074989795685, 
0.04160268232226372, -0.06879094243049622, 0.01282055489718914, 0.06826724112033844, 0.030835293233394623, 0.008324025198817253, -0.049843065440654755, 0.01699289120733738, 0.05018550157546997, -0.04575567692518234, 
-0.10093092918395996, -0.0038771377876400948, 0.06685894727706909, 0.028242327272892, 0.14707130193710327, 0.11348449438810349, 0.10161571949720383, 0.25026506185531616, 0.21139603853225708, 0.06891744583845139, 
0.06559331715106964, 0.16103368997573853, 0.1945025771856308, 0.11949120461940765, 0.008296718820929527, 0.05276702716946602, 0.01621745526790619, -0.01750253513455391, -0.11175145953893661, -0.06304161250591278, 
-0.1595401167869568, -0.3096786141395569, -0.12212857604026794, -0.11577200889587402, -0.07860392332077026, -0.03614559397101402, -0.059480275958776474, -0.09914371371269226, -0.13010704517364502, 0.023342514410614967, 
0.03322052210569382, -0.03948327898979187, 0.04126531630754471, 0.04490339010953903, 0.10970793664455414, 0.0990511029958725, 0.12303824722766876, 0.09915351867675781, 0.08532698452472687, 0.03370940685272217, 
-0.013579259626567364, 0.08036279678344727, -0.02435878850519657, -0.08010707050561905, -0.041856616735458374, -0.11809021234512329, -0.12653005123138428, 0.09130420535802841, 0.05660213530063629, -0.019051028415560722, 
0.009433216415345669, -0.08796684443950653, -0.10239199548959732, 0.0849454253911972, 0.11202481389045715, -0.03384914994239807, -0.07100430130958557, -0.03064274974167347, -0.07644589245319366, -0.048291683197021484, 
-0.12525589764118195, -0.23780623078346252, -0.006343083456158638, -0.0844656378030777, -0.1527324914932251, -0.07545088231563568, -0.03449193388223648, -0.04325846582651138, -0.07663334906101227, -0.07686764746904373, 
-0.00269504077732563, 0.10547465831041336, 0.11999443173408508, 0.12610125541687012, -0.03195991367101669, 0.03082660213112831, 0.06038753315806389, 0.11904136091470718, 0.1269625425338745, 0.019488561898469925, 
0.08450333029031754, -0.02404594235122204, 0.11086291819810867, 0.12218451499938965, 0.009135667234659195, 0.08429744094610214, -0.02110179141163826, -0.019154440611600876, -0.03404989093542099, -0.025490961968898773, 
0.04826357215642929, -0.0036535682156682014, -0.01160597987473011, -0.04569990932941437, -0.012706514447927475, 0.0075379908084869385, 0.1126469075679779, 0.12663879990577698, -0.05670171603560448, 0.041481710970401764, 
0.016074655577540398, 0.0397789366543293, 0.06262516230344772, -0.0023125410079956055, 0.044655412435531616, 0.02132413163781166, -0.018333829939365387, -0.054865024983882904, 0.02922942116856575, -0.010604925453662872, 
-0.09193375706672668, -0.08485309779644012, 0.019852682948112488, 0.06985548138618469, 0.00161655992269516, 0.09913815557956696, 0.09359495341777802, -0.004039723426103592, 0.04364469274878502, 0.12691821157932281, 
0.06597504019737244, 0.04429672285914421, 0.07987751811742783, -0.006436735391616821, -0.00023248419165611267, -0.026409795507788658, -0.05516061559319496, -0.10787690430879593, -0.09433034062385559, -0.11937029659748077, 
-0.08470438420772552, -0.028390459716320038, -0.04770854860544205, -0.005714703351259232, -0.07821275293827057, -0.020116087049245834, -0.0800633579492569, -0.05973487347364426, 0.010566245764493942, -0.031924888491630554, 
-0.018801018595695496, -0.033498167991638184, -0.050000838935375214, -0.0005749687552452087, -0.021488726139068604, -0.100592240691185, -0.09379277378320694, -0.035775989294052124, -0.014146756380796432, -0.09966575354337692, 
-0.030658336356282234, -0.03594636917114258, 0.004706878215074539, 0.020336803048849106, -0.07252807915210724, 0.04797575995326042, 0.03530822694301605, 0.026244111359119415, 0.04091271385550499, 0.04754030704498291, 
0.048881493508815765, 0.04321888089179993, 0.08302861452102661, -0.00964423269033432, -0.0005291029810905457, 0.03229065239429474, 0.04052484780550003, 0.020615361630916595, -0.015392199158668518, -0.04268956184387207, 
-0.08014889061450958, -0.09419228881597519, -0.06860434263944626, -0.061392441391944885, -0.01218331791460514, -0.0005821604281663895, -0.01974971778690815, -0.0003422833979129791, -0.03819279372692108, -0.017113372683525085, 
0.042956747114658356, 0.08956670761108398, -0.02819623425602913, 0.10958296805620193, 0.08631428331136703, -0.04565941169857979, 0.058993782848119736, -0.013751577585935593, 0.024977486580610275, 0.004684723913669586, 
-0.018545864149928093, 0.07353032380342484, -0.002164319157600403, 0.06651546061038971, 0.03969968110322952, 0.055632226169109344, 0.10141310095787048, -0.03271320089697838, 0.1111556887626648, 0.06468342989683151, 
0.05799296498298645, 0.09510402381420135, 0.04490673914551735, 0.03500225767493248, -0.002311423420906067, 0.031993746757507324, 0.01087883859872818, -0.030305631458759308, 0.0020607635378837585, -0.06878672540187836, 
-0.04158826544880867, 0.044946715235710144, -0.04474432021379471, 0.03709240257740021, 0.02836623601615429, -0.06074070185422897, -0.06531041115522385, -0.009542509913444519, 0.05753625929355621, 0.014312801882624626, 
-0.017812512814998627, -0.03571401536464691, -0.008939914405345917, 0.00455308984965086, -0.05985131487250328, -0.04230272024869919, -0.00716618075966835, -0.03372988849878311, -0.08679554611444473, -0.05096615478396416, 
-0.01052604615688324, 0.015025906264781952, -0.0351295992732048, -0.11010240018367767, -0.037383951246738434, -0.048445917665958405, 0.01332114264369011, 0.08884108066558838, 0.04201635718345642, -0.07320544123649597, 
-0.064607635140419, -0.01970572955906391, 0.01136401854455471, 0.029276013374328613, -0.08322635293006897, -0.020529381930828094, 0.024699022993445396, -0.03519415110349655, -0.05655510351061821, -0.04377456381917, 
-0.029165351763367653, -0.04906540364027023, -0.04053649306297302, -0.028630955144762993, -0.019798988476395607, 0.002958521246910095, -0.03885353356599808, -0.06409118324518204, -0.02117224596440792, 0.0846395492553711, 
0.09506013989448547, 0.021633166819810867, -0.01774219423532486, 0.02328098565340042, -0.01687679998576641, 0.028135471045970917, 0.10031399130821228, -0.04855778440833092, -0.0023729968816041946, 0.021761808544397354, 
0.007619233801960945, -0.02431379444897175, 0.04116168990731239, 0.040989868342876434, 0.017028924077749252, 0.08851035684347153, 0.05702053755521774, 0.04727427661418915, 0.02612517587840557, 0.0025485344231128693, 
0.017169568687677383, 0.08752427995204926, 0.019484851509332657, 0.02781977690756321, 0.04862403869628906, 0.023753389716148376, 0.043296895921230316, 0.02246902510523796, -0.01736336573958397, -0.009002409875392914, 
0.0056002140045166016, -0.04141738638281822, -0.011878319084644318, 0.043605975806713104, 0.047338247299194336, 0.021949075162410736, 0.00923224538564682, -0.020350661128759384, 0.00016140099614858627, 0.03287824988365173, 
-0.020264167338609695, 0.011312214657664299, 0.07325360924005508, -0.008749697357416153, 0.00683191791176796, 0.050119537860155106, 0.005337521433830261, 0.001270337961614132, 0.010972587391734123, 0.010821238160133362, 
0.043197207152843475, 0.039891816675662994, -0.04056325554847717, 0.002452416345477104, -0.019500667229294777, -0.03600526973605156, 0.03475039452314377, 0.0043318867683410645, -0.044451795518398285, -0.03068484552204609, 
-0.04708130285143852, -0.06816388666629791, -0.009506605565547943, -0.002615872770547867, -0.04272017627954483, -0.07083150744438171, -0.034788817167282104, 0.025788890197873116, 0.03261667490005493, -0.04186244681477547, 
-0.03662192076444626, -0.008339477702975273, 0.007706912234425545, -0.026587679982185364, 0.04844385385513306, 0.10540784150362015, -0.041730836033821106, -0.0006341468542814255, 0.02520427294075489, -0.0056476593017578125, 
0.002957291901111603, -0.011901428923010826, -0.05298548936843872, -0.024699926376342773, -0.028100168332457542, -0.03353452309966087, 0.005441056564450264, -0.003275565803050995, -0.0644037127494812, -0.051335908472537994, 
-0.019317876547574997, -0.06260248273611069, -0.010844465345144272, 0.012485467828810215, -0.029597723856568336, 0.009097634814679623, 0.041034549474716187, -0.04893279820680618, 0.0237965676933527, 0.00663658045232296, 
0.035086534917354584, 0.05130286514759064, 0.009938262403011322, 0.012420020997524261, -0.040450721979141235, 0.030868764966726303, 0.032436273992061615, 0.07059677690267563, -0.031004000455141068, -0.023537572473287582, 
0.04446139559149742, -0.057180166244506836, -0.02643704041838646, 0.017160270363092422, -0.033639293164014816, -0.0034402981400489807, -0.037623390555381775, 0.0004327744245529175, 0.09410882741212845, 0.06137504428625107, 
0.07860314100980759, 0.05632510036230087, 0.042022205889225006, 0.02592022530734539, 0.06725499033927917, 0.0651920810341835, 0.041607990860939026, -0.0066942982375621796, 0.05420876294374466, 0.0032739732414484024, 
-0.01719057932496071, 0.021486397832632065, -0.10452854633331299, -0.0437740758061409, 0.028920501470565796, -0.014043552801012993, -0.004738209769129753, -0.01823974773287773, 0.010777834802865982, 0.03719208389520645, 
0.013980187475681305, 0.05733799189329147, 0.05445294454693794, 0.1310664713382721, 0.029513003304600716, 0.007596850395202637, -0.026603449136018753, -0.03710125386714935, 0.0013483213260769844, 0.0020965170115232468, 
-0.005813299678266048, -0.05304979532957077, -0.0249648317694664, -0.03041590005159378, -0.007715819403529167, -0.034086644649505615, -0.017496544867753983, -0.1027945727109909, -0.026118410751223564, -0.02344677969813347, 
-0.017698105424642563, 0.06747943162918091, 0.007162683643400669, 0.04133123159408569, -0.04197578504681587, -0.03645746782422066, 0.01108129508793354, 0.04866281896829605, 0.01634697988629341, -0.06019052863121033, 
-0.09851866960525513, -0.04177159070968628, -0.06466469913721085, -0.08430491387844086, -0.0851370096206665, -0.05546849966049194, -0.020065082237124443, -0.12398649007081985, -0.102713443338871, -0.07471548020839691, 
0.03595324978232384, 0.019243281334638596, 0.11744940280914307, 0.010622850619256496, 0.02913101390004158, 0.06659059226512909, 0.09425030648708344, 0.05991390720009804, -0.06308378279209137, 0.04005023092031479, 
-0.07112564146518707, 0.06525668501853943, 0.0968519002199173, 0.03312814608216286, -0.05023258179426193, -0.09578175842761993, -0.0013486435636878014, 0.014925481751561165, -0.02984052337706089, 0.0007090531289577484, 
0.10822170227766037, 0.13186804950237274, -0.027242511510849, -0.08457939326763153, -0.0036975499242544174, 0.04090336337685585, 0.026317954063415527, -0.0345115028321743, -0.14227473735809326, -0.10684272646903992, 
-0.0970873087644577, -0.036944612860679626, -0.05912367254495621, -0.1888849437236786, 0.05799318850040436, 0.115545853972435, 0.11160814762115479, 0.09648382663726807, 0.03237448260188103, 0.07196550816297531, 
0.21723070740699768, 0.15063342452049255, -0.00035829097032546997, 0.09539330005645752, 0.14632895588874817, 0.1311618536710739, 0.05471310764551163, 0.05031740665435791, -0.06936463713645935, -0.0640362873673439, 
0.05282208323478699, -0.05055929347872734, -0.05564410239458084, -0.06920488178730011, -0.1507585048675537, 0.024282490834593773, 0.09698507189750671, 0.02491569332778454, -0.025938786566257477, -0.012955758720636368, 
0.053444091230630875, -0.019905973225831985, -0.0048664649948477745, -0.1538153886795044, -0.11917063593864441, 0.00036394596099853516, -0.041226111352443695, 0.00019479729235172272, 0.004194089211523533, -0.013324087485671043, 
0.11072845757007599, 0.1579800248146057, -0.04263314604759216, -0.14128606021404266, -0.04719002544879913, -0.024404844269156456, 0.05122508108615875, 0.14548364281654358, -0.012692632153630257, 0.03429453819990158, 
0.05225105211138725, -0.050412651151418686, -0.06037237495183945, 0.0617167130112648, 0.001980409026145935, -0.008477028459310532, 0.02123245969414711, -0.1853540688753128, -0.19173896312713623, 0.02204209566116333, 
0.07405158132314682, 0.016006819903850555, -0.002147190272808075, -0.18067599833011627, -0.17766791582107544, 0.015476588159799576, 0.10444900393486023, -0.1289687603712082, -0.014677304774522781, 0.06317257881164551, 
-0.08149252831935883, 0.13975366950035095, 0.16046589612960815, -0.024184275418519974, 0.08076964318752289, 0.14821572601795197, -0.13605839014053345, 0.018671181052923203, 0.039222925901412964, -0.049691472202539444, 
-0.11595465242862701, -0.10988688468933105, -0.1155313104391098, -0.09184402227401733, 0.15996921062469482, 0.04400673881173134, -0.04205073043704033, -0.06443307548761368, 0.03256703168153763, -0.07659869641065598, 
0.01502288319170475, 0.07492304593324661, 0.036735035479068756, 0.06727493554353714, 0.014021534472703934, 0.054641567170619965, 0.22662347555160522, 0.07539767026901245, -0.0037243105471134186, 0.11926953494548798, 
0.010223165154457092, 0.0557006374001503, 0.06345606595277786, 0.14819537103176117, -0.06872214376926422, 0.07592647522687912, -0.013519210740923882, -0.10197248309850693, 0.022350173443555832, 0.003254084847867489, 
-0.046584002673625946, -0.07987170666456223, -0.018319614231586456, -0.13371144235134125, -0.06503977626562119, -0.0211469829082489, -0.020769018679857254, 0.005398785695433617, -0.017940878868103027, -0.01706681028008461, 
0.050017260015010834, 0.05495676398277283, -0.011528050526976585, -0.1523188352584839, 0.03878045827150345, -0.05257792770862579, -0.09213005006313324, -0.028521325439214706, 0.03270681947469711, 0.06442475318908691, 
-0.03470681235194206, -0.058866847306489944, -0.02530328556895256, 0.1346728801727295, 0.12892326712608337, 0.0025136242620646954, 0.00524485856294632, -0.03247184306383133, -0.10112666338682175, 0.05726141482591629, 
0.04329298064112663, -0.02995928004384041, -0.08240300416946411, -0.037901196628808975, -0.11608562618494034, -0.08473734557628632, -0.09142133593559265, -0.1471339911222458, -0.15648728609085083, -0.047049567103385925, 
-0.08074772357940674, -0.05732378363609314, 0.005754433572292328, -0.040368348360061646, 0.025553075596690178, 0.028868719935417175, -0.014161022379994392, -0.058309707790613174, 0.10487623512744904, 0.09118077158927917, 
0.07096468657255173, 0.06483238935470581, 0.13220614194869995, -0.038855381309986115, 0.12899672985076904, 0.08460187911987305, -0.04369968920946121, -0.026818668469786644, -0.06163237988948822, -0.007994135841727257, 
-0.07031593471765518, 0.03422650322318077, -0.03595715016126633, 0.04986841231584549, -0.04820023849606514, -0.03352290391921997, -0.08170779794454575, 0.009103700518608093, 0.013022255152463913, 0.019290465861558914, 
0.06143743544816971, 0.02676444873213768, 0.02287960797548294, 0.05724351108074188, 0.11814455687999725, -0.008895929902791977, 0.05924936756491661, -0.06759517639875412, -0.03109242394566536, -0.07828032225370407, 
-0.08391299098730087, 0.013320610858500004, 0.1360577642917633, 0.0731598362326622, 0.0299573615193367, 0.09095218777656555, 0.05237425118684769, 0.10076066106557846, 0.07260110229253769, 0.007393889129161835, 
0.011735931970179081, 0.07904896885156631, 0.07069700956344604, 0.1486736536026001, 0.07011976093053818, -0.03416268900036812, 0.029782570898532867, -0.003945391625165939, -0.16812601685523987, -0.030738696455955505, 
0.08045830577611923, -0.045234013348817825, -0.08369903266429901, -0.024015743285417557, -0.07919403910636902, 0.03694087266921997, 0.13596054911613464, -0.08992250263690948, -0.08373039215803146, 0.0041136061772704124, 
-0.06216229498386383, -0.009734034538269043, 0.1036984920501709, -0.1493309587240219, -0.12047962099313736, 0.050177719444036484, -0.05465089902281761, 0.02443346381187439, 0.1316499263048172, -0.003096621483564377, 
-0.0684983879327774, 0.03573891893029213, -0.14687252044677734, -0.18185573816299438, -0.05169028788805008, -0.07464669644832611, -0.133755624294281, 0.059636443853378296, 0.04526340961456299, -0.10887400805950165, 
0.1789194941520691, 0.10848426818847656, -0.039334043860435486, -0.09198591113090515, 0.0038418667390942574, -0.0715470165014267, 0.001703564077615738, 0.1633502095937729, -0.14879655838012695, -0.09106037020683289, 
0.05434652417898178, -0.022263210266828537, -0.07235070317983627, 0.04026719555258751, -0.06162221357226372, -0.03584695979952812, -0.01696784794330597, 0.02016635239124298, 0.12055947631597519, 0.23497077822685242, 
0.21542727947235107, 0.011390581727027893, 0.16184769570827484, 0.08049876987934113, 0.0170020442456007, 0.1407816857099533, 0.014302249997854233, -0.14616386592388153, -0.012814139015972614, 0.0013521984219551086, 
-0.11004213243722916, -0.05681173875927925, -0.020585287362337112, -0.06973181664943695, -0.056472379714250565, -0.06310680508613586, -0.08225949853658676, 0.06042042374610901, 0.06018013879656792, -0.026647698134183884, 
0.016001980751752853, 0.03796634078025818, 0.06828296929597855, 0.13824966549873352, 0.05627691000699997, -0.11640530824661255, 0.04624045640230179, 0.09587632119655609, -0.11359786987304688, -0.09036466479301453, 
0.10648150742053986, 0.14358285069465637, 0.058128803968429565, 0.1340065598487854, -0.08447464555501938, -0.06274791061878204, 0.042865972965955734, -0.010350577533245087, 0.06858639419078827, 0.09179016947746277, 
-0.037053003907203674, 0.008626691997051239, 0.14062082767486572, -0.0046937428414821625, 0.06009568274021149, 0.07695777714252472, -0.013614211231470108, -0.02710053138434887, 0.07422883808612823, -0.007403239607810974, 
-0.07674944400787354, -0.12955524027347565, -0.06252308189868927, -0.017497945576906204, -0.07988279312849045, -0.028946997597813606, -0.1342800110578537, -0.19144828617572784, -0.08045163750648499, 0.006458510644733906, 
-0.07540826499462128, -0.03127842769026756, 0.06366495043039322, 0.056607089936733246, 0.03618159890174866, 0.09379642456769943, 0.08238641917705536, 0.031444936990737915, -0.04120424762368202, -0.1361650824546814, 
-0.06853216886520386, -0.07625830173492432, -0.23406726121902466, -0.11286915838718414, -0.11128776520490646, -0.1277785301208496, -0.04356495290994644, 0.02956688031554222, -0.010607482865452766, -0.07062362134456635, 
0.03418509662151337, 0.06720070540904999, 0.0012846887111663818, 0.006934588775038719, 0.12772202491760254, -0.011919798329472542, 0.041027773171663284, -0.03901127353310585, 0.02457852102816105, 0.047244563698768616, 
-0.0009939968585968018, -0.08282999694347382, -0.18161346018314362, -0.0495409294962883, -0.13893301784992218, -0.05305514857172966, -0.12836673855781555, -0.022349826991558075, -0.02213440090417862, 0.03052888810634613, 
0.04702690616250038, 0.05928868055343628, 0.0733974352478981, 0.0689549520611763, 0.1487157791852951, -0.0573313906788826, 0.1081371009349823, 0.004966238513588905, 0.1741362065076828, 0.21939300000667572, 
-0.007741361856460571, -0.025481030344963074, 0.07494606822729111, 0.15984651446342468, -0.023307716473937035, 0.012910783290863037, -0.050203174352645874, -0.03192932531237602, 0.1103101372718811, 0.0009515434503555298, 
-0.01874413713812828, 0.11058590561151505, 0.01939825713634491, 0.01642799749970436, 0.09026739001274109, 0.07134026288986206, 0.04112927243113518, 0.006651111412793398, -0.018844397738575935, -0.0004062098450958729, 
-0.02251606620848179, 0.061091531068086624, -0.09197258949279785, -0.16837450861930847, 0.02504017762839794, -0.07000383734703064, -0.13982529938220978, -0.054471004754304886, -0.0023672403767704964, -0.034885477274656296, 
-0.002999774180352688, 0.010789137333631516, 0.062358830124139786, -0.005593659356236458, 0.023465648293495178, 0.102083221077919, 0.08275113999843597, 0.058467235416173935, 0.034606341272592545, 0.029582878574728966, 
-0.02314545027911663, 0.05711642652750015, 0.03653157874941826, -0.1357758343219757, -0.13197961449623108, -0.06600278615951538, -0.13030558824539185, 0.06237885355949402, -0.05743708461523056, -0.11881038546562195, 
0.04489028453826904, -0.005383149720728397, -0.04536580294370651, -0.12000583112239838, 0.0316019132733345, -0.004828467965126038, -0.007195621728897095, -0.022295478731393814, 0.024610444903373718, -0.03595312684774399, 
-0.025506382808089256, 0.08123756945133209, -0.06919494271278381, -0.04326526075601578, 0.01661936193704605, 0.000785438809543848, 0.019729288294911385, 0.037350405007600784, -0.13049282133579254, -0.034562524408102036, 
-0.06535103172063828, 0.027110055088996887, 0.0505589060485363, -0.016534607857465744, 0.13402897119522095, 0.018130416050553322, 0.04029138386249542, 0.07421469688415527, 0.0885889008641243, 0.10789020359516144, 
0.005903076380491257, -0.08431553840637207, 0.032902784645557404, 0.0790802463889122, -0.03337647020816803, -0.06090399995446205, -0.011832185089588165, 0.031607240438461304, 0.05637504905462265, -0.030518803745508194, 
-0.13555538654327393, -0.06120825931429863, 0.003244645893573761, 0.008363498374819756, 0.09359175711870193, 0.06602014601230621, -0.02004210650920868, 0.09646543115377426, 0.1544155478477478, 0.05671796202659607, 
-0.0010612308979034424, 0.035312339663505554, 0.04376242682337761, 0.04538679122924805, -0.07718273252248764, -0.049273934215307236, 0.046759139746427536, -0.06807859987020493, -0.0464768223464489, 0.02131751924753189, 
0.053568482398986816, 0.05512600392103195, 0.03473268821835518, -0.033252134919166565, 0.08191628754138947, 0.09336239099502563, -0.030164286494255066, 0.011404495686292648, 0.03789311647415161, 0.08706894516944885, 
0.022489387542009354, 0.05890122428536415, 0.10085064172744751, -0.06004061549901962, -0.05019957199692726, 0.056856103241443634, -0.09256718307733536, -0.14150109887123108, -0.03241455927491188, -0.004155594855546951, 
0.012225918471813202, -0.027972141280770302, -0.09330997616052628, -0.12426170706748962, 0.006382044404745102, -0.03814077749848366, -0.08985812962055206, 0.0547834187746048, -0.07501852512359619, -0.062253884971141815, 
0.014520345255732536, -0.023963563144207, 0.08132351189851761, 0.11231249570846558, -0.038548216223716736, -0.15543489158153534, 0.030423907563090324, 0.04678035527467728, -0.05540401488542557, -0.053310077637434006, 
0.025224123150110245, -0.04694705083966255, -0.13538044691085815, -0.0035882778465747833, -0.07706145942211151, -0.0862274318933487, -0.004150977358222008, 0.03252730518579483, -0.0041509997099637985, 0.10495443642139435, 
0.08812156319618225, -0.07662519812583923, 0.004281209781765938, 0.022076839581131935, 0.02672361396253109, -0.04052135348320007, -0.026572318747639656, -0.027675818651914597, -0.04801661893725395, 0.09753160923719406, 
0.011472957208752632, 0.00018739700317382812, -0.047897715121507645, -0.06582403182983398, -0.0045922622084617615, -0.05652444064617157, 0.0017387187108397484, -0.0064455196261405945, 0.1186339482665062, 0.03518050163984299, 
-0.005282817408442497, 0.014768270775675774, 0.10415017604827881, 0.1580391824245453, -0.08953000605106354, 0.10229514539241791, 0.16446614265441895, -0.026698755100369453, -0.018335307016968727, 0.04160890355706215, 
0.038560591638088226, -0.008347202092409134, 0.10502538084983826, 0.004827005788683891, -0.21329627931118011, -0.01198570430278778, 0.003054353641346097, -0.029961489140987396, -0.0198680330067873, -0.051518991589546204, 
-0.07024670392274857, -0.001075293868780136, -0.0054763974621891975, -0.10416478663682938, 0.04679105430841446, 0.12744127213954926, 0.07808592915534973, -0.06433001160621643, 0.07225383818149567, 0.09958939999341965, 
0.06006316468119621, 0.06241171061992645, -0.03545152395963669, -0.11180242896080017, -0.021556895226240158, 0.07050830125808716, -0.11201296001672745, 0.004723748192191124, 0.11434660851955414, -0.06492370367050171, 
-0.020861435681581497, 0.1113118901848793, -0.08562914282083511, -0.13778367638587952, 0.12701597809791565, 0.06191021949052811, -0.05318383872509003, 0.053693968802690506, 0.15959002077579498, 0.06352509558200836, 
-0.06506478786468506, -0.012740787118673325, -0.052459120750427246, -0.08558240532875061, -0.029923051595687866, 0.008901026099920273, -0.1367928385734558, -0.13096779584884644, -0.04047247767448425, -0.06849749386310577, 
-0.08721707761287689, 0.022198719903826714, 0.12381960451602936, -0.11621195822954178, -0.15282313525676727, 0.048977479338645935, 0.010984532535076141, 0.06779968738555908, 0.06063535064458847, -0.03707776963710785, 
-0.018841227516531944, 0.028233936056494713, 0.050717391073703766, -0.05083100497722626, 0.027609769254922867, 0.04634472355246544, -0.021411532536149025, -0.023138200864195824, -0.10686778277158737, -0.0636606216430664, 
0.04937615990638733, -0.009723006747663021, -0.09548227488994598, 0.08066090941429138, 0.11572835594415665, 0.0018714023754000664, 0.12300733476877213, 0.13859187066555023, 0.13548336923122406, 0.021669482812285423, 
0.06865487992763519, -0.019517451524734497, -0.09298594295978546, 0.08326904475688934, 0.06255212426185608, 0.03780929744243622, 0.03474975749850273, 0.03654661029577255, -0.16125597059726715, 0.024859149008989334, 
0.00034357979893684387, 0.0031078383326530457, -0.03528112918138504, -0.09324381500482559, 0.15185564756393433, -0.024291180074214935, 0.042949482798576355, -0.03577914834022522, 0.07721222937107086, 0.005468945950269699, 
-0.04162482172250748, 0.10156940668821335, -0.0214446522295475, 0.09795566648244858, -0.04442369192838669, 0.026088261976838112, -0.08227206021547318, -0.03656923770904541, 0.13542059063911438, -0.020407823845744133, 
0.015066235326230526, -0.05357826501131058, 0.003581239841878414, -0.05189279094338417, 0.015911243855953217, 0.05529380589723587, -0.1023440957069397, 0.040228310972452164, -0.00864892452955246, -0.07151886820793152, 
0.007612012326717377, 0.015239477157592773, -0.06959552317857742, -0.04825698956847191, 0.07531470060348511, 0.0011920779943466187, -0.006235189735889435, 0.018994279205799103, -0.12619978189468384, -0.07873371243476868, 
0.09478262811899185, -0.0819191262125969, -0.1893257200717926, 0.014646902680397034, 0.005102179944515228, -0.07455844432115555, 0.049404267221689224, -0.02359136939048767, -0.139925017952919, -0.034604400396347046, 
-0.04224293678998947, -0.0573226734995842, 0.039781972765922546, 0.04109470546245575, 0.0020740218460559845, -0.06581185013055801, 0.044319696724414825, 0.07083012163639069, -0.033979758620262146, 0.08009963482618332, 
-0.003805447369813919, -0.1394890993833542, 0.028553830459713936, 0.02942122519016266, -0.08582223951816559, 0.05980195105075836, -0.11824598163366318, -0.20386210083961487, 0.025871096178889275, 0.027064146474003792, 
0.00411884905770421, 0.010565047152340412, -0.07113960385322571, -0.05731624364852905, 0.03408092260360718, -0.009158171713352203, 0.05433257669210434, 0.03739467263221741, 0.02780279889702797, -0.011943083256483078, 
-0.07846490293741226, 0.08367650210857391, -0.03898131847381592, -0.12394075840711594, 0.006049279123544693, -0.023425966501235962, 0.03267870843410492, 0.09539391100406647, 0.12214453518390656, 0.0926561951637268, 
0.020150713622570038, 0.08100815862417221, 0.1283050775527954, 0.10665519535541534, 0.10022084414958954, 0.1155548021197319, -0.042445436120033264, 0.015137732028961182, 0.03873898833990097, -0.043610744178295135, 
0.0566563680768013, -0.04172622784972191, -0.07926830649375916, -0.042629409581422806, -0.03498338162899017, -0.013269327580928802, 0.015632828697562218, -0.006728090345859528, 0.06521451473236084, -0.006820172071456909, 
-0.004403937608003616, 0.1502910554409027, -0.043327346444129944, 0.0023433268070220947, 0.06512416899204254, -0.06593681126832962, 0.07373126596212387, 0.02611653506755829, -0.10601919144392014, 0.047828081995248795, 
0.05634068325161934, -0.0404634065926075, 0.008814793080091476, 0.03959581255912781, 0.020642444491386414, -0.04889680817723274, -0.0616951547563076, 0.023582879453897476, 0.015740275382995605, 0.035075485706329346, 
-0.08075964450836182, -0.032087650150060654, 0.12161146104335785, -0.022686131298542023, 0.04293878376483917, 0.14415335655212402, -0.04107336699962616, -0.1757296323776245, 0.002560023218393326, -0.023532569408416748, 
-0.05067009478807449, 0.11022113263607025, -0.10523807257413864, -0.09126492589712143, 0.022949889302253723, -0.1097831130027771, -0.030757304280996323, -0.05468520522117615, -0.13485023379325867, -0.029046732932329178, 
-0.03889363259077072, -0.0014476440846920013, -0.009338214993476868, -0.018059423193335533, -0.03656291216611862, -0.06715687364339828, -0.0037483572959899902, 0.04673275351524353, 0.0901908278465271, -0.04619564488530159, 
-0.09305936098098755, -0.01880425028502941, 0.023201730102300644, -0.0032744910567998886, -0.010188398882746696, 0.03141847997903824, 0.10329663008451462, 0.0228654183447361, 0.031044453382492065, 0.06935817748308182, 
0.001571115106344223, 0.11562687158584595, 0.09036113321781158, 0.038704391568899155, 0.06543300300836563, -0.02186419442296028, -0.10327323526144028, 0.12310997396707535, 0.11890137940645218, 0.012379512190818787, 
0.05948229506611824, -0.08880994468927383, -0.06260199844837189, -0.035074301064014435, 0.02659974992275238, 0.195730522274971, 0.06455624103546143, -0.04936395213007927, 0.003119366243481636, -0.08154474198818207, 
-0.023990273475646973, 0.12418127804994583, 0.00535023957490921, -0.038122646510601044, -0.018694229423999786, -0.09836423397064209, 0.04795060306787491, 0.08991189301013947, -0.0043707191944122314, -0.012592373415827751, 
0.0013546273112297058, 0.00991042423993349, 0.06493335217237473, 0.03599762171506882, 0.018546970561146736, 0.08803889155387878, 0.01001380942761898, 0.1498657763004303, 0.09471306204795837, -0.09129376709461212, 
-0.06920315325260162, -0.09541842341423035, -0.02962428890168667, 0.04246290773153305, -0.03181193023920059, 0.1222233921289444, -0.014154307544231415, -0.159526526927948, 0.06633620709180832, -0.04204709827899933, 
-0.023431599140167236, -0.005593996495008469, -0.03162725269794464, 0.016945473849773407, -0.03620884567499161, -0.0669669657945633, -0.03482624515891075, 0.03326548635959625, 0.01242734119296074, -0.12369250506162643, 
-0.0074207112193107605, 0.03256293013691902, -0.08202162384986877, 0.023170525208115578, -0.1525982916355133, -0.05562356114387512, 0.15990455448627472, -0.05113368481397629, -0.06578069925308228, -0.05529250577092171, 
-0.058982737362384796, -0.09898588061332703, -0.12241550534963608, -0.009193507954478264, -0.015603778883814812, -0.05564014986157417, 0.05793484300374985, -0.04026700556278229, -0.08889365196228027, 0.07220716774463654, 
0.02971007488667965, 0.0515199713408947, -0.024262746796011925, -0.029917214065790176, -0.19607441127300262, -0.19165122509002686, -0.016197405755519867, -0.03668241947889328, 0.0102199362590909, -0.11890602111816406, 
-0.06530536711215973, -0.017154738306999207, -0.04404198005795479, -0.04065416753292084, 0.06392025202512741, 0.0865461528301239, 0.10694903880357742, 0.0728655532002449, -0.019262349233031273, 0.14742033183574677, 
0.006645171903073788, 0.07967174053192139, 0.05754674971103668, -0.0057619474828243256, -0.03715835139155388, -0.11651474982500076, 0.10526224225759506, -0.04696643352508545, -0.0620170533657074, 0.03777068108320236, 
0.10540413856506348, -0.010919440537691116, -0.011310126632452011, 0.06903722137212753, 0.05709504336118698, 0.06915099918842316, -0.013094337657094002, 0.10206817090511322, -0.014261147007346153, 0.05029330775141716, 
0.05587498098611832, -0.020412946119904518, 0.033438295125961304, 0.02433614246547222, 0.06372492760419846, 0.0739646703004837, 0.17924945056438446, -0.02261028066277504, -0.14627501368522644, 0.0047358255833387375, 
0.02858750708401203, 0.053286559879779816, 0.04128912836313248, 0.019295480102300644, 0.01577063277363777, -0.053992487490177155, -0.0334341935813427, -0.040702566504478455, 0.03873850405216217, 0.12870828807353973, 
0.03960621356964111, -0.009776292368769646, 0.018051642924547195, 0.09536856412887573, -0.012118872255086899, 0.1701011210680008, 0.0859837606549263, -0.08681520819664001, 0.005572831258177757, -0.04782857745885849, 
0.009519155137240887, -0.04243873432278633, 0.03174307942390442, -0.0229610875248909, -0.06722162663936615, -0.018016166985034943, 0.036586202681064606, -0.03586098551750183, -0.07818634808063507, 0.07204139232635498, 
0.055819664150476456, -0.005509662441909313, -0.01092682033777237, -0.018100077286362648, -0.05969695746898651, -0.10770662128925323, -0.054810889065265656, -0.04068933054804802, -0.0545027069747448, -0.006035478785634041, 
-0.010178038850426674, -0.06275675445795059, -0.15760624408721924, -0.07007558643817902, -0.05824477970600128, 0.005867393221706152, -0.03147419914603233, -0.04840819165110588, 0.0005533443763852119, -0.12714046239852905, 
-0.0038093701004981995, 0.0845913290977478, -0.06388567388057709, 0.03738725930452347, 0.08723990619182587, -0.045820899307727814, 0.0015376140363514423, 0.007980444468557835, -0.032574646174907684, -0.009977495297789574, 
0.06425964087247849, -0.030135996639728546, -0.11908604949712753, -0.006867806427180767, 0.05573391541838646, -0.020974114537239075, -0.04373495280742645, -0.0051493775099515915, -0.016202114522457123, 0.041987866163253784, 
0.01892620511353016, -0.030863380059599876, 0.09144415706396103, -0.01245635561645031, 0.028913484886288643, 0.14451472461223602, 0.038928400725126266, 0.01755041815340519, 0.08375143259763718, 0.023946160450577736, 
-0.0671127662062645, 0.014955824241042137, 0.013542670756578445, 0.007066117599606514, 0.07915584743022919, -0.0021399129182100296, -0.031090406700968742, -0.05643170699477196, -0.041417501866817474, 0.05964089184999466, 
-0.07417867332696915, 0.0462571419775486, 0.10874290764331818, -0.030933372676372528, -0.029853198677301407, 0.07406972348690033, 0.058381445705890656, 0.020709585398435593, 0.015655865892767906, 0.042852774262428284, 
0.08617417514324188, 0.0010619647800922394, -0.005817057099193335, 0.00903227273374796, -0.018884137272834778, 0.02415560930967331, -0.0026097372174263, -0.05968424305319786, 0.06077226251363754, 0.00733562745153904, 
-0.02526763267815113, 0.0030743740499019623, -0.019745251163840294, 0.032693251967430115, 0.03543441370129585, 0.02718600258231163, -0.03761696815490723, -0.07252481579780579, 0.056637175381183624, 0.030919207260012627, 
0.049444038420915604, 0.010909578762948513, 0.00374760664999485, 0.011563636362552643, -0.08463412523269653, -0.0647091269493103, -0.005242530256509781, 0.08667026460170746, -0.0587155818939209, -0.04720081016421318, 
-0.03450766205787659, -0.05982044339179993, 0.037574172019958496, -0.043508537113666534, -0.054074592888355255, -0.010345328599214554, -0.09070490300655365, 0.006320450454950333, 0.04843764007091522, -0.0889449417591095, 
-0.0070280274376273155, 0.013520770706236362, -0.07079508900642395, 0.00035256240516901016, 0.03854312747716904, -0.11797736585140228, -0.08574339747428894, 0.036061570048332214, -0.1135200783610344, -0.031204577535390854, 
0.11543652415275574, 0.08792556077241898, -0.03224191814661026, -0.1045684888958931, 0.03197047486901283, 0.02743392065167427, -0.05570899695158005, 0.0515744723379612, 0.03096708469092846, -0.06533055007457733, 
0.040423449128866196, 0.007899719290435314, 0.009434584528207779, -0.055126652121543884, -0.013302972540259361, 0.041628409177064896, -0.04184615612030029, 0.043823886662721634, 0.07846997678279877, 0.025105122476816177, 
-0.007669656537473202, 0.03772774338722229, 0.025062579661607742, -0.020926961675286293, 0.032581496983766556, 0.08767099678516388, -0.0030703898519277573, -0.03164251893758774, 0.0250239335000515, -0.00021428614854812622, 
0.014071531593799591, 0.14051249623298645, 0.04162828251719475, -0.07075174897909164, -0.009293165057897568, 0.05937616527080536, 0.06847089529037476, -0.017505506053566933, -0.022083833813667297, -0.03459889441728592, 
0.05215895548462868, 0.08583168685436249, -0.008733097463846207, 0.0488661453127861, 0.1268472969532013, -0.011336511000990868, -0.04694493114948273, 0.06928481161594391, 0.047140248119831085, 0.04004142805933952, 
0.04578378051519394, 0.03008476458489895, 0.0015144078060984612, 0.021431419998407364, -0.05512681230902672, 0.014230454340577126, 0.08968356251716614, -0.07567176967859268, -0.07472805678844452, -0.020262785255908966, 
-0.05902770161628723, -0.09764480590820312, 0.024320846423506737, -0.02068052813410759, -0.012753339484333992, 0.08745741099119186, -0.1468643844127655, -0.092980295419693, 0.1132107526063919, 0.05356629937887192, 
-0.06028140336275101, -0.05936860293149948, 0.010803554207086563, -0.10131031274795532, -0.04139358177781105, -0.020202623680233955, -0.02427300438284874, 0.07365850359201431, -0.03877846151590347, -0.0954761728644371, 
-0.09285223484039307, 0.004571804776787758, -0.0699533224105835, -0.12166336178779602, -0.00932818092405796, -0.010370416566729546, -0.03029211051762104, -0.06287790089845657, 0.0019550854340195656, 0.08818215131759644, 
0.02013702131807804, -0.02860337495803833, 0.05104540288448334, 0.01527953427284956, -0.007257141172885895, 0.03293689340353012, 0.034604404121637344, -0.027488641440868378, -0.036728858947753906, 0.008112477138638496, 
0.03853790834546089, 0.012236479669809341, -0.019962843507528305, -0.03216715529561043, -0.04505996033549309, 0.030180199071764946, -0.03908321633934975, -0.03594246134161949, -0.06353759765625, -0.07637830823659897, 
-0.0166102796792984, 0.005484048277139664, -0.0707097053527832, 0.019755704328417778, 0.09493981301784515, -0.03255707025527954, 0.013788294978439808, 0.055222608149051666, 0.1012110561132431, -0.030278723686933517, 
0.020961299538612366, 0.04909047484397888, -0.0011566625908017159, 0.07056921720504761, 0.02821914665400982, -0.04696299880743027, 0.05621340498328209, 0.0630512535572052, 0.017408061772584915, 0.10861917585134506, 
0.015959464013576508, -0.05159030854701996, -0.024177270010113716, 0.06213262304663658, 0.0008162008598446846, 0.011735345236957073, -0.0003187386319041252, 0.037978000938892365, 0.013242981396615505, -0.009535351768136024, 
0.07342012226581573, -0.030567187815904617, 0.06593138724565506, -0.03662633150815964, -0.06370536983013153, 0.01579919643700123, 0.0024045417085289955, 0.02422618865966797, 0.016602296382188797, 0.008818425238132477, 
0.00626615434885025, 0.022665202617645264, -0.00978582352399826, 0.09650072455406189, -0.011504417285323143, -0.05739995837211609, 0.028856460005044937, -0.004130106419324875, 0.035760436207056046, 0.04443217068910599, 
-0.034055691212415695, 0.03533738851547241, 0.1199239045381546, 0.04982396587729454, 0.029222607612609863, 0.05179743096232414, 0.03564295545220375, -0.08921504020690918, -0.09358807653188705, -0.004135455004870892, 
-0.027076195925474167, -0.048771876841783524, -0.01721026562154293, -0.049757614731788635, -0.1028919517993927, -0.08304787427186966, 0.0002075154334306717, -0.0322144441306591, -0.04963725060224533, -0.02641664445400238, 
0.004971577785909176, -0.011420534923672676, -0.025684969499707222, -0.027340514585375786, -0.02615586668252945, 0.03403637930750847, 0.03519061207771301, 0.049506668001413345, 0.008859602734446526, -0.040839120745658875, 
0.004136965610086918, -0.013081591576337814, -0.06383465975522995, 0.02188020385801792, -0.028176557272672653, -0.07831098884344101, -0.05749688297510147, 3.6908313632011414e-05, -0.04477649927139282, -0.023177608847618103, 
0.04418351501226425, 0.01521299034357071, -0.018981747329235077, -0.030233900994062424, 0.040568917989730835, -0.022707616910338402, 0.00046908482909202576, -0.04065953940153122, 0.03990287333726883, 0.07741568237543106, 
0.08295539021492004, 0.09250961244106293, -0.011645174585282803, 0.036239996552467346, -0.06365430355072021, 0.01758001372218132, 0.008268516510725021, -0.060578733682632446, -0.05196743458509445, -0.07090473920106888, 
-0.07758718729019165, -0.029928775504231453, 0.08579261600971222, -0.00780743733048439, -0.04088699072599411, 0.009757345542311668, -0.002692790701985359, 0.013807510957121849, 0.07274971902370453, 0.057661090046167374, 
0.00876307487487793, 0.00929342396557331, 0.12060963362455368, -0.007970606908202171, -0.0106029212474823, 0.0765775591135025, 0.0030854344367980957, -0.02027568221092224, -0.05225232243537903, 0.027692649513483047, 
-0.0024966923519968987, -0.04402078688144684, -0.05945374071598053, -0.06846597790718079, 0.004158633295446634, 0.0142745953053236, 0.0446353554725647, -0.011011974886059761, 0.011556420475244522, 0.048845209181308746, 
-0.012761224061250687, 0.06872908771038055, 0.03509528562426567, 0.0076566413044929504, 0.08936841040849686, 0.002757851965725422, -0.06419100612401962, 0.06199805438518524, -0.004874427802860737, -0.045140236616134644, 
-0.052564602345228195, -0.05200395733118057, -0.02348262071609497, -0.04998383671045303, -0.070975661277771, -0.06488153338432312, -0.08936651051044464, -0.07367958128452301, 0.00932401418685913, -0.021851634606719017, 
0.043253667652606964, -0.037771955132484436, -0.002521534450352192, 0.021111441776156425, 0.009817317128181458, 0.03420581296086311, 0.009260921739041805, 0.0006679687649011612, -0.0012705414555966854, 0.009960849769413471, 
-0.02411309815943241, 0.011577539145946503, -0.01985335350036621, -0.020523807033896446, -0.054958924651145935, -0.00021304935216903687, -0.02812226116657257, -0.02548355795443058, 0.027946893125772476, -0.0014666356146335602, 
-0.04643893241882324, 0.04073009639978409, 0.1236388087272644, 0.028403833508491516, 0.053116388618946075, 0.04845384508371353, 0.04666857793927193, 0.0339123010635376, 0.03245387226343155, -0.012772157788276672, 
-0.01651955023407936, 0.06944841891527176, 0.033112574368715286, -0.025544755160808563, -0.016542529687285423, -0.013322735205292702, -0.021589815616607666, 0.018016314134001732, -0.022423844784498215, -0.030383378267288208, 
0.06957972049713135, 0.04944173991680145, 0.028848007321357727, 0.00554762315005064, 0.058146268129348755, 0.04785161465406418, -0.014402005821466446, 0.040441472083330154, 0.04225742816925049, 0.045699868351221085, 
-0.018287837505340576, -0.00843340065330267, 0.028902772814035416, 0.036144182085990906, -0.005440478213131428, -0.08914518356323242, -0.0015023378655314445, 0.023740479722619057, 0.02200574241578579, 0.010035274550318718, 
-0.05734113231301308, -0.03508521616458893, 0.014493859373033047, -0.022499486804008484, -0.029851583763957024, 0.026328764855861664, -0.026925139129161835, 0.003924975171685219, 0.07854205369949341, 0.04600418359041214, 
-0.0340767428278923, 0.018904879689216614, 0.023849060758948326, 0.03159639984369278, 0.04801812395453453, -0.015328813344240189, 0.0015108948573470116, -0.01026900578290224, -0.04960399121046066, -0.050242312252521515, 
0.002452087588608265, -0.026081565767526627, 0.0019436664879322052, -0.04132083058357239, -0.04552282765507698, -0.0009965039789676666, 0.03657800704240799, 0.02891240268945694, -0.03788633644580841, 0.05146796628832817, 
-0.009991880506277084, -0.015021570958197117, -0.009537231177091599, -0.007605693768709898, 0.011181807145476341, -0.039657462388277054, -0.04831722006201744, -0.040487803518772125, -0.012840145267546177, -0.03382730484008789, 
0.012799989432096481, 0.011085888370871544, -0.0117808748036623, -0.025647811591625214, -0.04993295669555664, -0.02866297960281372, 0.016851797699928284, -0.05853523313999176, -0.09939537197351456, 0.04034316539764404, 
0.020303770899772644, -0.05242525041103363, -0.027436302974820137, -0.006041776388883591, -0.06156163662672043, -0.033626824617385864, 0.021357595920562744, 0.018867922946810722, 0.03462141379714012, 0.006806503050029278, 
-0.03329957276582718, -0.029622450470924377, 0.08551926165819168, 0.021286264061927795, 0.0023759249597787857, -0.016683317720890045, -0.04032490774989128, 0.04502852261066437, 0.0012192539870738983, -0.01048615574836731, 
-0.03158699721097946, 0.057864636182785034, -0.005666371434926987, 0.011602338403463364, 0.03370142728090286, 0.019099228084087372, 0.11246643960475922, 0.021042877808213234, -0.020869115367531776, 0.04266637563705444, 
0.0613720566034317, 0.01950913481414318, 0.0519498810172081, -0.00038299616426229477, 0.017142094671726227, -0.0009476523846387863, -0.020044898614287376, -0.007274650037288666, -0.03730598837137222, 0.008393091149628162, 
-0.01659020036458969, 0.013676892966032028, 0.0032302066683769226, -0.026711443439126015, -0.004660472273826599, -0.01422699075192213, 0.008038182742893696, -0.012433400377631187, -0.005619999021291733, 0.027577418833971024, 
0.008015220984816551, -0.005341903306543827, -0.01939396932721138, -0.021527735516428947, 0.014976512640714645, 0.05116395652294159, 0.023633619770407677, 0.013621194288134575, -0.013427197933197021, 0.0158985685557127, 
0.018448270857334137, -0.04228859394788742, 0.0408700630068779, 0.01653463765978813, -0.03181390464305878, 0.017535120248794556, -0.018155667930841446, -0.061126574873924255, -0.02726566046476364, -0.06553004682064056, 
-0.04391852021217346, 0.024927619844675064, -0.03640502691268921, -0.04620174318552017, -0.006997737102210522, 0.00446309894323349, -0.05418778955936432, -0.014980319887399673, -0.0007264353334903717, -0.007845452055335045, 
-0.0039435699582099915, -0.010134236887097359, -0.04563438519835472, -0.01839805766940117, -0.0012310966849327087, -0.053601350635290146, 0.024206459522247314, 0.007707685232162476, 0.010886752977967262, 0.012572873383760452, 
0.007621392607688904, -0.014302236959338188, -0.04306240379810333, -0.02082984708249569, 0.03058229386806488, -0.019466083496809006, 0.0040184492245316505, 0.04541274905204773, -0.004245128482580185, -0.015285627916455269, 
-0.0058220140635967255, 0.016625136137008667, -0.010330040007829666, -0.010606370866298676, -0.029687706381082535, 0.022671477869153023, -0.01483260840177536, 0.007385998964309692, -0.024580229073762894, -0.0035221362486481667, 
0.016618117690086365, -0.03882123529911041, 0.02138955146074295, 0.0008721696212887764, 0.025707058608531952, 0.0074687376618385315, -0.027047257870435715, -0.01834040880203247, 0.024448206648230553, 0.02845015376806259, 
0.0414162203669548, 0.026297658681869507, 0.03093857318162918, 0.019736425951123238, 0.03773433715105057, 0.008814560249447823, -0.039873018860816956, 0.0004973672330379486, -0.02318640984594822, 0.01665453612804413, 
0.027499638497829437, 0.05164177715778351, 0.02622871845960617, -0.027492716908454895, 0.03688773140311241, 0.020472180098295212, 0.02790912613272667, 0.05572635680437088, 0.022172335535287857, 0.05928922817111015, 
0.020771197974681854, 0.009731247089803219, 0.024991720914840698, 0.03244878351688385, 0.016152622178196907, -0.009299070574343204, -0.012880878522992134, -0.04090845584869385, -0.03969082236289978, -0.018206313252449036, 
-0.008597824722528458, -0.08854764699935913, 0.0004656622186303139, 0.009893035516142845, -0.007431849837303162, 0.007140403613448143, -0.03809169679880142, -0.011745253577828407, -0.026712566614151, -0.01954737678170204, 
-0.03133426606655121, 0.04826098680496216, -0.009038105607032776, -0.04789775609970093, 0.008848093450069427, -0.008425891399383545, -0.010499350726604462, -0.009082249365746975, -0.006637290120124817, 0.007113984785974026, 
0.003156013786792755, -0.010029975324869156, -0.002497851848602295, -0.00596908014267683, 0.023080337792634964, -0.04266434907913208, -0.009315107017755508, 0.005580175668001175, 0.006667211651802063, 0.023514211177825928, 
-0.0381627082824707, 0.006650004535913467, -0.012315582484006882, -0.002160964533686638, 0.015672365203499794, -0.026741674169898033, 0.01349952258169651, -0.0032888688147068024, -0.0057329535484313965, -0.008715194649994373, 
-0.019565057009458542, 0.004841441288590431, -0.03837858885526657, -0.01709037646651268, 0.021392367780208588, 0.015508279204368591, 0.0023954883217811584, -0.003392161801457405, -0.004292408004403114, -0.0036120787262916565, 
0.01049572043120861, 0.0329698771238327, 0.03178872913122177, 0.0007788036018610001, 0.011024802923202515, 0.011399243026971817, 2.2660940885543823e-05, 0.023592621088027954, 0.014929559081792831, -0.00539114885032177, 
-0.0108472416177392, 0.02325306087732315, 0.014582084491848946, 0.004486421123147011, 0.005244821310043335, 0.005589060485363007, 0.0237358957529068, 0.02207743376493454, 0.04102524369955063, 0.023767393082380295, 
0.019871436059474945, 0.005049661733210087, -0.006731852889060974, 0.010827104561030865, -0.001629645936191082, -0.0050978884100914, 0.038487665355205536, 0.0008682981133460999, -0.008445635437965393, 0.0074108559638261795, 
0.011704926379024982, 0.038770418614149094, 0.010286659002304077, 0.01780770532786846, -0.009416492655873299, -0.0023640645667910576, 0.04158247262239456, -0.007903732359409332, -0.029032716527581215, 0.02421051263809204, 
0.0005158260464668274, 0.011511549353599548, 0.04210630804300308, 0.03570470213890076, 0.03025941178202629, -0.002005845308303833, -0.020106330513954163, 0.03012387454509735, 0.04770832881331444, 0.0042510610073804855, 
0.009119050577282906, 0.029650148004293442, -0.007262013852596283, -0.06005389615893364, -0.01474255695939064, -0.021103575825691223, -0.04397795349359512, -0.07093266397714615, -0.05798245221376419, -0.0639399066567421, 
-0.05044960603117943, -0.03219705820083618, -0.051767781376838684, -0.06469114124774933, -0.04260079562664032, 0.02134600467979908, -0.01611180230975151, -0.012270934879779816, -0.0102623850107193, -0.01708226278424263, 
-0.02390352450311184, -0.010053914040327072, -0.015673547983169556, 0.028779786080121994, 0.014698395505547523, -0.049828313291072845, -0.0023890985175967216, 0.036773040890693665, 0.0038524791598320007, -0.029084332287311554, 
0.01770320162177086, 0.012035652995109558, 0.005324065685272217, -0.0003271978348493576, 0.0031631551682949066, 0.059881389141082764, 0.06352156400680542, -0.040007684379816055, -0.04283996671438217, 0.05139715224504471, 
0.02182185836136341, -0.03468753769993782, 0.0053618066012859344, -0.010803432203829288, -0.03965725377202034, -0.012650292366743088, -0.03494444116950035, -0.006935422308743, -0.011644527316093445, -0.033128730952739716, 
-0.002553560771048069, -0.003677554428577423, -0.020360123366117477, 0.030032578855752945, 0.04549200460314751, 0.0016859248280525208, 0.004368111491203308, 0.021868964657187462, 0.007322566583752632, 0.023704025894403458, 
0.05221516266465187, 0.01616324670612812, 0.014845610596239567, -0.009617611765861511, 0.016249926760792732, 0.022092631086707115, 0.010708479210734367, -0.03727683052420616, -0.03929460048675537, 0.04730900377035141, 
0.02828977257013321, 0.02091856859624386, 0.033668626099824905, -0.005307638086378574, 0.013329112902283669, 0.017586931586265564, -0.011541795916855335, 0.04089446738362312, 0.008063456043601036, 0.040184386074543, 
0.0020736069418489933, -0.01976364478468895, 0.023196808993816376, -0.0053388867527246475, 0.029122531414031982, -0.01731441542506218, 0.0020670033991336823, 0.00801568292081356, -0.009423729032278061, 0.027774084359407425, 
0.0077379485592246056, -0.042960502207279205, -0.048690490424633026, 0.012180507183074951, -0.022099141031503677, -0.01612098142504692, 9.797513484954834e-05, -0.02164703607559204, -0.020139792934060097, -0.007094849366694689, 
-0.01953866146504879, -0.008510527200996876, 0.005399581044912338, -0.02188497968018055, -0.0042886873707175255, -0.039191242307424545, -0.061940863728523254, -0.029834888875484467, 0.012598294764757156, -0.051886048167943954, 
-0.029345087707042694, 0.020684687420725822, -0.014505865052342415, -0.024223824962973595, -0.012990402057766914, -0.01446981355547905, -0.024171149358153343, -0.0066977739334106445, 0.006986680440604687, 0.02545125037431717, 
-0.0028792601078748703, -0.004106633365154266, 0.0034911008551716805, 0.011453729122877121, -0.011941850185394287, 0.0015781046822667122, -0.006466586142778397, -0.030738987028598785, -0.01129056140780449, -0.005221828818321228, 
-0.012768972665071487, -0.029348088428378105, -0.008119307458400726, -0.002075284719467163, -0.003520493395626545, -0.003686375916004181, 0.003549516201019287, 0.0050224908627569675, -0.001378611195832491, -0.0010603424161672592, 
0.02527494914829731, 0.020988501608371735, 0.006888162344694138, 0.03414211422204971, 0.04084054380655289, 0.02506118454039097, 0.04230501502752304, 0.036990463733673096, 0.03269291669130325, 0.01461874507367611, 
0.017676416784524918, 0.02613944746553898, 0.012385059148073196, 0.03415454924106598, -0.009221881628036499, -0.011315951123833656, -0.005362940952181816, 0.0034418050199747086, 0.0059058484621346, -0.0017163995653390884, 
0.001547735184431076, -0.006261609494686127, -0.006503626704216003, -0.0009782323613762856, 0.015275048092007637, 0.028263960033655167, 0.02279047854244709, -0.003423362970352173, 0.023933429270982742, 0.011814176104962826, 
0.011083275079727173, -0.0025714337825775146, -0.0027926005423069, 0.007154393941164017, -0.017664805054664612, -0.001684216782450676, 0.004263225942850113, 0.01844491809606552, 0.0120697021484375, 0.016795411705970764, 
0.0055511388927698135, 0.011734547093510628, 0.018798604607582092, -0.0014853104948997498, -0.014157883822917938, -0.004624748602509499, -0.0035894252359867096, -0.02795788273215294, -0.0037098973989486694, -0.012706094421446323, 
-0.022260265424847603, -0.01389989722520113, -0.015687361359596252, -0.029294131323695183, -0.04020446538925171, -0.026897095143795013, -0.020764365792274475, -0.041118770837783813, -0.03707847744226456, -0.007976852357387543, 
-0.001648496836423874, -0.0036290306597948074, -0.022974297404289246, -0.01840997114777565, -0.013644378632307053, -0.009605180472135544, -0.018920332193374634, -0.023048974573612213, -0.006618384271860123, -0.013072438538074493, 
-0.020058494061231613, -0.01343262568116188, -0.012130727991461754, -0.01719551347196102, -0.014009945094585419, -0.018886521458625793, -0.010389786213636398, -0.01855367049574852, -0.027214758098125458, -0.019433245062828064, 
-0.015927739441394806, -0.00786370038986206, -0.020119791850447655, -0.006550604477524757, 0.019597770646214485, 0.017064927145838737, 0.011494725942611694, 0.018876750022172928, 0.04039067402482033, 0.036210983991622925, 
0.021673835813999176, 0.016736004501581192, 0.024679582566022873, 0.023204520344734192, 0.024590875953435898, 0.02539963461458683, 0.019967012107372284, 0.01302347332239151, -0.0010221004486083984, 0.011468693614006042, 
0.006171846762299538, 0.005989566445350647, 0.016207098960876465, 0.016046583652496338, 0.0084611177444458, 0.01876453123986721, 0.014264027588069439, 0.011978425085544586, 0.023563139140605927, 0.01064025703817606, 
0.007025262340903282, -0.005430486053228378, 0.007999157533049583, 0.006866471841931343, 0.0001901388168334961, 0.0013295859098434448, 0.005504518747329712, 0.020973332226276398, 0.02577626332640648, 0.03566322475671768, 
0.024678455665707588, 0.019049275666475296, 0.02754911407828331, 0.03909978270530701, 0.02588309906423092, 0.004829226527363062, 0.011554213240742683, 0.009244998916983604, -0.001691540703177452, -0.004273250699043274, 
0.009883338585495949, 0.02577999047935009, -0.0012349225580692291, -0.025094376876950264, -0.002083280123770237, -0.020353451371192932, -0.029068462550640106, -0.013863150961697102, -0.011359878815710545, -0.009904960170388222, 
-0.02254876121878624, -0.007703319191932678, -0.006601681932806969, -0.007138462271541357, -0.016092825680971146, -0.02276816964149475, -0.008054975420236588, -0.0024954453110694885, -0.016977423802018166, -0.016606740653514862, 
-0.001277809962630272, -0.017404630780220032, -0.006621240638196468, -0.0038037430495023727, -0.021136198192834854, -0.0200297050178051, -0.013396580703556538, -0.012201579287648201, -0.013906093314290047, -0.01796969398856163, 
-0.016628075391054153, -0.03196302056312561, -0.039133645594120026, -0.013674166053533554, -0.02020893432199955, -0.016713308170437813, -0.010597966611385345, -0.0044265612959861755, -0.006436504423618317, -0.02505110204219818, 
-0.011720096692442894, -0.0045986175537109375, -0.006755579262971878, -0.006342872977256775, -0.0038526710122823715, -0.005437192507088184, -0.0057085007429122925, -0.006137087941169739, -0.006854083389043808, -0.007299020886421204, 
-0.006671866402029991, -0.0034190877340734005, -0.0011397176422178745, 0.003941473551094532, 0.002925669774413109, 0.007358982227742672, 0.004192635882645845, 0.004349181428551674, 0.010193025693297386, 0.008698105812072754, 
0.01083268504589796, 0.015132622793316841, 0.019605426117777824, 0.015058059245347977, 0.020339487120509148, 0.018691718578338623, 0.010328180156648159, 0.028097055852413177, 0.03580068051815033, 0.027769168838858604, 
0.03022526018321514, 0.040004629641771317, 0.03469676524400711, 0.025440407916903496, 0.027324190363287926, 0.031050994992256165, 0.03379800543189049, 0.028554020449519157, 0.03150981664657593, 0.03194955736398697, 
0.036371905356645584, 0.027025261893868446, 0.013568071648478508, 0.022718794643878937, 0.019510608166456223, 0.01551958080381155, 0.011540582403540611, 0.014977119863033295, 0.007435984909534454, 0.004251901060342789, 
0.002383721061050892, 0.012025336734950542, 0.010463926941156387, 0.0037871384993195534, 0.001517564058303833, -0.0021167267113924026, 0.013327354565262794, 0.0011630188673734665, 0.001839352771639824, -0.002858651801943779, 
-0.0030088722705841064, -0.009534351527690887, -0.01217099092900753, -0.013001348823308945, -0.011245420202612877, -0.012125782668590546, -0.017484869807958603, -0.0224705059081316, -0.02699209377169609, -0.019783765077590942, 
-0.023432360962033272, -0.025637619197368622, -0.027081463485956192, -0.020978674292564392, -0.026060158386826515, -0.017801493406295776, -0.028504760935902596, -0.021845385432243347, -0.01823197305202484, -0.015683280304074287, 
-0.024153513833880424, -0.01965281367301941, -0.019081391394138336, -0.028034374117851257, -0.0136338509619236, -0.01892777532339096, -0.015435614623129368, -0.02434360980987549, -0.018106672912836075, -0.0227954164147377, 
-0.012151898816227913, -0.014695759862661362, -0.0116463303565979, -0.014294895343482494, -0.00801602192223072, -0.01268552802503109, -0.018901128321886063, -0.007040292955935001, -0.017849601805210114, -0.0057626329362392426, 
-0.007767036557197571, -0.012230617925524712, -0.01265285350382328, 0.0038343118503689766, 0.0012118369340896606, -0.009488528594374657, -0.005350377410650253, 0.003684363327920437, -0.0007537715137004852, 0.00017899367958307266, 
0.006198015995323658, 0.007026728242635727, 0.002546936273574829, 0.0057707857340574265, 0.019123125821352005, 0.008562054485082626, 0.009126978926360607, 0.006319345906376839, 0.00048484373837709427, 0.004830135032534599, 
0.018341505900025368, 0.010725042782723904, 0.017419390380382538, 0.022789660841226578, 0.014353418722748756, 0.01013258844614029, 0.01539340615272522, 0.021943217143416405, 0.010946528986096382, 0.024315737187862396, 
0.0065908716060221195, 0.02150837890803814, 0.01700468920171261, 0.014630889520049095, 0.030410803854465485, 0.025008942931890488, 0.009860135614871979, 0.010892207734286785, 0.022889848798513412, 0.009237565100193024, 
0.019218258559703827, 0.013229738920927048, 0.013353778049349785, 0.010325295850634575, 0.026202157139778137, 0.002407948486506939, 0.010072450153529644, 0.00781339593231678, 0.0031242272816598415, 0.008587329648435116, 
0.004668938461691141, 0.0053209662437438965, -0.0033219344913959503, -8.68476927280426e-05, -0.013398397713899612, 0.003577607683837414, -0.010426978580653667, -0.012105843052268028, -0.007571258582174778, 0.001982863061130047, 
-0.0068992190062999725, -0.00785558670759201, -0.007499301806092262, -0.011478830128908157, -0.015131881460547447, -0.018555019050836563, -0.014146704226732254, -0.014355591498315334, -0.012452959083020687, -0.025269880890846252, 
-0.02081727609038353, -0.017947908490896225, -0.01392428856343031, -0.01778191141784191, -0.029966983944177628, -0.0174849946051836, -0.035145651549100876, -0.04259634390473366, -0.02208738774061203, -0.029702579602599144, 
-0.029331859201192856, -0.03487328812479973, -0.028196148574352264, -0.0318480022251606, -0.026619618758559227, -0.015682872384786606, -0.015001685358583927, -0.018664948642253876, -0.028641916811466217, -0.02126115933060646, 
-0.003648011479526758, -0.004785957746207714, -0.019370058551430702, 0.0019577699713408947, 0.0031498614698648453, 0.00028650183230638504, 0.007475920021533966, 0.011573271825909615, 0.01767463982105255, -0.0013482198119163513, 
0.009625147096812725, 0.007744039408862591, 0.02359386906027794, 0.012691102921962738, -0.0006755022332072258, 0.016384266316890717, 0.015755699947476387, 0.019303351640701294, 0.0030077900737524033, 0.009804271161556244, 
0.007050029933452606, 0.009318137541413307, 0.0021425886079669, 0.008162070997059345, 0.01319420337677002, 0.0077629657462239265, 0.0063298167660832405, 0.009815195575356483, 0.004482405260205269, 0.007507300470024347, 
0.01744012162089348, 0.02006015181541443, 0.011355124413967133, 0.00874563679099083, 0.024933375418186188, 0.018358590081334114, 0.031407907605171204, 0.01921817846596241, 0.01673196069896221, 0.016766956076025963, 
0.019046353176236153, 0.022372502833604813, 0.01847987249493599, 0.007776034530252218, 0.006185621023178101, 0.005733164958655834, 0.0032461443915963173, 0.01635279506444931, 0.013710650615394115, 0.006715946830809116, 
-0.010742929764091969, 0.00047268718481063843, -0.0017016083002090454, -0.0012719649821519852, -0.013555532321333885, -0.016890905797481537, -0.006589763797819614, -0.010488450527191162, -0.01219123788177967, -0.012364537455141544, 
-0.007542215287685394, -0.009269313886761665, -0.016504112631082535, -0.021127663552761078, -0.0029188981279730797, -0.004051662050187588, -0.00994807854294777, -0.015522979199886322, -0.009450685232877731, -0.009025823324918747, 
-0.012759503908455372, -0.009903128258883953, -0.0070433234795928, -0.015723617747426033, -0.023567788302898407, -0.012726161628961563, -0.011016448959708214, -0.010465683415532112, -0.013869496993720531, -0.01240658387541771, 
-0.008563383482396603, -0.010557440109550953, -0.008036548271775246, -0.009278986603021622, -0.010664273984730244, -0.014631984755396843, -0.020170148462057114, -0.011904722079634666, -0.01419201958924532, -0.01567206159234047, 
-0.01215494703501463, -0.010488556697964668, -0.008642570115625858, -0.011170324869453907, -0.010407916270196438, -0.0012437035329639912, -0.0004891692660748959, -0.005119809880852699, -0.004597971681505442, 0.00354135874658823, 
0.009709660895168781, 0.00966124702244997, 0.011671286076307297, 0.012432686053216457, 0.016937226057052612, 0.014553080312907696, 0.014933806844055653, 0.016539976000785828, 0.014646975323557854, 0.014862787909805775, 
0.012977045960724354, 0.01957242749631405, 0.016415804624557495, 0.017249900847673416, 0.02011186070740223, 0.01936362311244011, 0.027054693549871445, 0.022341884672641754, 0.021134326234459877, 0.02756170928478241, 
0.03738240525126457, 0.0341796949505806, 0.03516959398984909, 0.036455415189266205, 0.036360107362270355, 0.02768637239933014, 0.013455942273139954, 0.012791979126632214, 0.0139000304043293, 0.020465265959501266, 
0.019576264545321465, 0.017297547310590744, 0.010560073889791965, 0.009563966654241085, 0.013565491884946823, 0.012770202942192554, 0.015927504748106003, 0.01053391583263874, 0.005428677424788475, 0.004084796644747257, 
0.007994361221790314, 0.012118506245315075, -0.00010938616469502449, -0.000504007562994957, -0.0038961018435657024, -0.004846250172704458, 0.002216530032455921, 0.002735366113483906, -0.0009244680404663086, -0.007096442393958569, 
-0.01445954293012619, -0.01582801155745983, -0.015107737854123116, -0.011949099600315094, -0.007465077564120293, -0.012717134319245815, -0.015625230967998505, -0.019938863813877106, -0.020192231982946396, -0.02562304213643074, 
-0.02046717330813408, -0.021660754457116127, -0.026571370661258698, -0.029476482421159744, -0.02883911319077015, -0.02672290802001953, -0.03354316204786301, -0.0284946970641613, -0.026661328971385956, -0.03025750070810318, 
-0.03477352485060692, -0.026152465492486954, -0.019486630335450172, -0.026287106797099113, -0.028201742097735405, -0.02212362177670002, -0.024053145200014114, -0.021546639502048492, -0.01926877535879612, -0.018852876499295235, 
-0.012837073765695095, -0.010186661966145039, -0.00634106807410717, -0.008018513210117817, -0.0031882254406809807, 8.078757673501968e-05, -0.0031184321269392967, -0.0019417721778154373, 0.0034492071717977524, 0.005564206279814243, 
0.0027728062123060226, 0.0035271169617772102, 0.006139414384961128, 0.0034198155626654625, 0.00473805982619524, 0.00863973144441843, 0.006216262001544237, 0.009379110299050808, 0.014967575669288635, 0.01505085825920105, 
0.011193696409463882, 0.010539036244153976, 0.010106317698955536, 0.0153940599411726, 0.021892890334129333, 0.02178099751472473, 0.023727068677544594, 0.0316825695335865, 0.03287559747695923, 0.030953239649534225, 
0.03278038650751114, 0.03405715525150299, 0.03447354957461357, 0.03136352077126503, 0.037086568772792816, 0.03749633580446243, 0.039964258670806885, 0.03479569032788277, 0.028779994696378708, 0.029462339356541634, 
0.027762271463871002, 0.030020976439118385, 0.03114345483481884, 0.030539529398083687, 0.023442115634679794, 0.017499802634119987, 0.014911369420588017, 0.016518671065568924, 0.012473780661821365, 0.012184074148535728, 
0.011574525386095047, 0.009621982462704182, 0.007627387531101704, 0.005674206186085939, 0.004104861058294773, 0.002073066309094429, 0.00599878653883934, 0.006698420271277428, 0.004599830135703087, 0.0036580022424459457, 
0.004507813602685928, 0.000816655345261097, -0.0008658850565552711, 0.0006534699350595474, -0.0063714804127812386, -0.01184730976819992, -0.010243136435747147, -0.01293383352458477, -0.012071838602423668, -0.01755460724234581, 
-0.025523263961076736, -0.026933416724205017, -0.022831030189990997, -0.01993025466799736, -0.02397567592561245, -0.01956496760249138, -0.022720472887158394, -0.030202990397810936, -0.03172139450907707, -0.028105560690164566, 
-0.03124195709824562, -0.03316252678632736, -0.0322004109621048, -0.030236924067139626, -0.026765163987874985, -0.027998093515634537, -0.026824243366718292, -0.029423309490084648, -0.029086820781230927, -0.033130012452602386, 
-0.03001071885228157, -0.02644173428416252, -0.026545017957687378, -0.02570372447371483, -0.027385642752051353, -0.02589179389178753, -0.028542831540107727, -0.0298105888068676, -0.030866499990224838, -0.03191341832280159, 
-0.02732931077480316, -0.02673204243183136, -0.02770538628101349, -0.02519039623439312, -0.02115521766245365, -0.01763550564646721, -0.015551598742604256, -0.01426774449646473, -0.01199839822947979, -0.014681202359497547, 
-0.01235888060182333, -0.002924247644841671, -0.005484800785779953, -0.004654623568058014, 0.0014790091663599014, 0.008679458871483803, 0.008637819439172745, 0.013446648605167866, 0.01630144566297531, 0.013356010429561138, 
0.01417975127696991, 0.01592255011200905, 0.017990024760365486, 0.01922588422894478, 0.02884688228368759, 0.03070065937936306, 0.03381996601819992, 0.037131167948246, 0.031044375151395798, 0.026054510846734047, 
0.03285135328769684, 0.036180853843688965, 0.031839411705732346, 0.03580598160624504, 0.036137133836746216, 0.035688694566488266, 0.0362568236887455, 0.03378388285636902, 0.028029419481754303, 0.026699524372816086, 
0.02493712492287159, 0.019585292786359787, 0.022867737337946892, 0.02564215660095215, 0.027919040992856026, 0.02796153351664543, 0.024178821593523026, 0.02069191262125969, 0.018176592886447906, 0.014046057127416134, 
0.010777625255286694, 0.010183369740843773, 0.00816288497298956, 0.0075600589625537395, 0.002487803343683481, -0.00032670143991708755, -0.0021057389676570892, -0.0038350317627191544, -0.0017076879739761353, 0.002795383334159851, 
0.0038048867136240005, -0.0031777145341038704, -0.004088209941983223, -0.002587377093732357, -0.001938006840646267, -0.0043621184304356575, -0.0032734340056777, -0.002976447343826294, -0.0033691832795739174, 0.0017386050894856453, 
-0.0028683701530098915, -0.01071452721953392, -0.009520572610199451, -0.007229352369904518, -0.01427789032459259, -0.02321341075003147, -0.027418335899710655, -0.028078142553567886, -0.02843203954398632, -0.02600669302046299, 
-0.02254834584891796, -0.02333408035337925, -0.027432387694716454, -0.027516869828104973, -0.02725326269865036, -0.023894425481557846, -0.025662269443273544, -0.03161850571632385, -0.03161555156111717, -0.027098074555397034, 
-0.02322843112051487, -0.021610990166664124, -0.014599654823541641, -0.016888534650206566, -0.020197169855237007, -0.019396498799324036, -0.01584470272064209, -0.012041278183460236, -0.005941755138337612, -0.0035184170119464397, 
-0.007584830746054649, -0.005380924325436354, -0.00042981095612049103, -0.0022802911698818207, -0.005196677520871162, 0.0012997305020689964, 0.005169724114239216, 0.0026703951880335808, 0.0051023587584495544, 0.010404959321022034, 
0.004115338437259197, 0.0016045048832893372, 0.009887774474918842, 0.010994825512170792, 0.00528343953192234, 0.008957716636359692, 0.015823721885681152, 0.015379287302494049, 0.017891161143779755, 0.017299439758062363, 
0.014379980973899364, 0.017808154225349426, 0.02543460763990879, 0.02611546404659748, 0.01987692341208458, 0.024815727025270462, 0.029638368636369705, 0.029531870037317276, 0.02799934707581997, 0.027796847745776176, 
0.024358078837394714, 0.018625479191541672, 0.02430145815014839, 0.026749849319458008, 0.02358776517212391, 0.021770203486084938, 0.020526688545942307, 0.02137363702058792, 0.02053936943411827, 0.021768318489193916, 
0.020728297531604767, 0.018295908346772194, 0.014958346262574196, 0.00965193472802639, 0.006931890733540058, 0.003904839977622032, 0.0014385110698640347, -0.0017238776199519634, 0.0004818597808480263, -0.00043993396684527397, 
-0.004712795373052359, -0.008325524628162384, -0.015071386471390724, -0.019837239757180214, -0.01715216599404812, -0.017176588997244835, -0.014282947406172752, -0.014859871938824654, -0.016563497483730316, -0.014962836168706417, 
-0.01560478936880827, -0.017884161323308945, -0.026274265721440315, -0.023519672453403473, -0.024567443877458572, -0.022441204637289047, -0.023793742060661316, -0.03204041346907616, -0.02965105138719082, -0.02826053276658058, 
-0.02634601481258869, -0.03052949532866478, -0.028998173773288727, -0.020881494507193565, -0.01976994425058365, -0.016933584585785866, -0.02062109299004078, -0.02178553119301796, -0.021252930164337158, -0.02126617543399334, 
-0.02129661850631237, -0.021572183817625046, -0.021654682233929634, -0.02257848158478737, -0.020939510315656662, -0.021149760112166405, -0.021256176754832268, -0.02293507382273674, -0.020642921328544617, -0.019449016079306602, 
-0.020544547587633133, -0.020360099151730537, -0.014841588214039803, -0.016973868012428284, -0.019270554184913635, -0.016494929790496826, -0.016582991927862167, -0.014683404937386513, -0.011009938083589077, -0.008132919669151306, 
-0.010180124081671238, -0.003563918638974428, 0.0025844364427030087, 0.0038851420395076275, 0.006674341857433319, 0.009088816121220589, 0.01019233651459217, 0.013848971575498581, 0.020518021658062935, 0.016529398038983345, 
0.02169121988117695, 0.032732367515563965, 0.03811990097165108, 0.036479394882917404, 0.03634047135710716, 0.040231164544820786, 0.032292090356349945, 0.03327074646949768, 0.03750578314065933, 0.033755261451005936, 
0.030316203832626343, 0.033953774720430374, 0.03456372767686844, 0.03505004942417145, 0.040957316756248474, 0.039781197905540466, 0.039306025952100754, 0.03835485130548477, 0.03955511376261711, 0.04442692548036575, 
0.040422260761260986, 0.03958401829004288, 0.04125843942165375, 0.035643570125103, 0.03542141616344452, 0.04173358529806137, 0.04107875004410744, 0.03938731551170349, 0.03110530599951744, 0.030734993517398834, 
0.03237469494342804, 0.02637389674782753, 0.030129101127386093, 0.028640709817409515, 0.023519910871982574, 0.017722809687256813, 0.01882905885577202, 0.01591360941529274, 0.014313552528619766, 0.012227408587932587, 
0.008894987404346466, 0.005186954513192177, -0.003639497794210911, 0.0013239644467830658, -0.00584658607840538, -0.009750597178936005, -0.008988874033093452, -0.01946876011788845, -0.018625983968377113, -0.020119283348321915, 
-0.028094301000237465, -0.029427271336317062, -0.02671598456799984, -0.03039081022143364, -0.03597809746861458, -0.035697244107723236, -0.029772277921438217, -0.03196403756737709, -0.03192576766014099, -0.031618960201740265, 
-0.03393004089593887, -0.033629246056079865, -0.0342983677983284, -0.03789404779672623, -0.042083658277988434, -0.035104554146528244, -0.03504859283566475, -0.03390917927026749, -0.03377202898263931, -0.035351552069187164, 
-0.03237585350871086, -0.035228755325078964, -0.035462066531181335, -0.03374241292476654, -0.03205716609954834, -0.03281223028898239, -0.03146921098232269, -0.026329753920435905, -0.026676703244447708, -0.02273508906364441, 
-0.016305040568113327, -0.012158367782831192, -0.020413022488355637, -0.017762836068868637, -0.012872129678726196, -0.017681211233139038, -0.010660199448466301, -0.007620660588145256, -0.004366932436823845, 0.0001630326732993126, 
0.004558485001325607, 0.0034137414768338203, 0.0022877296432852745, 0.003026619553565979, 0.00847342424094677, 0.011749692261219025, 0.011551108211278915, 0.015774745494127274, 0.0176419485360384, 0.016437258571386337, 
0.015477431006729603, 0.018088150769472122, 0.01992705464363098, 0.02190861850976944, 0.03138755261898041, 0.033721476793289185, 0.03633757308125496, 0.04211011528968811, 0.034180898219347, 0.030998356640338898, 
0.0337199941277504, 0.0320805087685585, 0.026538243517279625, 0.03347494453191757, 0.03653053939342499, 0.032894060015678406, 0.02943640574812889, 0.029236845672130585, 0.025342648848891258, 0.020252550020813942, 
0.02474823221564293, 0.0244277473539114, 0.0230434387922287, 0.02377619408071041, 0.023675795644521713, 0.016311947256326675, 0.01611504517495632, 0.01823175884783268, 0.012217672541737556, 0.007489735260605812, 
0.012993552722036839, 0.014647498726844788, 0.010761651210486889, 0.012994926422834396, 0.006264884956181049, -0.001139078289270401, 0.001271805725991726, -0.0022506248205900192, -0.003617941401898861, -0.0035219567362219095, 
-0.003663746640086174, 0.0006355689838528633, -1.3224780559539795e-07, -0.004487955942749977, -0.003820926882326603, -0.004255611915141344, -0.006143319886177778, -0.006240140646696091, -0.008063452318310738, -0.008022637106478214, 
-0.009808876551687717, -0.012269113212823868, -0.012586314231157303, -0.016560016199946404, -0.02374344691634178, -0.02051808312535286, -0.022622765973210335, -0.02441522851586342, -0.021711934357881546, -0.023303423076868057, 
-0.027435462921857834, -0.02684428170323372, -0.0258841123431921, -0.03191828727722168, -0.029798049479722977, -0.02798588201403618, -0.03381521999835968, -0.035497259348630905, -0.02992410399019718, -0.026968542486429214, 
-0.030173607170581818, -0.031239215284585953, -0.02974451333284378, -0.027418741956353188, -0.025873862206935883, -0.03160618618130684, -0.028898458927869797, -0.019461175426840782, -0.017260128632187843, -0.024716267362236977, 
-0.02331330068409443, -0.012422062456607819, -0.008356764912605286, -0.010995922610163689, -0.006747076287865639, -0.0022117318585515022, -0.0028748605400323868, 0.0014554131776094437, 0.00038753729313611984, 0.0006360495463013649, 
-0.00020242296159267426, 0.005357054993510246, 0.0100704962387681, 0.011805221438407898, 0.016715634614229202, 0.014429382048547268, 0.016200266778469086, 0.022082440555095673, 0.026806192472577095, 0.026925234124064445, 
0.029835309833288193, 0.03056316450238228, 0.02448344975709915, 0.027198191732168198, 0.035305656492710114, 0.034179870039224625, 0.022440336644649506, 0.02203339710831642, 0.02756594307720661, 0.02727128192782402, 
0.02665829472243786, 0.028497589752078056, 0.02683199569582939, 0.02624904178082943, 0.02906426228582859, 0.028955679386854172, 0.02864469215273857, 0.02348022535443306, 0.021636761724948883, 0.021214855834841728, 
0.020594658330082893, 0.021806631237268448, 0.021889999508857727, 0.021779805421829224, 0.02137932926416397, 0.01702803745865822, 0.01589120924472809, 0.014458433724939823, 0.010104774497449398, 0.009156739339232445, 
0.009055486880242825, 0.006977999582886696, 0.0027972678653895855, -0.0030966773629188538, -0.005480058491230011, -0.005947609432041645, -0.011302914470434189, -0.01656477153301239, -0.01921776868402958, -0.02378050982952118, 
-0.025757037103176117, -0.022291146218776703, -0.030841562896966934, -0.03831201419234276, -0.03616377338767052, -0.034464649856090546, -0.03278639167547226, -0.03461217135190964, -0.03301333263516426, -0.033267807215452194, 
-0.03398492932319641, -0.025634746998548508, -0.024637969210743904, -0.025388281792402267, -0.023749954998493195, -0.019055800512433052, -0.018379701301455498, -0.02085244655609131, -0.017291676253080368, -0.01259907241910696, 
-0.014230036176741123, -0.015333633869886398, -0.013195461593568325, -0.016410138458013535, -0.01262750755995512, -0.010989788919687271, -0.009614525362849236, -0.007333539426326752, -0.005516177974641323, -0.00811605341732502, 
-0.010700248181819916, -0.005378649570047855, -0.009786919690668583, -0.011078760027885437, -0.0076485443860292435, -0.011218995787203312, -0.009319678880274296, -0.006744753569364548, -0.005919256713241339, -0.00542790163308382, 
-0.006298551335930824, -0.004919754806905985, 0.00326944375410676, 0.00939225498586893, 0.010027388110756874, 0.010159194469451904, 0.009568051435053349, 0.01409243606030941, 0.015136417001485825, 0.012604252435266972, 
0.01228426769375801, 0.013747409917414188, 0.0170375294983387, 0.019156623631715775, 0.018493883311748505, 0.019908543676137924, 0.02072807215154171, 0.020807521417737007, 0.020537495613098145, 0.02196970023214817, 
0.021623073145747185, 0.021873660385608673, 0.02104322239756584, 0.01683848723769188, 0.015466956421732903, 0.017651230096817017, 0.017635826021432877, 0.013720706105232239, 0.011419594287872314, 0.012619107961654663, 
0.01684623956680298, 0.010183678939938545, 0.00906586367636919, 0.013306367211043835, 0.015129868872463703, 0.01298963651061058, 0.01120840385556221, 0.01252223365008831, 0.0104245375841856, 0.010188375599682331, 
0.010461810044944286, 0.011299991980195045, 0.007838483899831772, 0.009861413389444351, 0.013894862495362759, 0.013451249338686466, 0.016132453456521034, 0.012714764103293419, 0.0058089992962777615, 0.0069633303210139275, 
0.008530433289706707, 0.005485255271196365, 0.003489013761281967, 0.0027374019846320152, -0.00015489710494875908, -0.0005603358149528503, -0.0020680343732237816, -0.006218407768756151, -0.009463082067668438, -0.013334738090634346, 
-0.01648317463696003, -0.015863783657550812, -0.018648041412234306, -0.02688627317547798, -0.026202786713838577, -0.02474178373813629, -0.028172360733151436, -0.02871064469218254, -0.030983980745077133, -0.03708982467651367, 
-0.04023940861225128, -0.03402409702539444, -0.030400242656469345, -0.03128200024366379, -0.027789629995822906, -0.025369901210069656, -0.02318166196346283, -0.02251904085278511, -0.02096620574593544, -0.01533468533307314, 
-0.013170581310987473, -0.01053474098443985, -0.008272305130958557, -0.008466962724924088, -0.009472491219639778, -0.009501708671450615, -0.0029656970873475075, 8.171983063220978e-05, 0.0002621430903673172, 0.00470380112528801, 
0.005581403151154518, 0.007699735462665558, 0.010512332431972027, 0.012541547417640686, 0.011574436910450459, 0.008528560400009155, 0.015150721184909344, 0.01634778082370758, 0.014650809578597546, 0.014263290911912918, 
0.017624422907829285, 0.022063814103603363, 0.019158508628606796, 0.016138670966029167, 0.01628059335052967, 0.02138986811041832, 0.027151845395565033, 0.031082160770893097, 0.02545314095914364, 0.018711179494857788, 
0.022922150790691376, 0.0235444325953722, 0.023874443024396896, 0.02556249126791954, 0.02449232153594494, 0.02468901500105858, 0.025825506076216698, 0.02939879149198532, 0.027891619130969048, 0.02768305130302906, 
0.027462270110845566, 0.023515349254012108, 0.020311536267399788, 0.02334536612033844, 0.0252499058842659, 0.02360323816537857, 0.021047061309218407, 0.022631695494055748, 0.023130496963858604, 0.016064900904893875, 
0.009935312904417515, 0.008552570827305317, 0.011350961402058601, 0.00756719708442688, 0.004693588241934776, 0.003558943048119545, 0.0008634254336357117, -0.0018444554880261421, -0.005952978506684303, -0.006917738355696201, 
-0.0095010781660676, -0.013432558625936508, -0.013108871877193451, -0.014338582754135132, -0.01928875967860222, -0.019654512405395508, -0.01992849074304104, -0.022748500108718872, -0.01789284311234951, -0.018783876672387123, 
-0.020053621381521225, -0.022499404847621918, -0.020473338663578033, -0.02014232613146305, -0.024743176996707916, -0.02673475071787834, -0.025739498436450958, -0.022540995851159096, -0.024685289710760117, -0.018449120223522186, 
-0.02554386667907238, -0.0306156687438488, -0.027647018432617188, -0.03147547319531441, -0.029913660138845444, -0.02665565349161625, -0.029026081785559654, -0.028800103813409805, -0.025081712752580643, -0.0256887786090374, 
-0.029329514130949974, -0.03170746564865112, -0.028689350932836533, -0.02804996632039547, -0.02976873703300953, -0.02534330263733864, -0.024628493934869766, -0.02593882940709591, -0.021614085882902145, -0.022512761875987053, 
-0.020190242677927017, -0.017599137499928474, -0.01539641059935093, -0.011870228685438633, -0.004345450550317764, 0.002583324909210205, 0.0032117795199155807, 0.0033941324800252914, 0.0056875357404351234, 0.009384534321725368, 
0.007754305377602577, 0.010551787912845612, 0.014706502668559551, 0.016288474202156067, 0.020118437707424164, 0.02010633796453476, 0.02247213013470173, 0.02593686245381832, 0.02680373750627041, 0.031132619827985764, 
0.02655162289738655, 0.02562057413160801, 0.025813551619648933, 0.028515800833702087, 0.03148096427321434, 0.032170988619327545, 0.03612418845295906, 0.03193981200456619, 0.030849497765302658, 0.026625650003552437, 
0.026599634438753128, 0.027418196201324463, 0.025174636393785477, 0.02263869158923626, 0.021428639069199562, 0.022825423628091812, 0.02325197495520115, 0.02319807931780815, 0.023385733366012573, 0.022495249286293983, 
0.020481184124946594, 0.01855611428618431, 0.015469477511942387, 0.018454547971487045, 0.018484510481357574, 0.018332544714212418, 0.018550507724285126, 0.015161553397774696, 0.009794594720005989, 0.0074991993606090546, 
0.007518112659454346, 0.00705590657889843, 0.002822818234562874, -0.0011815372854471207, 0.002520998939871788, 0.0022421348839998245, 0.0008104667067527771, 0.002544127404689789, 0.00167076475918293, -0.001094430685043335, 
-0.0018594898283481598, -0.005865041166543961, -0.009024685248732567, -0.011006730608642101, -0.01727413199841976, -0.016883471980690956, -0.015910327434539795, -0.017964625731110573, -0.01923721842467785, -0.019165579229593277, 
-0.019914254546165466, -0.02727973833680153, -0.026964526623487473, -0.02253175899386406, -0.020583830773830414, -0.02075740322470665, -0.022066807374358177, -0.024216771125793457, -0.02452937886118889, -0.021887652575969696, 
-0.022446945309638977, -0.02433946542441845, -0.025323279201984406, -0.025619104504585266, -0.02774702198803425, -0.02778434008359909, -0.026389844715595245, -0.023375559598207474, -0.02048783004283905, -0.020592492073774338, 
-0.022357145324349403, -0.018709862604737282, -0.017851315438747406, -0.018066557124257088, -0.015346640720963478, -0.015238502062857151, -0.0149920005351305, -0.013379191048443317, -0.00923273153603077, -0.007435256615281105, 
-0.006248164921998978, -0.00845850445330143, -0.00709429569542408, -0.002162747085094452, -0.0008202595636248589, 0.0009876871481537819, 0.0014712018892168999, 0.00462702102959156, 0.006318333558738232, 0.006997872143983841, 
0.01237994059920311, 0.015629885718226433, 0.016461089253425598, 0.015424977988004684, 0.01761658489704132, 0.020725322887301445, 0.02103583887219429, 0.0250045545399189, 0.02543165162205696, 0.024225156754255295, 
0.028733639046549797, 0.031194541603326797, 0.028538350015878677, 0.02367563545703888, 0.024046871811151505, 0.02624695934355259, 0.028887636959552765, 0.030747119337320328, 0.031661104410886765, 0.033188071101903915, 
0.03374418616294861, 0.03395248204469681, 0.0329955518245697, 0.03328893706202507, 0.03253152593970299, 0.03015763685107231, 0.028375398367643356, 0.02920784242451191, 0.028678547590970993, 0.023454248905181885, 
0.01783362776041031, 0.01743563823401928, 0.016410185024142265, 0.013871975243091583, 0.010269898921251297, 0.011187667027115822, 0.00885072723031044, 0.004526862874627113, 0.002844393253326416, 0.00021340139210224152, 
-0.0034422650933265686, -0.0050048306584358215, -0.005504149943590164, -0.011876369826495647, -0.012249765917658806, -0.013848217204213142, -0.017441287636756897, -0.01870458945631981, -0.014841113239526749, -0.017952967435121536, 
-0.023139018565416336, -0.024019736796617508, -0.02327672764658928, -0.023586206138134003, -0.02467498555779457, -0.020188532769680023, -0.020217236131429672, -0.019069122150540352, -0.02157994545996189, -0.02196740359067917, 
-0.023599155247211456, -0.02843165211379528, -0.028277289122343063, -0.029438264667987823, -0.027058500796556473, -0.02477310597896576, -0.027697188779711723, -0.026606004685163498, -0.026504091918468475, -0.02791461907327175, 
-0.02580876648426056, -0.02609158679842949, -0.02593599446117878, -0.02636348456144333, -0.02371087111532688, -0.018709249794483185, -0.01971401274204254, -0.019823282957077026, -0.016863804310560226, -0.015051675029098988, 
-0.016194386407732964, -0.018769370391964912, -0.013741858303546906, -0.00745687261223793, -0.00619170069694519, -0.006037920713424683, -0.003797002136707306, -0.0033133458346128464, -0.0039634183049201965, -0.002608953043818474, 
-0.004431720823049545, -0.004680372774600983, 0.0004360862076282501, 0.005181116983294487, 0.0023491457104682922, 6.301701068878174e-05, 0.003885985352098942, 0.006295241415500641, 0.008343247696757317, 0.013225963339209557, 
0.014491221867501736, 0.012287690304219723, 0.01199214905500412, 0.011330259963870049, 0.014919571578502655, 0.020192688331007957, 0.01931440457701683, 0.019221745431423187, 0.01873375102877617, 0.019474925473332405, 
0.020246800035238266, 0.02037648856639862, 0.02101317048072815, 0.01466608140617609, 0.013029221445322037, 0.018836352974176407, 0.024740898981690407, 0.024290990084409714, 0.02544189617037773, 0.027705814689397812, 
0.02182663418352604, 0.018664900213479996, 0.020174499601125717, 0.01857319474220276, 0.01248057372868061, 0.008413832634687424, 0.01148872822523117, 0.014342052862048149, 0.012818197719752789, 0.012870138511061668, 
0.012940184213221073, 0.014425603672862053, 0.015011077746748924, 0.014829668216407299, 0.01395498588681221, 0.00845140591263771, 0.004659920930862427, 0.005071152932941914, 0.004913765005767345, 0.0031805643811821938, 
0.0007803020998835564, -0.00258488766849041, -0.0035702986642718315, 0.0002533048391342163, -0.0003578606992959976, -0.002159089781343937, -0.0018649054691195488, -0.0020667612552642822, -0.004656714387238026, -0.007806443143635988, 
-0.007697552442550659, -0.008462752215564251, -0.009391352534294128, -0.009602649137377739, -0.009352524764835835, -0.011960688978433609, -0.014604146592319012, -0.014964573085308075, -0.013342873193323612, -0.01239107083529234, 
-0.01302824355661869, -0.014916757121682167, -0.01305630523711443, -0.010538656264543533, -0.012484638020396233, -0.015198927372694016, -0.013484325259923935, -0.011370662599802017, -0.011799732223153114, -0.008028740994632244, 
-0.008091232739388943, -0.010837480425834656, -0.00958399660885334, -0.008959857746958733, -0.012472300790250301, -0.011753067374229431, -0.010085362941026688, -0.014793961308896542, -0.013123039156198502, -0.009719258174300194, 
-0.010377340018749237, -0.0073738787323236465, -0.0023283977061510086, -0.001611987128853798, -0.002808603458106518, 0.0012072715908288956, 0.004299843683838844, 0.001931784674525261, -0.0009376686066389084, 0.003805645741522312, 
0.009449891746044159, 0.012538759037852287, 0.01699412241578102, 0.017464863136410713, 0.018803076818585396, 0.022049766033887863, 0.018301691859960556, 0.01596110314130783, 0.016904545947909355, 0.016251569613814354, 
0.01426009926944971, 0.012568668462336063, 0.014640677720308304, 0.014608534052968025, 0.014990468509495258, 0.01480353157967329, 0.01717725209891796, 0.015067987143993378, 0.010650455951690674, 0.008302377536892891, 
0.004961371421813965, 0.009520982392132282, 0.012134340591728687, 0.007760196924209595, 0.006958273239433765, 0.008051705546677113, 0.0058277677744627, 0.005680875852704048, 0.006424862891435623, 0.005164654925465584, 
0.004563404247164726, 0.0060106003656983376, 0.003278353251516819, 0.002820294350385666, 0.003734072670340538, 0.005012011155486107, 0.00593973696231842, 0.002729548141360283, -0.0010842829942703247, -0.00059555284678936, 
0.0006392113864421844, -0.0024385470896959305, -0.005241291597485542, -0.005480503663420677, -0.005583517253398895, -0.010860374197363853, -0.009149322286248207, -0.004482110030949116, -0.0076277004554867744, -0.012188718654215336, 
-0.013609983026981354, -0.015337986871600151, -0.013991354033350945, -0.012196973897516727, -0.012842945754528046, -0.01746155321598053, -0.018793653696775436, -0.017122391611337662, -0.018825115635991096, -0.016558833420276642, 
-0.013097388669848442, -0.014397874474525452, -0.0175371952354908, -0.01368158869445324, -0.013994027860462666, -0.015518329106271267, -0.012755569070577621, -0.0131526505574584, -0.013663885183632374, -0.011959290131926537, 
-0.010467574000358582, -0.010892968624830246, -0.012806269340217113, -0.01145698968321085, -0.010639723390340805, -0.009360604919493198, -0.007934663444757462, -0.010662692598998547, -0.014533180743455887, -0.012121113017201424, 
-0.006268389523029327, -0.005769314244389534, -0.0046107713133096695, -0.007033286616206169, -0.00763910636305809, -0.004666823893785477, -0.0028322767466306686, -0.005275268107652664, -0.003970984369516373, 0.0009739156812429428, 
0.0016214735805988312, 0.00023983418941497803, 0.002785559743642807, 0.007215885445475578, 0.004158562049269676, 0.0059133004397153854, 0.007726095616817474, 0.0072978585958480835, 0.009805381298065186, 0.013217148371040821, 
0.012536367401480675, 0.010249000042676926, 0.013453862629830837, 0.014037178829312325, 0.012450745329260826, 0.01538508664816618, 0.015993241220712662, 0.014456088654696941, 0.01632297784090042, 0.018406465649604797, 
0.01874982938170433, 0.016346512362360954, 0.017749957740306854, 0.017577532678842545, 0.01457282342016697, 0.01571040228009224, 0.01484652515500784, 0.011640235781669617, 0.011307740584015846, 0.012286145240068436, 
0.012357732281088829, 0.012891724705696106, 0.0118100019171834, 0.01094072125852108, 0.010526747442781925, 0.009263819083571434, 0.005768716335296631, 0.006389198824763298, 0.007199372164905071, 0.00277593731880188, 
0.002987183630466461, 0.00393305066972971, 0.0021280907094478607, -0.0023259706795215607, 0.0010376013815402985, 0.0033477284014225006, 0.0016165710985660553, -0.00013586506247520447, -0.002720603719353676, -0.0027481205761432648, 
-0.005990608595311642, -0.0036397315561771393, -0.002902362495660782, -0.0026557771489024162, 0.0009298175573348999, -0.0019955141469836235, -0.007253442890942097, -0.007439509034156799, -0.007766957860440016, -0.012322274968028069, 
-0.012403912842273712, -0.010250631719827652, -0.013591605238616467, -0.013600727543234825, -0.012922486290335655, -0.012894375249743462, -0.012017539702355862, -0.011089947074651718, -0.012453757226467133, -0.013540886342525482, 
-0.008770443499088287, -0.011006083339452744, -0.01435902714729309, -0.013156982138752937, -0.012664487585425377, -0.014400959014892578, -0.012640709057450294, -0.01408778689801693, -0.014681791886687279, -0.009520703926682472, 
-0.007935995236039162, -0.005641171708703041, -0.006083657965064049, -0.00959097407758236, -0.011278762482106686, -0.006970416754484177, -0.002799578011035919, -0.004551112651824951, -0.0014461278915405273, -0.0017708390951156616, 
-0.003466043621301651, -0.0003642849624156952, 0.0036846203729510307, 0.008815818466246128, 0.009043280966579914, 0.00844934955239296, 0.008078319020569324, 0.01189750898629427, 0.013735448010265827, 0.010087613016366959, 
0.009077347815036774, 0.011910596862435341, 0.01580227166414261, 0.01885295659303665, 0.018440619111061096, 0.02133351005613804, 0.02399616502225399, 0.023511681705713272, 0.020531252026557922, 0.01866934262216091, 
0.01819736883044243, 0.01680002734065056, 0.016600074246525764, 0.0162943284958601, 0.018228404223918915, 0.01760837435722351, 0.01759171299636364, 0.01710454374551773, 0.019456271082162857, 0.019145488739013672, 
0.019802551716566086, 0.01701437123119831, 0.01184292696416378, 0.01351616345345974, 0.01221961248666048, 0.01143072172999382, 0.012709586881101131, 0.009696881286799908, 0.0085957832634449, 0.009019419550895691, 
0.006133563816547394, -0.002000153064727783, -0.003578042611479759, 0.0038211457431316376, 0.003920815885066986, -0.0020171068608760834, -0.004371140152215958, 0.002231994643807411, -0.00081600621342659, -0.0073968637734651566, 
-0.014152731746435165, -0.012746928259730339, -0.0075847078114748, -0.01228355523198843, -0.014892016537487507, -0.019213706254959106, -0.022236695513129234, -0.023313041776418686, -0.024602971971035004, -0.031470030546188354, 
-0.031762875616550446, -0.029183700680732727, -0.029616810381412506, -0.025018637999892235, -0.030540117993950844, -0.035455141216516495, -0.03382578492164612, -0.029245564714074135, -0.025241784751415253, -0.027819493785500526, 
-0.03477724269032478, -0.03568220138549805, -0.030215824022889137, -0.032353464514017105, -0.03113488107919693, -0.026063034310936928, -0.02494567632675171, -0.020503833889961243, -0.01550404354929924, -0.015860887244343758, 
-0.017213348299264908, -0.018533091992139816, -0.022370077669620514, -0.018522515892982483, -0.014769917353987694, -0.01417914405465126, -0.009023468941450119, -0.006239201873540878, -0.006275732070207596, -0.004536494612693787, 
-0.005024636164307594, -0.007838284596800804, -0.006378667429089546, -0.004530500620603561, 0.002458399161696434, 0.004536531865596771, 0.006014518439769745, 0.007409915328025818, 0.004130726680159569, 0.000823974609375, 
0.0026274919509887695, 0.010706843808293343, 0.011342305690050125, 0.0144940335303545, 0.021616198122501373, 0.019535280764102936, 0.011959625408053398, 0.011728385463356972, 0.01548241451382637, 0.01306014321744442, 
0.010984515771269798, 0.013212854042649269, 0.02131599187850952, 0.02261582389473915, 0.021052028983831406, 0.024130159988999367, 0.019782040268182755, 0.02199750393629074, 0.022471114993095398, 0.02219972014427185, 
0.027105938643217087, 0.028763554990291595, 0.022874586284160614, 0.01815035566687584, 0.024965636432170868, 0.02555834874510765, 0.022515352815389633, 0.02468130923807621, 0.02621004730463028, 0.02642170339822769, 
0.026285320520401, 0.024834973737597466, 0.022359933704137802, 0.02487606182694435, 0.0232289656996727, 0.023607153445482254, 0.022107262164354324, 0.015171054750680923, 0.017969105392694473, 0.020225590094923973, 
0.009525010362267494, -0.0012362804263830185, 0.004641290754079819, 0.015209446661174297, 0.011240296997129917, -0.0033428706228733063, -0.002965018153190613, -0.001771446317434311, -0.005837573669850826, -0.008251058869063854, 
-0.01807582564651966, -0.023867670446634293, -0.013645287603139877, -0.008832304738461971, -0.01238502748310566, -0.020940782502293587, -0.021432196721434593, -0.011965781450271606, -0.0186261348426342, -0.02485704980790615, 
-0.023967232555150986, -0.01753152161836624, -0.011746642179787159, -0.019178040325641632, -0.02929205447435379, -0.0192530807107687, -0.013958575204014778, -0.02339266799390316, -0.025874707847833633, -0.019256968051195145, 
-0.014509236440062523, -0.012225963175296783, -0.008814595639705658, -0.014197140000760555, -0.018936200067400932, -0.019766928628087044, -0.014936413615942001, -0.013751499354839325, -0.01663259044289589, -0.010755613446235657, 
-0.004748037084937096, -0.005191531032323837, -0.011256391182541847, -0.014673952013254166, -0.00995017308741808, -0.004674363881349564, -0.003147977404296398, -0.005296941846609116, 0.002807995304465294, 0.013230569660663605, 
0.010705257765948772, 0.010753213427960873, 0.01460272166877985, 0.013709408231079578, 0.016083307564258575, 0.024370932951569557, 0.025670446455478668, 0.021201463416218758, 0.026741385459899902, 0.03543713316321373, 
0.03322306275367737, 0.028252294287085533, 0.02721088007092476, 0.032268259674310684, 0.03668564558029175, 0.03247157484292984, 0.03391997516155243, 0.03901086002588272, 0.042405322194099426, 0.03937695920467377, 
0.029870958998799324, 0.032762184739112854, 0.03574451059103012, 0.033725839108228683, 0.031880978494882584, 0.03191689774394035, 0.027732007205486298, 0.030576054006814957, 0.03074270486831665, 0.023190272971987724, 
0.027259275317192078, 0.0243162102997303, 0.023291608318686485, 0.02348167635500431, 0.01766359806060791, 0.011185696348547935, 0.016208216547966003, 0.01722590997815132, 0.005905821919441223, 0.0035472922027111053, 
0.006737549789249897, 0.0043298713862895966, -0.000902772881090641, -0.00383183266967535, -0.0068617649376392365, -0.009161558002233505, -0.012188207358121872, -0.017071722075343132, -0.02740693837404251, -0.0283830389380455, 
-0.0255887433886528, -0.028810301795601845, -0.03368845954537392, -0.03336264193058014, -0.03547538444399834, -0.043165531009435654, -0.04480687901377678, -0.04424411430954933, -0.04931865632534027, -0.049974046647548676, 
-0.045879676938056946, -0.04697567969560623, -0.0518488809466362, -0.05583720654249191, -0.05353417992591858, -0.05107402801513672, -0.05039233714342117, -0.05841702222824097, -0.05503718927502632, -0.047232404351234436, 
-0.044382840394973755, -0.04436445236206055, -0.04412752389907837, -0.04104313254356384, -0.03918251395225525, -0.036269254982471466, -0.0380626879632473, -0.03365702927112579, -0.03090900182723999, -0.02895999141037464, 
-0.027910683304071426, -0.023659346625208855, -0.01882181316614151, -0.021058402955532074, -0.012390336021780968, -0.007027952000498772, -0.011688201688230038, -0.008837923407554626, -0.0007465966045856476, 0.003965110518038273, 
0.0034020720049738884, 0.0056965649127960205, 0.006922136060893536, 0.013417343609035015, 0.019443370401859283, 0.01788446493446827, 0.020044727250933647, 0.03050938807427883, 0.037392549216747284, 0.033072326332330704, 
0.03550131618976593, 0.039135415107011795, 0.03656482696533203, 0.03440837562084198, 0.03582949936389923, 0.03806079551577568, 0.037886448204517365, 0.036080311983823776, 0.035345952957868576, 0.03343941271305084, 
0.03615375608205795, 0.04012938588857651, 0.03483987972140312, 0.030195625498890877, 0.03371617570519447, 0.03291536867618561, 0.02921859547495842, 0.028054799884557724, 0.02598511055111885, 0.02866397611796856, 
0.0362953282892704, 0.03158750385046005, 0.026338918134570122, 0.02587161958217621, 0.01998838037252426, 0.013998650014400482, 0.013557006604969501, 0.018871773034334183, 0.01812715083360672, 0.015370428562164307, 
0.008002938702702522, 0.010064329020678997, 0.011218777857720852, 0.0038584303110837936, 0.0013931533321738243, 0.0010284418240189552, 0.0011700447648763657, 0.0004447782412171364, -0.00396569911390543, -0.005415752530097961, 
-0.005419211462140083, -0.010691728442907333, -0.013874745927751064, -0.008423181250691414, -0.012412258423864841, -0.022355975583195686, -0.02149060368537903, -0.020740728825330734, -0.01612021215260029, -0.018567683175206184, 
-0.020140131935477257, -0.019049081951379776, -0.024632051587104797, -0.029682457447052002, -0.028796380385756493, -0.03072940930724144, -0.032776474952697754, -0.03553136810660362, -0.03467203304171562, -0.03249890357255936, 
-0.033341340720653534, -0.03504118323326111, -0.03597128018736839, -0.03928142040967941, -0.04262036457657814, -0.03516555577516556, -0.03637559711933136, -0.03580399975180626, -0.03594059497117996, -0.038408588618040085, 
-0.04080234467983246, -0.039400167763233185, -0.0351833775639534, -0.03633914515376091, -0.03812004625797272, -0.03018852509558201, -0.028828205540776253, -0.026081889867782593, -0.02020074799656868, -0.019692495465278625, 
-0.010784659534692764, -0.007701407186686993, -0.009161248803138733, -0.0070210956037044525, 0.0014272890985012054, -0.00027801841497421265, 0.0008292291313409805, 0.007977139204740524, 0.01072997972369194, 0.013662880286574364, 
0.016460133716464043, 0.019787032157182693, 0.01834360882639885, 0.01993965357542038, 0.029033951461315155, 0.026584308594465256, 0.026031846180558205, 0.029951535165309906, 0.03202792629599571, 0.03386195749044418, 
0.03180056810379028, 0.0336226187646389, 0.0363171361386776, 0.03649042174220085, 0.0385991595685482, 0.042361874133348465, 0.04566918686032295, 0.04836929589509964, 0.04531728848814964, 0.04746251553297043, 
0.04914148896932602, 0.04627087339758873, 0.04109378904104233, 0.04082443192601204, 0.039431773126125336, 0.039891645312309265, 0.039359040558338165, 0.03592107445001602, 0.03509635105729103, 0.03554333746433258, 
0.03448373079299927, 0.029868053272366524, 0.029546985402703285, 0.026949694380164146, 0.02570590190589428, 0.024683590978384018, 0.015911836177110672, 0.011390208266675472, 0.01262112706899643, 0.004302999470382929, 
0.0024016695097088814, 0.004600787069648504, 0.002294418402016163, -0.002862313762307167, -0.002621673047542572, -0.0067327930592000484, -0.010523234494030476, -0.01387688796967268, -0.021031558513641357, -0.02569509483873844, 
-0.030191883444786072, -0.03066437691450119, -0.03483176231384277, -0.0351107195019722, -0.033066824078559875, -0.03471849486231804, -0.03495951369404793, -0.03753504529595375, -0.04303818941116333, -0.03595566004514694, 
-0.03286372870206833, -0.037885937839746475, -0.04078942537307739, -0.03985939919948578, -0.03864062577486038, -0.04226579889655113, -0.04638119786977768, -0.04438424110412598, -0.04118727520108223, -0.03533456474542618, 
-0.03259305655956268, -0.032552607357501984, -0.029547113925218582, -0.029165897518396378, -0.026789121329784393, -0.02303120493888855, -0.02792472392320633, -0.029137155041098595, -0.018241608515381813, -0.0190509632229805, 
-0.02118109166622162, -0.02090846374630928, -0.017596449702978134, -0.012991460040211678, -0.012096049264073372, -0.01404148805886507, -0.009688619524240494, -0.004194498527795076, -0.0019163237884640694, -0.00011060107499361038, 
0.0008242903277277946, 0.004369885660707951, 0.004360577091574669, 0.007150798104703426, 0.012547120451927185, 0.01431412436068058, 0.018749568611383438, 0.02368018962442875, 0.026645174250006676, 0.028817877173423767, 
0.027790356427431107, 0.033112045377492905, 0.03440244495868683, 0.02743046171963215, 0.030480526387691498, 0.03744246065616608, 0.03456626087427139, 0.03405185043811798, 0.03166431561112404, 0.0344015471637249, 
0.03622093051671982, 0.032723553478717804, 0.036268025636672974, 0.033870916813611984, 0.03288991004228592, 0.0348503440618515, 0.03145755082368851, 0.028964422643184662, 0.02722388505935669, 0.025643667206168175, 
0.02924918942153454, 0.029158953577280045, 0.024726349860429764, 0.02430054359138012, 0.022787176072597504, 0.016741544008255005, 0.014213614165782928, 0.019833087921142578, 0.021649038419127464, 0.012843205593526363, 
0.01005799975246191, 0.011956315487623215, 0.01107281818985939, 0.0076746451668441296, 0.005255599040538073, 0.0025015315040946007, -0.0025785337202250957, -0.005462301895022392, -0.006041378248482943, -0.003858243115246296, 
-0.007152308709919453, -0.010701945051550865, -0.011107152327895164, -0.013960601761937141, -0.02202095091342926, -0.02816190756857395, -0.026632975786924362, -0.026224324479699135, -0.028703339397907257, -0.02856479398906231, 
-0.029857713729143143, -0.03225066140294075, -0.032351765781641006, -0.03491935878992081, -0.034788358956575394, -0.03497868776321411, -0.04033610224723816, -0.03893231973052025, -0.03655724972486496, -0.03696971386671066, 
-0.03421943262219429, -0.03585389629006386, -0.038643889129161835, -0.043015141040086746, -0.04125887155532837, -0.03250359743833542, -0.032137662172317505, -0.035748258233070374, -0.029454883188009262, -0.023734821006655693, 
-0.02805527113378048, -0.026063840836286545, -0.01905614323914051, -0.02039450965821743, -0.022399617359042168, -0.02066425234079361, -0.019085953012108803, -0.016605185344815254, -0.01455562561750412, -0.013057820498943329, 
-0.01414363645017147, -0.012049839831888676, -0.006785389501601458, -0.004253431688994169, -0.00012104958295822144, 0.0032489225268363953, 0.00821838527917862, 0.01300746202468872, 0.010731765069067478, 0.013723365031182766, 
0.01597272977232933, 0.02029014378786087, 0.026470521464943886, 0.024393033236265182, 0.022596944123506546, 0.025223033502697945, 0.028156649321317673, 0.03165411204099655, 0.032505422830581665, 0.03497340530157089, 
0.04095909744501114, 0.03764725103974342, 0.03578619658946991, 0.03754927217960358, 0.03973665088415146, 0.03799930214881897, 0.03899586573243141, 0.046223729848861694, 0.0457092821598053, 0.04443763941526413, 
0.04524227976799011, 0.04247892647981644, 0.04023037105798721, 0.038109008222818375, 0.03606107831001282, 0.03928297385573387, 0.0407286137342453, 0.036924369633197784, 0.03622201830148697, 0.03401394188404083, 
0.029059158638119698, 0.025694411247968674, 0.02514098584651947, 0.019306249916553497, 0.013542739674448967, 0.012938367202877998, 0.01154690608382225, 0.010403790511190891, 0.005377344321459532, 0.0031940722838044167, 
0.004046248272061348, 0.0020619709976017475, -0.006030172109603882, -0.006873327307403088, -0.005683505441993475, -0.010329796001315117, -0.012128442525863647, -0.012807419523596764, -0.014314355328679085, -0.016361985355615616, 
-0.019390061497688293, -0.023765401914715767, -0.023844528943300247, -0.02766583114862442, -0.02955036610364914, -0.024506088346242905, -0.026102278381586075, -0.03212205320596695, -0.03470838814973831, -0.03537549078464508, 
-0.035406000912189484, -0.034494902938604355, -0.03528840094804764, -0.03462737053632736, -0.03426678478717804, -0.036016739904880524, -0.03529134392738342, -0.03656672313809395, -0.035819388926029205, -0.03524091839790344, 
-0.03565145283937454, -0.03514321893453598, -0.03431795537471771, -0.029034487903118134, -0.027802718803286552, -0.029107503592967987, -0.02633095532655716, -0.02609954960644245, -0.02506273239850998, -0.019498353824019432, 
-0.01794048584997654, -0.015530896373093128, -0.013051009736955166, -0.00955295655876398, -0.007743513211607933, -0.008294896222651005, -0.0054984427988529205, -0.00027712155133485794, 0.0035689505748450756, 0.0038700131699442863, 
0.00480437558144331, 0.006949468515813351, 0.013639556244015694, 0.016390793025493622, 0.017753828316926956, 0.021276229992508888, 0.020245207473635674, 0.021544110029935837, 0.02325299009680748, 0.022426603361964226, 
0.02666809782385826, 0.0290805846452713, 0.02853306010365486, 0.030635474249720573, 0.029869444668293, 0.028182465583086014, 0.026785217225551605, 0.028327789157629013, 0.03233132138848305, 0.031200218945741653, 
0.02970809116959572, 0.03134351968765259, 0.03645568713545799, 0.037218574434518814, 0.029261860996484756, 0.027984123677015305, 0.03159603476524353, 0.029688913375139236, 0.02677765116095543, 0.02511831745505333, 
0.025624150410294533, 0.024910783395171165, 0.022400544956326485, 0.02046428993344307, 0.018697133287787437, 0.019565735012292862, 0.016340695321559906, 0.013395826332271099, 0.012688746675848961, 0.013760710135102272, 
0.011358880437910557, 0.006431239657104015, 0.006071350537240505, 0.006937253288924694, 0.005459840875118971, 0.00124063016846776, 0.0004053385928273201, -0.005427737720310688, -0.009116093628108501, -0.010985074564814568, 
-0.013975773006677628, -0.01811755821108818, -0.022969212383031845, -0.022178109735250473, -0.021437272429466248, -0.024906478822231293, -0.029374051839113235, -0.03030707687139511, -0.03278857097029686, -0.03632733225822449, 
-0.03880085051059723, -0.03628372400999069, -0.03820766881108284, -0.038803815841674805, -0.03687759116292, -0.04031701385974884, -0.03972306847572327, -0.04261389747262001, -0.04508667066693306, -0.04697943478822708, 
-0.050745658576488495, -0.05135210603475571, -0.05034675449132919, -0.04975438863039017, -0.05044484883546829, -0.04489133134484291, -0.03826410323381424, -0.03851080685853958, -0.0411679781973362, -0.04063020646572113, 
-0.038856346160173416, -0.03862273693084717, -0.03672134131193161, -0.033151257783174515, -0.030334243550896645, -0.02915034070611, -0.023913666605949402, -0.01942090317606926, -0.015897855162620544, -0.010068237781524658, 
-0.011552352458238602, -0.009786439128220081, -0.004478903952986002, -0.0014296770095825195, 0.0016301991418004036, 0.007489821873605251, 0.011796773411333561, 0.012194516137242317, 0.013357224874198437, 0.01222722977399826, 
0.017816225066781044, 0.022622983902692795, 0.02563704550266266, 0.030054675415158272, 0.031091494485735893, 0.03286692500114441, 0.034767355769872665, 0.035890132188797, 0.03733120858669281, 0.03957618772983551, 
0.04043004661798477, 0.043414052575826645, 0.046220649033784866, 0.04577387124300003, 0.04350738599896431, 0.04366367682814598, 0.04647565260529518, 0.04542115330696106, 0.046824440360069275, 0.046176034957170486, 
0.04406499117612839, 0.043867845088243484, 0.0423765704035759, 0.044159580022096634, 0.04354739189147949, 0.04095517843961716, 0.042030856013298035, 0.04403656721115112, 0.043245188891887665, 0.03852525353431702, 
0.03834179788827896, 0.03712358698248863, 0.031674016267061234, 0.03252112865447998, 0.027518458664417267, 0.023163076490163803, 0.017824433743953705, 0.009389607235789299, 0.009356299415230751, 0.008457724004983902, 
0.006079713813960552, 0.004652484320104122, 0.0012180008925497532, -0.004097141325473785, -0.005581772420555353, -0.0076097361743450165, -0.012192318215966225, -0.01587461680173874, -0.016693634912371635, -0.020591478794813156, 
-0.021775372326374054, -0.02109970711171627, -0.02734202891588211, -0.026904353871941566, -0.026602331548929214, -0.025828884914517403, -0.026199132204055786, -0.028290554881095886, -0.02992466278374195, -0.03098524734377861, 
-0.03456613048911095, -0.036533333361148834, -0.03331717848777771, -0.03057893179357052, -0.02600979432463646, -0.026893295347690582, -0.026355385780334473, -0.0270632766187191, -0.024630114436149597, -0.026924019679427147, 
-0.026492170989513397, -0.024787940084934235, -0.026003826409578323, -0.021932724863290787, -0.022238019853830338, -0.021953077986836433, -0.020209021866321564, -0.017668742686510086, -0.017661161720752716, -0.014558878727257252, 
-0.013311732560396194, -0.013226659968495369, -0.010508153587579727, -0.010339440777897835, -0.012044827453792095, -0.006860197521746159, -0.00446827057749033, -0.003368249163031578, -0.001935051754117012, -0.003426522947847843, 
0.0025145187973976135, 0.0036314046010375023, 0.00180792436003685, 0.004150522872805595, 0.005200001411139965, 0.006371034774929285, 0.008822513744235039, 0.011079465970396996, 0.011717984452843666, 0.01177175808697939, 
0.012353182770311832, 0.016731221228837967, 0.02331716939806938, 0.020829061046242714, 0.020399659872055054, 0.023258082568645477, 0.022535763680934906, 0.025630008429288864, 0.031072691082954407, 0.032459795475006104, 
0.034992363303899765, 0.0357743501663208, 0.0332951582968235, 0.0332658514380455, 0.030816594138741493, 0.028397344052791595, 0.027829041704535484, 0.026482097804546356, 0.030744075775146484, 0.028145048767328262, 
0.02216595783829689, 0.024332460016012192, 0.020593613386154175, 0.01995277591049671, 0.01973297819495201, 0.018689682707190514, 0.019229361787438393, 0.017915673553943634, 0.017466094344854355, 0.013195119798183441, 
0.011468986049294472, 0.008895369246602058, 0.00205068476498127, -0.0010183518752455711, -0.003731047734618187, -0.00030668918043375015, -0.0017937077209353447, -0.00861403439193964, -0.012517469935119152, -0.015432976186275482, 
-0.015317052602767944, -0.018090449273586273, -0.01880623959004879, -0.02148216776549816, -0.02198018878698349, -0.023651819676160812, -0.029806409031152725, -0.028373228386044502, -0.026013748720288277, -0.027722973376512527, 
-0.027565376833081245, -0.0329214371740818, -0.035002700984478, -0.030320381745696068, -0.029229499399662018, -0.03426605090498924, -0.03751267120242119, -0.0319620743393898, -0.03203786909580231, -0.032185979187488556, 
-0.03271057456731796, -0.02917778305709362, -0.02894769422709942, -0.035628221929073334, -0.03578483313322067, -0.031033268198370934, -0.03164916858077049, -0.03558303043246269, -0.03593809902667999, -0.0310064684599638, 
-0.02475557290017605, -0.024075817316770554, -0.020087404176592827, -0.024704474955797195, -0.019224505871534348, -0.012422613799571991, -0.011265510693192482, -0.00815250352025032, -0.006612342782318592, -0.0005062194541096687, 
-0.0008607050403952599, -0.0015837885439395905, 0.0007743211463093758, 0.007742320187389851, 0.0037091635167598724, 0.0017085829749703407, 0.013502243906259537, 0.017781397327780724, 0.020238414406776428, 0.02546413242816925, 
0.025547096505761147, 0.027587831020355225, 0.024213718250393867, 0.024730324745178223, 0.027679787948727608, 0.02647961862385273, 0.028049223124980927, 0.025653302669525146, 0.028432369232177734, 0.028586983680725098, 
0.03201693668961525, 0.031027451157569885, 0.028096459805965424, 0.030897285789251328, 0.033998701721429825, 0.03444237262010574, 0.029382959008216858, 0.033412810415029526, 0.036187902092933655, 0.03235064446926117, 
0.030910959467291832, 0.03392907977104187, 0.033407606184482574, 0.02953796461224556, 0.027712473645806313, 0.02539808861911297, 0.02725256234407425, 0.029159829020500183, 0.023908529430627823, 0.020247388631105423, 
0.017465172335505486, 0.018119536340236664, 0.016623282805085182, 0.012548429891467094, 0.013258697465062141, 0.009625151753425598, 0.00518158171325922, 0.0036199111491441727, 0.004563998430967331, 0.0018783295527100563, 
-0.0031443312764167786, -0.00814899429678917, -0.007491583004593849, -0.010400588624179363, -0.016682198271155357, -0.01925083063542843, -0.02430909126996994, -0.022588158026337624, -0.018591444939374924, -0.020013097673654556, 
-0.022295894101262093, -0.01751621440052986, -0.01846814528107643, -0.02144024893641472, -0.020407728850841522, -0.022896450012922287, -0.025624994188547134, -0.0244145430624485, -0.024448052048683167, -0.027665918692946434, 
-0.027074221521615982, -0.032068587839603424, -0.031602442264556885, -0.029071033000946045, -0.030309759080410004, -0.027093231678009033, -0.026932988315820694, -0.02594754658639431, -0.018102489411830902, -0.014523417688906193, 
-0.018519654870033264, -0.01967766135931015, -0.018588518723845482, -0.015157371759414673, -0.014836582355201244, -0.01355823129415512, -0.01097759697586298, -0.00722648948431015, -0.007383972406387329, -0.014680864289402962, 
-0.012917468324303627, -0.010297313332557678, -0.013117674738168716, -0.013773858547210693, -0.00853559747338295, -0.0040877945721149445, -0.006401653401553631, -0.0064561814069747925, -0.00485973060131073, -0.00024090241640806198, 
0.003841179423034191, 0.005260709673166275, 0.003125520423054695, 0.006769826635718346, 0.009131740778684616, 0.008017133921384811, 0.014669970609247684, 0.011354283429682255, 0.010064303874969482, 0.007865076884627342, 
0.010349126532673836, 0.012449363246560097, 0.010248321108520031, 0.014224303886294365, 0.01700533553957939, 0.01930885761976242, 0.023764021694660187, 0.021065661683678627, 0.01950334757566452, 0.019039303064346313, 
0.018237410113215446, 0.02782324329018593, 0.023997079581022263, 0.021096091717481613, 0.01906048320233822, 0.018569305539131165, 0.01657246984541416, 0.011629783548414707, 0.012799350544810295, 0.014096757397055626, 
0.015178918838500977, 0.015135539695620537, 0.015922004356980324, 0.017431743443012238, 0.019035814329981804, 0.013903401792049408, 0.014364195056259632, 0.014945978298783302, 0.014865517616271973, 0.013744422234594822, 
0.012036105617880821, 0.012037320993840694, 0.01080583781003952, 0.013874520547688007, 0.0070135341957211494, 0.0021056802943348885, 0.004278451669961214, 0.0032280609011650085, -0.0011904612183570862, -0.0022921431809663773, 
-0.001412695273756981, -0.003537392243742943, -0.003929076716303825, -0.005195949226617813, -0.008168244734406471, -0.007795682176947594, -0.012426194734871387, -0.017717765644192696, -0.014246251434087753, -0.01759616658091545, 
-0.0196366086602211, -0.023610811680555344, -0.019530897960066795, -0.017780613154172897, -0.020005052909255028, -0.014484785497188568, -0.01497017964720726, -0.014820633456110954, -0.017928902059793472, -0.017572740092873573, 
-0.018351731821894646, -0.016979709267616272, -0.010816918686032295, -0.014515683986246586, -0.017692532390356064, -0.01521009299904108, -0.009772459045052528, -0.009972786530852318, -0.011081920005381107, -0.0066641950979828835, 
-0.010966095142066479, -0.006214730441570282, -0.006750686094164848, -0.0034971311688423157, 0.00028049200773239136, -0.002768067643046379, 0.001927986741065979, 0.0028444211930036545, 0.01040901429951191, 0.013170618563890457, 
0.009171908721327782, 0.009442420676350594, 0.012028571218252182, 0.012882214970886707, 0.014462613500654697, 0.0162469781935215, 0.02183522656559944, 0.022983010858297348, 0.0193516593426466, 0.015903528779745102, 
0.018582908436655998, 0.0209455955773592, 0.024297816678881645, 0.024762462824583054, 0.02807758003473282, 0.024329951032996178, 0.024419602006673813, 0.030056972056627274, 0.0229068323969841, 0.01508362963795662, 
0.013614125549793243, 0.030629746615886688, 0.0238037109375, 0.019339080899953842, 0.020882505923509598, 0.02126627042889595, 0.02487572841346264, 0.020736025646328926, 0.021083053201436996, 0.026095958426594734, 
0.030355162918567657, 0.022784583270549774, 0.016328491270542145, 0.018587911501526833, 0.01733359880745411, 0.01266586221754551, 0.012926538474857807, 0.0085578802973032, 0.00752636231482029, 0.005864662118256092, 
0.0026952866464853287, 0.0006633643060922623, -6.193295121192932e-05, -0.009048350155353546, -0.013790613040328026, -0.006587117910385132, -0.009220961481332779, -0.011702625080943108, -0.013761894777417183, -0.014202844351530075, 
-0.011930176988244057, -0.013952960260212421, -0.020796265453100204, -0.024575697258114815, -0.02691507712006569, -0.025434015318751335, -0.027423616498708725, -0.03108806535601616, -0.027095312252640724, -0.027758914977312088, 
-0.022565728053450584, -0.0292808935046196, -0.03235679119825363, -0.027552474290132523, -0.03283507749438286, -0.032420068979263306, -0.02959410287439823, -0.03607260435819626, -0.03625936806201935, -0.03542013466358185, 
-0.041989415884017944, -0.04103449732065201, -0.040236588567495346, -0.03688247129321098, -0.038411810994148254, -0.03986693173646927, -0.047840140759944916, -0.047254301607608795, -0.037158846855163574, -0.03741797059774399, 
-0.03540971502661705, -0.02677440457046032, -0.027856767177581787, -0.030400890856981277, -0.0216531865298748, -0.01883058063685894, -0.024679485708475113, -0.023724878206849098, -0.017314504832029343, -0.01677064225077629, 
-0.014993231743574142, -0.01879141852259636, -0.014276393689215183, -0.006952254101634026, -0.007864358834922314, -0.010695313103497028, -0.0037761274725198746, 0.00464947335422039, -6.384961307048798e-05, -0.0016830898821353912, 
0.0013793939724564552, 0.00917772762477398, 0.005776582285761833, 0.008711624890565872, 0.01954754628241062, 0.02406053990125656, 0.018384136259555817, 0.01358719915151596, 0.022679124027490616, 0.02481834962964058, 
0.024447904899716377, 0.02273097075521946, 0.031153038144111633, 0.029472410678863525, 0.029055269435048103, 0.03275427222251892, 0.028099123388528824, 0.026366647332906723, 0.03210359811782837, 0.03307173773646355, 
0.028929658234119415, 0.03476518392562866, 0.03082641214132309, 0.033117279410362244, 0.030909504741430283, 0.022067736834287643, 0.021166659891605377, 0.024193136021494865, 0.017889413982629776, 0.01411399058997631, 
0.016063164919614792, 0.015453755855560303, 0.015696097165346146, 0.013531485572457314, 0.009960492141544819, 0.010700900107622147, 0.010164394974708557, 0.010224220342934132, 0.013256572186946869, 0.012560836039483547, 
0.010944489389657974, 0.0031990278512239456, 0.003317972645163536, 0.007589297369122505, 0.006789849139750004, -0.002483934164047241, 0.0008174479007720947, 0.004186157137155533, 0.0014576157554984093, 0.0001439843326807022, 
0.0003393460065126419, -0.0027054715901613235, -0.006639967672526836, -0.007617255672812462, -0.012920435518026352, -0.005429891869425774, -0.005115441046655178, -0.012280771508812904, -0.013256827369332314, -0.009506134316325188, 
-0.016427354887127876, -0.01944694295525551, -0.017731403931975365, -0.0162050724029541, -0.00990738719701767, -0.01182265393435955, -0.01755405217409134, -0.01885082758963108, -0.015483863651752472, -0.01424387563019991, 
-0.01808910444378853, -0.020911984145641327, -0.017142191529273987, -0.01770264282822609, -0.021359317004680634, -0.018887940794229507, -0.012602278962731361, -0.013716279529035091, -0.013567398302257061, -0.012240419164299965, 
-0.01752486452460289, -0.02060374617576599, -0.014787409454584122, -0.016380490735173225, -0.016630209982395172, -0.011495777405798435, -0.012111285701394081, -0.009347074665129185, -0.007696947082877159, -0.004152094945311546, 
0.0014431774616241455, 0.005044971592724323, 0.005065295845270157, 0.0031673405319452286, 0.007149586454033852, 0.00833849050104618, 0.007463363930583, 0.010679442435503006, 0.010263276286423206, 0.013641142286360264, 
0.012917714193463326, 0.012006450444459915, 0.012031296268105507, 0.014988474547863007, 0.015383690595626831, 0.01593310385942459, 0.019883934408426285, 0.01858123391866684, 0.01584179326891899, 0.013893138617277145, 
0.014860556460916996, 0.01717754825949669, 0.025833498686552048, 0.024710364639759064, 0.02480063959956169, 0.023383725434541702, 0.02824404463171959, 0.03473975881934166, 0.03033989854156971, 0.025766458362340927, 
0.026407919824123383, 0.030186111107468605, 0.03268643096089363, 0.03365756943821907, 0.02460659295320511, 0.02266494370996952, 0.023638879880309105, 0.019959047436714172, 0.01863708719611168, 0.01901083253324032, 
0.01570964604616165, 0.02080495096743107, 0.019094545394182205, 0.014258543029427528, 0.009108887985348701, 0.011594911105930805, 0.010518939234316349, 0.0006467718631029129, 9.55536961555481e-05, -0.0022637080401182175, 
-0.00011062528938055038, -0.00894674751907587, -0.012966378591954708, -0.017435777932405472, -0.01927531510591507, -0.018944602459669113, -0.017649676650762558, -0.021178513765335083, -0.024701856076717377, -0.022073619067668915, 
-0.02641380950808525, -0.026626555249094963, -0.027402004227042198, -0.024597061797976494, -0.028154103085398674, -0.027666650712490082, -0.02664162404835224, -0.032124560326337814, -0.03083752654492855, -0.026498962193727493, 
-0.025898952037096024, -0.03143869712948799, -0.027754493057727814, -0.02404768019914627, -0.03133133798837662, -0.0335346944630146, -0.029793472960591316, -0.027373792603611946, -0.029308265075087547, -0.02336519956588745, 
-0.020476002246141434, -0.022928772494196892, -0.02452579326927662, -0.02530561573803425, -0.02280505746603012, -0.020951885730028152, -0.01892523653805256, -0.021176982671022415, -0.02118350937962532, -0.016774162650108337, 
-0.013504134491086006, -0.01792983151972294, -0.0171651653945446, -0.011229224503040314, -0.010625235736370087, -0.010328158736228943, -0.0042717549949884415, 0.0021027345210313797, 0.0034823808819055557, 0.0025147628039121628, 
-0.0025351103395223618, -0.0002749301493167877, 0.00529785081744194, 0.0017180554568767548, 0.0022912416607141495, 0.006737705320119858, 0.008022400550544262, 0.0013525467365980148, 0.00012618675827980042, 0.008730965666472912, 
0.017915256321430206, 0.019773822277784348, 0.0135990921407938, 0.011247416958212852, 0.012210107408463955, 0.016779381781816483, 0.020118363201618195, 0.02355242148041725, 0.020556550472974777, 0.019222956150770187, 
0.0203896202147007, 0.01687074452638626, 0.013861335813999176, 0.018049728125333786, 0.01937205158174038, 0.019896335899829865, 0.022707033902406693, 0.024498848244547844, 0.02091650851070881, 0.01914091408252716, 
0.022752178832888603, 0.018752437084913254, 0.018208421766757965, 0.017688123509287834, 0.017486438155174255, 0.016853466629981995, 0.0181441530585289, 0.020861241966485977, 0.021255481988191605, 0.016009822487831116, 
0.018124915659427643, 0.01782175898551941, 0.004588384181261063, 0.006522089242935181, 0.015422623604536057, 0.008360091596841812, 0.0037861093878746033, 0.0035691671073436737, 0.0058568324893713, 0.0033583082258701324, 
0.0008643623441457748, 0.001843119040131569, -0.0017438773065805435, 0.0005539879202842712, -0.002180621027946472, 4.358217120170593e-05, 0.004287635907530785, 0.0005036341026425362, -0.007831576280295849, -0.007713492959737778, 
-0.0071404799818992615, -0.005879310891032219, -0.006184950936585665, -0.010329321026802063, -0.010194496251642704, -0.006352168973535299, -0.010870681144297123, -0.016606442630290985, -0.009463910944759846, -0.008309396915137768, 
-0.013673516921699047, -0.020853925496339798, -0.02091072127223015, -0.02089051343500614, -0.01733720488846302, -0.016377083957195282, -0.02108616940677166, -0.015295569784939289, -0.008132291957736015, -0.012686038389801979, 
-0.014538019895553589, -0.011510353535413742, -0.011028766632080078, -0.009664606302976608, -0.010497823357582092, -0.0052787307649850845, -0.006250662729144096, -0.007249300368130207, -0.004558811895549297, -0.006820479407906532, 
-0.00914156623184681, -0.004123273305594921, 0.003299061208963394, 0.0057193078100681305, 0.004035541787743568, 0.008334580808877945, 0.0189597737044096, 0.013880272395908833, 0.014327678829431534, 0.013382363133132458, 
0.011275001801550388, 0.01585332490503788, 0.019204843789339066, 0.02037382312119007, 0.019741766154766083, 0.024583254009485245, 0.020180296152830124, 0.016914332285523415, 0.021903621032834053, 0.029507266357541084, 
0.027095802128314972, 0.025042450055480003, 0.026453234255313873, 0.025062426924705505, 0.026844989508390427, 0.027387242764234543, 0.02715163305401802, 0.018042519688606262, 0.01251124870032072, 0.016591759398579597, 
0.017836082726716995, 0.011981185525655746, 0.010244863107800484, 0.012563643045723438, 0.00909609254449606, 0.0018534474074840546, 0.0004840213805437088, 0.0013607088476419449, -0.003064442425966263, 0.0002083219587802887, 
-0.000989805907011032, -0.007946547120809555, -0.00509455893188715, -0.0026360321789979935, -0.005238795652985573, -0.008425114676356316, -0.014489833265542984, -0.012437807396054268, -0.006464916281402111, -0.009927376173436642, 
-0.013646900653839111, -0.0158127062022686, -0.016981326043605804, -0.021908685564994812, -0.02223302610218525, -0.019412130117416382, -0.021256180480122566, -0.019009197130799294, -0.015034059062600136, -0.014723125845193863, 
-0.01831148937344551, -0.021968455985188484, -0.02145935781300068, -0.021515320986509323, -0.02055485174059868, -0.01700419932603836, -0.018188484013080597, -0.02360021322965622, -0.023749839514493942, -0.0171388927847147, 
-0.022271426394581795, -0.02116609923541546, -0.017157094553112984, -0.016860414296388626, -0.017082171514630318, -0.01577872782945633, -0.006858258508145809, -0.007412572391331196, -0.00605715811252594, -0.005519948899745941, 
-0.011707576923072338, -0.010442077182233334, -0.00039940886199474335, 0.002252275124192238, -0.002645544707775116, 0.0020095445215702057, 0.004858618602156639, 0.0011904872953891754, 0.0014035291969776154, 0.003962783142924309, 
0.004953288473188877, 0.003618561662733555, 0.010630985721945763, 0.014014605432748795, 0.012750810012221336, 0.014878933317959309, 0.019385214895009995, 0.021660976111888885, 0.018114592880010605, 0.01504536159336567, 
0.01674005761742592, 0.015934988856315613, 0.01970013603568077, 0.029224490746855736, 0.028256986290216446, 0.024633418768644333, 0.023803822696208954, 0.02018401399254799, 0.019201256334781647, 0.023509236052632332, 
0.022490080446004868, 0.020311780273914337, 0.02315441519021988, 0.03129461407661438, 0.03402409702539444, 0.028148403391242027, 0.027094582095742226, 0.029917635023593903, 0.023057004436850548, 0.018927503377199173, 
0.01636582612991333, 0.022660521790385246, 0.02704671584069729, 0.025997500866651535, 0.020345808938145638, 0.012581679038703442, 0.014161383733153343, 0.0031041568145155907, -0.008924327790737152, -0.014523391611874104, 
-0.005686575546860695, -0.00021831132471561432, -0.00751790776848793, -0.014082089997828007, -0.009782904759049416, -0.007428827695548534, -0.021241445094347, -0.03520544618368149, -0.034564584493637085, -0.034364718943834305, 
-0.032191887497901917, -0.028534024953842163, -0.030808303505182266, -0.02457030862569809, -0.02164476178586483, -0.02594926953315735, -0.035648979246616364, -0.0339655764400959, -0.03859107568860054, -0.043101951479911804, 
-0.037694089114665985, -0.035821519792079926, -0.03456799313426018, -0.03655832260847092, -0.03518516570329666, -0.03942783176898956, -0.03702569380402565, -0.030107684433460236, -0.03176897391676903, -0.03486781194806099, 
-0.035918399691581726, -0.030125020071864128, -0.03249039500951767, -0.03830160200595856, -0.03658245876431465, -0.032073721289634705, -0.03326312452554703, -0.03480686992406845, -0.02825818583369255, -0.0240459181368351, 
-0.03059908002614975, -0.03385230153799057, -0.02663775347173214, -0.0222490094602108, -0.024194199591875076, -0.028097234666347504, -0.019329972565174103, -0.012641757726669312, -0.007438434287905693, -0.006254522129893303, 
-0.014483434148132801, -0.015919072553515434, -0.009051971137523651, -0.0031770458444952965, -0.0026483051478862762, -0.00494824443012476, -0.0005775559693574905, -0.0015257569029927254, 0.0029635364189743996, 0.011514278128743172, 
0.006823687814176083, 0.008342818357050419, 0.014248228631913662, 0.021421954035758972, 0.013972929678857327, 0.017593834549188614, 0.026526816189289093, 0.029505591839551926, 0.029610756784677505, 0.02752358466386795, 
0.03342331200838089, 0.030375972390174866, 0.02717682160437107, 0.028860755264759064, 0.02692386880517006, 0.03134438768029213, 0.04289275407791138, 0.03829963505268097, 0.031069345772266388, 0.03141418844461441, 
0.03518051654100418, 0.03015916794538498, 0.02885083295404911, 0.02922958694398403, 0.02606528252363205, 0.02755877748131752, 0.02587578073143959, 0.031215012073516846, 0.03515931963920593, 0.033641114830970764, 
0.028517479076981544, 0.025465402752161026, 0.02844301611185074, 0.022687550634145737, 0.018479518592357635, 0.020401885733008385, 0.016946397721767426, 0.014845229685306549, 0.012176068499684334, 0.015320152044296265, 
0.013600407168269157, 0.012759627774357796, 0.013229358941316605, 0.0028067808598279953, -0.0069800447672605515, -0.005041719414293766, -0.010536611080169678, -0.01510596927255392, -0.010059121996164322, -0.011879499070346355, 
-0.01209267508238554, -0.011606014333665371, -0.008632558397948742, -0.019059669226408005, -0.028959903866052628, -0.026746980845928192, -0.024241028353571892, -0.027840901166200638, -0.026948582381010056, -0.025927327573299408, 
-0.026101455092430115, -0.023972325026988983, -0.02481355518102646, -0.026815906167030334, -0.02568507194519043, -0.028250282630324364, -0.03212890028953552, -0.0310220830142498, -0.02470574341714382, -0.01635586842894554, 
-0.02065323479473591, -0.02714327536523342, -0.028926406055688858, -0.022421224042773247, -0.025475863367319107, -0.02932465448975563, -0.02118481881916523, -0.008746090345084667, -0.006937356665730476, -0.005020153243094683, 
-0.0013879723846912384, -0.0008384943939745426, -0.0019382890313863754, 0.0007170485332608223, 0.002487311139702797, 0.004192350897938013, 0.010726886801421642, 0.002642836421728134, -0.0011000866070389748, 0.003845478408038616, 
0.013988060876727104, 0.009703515097498894, 0.009034264832735062, 0.012585291638970375, 0.016246352344751358, 0.018986808136105537, 0.015304732136428356, 0.01722530461847782, 0.021342631429433823, 0.02838321402668953, 
0.025086764246225357, 0.025511976331472397, 0.034767355769872665, 0.03883256018161774, 0.03995164483785629, 0.036599867045879364, 0.03734258562326431, 0.03898390009999275, 0.03816395252943039, 0.043652839958667755, 
0.0410180389881134, 0.04349714517593384, 0.04758129641413689, 0.046389080584049225, 0.04179557412862778, 0.04165274649858475, 0.041622236371040344, 0.04341049864888191, 0.04389319568872452, 0.03825048357248306, 
0.03831930831074715, 0.03662455081939697, 0.03631221503019333, 0.032756321132183075, 0.03223009780049324, 0.035212475806474686, 0.03036940097808838, 0.022759299725294113, 0.01770075410604477, 0.01753019541501999, 
0.015003984794020653, 0.008077256381511688, 0.010984284803271294, 0.005508961621671915, 0.005065299570560455, 0.007294175680726767, 0.0036765439435839653, 0.0001320093870162964, -0.0066968947649002075, -0.0027344366535544395, 
0.0006680237129330635, -0.003458114340901375, -0.009869307279586792, -0.009154088795185089, -0.006945371627807617, -0.015730082988739014, -0.018338145688176155, -0.018774814903736115, -0.02457779459655285, -0.02948598936200142, 
-0.0322287455201149, -0.0292774960398674, -0.03480777516961098, -0.03442561626434326, -0.03163041174411774, -0.03051748126745224, -0.03583583980798721, -0.044811002910137177, -0.04557614028453827, -0.04522983729839325, 
-0.04171480983495712, -0.04141503572463989, -0.04003039747476578, -0.037238191813230515, -0.0357208289206028, -0.0344342477619648, -0.03555038943886757, -0.03911666199564934, -0.0417812280356884, -0.03721185773611069, 
-0.032133013010025024, -0.036305610090494156, -0.038070909678936005, -0.030864283442497253, -0.020890837535262108, -0.021547798067331314, -0.02703392319381237, -0.026991119608283043, -0.02213059552013874, -0.019649572670459747, 
-0.01772015541791916, -0.019199470058083534, -0.019059840589761734, -0.012063206173479557, -0.01873902603983879, -0.016385624185204506, -0.007144941948354244, -0.007328953593969345, -0.0035229241475462914, -0.0006275791674852371, 
-0.000996823888272047, 0.0028329119086265564, 0.008038156665861607, 0.010016148909926414, 0.006235962733626366, 0.0029217787086963654, 0.009874671697616577, 0.016047708690166473, 0.013849450275301933, 0.009140865877270699, 
0.013226911425590515, 0.02285803109407425, 0.01950962468981743, 0.013054007664322853, 0.01766878180205822, 0.028897013515233994, 0.025022411718964577, 0.01864822395145893, 0.022147497162222862, 0.0251267459243536, 
0.02980990707874298, 0.025256602093577385, 0.021063655614852905, 0.0240947213023901, 0.031204311177134514, 0.030249468982219696, 0.02806904911994934, 0.021611414849758148, 0.023900646716356277, 0.035455893725156784, 
0.03215169161558151, 0.026970745995640755, 0.02852451056241989, 0.02824348397552967, 0.019180625677108765, 0.02351984567940235, 0.023802919313311577, 0.014468301087617874, 0.017831280827522278, 0.018386783078312874, 
0.010084381327033043, 0.012027569115161896, 0.013171054422855377, 0.003566056489944458, 0.0009580031037330627, 0.005229867994785309, 0.0024720467627048492, -0.005267251282930374, -0.009901387616991997, -0.014414901845157146, 
-0.022135084494948387, -0.022991137579083443, -0.02326110377907753, -0.02396111935377121, -0.023388927802443504, -0.02604026533663273, -0.018950004130601883, -0.01715218648314476, -0.0286840982735157, -0.03593357652425766, 
-0.030173873528838158, -0.027923565357923508, -0.02827654778957367, -0.028585825115442276, -0.03447213023900986, -0.03269706293940544, -0.033208876848220825, -0.03866741433739662, -0.04145120084285736, -0.04169687628746033, 
-0.03381302207708359, -0.031104572117328644, -0.030713669955730438, -0.02719828113913536, -0.02715575322508812, -0.02451324090361595, -0.02655072510242462, -0.03254447132349014, -0.026432843878865242, -0.02203270234167576, 
-0.02648763358592987, -0.028397295624017715, -0.018518317490816116, -0.008684493601322174, -0.01081680878996849, -0.011994067579507828, -0.012734528630971909, -0.006831670179963112, -0.001986641436815262, -0.003134004771709442, 
-0.004422664642333984, 0.0010287538170814514, 0.011096682399511337, 0.014456937089562416, 0.010924514383077621, 0.012954376637935638, 0.016937237232923508, 0.016235537827014923, 0.012954279780387878, 0.013916537165641785, 
0.018185153603553772, 0.018824409693479538, 0.026035767048597336, 0.03160540759563446, 0.03441799432039261, 0.03640816733241081, 0.03320679813623428, 0.03289508447051048, 0.03401432931423187, 0.0302080437541008, 
0.030814260244369507, 0.03116311877965927, 0.036764051765203476, 0.03610729053616524, 0.03441588580608368, 0.03720249980688095, 0.035002030432224274, 0.03447669744491577, 0.037884268909692764, 0.03684943914413452, 
0.027524080127477646, 0.02894275262951851, 0.028357379138469696, 0.02407885156571865, 0.01919940859079361, 0.014491530135273933, 0.012137919664382935, 0.013220325112342834, 0.011211957782506943, 0.011073857545852661, 
0.011461388319730759, 0.012401938438415527, 0.008377786725759506, 0.002733945846557617, 0.007784411311149597, -0.0008328147232532501, -0.008730720728635788, -0.005878269672393799, -0.005429388955235481, -0.010514345020055771, 
-0.009457513689994812, -0.003004089929163456, -0.007466007024049759, -0.011609714478254318, -0.011553982272744179, -0.014626933261752129, -0.015162312425673008, -0.016486963257193565, -0.020068688318133354, -0.0242143664509058, 
-0.022622572258114815, -0.018140317872166634, -0.01984701119363308, -0.020029421895742416, -0.022374562919139862, -0.025372983887791634, -0.02381722815334797, -0.024885185062885284, -0.024525975808501244, -0.025333642959594727, 
-0.020936498418450356, -0.02093602903187275, -0.02468031644821167, -0.02660781517624855, -0.026914553716778755, -0.02341698855161667, -0.027274662628769875, -0.02826954424381256, -0.02581973932683468, -0.015818463638424873, 
-0.0133054219186306, -0.013715212233364582, -0.01355722639709711, -0.016638804227113724, -0.014464345760643482, -0.011238925158977509, -0.011762479320168495, -0.008302386850118637, -0.007332326844334602, -0.009733467362821102, 
-0.006180775351822376, -0.0011581936851143837, 0.0037657301872968674, -0.003739415667951107, -0.0021623168140649796, 0.0026197321712970734, 0.009664403274655342, 0.012131279334425926, 0.012587388977408409, 0.014966131187975407, 
0.01202258002012968, 0.014334957115352154, 0.012664668262004852, 0.015210320241749287, 0.015490747056901455, 0.017695993185043335, 0.020864786580204964, 0.027593854814767838, 0.02791702002286911, 0.021141989156603813, 
0.024260640144348145, 0.02298619970679283, 0.023182280361652374, 0.02303607016801834, 0.022540636360645294, 0.02981637977063656, 0.0337691530585289, 0.034812863916158676, 0.039894189685583115, 0.03893597051501274, 
0.03663797676563263, 0.037519410252571106, 0.033507090061903, 0.029417235404253006, 0.027430295944213867, 0.02587883360683918, 0.026875104755163193, 0.02735176309943199, 0.022615816444158554, 0.021413542330265045, 
0.014794657938182354, 0.006543985567986965, 0.008678515441715717, 0.012776070274412632, 0.010189030319452286, 0.00022023357450962067, -0.0015583978965878487, -0.0024702269583940506, -0.0051606036722660065, -0.006673149764537811, 
-0.011772741563618183, -0.013196710497140884, -0.01596846990287304, -0.020343562588095665, -0.020096005871891975, -0.017647847533226013, -0.01808849535882473, -0.023611821234226227, -0.022344272583723068, -0.020928267389535904, 
-0.027341745793819427, -0.030520278960466385, -0.028466517105698586, -0.027839627116918564, -0.03031000867486, -0.024337271228432655, -0.021921034902334213, -0.02637336589396, -0.025554601103067398, -0.027269504964351654, 
-0.029698042199015617, -0.03184816613793373, -0.03228282555937767, -0.03002937138080597, -0.03229735046625137, -0.029577981680631638, -0.026340102776885033, -0.02835928276181221, -0.02817614935338497, -0.03345414623618126, 
-0.0307184886187315, -0.021454045549035072, -0.017083628103137016, -0.008659204468131065, -0.00770419929176569, -0.01254956889897585, -0.012819853611290455, -0.006327452138066292, -0.008545245975255966, -0.011997457593679428, 
-0.0038285828195512295, -0.0012233005836606026, 0.005483362823724747, 0.011728819459676743, 0.014520522207021713, 0.016268402338027954, 0.016967065632343292, 0.018820801749825478, 0.012962155044078827, 0.008255844935774803, 
0.011043785139918327, 0.013743865303695202, 0.015488819219172001, 0.016386765986680984, 0.014749028719961643, 0.017998341470956802, 0.024237999692559242, 0.022476719692349434, 0.024333544075489044, 0.024273596704006195, 
0.02224212884902954, 0.028600728139281273, 0.03177240118384361, 0.022898292168974876, 0.01688431017100811, 0.024402249604463577, 0.02548912726342678, 0.026201670989394188, 0.023595163598656654, 0.019761960953474045, 
0.017360877245664597, 0.01989341527223587, 0.024309903383255005, 0.02095397189259529, 0.014650890603661537, 0.01578383333981037, 0.013656392693519592, 0.013582643121480942, 0.01242821291089058, 0.004122251644730568, 
0.006975410506129265, 0.0004863152280449867, 0.00023010605946183205, -0.00032844673842191696, -0.0014731977134943008, -0.0008706338703632355, -0.003427925519645214, -0.009293954819440842, -0.004629725590348244, 0.0019226227886974812, 
-0.008008926175534725, -0.009859154000878334, -0.012815384194254875, -0.016094576567411423, -0.011783698573708534, -0.012256051413714886, -0.01798877865076065, -0.017170555889606476, -0.015232869423925877, -0.014492933638393879, 
-0.014841804280877113, -0.016634289175271988, -0.017013657838106155, -0.022257646545767784, -0.026942679658532143, -0.02095247246325016, -0.0203215554356575, -0.020285451784729958, -0.023086071014404297, -0.031088147312402725, 
-0.027805224061012268, -0.02248200587928295, -0.025833556428551674, -0.031199531629681587, -0.0306473970413208, -0.02639913372695446, -0.026358991861343384, -0.026177290827035904, -0.02747504971921444, -0.028275594115257263, 
-0.026797045022249222, -0.02866905741393566, -0.026473775506019592, -0.025860393419861794, -0.023009352385997772, -0.02162749320268631, -0.015926608815789223, -0.009578055702149868, -0.008389469236135483, -0.010658902116119862, 
-0.010333340615034103, -0.004128415137529373, 0.004234751686453819, 0.004873430356383324, 0.0005753273144364357, 0.005711641162633896, 0.009060370735824108, 0.007707457989454269, 0.006236400455236435, 0.004630180075764656, 
0.00874403491616249, 0.020965270698070526, 0.024682335555553436, 0.029233209788799286, 0.02652415633201599, 0.026191769167780876, 0.03203307464718819, 0.038808610290288925, 0.03673949092626572, 0.03322705999016762, 
0.04242313280701637, 0.03939531743526459, 0.03927121311426163, 0.04211990907788277, 0.04770687222480774, 0.047235630452632904, 0.04654505103826523, 0.048642173409461975, 0.04624983295798302, 0.04110090434551239, 
0.038232773542404175, 0.03770595043897629, 0.03775269538164139, 0.03825158625841141, 0.03763143718242645, 0.04513763636350632, 0.04220838472247124, 0.04763481020927429, 0.06583592295646667, 0.07425373792648315, 
0.02782610058784485, -0.06708699464797974, -0.1487899124622345, -0.1693967580795288, -0.13213950395584106, -0.04769160598516464, 0.05159420520067215, 0.14823506772518158, 0.19957520067691803, 0.16516020894050598, 
0.10547607392072678, 0.05894084274768829, 0.030309394001960754, -0.008425615727901459, 0.019533609971404076, 0.08003221452236176, 0.08293959498405457, -0.000799170695245266, -0.06580588966608047, -0.036184582859277725, 
0.006616841536015272, 0.058390308171510696, 0.18691599369049072, 0.3176993727684021, 0.3788514733314514, 0.3605530560016632, 0.2616453766822815, 0.13384824991226196, -0.041013553738594055, -0.22185857594013214, 
-0.40417516231536865, -0.46071678400039673, -0.3816888928413391, -0.2547096610069275, -0.15261094272136688, -0.08703219890594482, -0.04799681529402733, -0.1439495086669922, -0.32794544100761414, -0.45606958866119385, 
-0.5080199241638184, -0.49117857217788696, -0.3992939889431, -0.2991475462913513, -0.2179204821586609, -0.14847233891487122, -0.04826395958662033, 0.005931531079113483, 0.0028107864782214165, 0.010626433417201042, 
0.06135975569486618, 0.13717922568321228, 0.19761015474796295, 0.2789725065231323, 0.3852725625038147, 0.4472343921661377, 0.45308321714401245, 0.4105585813522339, 0.3184363842010498, 0.24069589376449585, 
0.19006460905075073, 0.1733332872390747, 0.20132339000701904, 0.23868851363658905, 0.21607953310012817, 0.10993198305368423, -0.025602638721466064, -0.1308194100856781, -0.217521071434021, -0.256709486246109, 
-0.2317609339952469, -0.17526599764823914, -0.1136130839586258, -0.08919072896242142, -0.12670689821243286, -0.20675183832645416, -0.27537330985069275, -0.2819465398788452, -0.2436181902885437, -0.195452481508255, 
-0.11092908680438995, 0.0039055729284882545, 0.08149594068527222, 0.10876181721687317, 0.11284557729959488, 0.10737253725528717, 0.11963071674108505, 0.115153007209301, 0.11461558938026428, 0.12972977757453918, 
0.16883648931980133, 0.1723412424325943, 0.1390594094991684, 0.11230801790952682, 0.1300007551908493, 0.16888433694839478, 0.1746111512184143, 0.15777380764484406, 0.13909128308296204, 0.11620157212018967, 
0.033814311027526855, -0.10509216785430908, -0.20715439319610596, -0.17756688594818115, -0.06792664527893066, 0.023303192108869553, -0.0954018086194992, -0.11353586614131927, -0.014866584911942482, -0.07052723318338394, 
-0.10825884342193604, -0.10534470528364182, -0.05392032861709595, -0.08106256276369095, -0.06278800964355469, -0.002888912335038185, -0.03364744037389755, 0.004096876829862595, -0.09642556309700012, -0.08827079087495804, 
-0.08260097354650497, -0.0651138424873352, -0.032313618808984756, -0.1365797370672226, -0.023167584091424942, 0.002109173685312271, 0.016839968040585518, 0.05999016389250755, 0.026206212118268013, 0.03584577888250351, 
0.06309234350919724, -0.0018996447324752808, -0.029856812208890915, -0.11779387295246124, -0.04132486507296562, 0.0004961518570780754, -0.09605379402637482, -0.1007743626832962, -0.0018301019445061684, 0.05912169814109802, 
-0.09845913201570511, -0.009805593639612198, 0.06868626922369003, 0.0880240797996521, 0.040773626416921616, 0.006171926856040955, 0.01964760757982731, 0.025857066735625267, 0.03735070675611496, 0.014054154977202415, 
0.07233292609453201, -0.0008928356692194939, -0.07801911234855652, 0.01995329186320305, -0.010902638547122478, -0.03217940405011177, -0.08102697879076004, -0.023006638512015343, -0.020613091066479683, -0.04983428493142128, 
0.05120019242167473, -0.06695844978094101, -0.009724095463752747, -0.03131861239671707, -0.03924406319856644, -0.05172712355852127, -0.06374633312225342, -0.020661737769842148, -0.011453349143266678, 0.04318423196673393, 
-0.038662638515233994, -0.028784360736608505, 0.05833524465560913, 0.1295640766620636, 0.07002239674329758, 0.07103415578603745, 0.09193848073482513, 0.10592911392450333, 0.13304133713245392, 0.09295587241649628, 
0.22058016061782837, 0.1995466649532318, 0.04684443399310112, 0.08679095655679703, 0.17542243003845215, 0.14472773671150208, 0.07826922833919525, -0.003960287664085627, 0.02709818258881569, 0.04017406702041626, 
0.016039544716477394, -0.027946768328547478, -0.05077424645423889, 0.03488710895180702, -0.0036613112315535545, -0.01810063049197197, -0.024782707914710045, -0.06056468188762665, -0.0810425877571106, 0.0022602975368499756, 
-0.010100824758410454, -0.08727432042360306, -0.0392008051276207, 0.01575031317770481, 0.0043129753321409225, -0.03165243938565254, -0.052237749099731445, -0.028575386852025986, 0.03277263790369034, -0.00024092942476272583, 
-0.0797809511423111, -0.05919867008924484, 0.030603203922510147, -0.020362211391329765, -0.03230069577693939, 0.008135363459587097, -0.0051808785647153854, 0.025486908853054047, 0.03644777834415436, -0.014574101194739342, 
-0.03246414288878441, -0.01642022281885147, -0.0007648570463061333, -0.05113661661744118, -0.03257439285516739, 0.024147117510437965, 0.004396106116473675, -0.027757303789258003, -0.011082659475505352, 0.03781861439347267, 
0.008576426655054092, -0.045926280319690704, -0.066804438829422, -0.028471969068050385, -0.08488908410072327, -0.10289400815963745, -0.06431947648525238, -0.03707035630941391, -0.05607350915670395, -0.1013576090335846, 
-0.11101721227169037, -0.04262098670005798, 0.024457715451717377, -0.009015985764563084, -0.03002968803048134, -0.024904735386371613, 0.03402145579457283, 0.04226546734571457, 0.0018854886293411255, 0.004248492419719696, 
0.030570250004529953, 0.0606529675424099, 0.07200513035058975, 0.028733642771840096, 0.036771196871995926, 0.054391805082559586, -0.016558531671762466, -0.04721662402153015, 0.03561404347419739, 0.06880588084459305, 
0.0121947405859828, -0.018398627638816833, 0.006687546148896217, 0.03261784464120865, -0.021676864475011826, -0.05173289775848389, 0.024003468453884125, 0.05442424491047859, 0.004002587869763374, -0.028855670243501663, 
0.032811619341373444, -0.005540057085454464, -0.11297719180583954, -0.09523679316043854, -0.08057162910699844, -0.034945420920848846, -0.011704884469509125, -0.021467240527272224, -0.010570349171757698, 0.03147473186254501, 
0.04654573276638985, -0.024376485496759415, -0.0331617072224617, 0.03479901701211929, 0.05917651206254959, 0.03194567188620567, 0.048965439200401306, 0.05967869609594345, 0.07221411168575287, 0.08277406543493271, 
0.03484102711081505, 0.024225573986768723, 0.09117956459522247, 0.13790324330329895, 0.07751541584730148, 0.019685247913002968, 0.03981263190507889, 0.08254338800907135, 0.03466403856873512, 0.012927191331982613, 
-0.0009466735646128654, -0.0030493754893541336, -0.007446121424436569, -0.005965536460280418, 0.03947763890028, -0.007588987238705158, -0.025495342910289764, -0.04184015095233917, -0.03223346546292305, -0.039558324962854385, 
-0.05204673111438751, -0.04107692837715149, -0.04808266460895538, -0.013437444344162941, -0.01940755359828472, -0.04410972818732262, -0.02861505001783371, -0.029683630913496017, -0.0711163654923439, -0.07653529942035675, 
-0.027195870876312256, -0.020634744316339493, -0.043417368084192276, -0.06806960701942444, -0.1002821996808052, -0.11959437280893326, -0.06920170783996582, -0.03904789686203003, -0.09070710837841034, -0.08675646781921387, 
-0.04944459721446037, -0.0010661371052265167, 0.012210703454911709, 0.0007928963750600815, -0.018205245956778526, -0.07060878723859787, -0.061554815620183945, -0.0033606328070163727, 0.01751876249909401, -0.013414038345217705, 
0.007252845913171768, 0.013194284401834011, -0.04489671066403389, -0.04322705417871475, 0.018539059907197952, 0.04458900913596153, 0.02121090143918991, 0.01748543418943882, 0.01233878917992115, 0.02738083153963089, 
0.042481038719415665, 0.02321251854300499, -0.007188165560364723, 0.012945863418281078, 0.035496991127729416, 0.05723883956670761, 0.06263941526412964, 0.035158198326826096, -0.017884040251374245, -0.030913423746824265, 
0.012690378352999687, 0.016994554549455643, -0.003020630218088627, 0.0060837287455797195, 0.05588700994849205, 0.046217381954193115, 0.031740620732307434, 0.0379144549369812, 0.06092428043484688, 0.02963445708155632, 
0.017639195546507835, 0.01281425729393959, 0.005924813449382782, 0.024596363306045532, 0.008103519678115845, -0.02282879501581192, -0.03274257481098175, 0.03353039175271988, 0.05064170062541962, 0.023485690355300903, 
-0.01065199077129364, 0.00905713438987732, 0.03435199707746506, 0.01741548627614975, 0.0037292763590812683, 0.0180901437997818, 0.0364777036011219, 0.030732207000255585, 0.03462931886315346, 0.0300856102257967, 
0.04320274665951729, 0.03972325846552849, -0.00867609865963459, -0.018197277560830116, 0.01579398289322853, 0.007343549281358719, 0.0022260136902332306, 0.036885157227516174, 0.062073566019535065, 0.04500648379325867, 
0.04018724337220192, 0.05977015569806099, 0.047041844576597214, -0.004214371554553509, -0.043393999338150024, -0.02571905218064785, 0.029363879933953285, 0.02968328446149826, -0.01801598258316517, -0.034414175897836685, 
-0.029738139361143112, 0.009868241846561432, 0.002301417291164398, -0.03231658786535263, -0.044113628566265106, -0.021397678181529045, -0.028622489422559738, -0.0567975677549839, -0.04496780410408974, -0.02300272136926651, 
-0.024941209703683853, -0.04540129005908966, -0.04163961857557297, -0.03798762708902359, -0.0040028952062129974, -0.015392143279314041, -0.023141566663980484, -0.03515283390879631, -0.03907735273241997, -0.04027840495109558, 
-0.015378657728433609, 0.005750058218836784, -0.025281205773353577, -0.02295871265232563, 0.004937293007969856, 0.028642356395721436, -0.005646742880344391, -0.015836812555789948, -0.03426022827625275, -0.013255979865789413, 
-0.004023503512144089, -0.0033715730533003807, -0.010053074918687344, -0.010338373482227325, 0.00466524250805378, -0.013086223043501377, 0.0024392278864979744, -0.0022191228345036507, 0.015235235914587975, 0.010807575657963753, 
-0.006217820569872856, -0.016500448808073997, 0.01735323667526245, 0.05765122175216675, 0.02955194190144539, 0.012185764499008656, 0.020604107528924942, 0.045609716325998306, 0.04891104996204376, 0.028668370097875595, 
0.008264103904366493, -0.001128511969000101, 0.012986520305275917, 0.008519884198904037, 0.014329511672258377, 0.0060014259070158005, 0.0020188235212117434, 0.026475880295038223, 0.01839655637741089, 0.003974240273237228, 
0.006917309481650591, 0.02105852961540222, -0.01544254645705223, -0.017114851623773575, 0.004651198163628578, 0.0129185039550066, 0.0034041418693959713, -0.0015978766605257988, 0.0008078105747699738, -0.004431731998920441,
        };
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };
        Tensor<f32, 2> sig_prep = mfcc_spectrogram_for_learning(sig, 8000.0f);

        Tensor<f32, 1> embed = inference(sig_prep);
        assert(embed.dim<0>() == 16);
        assert(std::abs(embed(0)  -  1.2245) < 0.175);
        assert(std::abs(embed(1)  - -0.0104) < 0.175);
        assert(std::abs(embed(2)  -  2.6651) < 0.175);
        assert(std::abs(embed(3)  -  1.1899) < 0.175);
        assert(std::abs(embed(4)  - -0.0745) < 0.175);
        assert(std::abs(embed(5)  - -1.2355) < 0.175);
        assert(std::abs(embed(6)  -  1.3877) < 0.175);
        assert(std::abs(embed(7)  - -1.1311) < 0.175);
        assert(std::abs(embed(8)  -  1.0021) < 0.175);
        assert(std::abs(embed(9)  - -0.7951) < 0.175);
        assert(std::abs(embed(10) -  0.2675) < 0.175);
        assert(std::abs(embed(11) -  0.9879) < 0.175);
        assert(std::abs(embed(12) - -0.0664) < 0.175);
        assert(std::abs(embed(13) - -0.2827) < 0.175);
        assert(std::abs(embed(14) -  0.2799) < 0.175);
        assert(std::abs(embed(15) -  0.3221) < 0.175);
    } CATCH({
        std::cout << "!!!! inference error: " << x.what() << '\n';
        throw;
    })

    std::cout << "passed all tests! (no output means good)\n";
}
