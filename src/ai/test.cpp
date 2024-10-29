#include <iostream>
#include <cassert>
#include <cmath>

#include "./tensor.h"
#include "./util.h"

template<typename T>
void deleter(T *v) { delete[] v; }

int main() {
    std::cout << "starting tests...\n";

    try { // tensor
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
    } catch (const std::exception &x) {
        std::cout << "!!!! tensor error: " << x.what() << '\n';
        throw;
    }

    try { // complex
        c32 a = c32 {5, 7} * c32 {-4, 1};
        assert(a.real == -27 && a.imag == -23);
        c32 b = c32 {6, 2} + c32 { 4, -8 };
        assert(b.real == 10 && b.imag == -6);
        c32 c = c32 {6, 2} * 2.0f;
        assert(c.real == 12 && c.imag == 4);
        c32 d = 3.0f * c32 {4, -2};
        assert(d.real == 12 && d.imag == -6);
    } catch (const std::exception &x) {
        std::cout << "!!!! complex error: " << x.what() << '\n';
        throw;
    }

    try { // fft
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
    } catch (const std::exception &x) {
        std::cout << "!!!! fft error: " << x.what() << '\n';
        throw;
    }

    try { // rfft
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
    } catch (const std::exception &x) {
        std::cout << "!!!! rfft error: " << x.what() << '\n';
        throw;
    }

    try { // low_pass_filter
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
    } catch (const std::exception &x) {
        std::cout << "!!!! low_pass_filter error: " << x.what() << '\n';
        throw;
    }

    try { // normalize_audio
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
    } catch (const std::exception &x) {
        std::cout << "!!!! normalize_audio error: " << x.what() << '\n';
        throw;
    }

    try { // overlapping_chunks
        f32 sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1};
        Tensor<f32, 1> sig { sig_raw, nullptr, sizeof(sig_raw) / sizeof(*sig_raw) };

        auto chunks = overlapping_chunks(sig, 4);

        assert(chunks.size() == 4);
        assert(chunks[0].dim<0>() == 4);
        assert(std::abs(chunks[0](0) - 1) < 0.01);
        assert(std::abs(chunks[0](1) - 2) < 0.01);
        assert(std::abs(chunks[0](2) - 3) < 0.01);
        assert(std::abs(chunks[0](3) - 4) < 0.01);
        assert(chunks[1].dim<0>() == 4);
        assert(std::abs(chunks[1](0) - 3) < 0.01);
        assert(std::abs(chunks[1](1) - 4) < 0.01);
        assert(std::abs(chunks[1](2) - 5) < 0.01);
        assert(std::abs(chunks[1](3) - 6) < 0.01);
        assert(chunks[2].dim<0>() == 4);
        assert(std::abs(chunks[2](0) - 5) < 0.01);
        assert(std::abs(chunks[2](1) - 6) < 0.01);
        assert(std::abs(chunks[2](2) - 2) < 0.01);
        assert(std::abs(chunks[2](3) - 3) < 0.01);
        assert(chunks[3].dim<0>() == 4);
        assert(std::abs(chunks[3](0) - 2) < 0.01);
        assert(std::abs(chunks[3](1) - 3) < 0.01);
        assert(std::abs(chunks[3](2) - 8) < 0.01);
        assert(std::abs(chunks[3](3) - 1) < 0.01);

    } catch (const std::exception &x) {
        std::cout << "!!!! overlapping_chunks error: " << x.what() << '\n';
        throw;
    }

    try { // mul_hann_window
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
    } catch (const std::exception &x) {
        std::cout << "!!!! mul_hann_window error: " << x.what() << '\n';
        throw;
    }

    try { // spectrogram
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
    } catch (const std::exception &x) {
        std::cout << "!!!! spectrogram error: " << x.what() << '\n';
        throw;
    }

    try { // freq_to_mel / mel_to_freq
        assert(std::abs(freq_to_mel(921.0f) - 946.3624) < 0.01);
        assert(std::abs(freq_to_mel(391.0f) - 500.1284) < 0.01);

        assert(std::abs(mel_to_freq(129.0f) - 84.8899) < 0.01);
        assert(std::abs(mel_to_freq(1236.0f) - 1396.0235) < 0.01);
    } catch (const std::exception &x) {
        std::cout << "!!!! freq_to_mel / mel_to_freq error: " << x.what() << '\n';
        throw;
    }

    try { // dct
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
    } catch (const std::exception &x) {
        std::cout << "!!!! dct error: " << x.what() << '\n';
        throw;
    }

    try { // linspace
        Tensor<f32, 1> p = linspace(23.0f, 175.0f, 7);
        assert(p.dim<0>() == 7);
        assert(std::abs(p(0) -  23.0000) < 0.01);
        assert(std::abs(p(1) -  48.3333) < 0.01);
        assert(std::abs(p(2) -  73.6666) < 0.01);
        assert(std::abs(p(3) -  99.0000) < 0.01);
        assert(std::abs(p(4) - 124.3333) < 0.01);
        assert(std::abs(p(5) - 149.6666) < 0.01);
        assert(std::abs(p(6) - 175.0000) < 0.01);
    } catch (const std::exception &x) {
        std::cout << "!!!! linspace error: " << x.what() << '\n';
        throw;
    }

    try { // transpose
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
    } catch (const std::exception &x) {
        std::cout << "!!!! transpose error: " << x.what() << '\n';
        throw;
    }

    try { // matmul
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
    } catch (const std::exception &x) {
        std::cout << "!!!! matmul error: " << x.what() << '\n';
        throw;
    }

    try { // mfcc spectrogram
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
    } catch (const std::exception &x) {
        std::cout << "!!!! mfcc spectrogram error: " << x.what() << '\n';
        throw;
    }

    try { // mfcc spectrogram for learning
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
    } catch (const std::exception &x) {
        std::cout << "!!!! mfcc spectrogram for learning error: " << x.what() << '\n';
        throw;
    }

    std::cout << "passed all tests! (no output means good)\n";
}
