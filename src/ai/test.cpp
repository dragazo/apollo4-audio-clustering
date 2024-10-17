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
        throw x;
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
        throw x;
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
        throw x;
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
        throw x;
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
    }

    std::cout << "passed all tests! (no output means good)\n";
}
