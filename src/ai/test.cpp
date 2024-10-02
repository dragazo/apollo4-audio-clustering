#include <iostream>
#include <cassert>
#include <cmath>

#include "./tensor.h"
#include "./util.h"

template<typename T>
void deleter(T *v) { delete[] v; }

int main() {
    std::cout << "starting tests...\n";

    { // tensor
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
    }

    { // rfft
        f32 r_sig_raw[] = {1, 2, 3, 4, 5, 6, 2, 3, 8, 1};
        Tensor<f32, 1> r_sig { r_sig_raw, nullptr, sizeof(r_sig_raw) / sizeof(*r_sig_raw) };
        Tensor<c32, 1> r_sig_rfft = rfft(r_sig);
        assert(r_sig_rfft.dim<0>() == 6);
        for (u32 i = 0; i < r_sig_rfft.dim<0>(); ++i) {
            std::cout << r_sig_rfft(i).real << " + " << r_sig_rfft(i).imag << "i\n";
        }

        assert(std::abs(r_sig_rfft(0).real - 35.0000000000000000) < 0.01 && std::abs(r_sig_rfft(0).imag - 0.0000000000000000) < 0.01);
        assert(std::abs(r_sig_rfft(1).real - -7.0000000000000000) < 0.01 && std::abs(r_sig_rfft(1).imag - 1.4530850560107220) < 0.01);
        assert(std::abs(r_sig_rfft(2).real - -4.4721359549995805) < 0.01 && std::abs(r_sig_rfft(2).imag - 5.4288245463451460) < 0.01);
        assert(std::abs(r_sig_rfft(3).real - -7.0000000000000000) < 0.01 && std::abs(r_sig_rfft(3).imag - -6.155367074350506) < 0.01);
        assert(std::abs(r_sig_rfft(4).real - 4.47213595499958000) < 0.01 && std::abs(r_sig_rfft(4).imag - -4.530768593185974) < 0.01);
        assert(std::abs(r_sig_rfft(5).real - 3.00000000000000000) < 0.01 && std::abs(r_sig_rfft(5).imag - 0.0000000000000000) < 0.01);
    }

    std::cout << "passed all tests!\n";
}
