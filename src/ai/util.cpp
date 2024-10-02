#include <cmath>

#include "./util.h"

#define PI 3.14159265358979323846f

// TODO: make this actually an rfft
// currently is just rft cause the correction for non-power-of-2 fft is complicated
Tensor<c32, 1> rfft(const Tensor<f32, 1> &a) {
    u32 res_size = a.dim<0>() / 2 + 1;
    Tensor<c32, 1> res { new c32[res_size], [](auto *v) { delete[] v; }, res_size };
    for (u32 k = 0; k < res_size; k++) {
        f32 real_sum = 0.0;
        f32 imag_sum = 0.0;
        for (u32 n = 0; n < a.dim<0>(); n++) {
            f32 angle = 2 * PI * k * n / a.dim<0>();
            real_sum += a(n) * cosf(angle);
            imag_sum -= a(n) * sinf(angle);  // FFT uses negative sine
        }
        res(k).real = real_sum;
        res(k).imag = imag_sum;
    }
    return res;
}
