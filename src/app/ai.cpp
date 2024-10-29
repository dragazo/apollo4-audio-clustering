#include "../ai/util.h"

extern "C" {
    void preprocess(float *input, unsigned input_len, unsigned sample_rate, float **output, unsigned *output_rows, unsigned *output_cols, void (**output_deleter)(float*)) {
        Tensor<f32, 1> input_tensor { input, nullptr, input_len };
        Tensor<f32, 2> res = mfcc_spectrogram_for_learning(input_tensor, (f32)sample_rate);
        *output_rows = res.dim<0>();
        *output_cols = res.dim<1>();
        std::move(res).leak(*output, *output_deleter);
    }
}
