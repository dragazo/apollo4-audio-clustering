#include "../ai/util.h"
#include "../ai/tf.h"

extern "C" {
    void preprocess_and_encode(float *input, unsigned input_len, float sample_rate, float *output) {
        Tensor<f32, 1> input_tensor { input, nullptr, input_len };
        Tensor<f32, 2> prepped = mfcc_spectrogram_for_learning(input_tensor, (f32)sample_rate);
        Tensor<f32, 1> res = inference(prepped);
        for (u32 i = 0; i < res.dim<0>(); ++i) output[i] = res(i);
    }
}
