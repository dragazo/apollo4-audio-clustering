#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "./model.h"
#include "./tensor.h"

constexpr u32 tensor_arena_size = 512 * 1024;
u8 tensor_arena[tensor_arena_size];

Tensor<f32, 1> inference(const Tensor<f32, 2> &input) {
    const tflite::Model *model = tflite::GetModel(model_tflite);
    if (model->version() != TFLITE_SCHEMA_VERSION) throw std::runtime_error("wrong model schema version");

    tflite::MicroMutableOpResolver<7> resolver;
    if (resolver.AddQuantize() != kTfLiteOk) throw std::runtime_error("failed to add quantize op to resolver");
    if (resolver.AddReshape() != kTfLiteOk) throw std::runtime_error("failed to add reshape op to resolver");
    if (resolver.AddConv2D() != kTfLiteOk) throw std::runtime_error("failed to add conv2d op to resolver");
    if (resolver.AddTranspose() != kTfLiteOk) throw std::runtime_error("failed to add transpose op to resolver");
    if (resolver.AddLeakyRelu() != kTfLiteOk) throw std::runtime_error("failed to add leaky relu op to resolver");
    if (resolver.AddFullyConnected() != kTfLiteOk) throw std::runtime_error("failed to add fully connected op to resolver");
    if (resolver.AddDequantize() != kTfLiteOk) throw std::runtime_error("failed to add dequantize op to resolver");

    tflite::MicroInterpreter interpreter { model, resolver, tensor_arena, tensor_arena_size };
    if (interpreter.AllocateTensors() != kTfLiteOk) throw std::runtime_error("failed to allocate tensors");

    Tensor<f32, 1> res { nullptr, nullptr, 0 };
    return res;
}
