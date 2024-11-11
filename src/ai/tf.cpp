#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "./model.h"
#include "./tensor.h"

constexpr u32 tensor_arena_size = 512 * 1024;
u8 tensor_arena[tensor_arena_size];

Tensor<f32, 1> inference(const Tensor<f32, 2> &input) {
    static const tflite::Model *model = tflite::GetModel(model_tflite);

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        throw std::runtime_error("wrong model schema version");
    }

    tflite::MicroMutableOpResolver<6> resolver;
    if (resolver.AddQuantize() != kTfLiteOk) throw std::runtime_error("failed to add op to resolver");
    tflite::MicroInterpreter interpreter { model, resolver, tensor_arena, tensor_arena_size };

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        throw std::runtime_error("failed to allocate tensors");
    }

    Tensor<f32, 1> res { nullptr, nullptr, 0 };
    return res;
}
