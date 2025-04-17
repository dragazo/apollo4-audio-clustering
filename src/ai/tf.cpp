#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "./model.h"
#include "./tensor.h"

constexpr u32 tensor_arena_size = 284 * 1024;
u8 tensor_arena[tensor_arena_size];

Tensor<f32, 1> inference(const Tensor<f32, 2> &x) {
    static struct Cache {
        const tflite::Model *model;
        tflite::MicroMutableOpResolver<7> resolver;
        char interpreter[sizeof(tflite::MicroInterpreter)];
        Cache() {
            model = tflite::GetModel(model_tflite);
            if (model->version() != TFLITE_SCHEMA_VERSION) THROW(std::runtime_error("wrong model schema version"));

            if (resolver.AddQuantize() != kTfLiteOk) THROW(std::runtime_error("failed to add quantize op to resolver"));
            if (resolver.AddReshape() != kTfLiteOk) THROW(std::runtime_error("failed to add reshape op to resolver"));
            if (resolver.AddConv2D() != kTfLiteOk) THROW(std::runtime_error("failed to add conv2d op to resolver"));
            if (resolver.AddTranspose() != kTfLiteOk) THROW(std::runtime_error("failed to add transpose op to resolver"));
            if (resolver.AddLeakyRelu() != kTfLiteOk) THROW(std::runtime_error("failed to add leaky relu op to resolver"));
            if (resolver.AddFullyConnected() != kTfLiteOk) THROW(std::runtime_error("failed to add fully connected op to resolver"));
            if (resolver.AddDequantize() != kTfLiteOk) THROW(std::runtime_error("failed to add dequantize op to resolver"));

            new (interpreter) tflite::MicroInterpreter { model, resolver, tensor_arena, tensor_arena_size };
            if (reinterpret_cast<tflite::MicroInterpreter*>(interpreter)->AllocateTensors() != kTfLiteOk) THROW(std::runtime_error("failed to allocate tensors"));
        }
    } cache;

    tflite::MicroInterpreter &interpreter = *reinterpret_cast<tflite::MicroInterpreter*>(cache.interpreter);

    TfLiteTensor *input = interpreter.input(0);
    TfLiteTensor *output = interpreter.output(0);

    if (input->type != kTfLiteFloat32) THROW(std::runtime_error("model input is not f32"));
    if (input->dims->size != 3 || input->dims->data[0] != 1 || input->dims->data[1] != x.dim<0>() || input->dims->data[2] != x.dim<1>()) THROW(std::runtime_error("input wrong shape"));
    for (u32 i = x.size(); i-- > 0; ) input->data.f[i] = (&x(0,0))[i];

    if (interpreter.Invoke() != kTfLiteOk) THROW(std::runtime_error("failed to execute model"));

    if (output->type != kTfLiteFloat32) THROW(std::runtime_error("model output is not f32"));
    if (output->dims->size != 2 || output->dims->data[0] != 1) THROW(std::runtime_error("output wrong shape"));
    Tensor<f32, 1> res { new f32[output->dims->data[1]], [](auto *p) { delete[] p; }, output->dims->data[1] };
    for (u32 i = 0; i < res.dim<0>(); ++i) res(i) = output->data.f[i];
    return res;
}
