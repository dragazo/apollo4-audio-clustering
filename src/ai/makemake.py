import os

build_dir = 'build'

inc = [
    'tflite-micro/',
    'flatbuffers/include/',
    'gemmlowp/',
    'ruy/',
]

src = [
    'tflite-micro/tensorflow/lite/micro/arena_allocator/recording_single_arena_buffer_allocator.cc',
    'tflite-micro/tensorflow/lite/micro/arena_allocator/non_persistent_arena_buffer_allocator.cc',
    'tflite-micro/tensorflow/lite/micro/arena_allocator/persistent_arena_buffer_allocator.cc',
    'tflite-micro/tensorflow/lite/micro/arena_allocator/single_arena_buffer_allocator.cc',
    'tflite-micro/tensorflow/lite/micro/tflite_bridge/flatbuffer_conversions_bridge.cc',
    'tflite-micro/tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc',
    'tflite-micro/tensorflow/lite/micro/memory_planner/linear_memory_planner.cc',
    'tflite-micro/tensorflow/lite/micro/tflite_bridge/micro_error_reporter.cc',
    'tflite-micro/tensorflow/lite/kernels/internal/portable_tensor_utils.cc',
    'tflite-micro/tensorflow/compiler/mlir/lite/core/api/error_reporter.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/fully_connected_common.cc',
    'tflite-micro/tensorflow/lite/kernels/internal/quantization_util.cc',
    'tflite-micro/tensorflow/compiler/mlir/lite/schema/schema_utils.cc',
    'tflite-micro/tensorflow/lite/core/api/flatbuffer_conversions.cc',
    'tflite-micro/tensorflow/lite/micro/micro_interpreter_context.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/leaky_relu_common.cc',
    'tflite-micro/tensorflow/lite/micro/recording_micro_allocator.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/dequantize_common.cc',
    'tflite-micro/tensorflow/lite/kernels/internal/tensor_ctypes.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/fully_connected.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/quantize_common.cc',
    'tflite-micro/tensorflow/lite/micro/micro_interpreter_graph.cc',
    'tflite-micro/tensorflow/lite/micro/micro_resource_variable.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/reshape_common.cc',
    'tflite-micro/tensorflow/lite/micro/micro_allocation_info.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/conv_common.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/kernel_util.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/dequantize.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/leaky_relu.cc',
    'tflite-micro/tensorflow/lite/kernels/internal/common.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/transpose.cc',
    'tflite-micro/tensorflow/lite/micro/micro_interpreter.cc',
    'tflite-micro/tensorflow/lite/micro/micro_op_resolver.cc',
    'tflite-micro/tensorflow/lite/micro/flatbuffer_utils.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/quantize.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/reshape.cc',
    'tflite-micro/tensorflow/lite/micro/micro_allocator.cc',
    'tflite-micro/tensorflow/lite/micro/memory_helpers.cc',
    'tflite-micro/tensorflow/lite/kernels/kernel_util.cc',
    'tflite-micro/tensorflow/lite/micro/micro_context.cc',
    'tflite-micro/tensorflow/lite/micro/kernels/conv.cc',
    'tflite-micro/tensorflow/lite/micro/micro_utils.cc',
    'tflite-micro/tensorflow/lite/micro/debug_log.cc',
    'tflite-micro/tensorflow/lite/micro/micro_log.cc',
    'tflite-micro/tensorflow/lite/core/c/common.cc',
    'tflite-micro/tensorflow/lite/array.cc',
]
src.extend(x for x in os.listdir('.') if x.endswith('.cpp'))
assert sorted(set(src)) == sorted(src), 'found duplicate src entries'
src.sort(key = lambda x: -os.path.getsize(x))

with open('Makefile', 'w') as f:
    cxx = f'g++ {" ".join(f"-I{x}" for x in inc)}'

    all_objs = " ".join(f"{build_dir}/{x[:x.rfind('.')]}.o" for x in src)
    f.write(f'test: {all_objs}\n\t{cxx} {all_objs} -o test\n')

    f.write(f'clean:\n\trm -rf {build_dir}\n')

    for src in src:
        src_dir = src[:src.rfind('/')]
        src_no_ext = src[:src.rfind('.')]

        f.write(f'{build_dir}/{src_no_ext}.o: {src}\n\tmkdir -p {build_dir}/{src_dir} && {cxx} {src} -c -o build/{src_no_ext}.o\n')
