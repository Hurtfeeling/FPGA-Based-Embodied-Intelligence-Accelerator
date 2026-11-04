#include <onnxruntime_cxx_api.h>
#include <array>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string softmax_error;

struct CpuSoftmax {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "smolvla_cpu_softmax"};
    Ort::MemoryInfo memory{Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault)};
    std::unique_ptr<Ort::Session> session;

    CpuSoftmax(const char* path, int threads) {
        if (!path || !*path || threads < 1 || threads > 4)
            throw std::runtime_error("Softmax requires a model path and 1..4 CPU threads");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads);
        options.SetInterOpNumThreads(1);
        options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
        options.DisableCpuMemArena();
        options.DisableMemPattern();
        // Yield idle workers while the PL and packing code use the same CPUs.
        options.AddConfigEntry("session.intra_op.allow_spinning", "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", "0");
        session.reset(new Ort::Session(env, path, options));
        if (session->GetInputCount() != 1 || session->GetOutputCount() != 1)
            throw std::runtime_error("Expected a single-input, single-output Softmax model");
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session->GetInputNameAllocated(0, allocator);
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        if (!input_name || !output_name || std::string(input_name.get()) != "X" ||
            std::string(output_name.get()) != "Y")
            throw std::runtime_error("Expected Softmax tensor names X and Y");
        auto input_info = session->GetInputTypeInfo(0);
        auto output_info = session->GetOutputTypeInfo(0);
        auto input_type = input_info.GetTensorTypeAndShapeInfo();
        auto output_type = output_info.GetTensorTypeAndShapeInfo();
        if (input_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            output_type.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            input_type.GetDimensionsCount() != 2 || output_type.GetDimensionsCount() != 2)
            throw std::runtime_error("Softmax model must use rank-two FP32 input and output");
    }
};
}

extern "C" {
const char* smol_softmax_error() { return softmax_error.c_str(); }

void* smol_softmax_create(const char* path, int threads) {
    try {
        softmax_error.clear();
        return new CpuSoftmax(path, threads);
    } catch (const std::exception& error) {
        softmax_error = error.what();
        return nullptr;
    }
}

int smol_softmax_run(void* handle, const float* src, float* dst, int64_t rows, int64_t k) {
    try {
        softmax_error.clear();
        if (!handle || !src || !dst || rows <= 0 || k <= 0)
            throw std::runtime_error("Softmax requires live tensors and positive dimensions");
        const size_t max_elements = std::numeric_limits<size_t>::max() / sizeof(float);
        if (static_cast<uint64_t>(rows) > max_elements / static_cast<uint64_t>(k))
            throw std::runtime_error("Softmax tensor byte count overflows size_t");
        const size_t elements = static_cast<size_t>(rows) * static_cast<size_t>(k);
        const size_t bytes = elements * sizeof(float);
        const uintptr_t a = reinterpret_cast<uintptr_t>(src);
        const uintptr_t b = reinterpret_cast<uintptr_t>(dst);
        if ((a <= b && b - a < bytes) || (b <= a && a - b < bytes))
            throw std::runtime_error("Softmax input and output must not overlap");
        auto& model = *static_cast<CpuSoftmax*>(handle);
        const std::array<int64_t, 2> shape{{rows, k}};
        // Both OrtValues wrap caller storage. ORT writes Y directly into dst;
        // there is no intermediate allocation or host copy of the result.
        auto input = Ort::Value::CreateTensor<float>(
            model.memory, const_cast<float*>(src), elements, shape.data(), shape.size());
        auto output = Ort::Value::CreateTensor<float>(
            model.memory, dst, elements, shape.data(), shape.size());
        const char* inputs[]{"X"};
        const char* outputs[]{"Y"};
        model.session->Run(Ort::RunOptions{nullptr}, inputs, &input, 1, outputs, &output, 1);
        if (output.GetTensorMutableData<float>() != dst)
            throw std::runtime_error("ORT replaced the externally supplied Softmax output buffer");
        return 0;
    } catch (const std::exception& error) {
        softmax_error = error.what();
        return -1;
    }
}

void smol_softmax_destroy(void* handle) { delete static_cast<CpuSoftmax*>(handle); }
}
