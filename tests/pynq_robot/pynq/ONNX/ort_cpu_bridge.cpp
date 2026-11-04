#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include <cctype>
#include <vector>
#include "pynq_custom_ops.h"

static thread_local std::string last_error;

static std::string EnvString(const char* name, const char* fallback) {
    const char* value = std::getenv(name);
    return value && *value ? std::string(value) : std::string(fallback);
}

static int EnvInt(const char* name, int fallback, int minimum, int maximum) {
    const std::string text = EnvString(name, "");
    if (text.empty()) return fallback;
    size_t used = 0;
    const int value = std::stoi(text, &used);
    if (used != text.size() || value < minimum || value > maximum)
        throw std::runtime_error(std::string("Invalid ") + name + ": " + text);
    return value;
}

static bool EnvBool(const char* name, bool fallback) {
    std::string value = EnvString(name, "");
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value.empty()) return fallback;
    if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
    if (value == "0" || value == "false" || value == "no" || value == "off") return false;
    throw std::runtime_error(std::string("Invalid boolean ") + name + ": " + value);
}

static ExecutionMode EnvExecutionMode() {
    std::string value = EnvString("SMOL_ORT_EXECUTION_MODE", "sequential");
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "sequential") return ExecutionMode::ORT_SEQUENTIAL;
    if (value == "parallel") return ExecutionMode::ORT_PARALLEL;
    throw std::runtime_error("SMOL_ORT_EXECUTION_MODE must be sequential or parallel");
}

static GraphOptimizationLevel EnvOptimizationLevel() {
    std::string value = EnvString("SMOL_ORT_GRAPH_OPT", "disable");
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "disable" || value == "disabled") return GraphOptimizationLevel::ORT_DISABLE_ALL;
    if (value == "basic") return GraphOptimizationLevel::ORT_ENABLE_BASIC;
    if (value == "extended") return GraphOptimizationLevel::ORT_ENABLE_EXTENDED;
    if (value == "all") return GraphOptimizationLevel::ORT_ENABLE_ALL;
    throw std::runtime_error("SMOL_ORT_GRAPH_OPT must be disable, basic, extended or all");
}

struct Model {
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "smolvla_two_camera_pynq_eval"};
    std::unique_ptr<PynqOp> gemm;
    std::unique_ptr<PynqOp> softmax;
    std::unique_ptr<Ort::CustomOpDomain> domain;
    std::unique_ptr<Ort::Session> session;

    Model(const char* path, int threads, PynqCallback callback = nullptr) {
        if (!path || threads < 1) throw std::runtime_error("A valid model path and positive thread count are required");
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(EnvInt("SMOL_ORT_INTRA_OP_THREADS", threads, 1, 64));
        options.SetInterOpNumThreads(EnvInt("SMOL_ORT_INTER_OP_THREADS", 1, 1, 64));
        options.SetExecutionMode(EnvExecutionMode());
        options.SetGraphOptimizationLevel(EnvOptimizationLevel());
        const bool spin = EnvBool("SMOL_ORT_ALLOW_SPINNING", true);
        options.AddConfigEntry("session.intra_op.allow_spinning", spin ? "1" : "0");
        options.AddConfigEntry("session.inter_op.allow_spinning", spin ? "1" : "0");
        if (EnvBool("SMOL_ORT_CPU_ARENA", false)) options.EnableCpuMemArena();
        else options.DisableCpuMemArena();
        if (EnvBool("SMOL_ORT_MEMORY_PATTERN", false)) options.EnableMemPattern();
        else options.DisableMemPattern();
        if (callback) {
            gemm.reset(new PynqOp(0, callback));
            softmax.reset(new PynqOp(1, callback));
            domain.reset(new Ort::CustomOpDomain("pynq"));
            domain->Add(gemm.get());
            domain->Add(softmax.get());
            options.Add(*domain);
        }
        session.reset(new Ort::Session(env, path, options));
        ValidateSchema();
    }

    void ValidateSchema() {
        if (session->GetInputCount() != 6 || session->GetOutputCount() != 1)
            throw std::runtime_error("Expected six inputs and one output for the two-camera model");
        Ort::AllocatorWithDefaultOptions allocator;
        std::set<std::string> remaining{"camera2", "camera3", "state", "language_tokens",
                                        "language_mask", "noise"};
        for (size_t i = 0; i < session->GetInputCount(); ++i) {
            auto name = session->GetInputNameAllocated(i, allocator);
            if (!name || !remaining.erase(name.get()))
                throw std::runtime_error("Unexpected or repeated model input name");
        }
        auto output_name = session->GetOutputNameAllocated(0, allocator);
        if (!output_name || std::string(output_name.get()) != "actions")
            throw std::runtime_error("Expected output named actions");
    }
};

extern "C" {
const char* smol_error() { return last_error.c_str(); }
const char* smol_runtime_version() { return OrtGetApiBase()->GetVersionString(); }

void* smol_create(const char* path, int threads) {
    try { last_error.clear(); return new Model(path, threads); }
    catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}

void* smol_create_hybrid(const char* path, int threads, PynqCallback callback) {
    try {
        last_error.clear();
        if (!callback) throw std::runtime_error("A hybrid callback is required");
        return new Model(path, threads, callback);
    } catch (const std::exception& error) { last_error = error.what(); return nullptr; }
}

// Six graph inputs: camera2, camera3, state, language_tokens, language_mask, noise.
int smol_run(void* handle, float* camera2, float* camera3, float* state,
             int64_t* tokens, bool* mask, float* noise, float* output) {
    try {
        last_error.clear();
        if (!handle || !camera2 || !camera3 || !state || !tokens || !mask || !noise || !output)
            throw std::runtime_error("Null model handle or tensor pointer");
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        const std::array<int64_t, 4> image_shape{1, 3, 480, 640};
        const std::array<int64_t, 2> state_shape{1, 6}, text_shape{1, 48};
        const std::array<int64_t, 3> noise_shape{1, 50, 32};
        std::array<Ort::Value, 6> inputs{
            Ort::Value::CreateTensor<float>(memory, camera2, 3 * 480 * 640, image_shape.data(), 4),
            Ort::Value::CreateTensor<float>(memory, camera3, 3 * 480 * 640, image_shape.data(), 4),
            Ort::Value::CreateTensor<float>(memory, state, 6, state_shape.data(), 2),
            Ort::Value::CreateTensor<int64_t>(memory, tokens, 48, text_shape.data(), 2),
            Ort::Value::CreateTensor<bool>(memory, mask, 48, text_shape.data(), 2),
            Ort::Value::CreateTensor<float>(memory, noise, 50 * 32, noise_shape.data(), 3)};
        const char* names[]{"camera2", "camera3", "state", "language_tokens", "language_mask", "noise"};
        const char* outputs[]{"actions"};
        auto result = static_cast<Model*>(handle)->session->Run(
            Ort::RunOptions{nullptr}, names, inputs.data(), inputs.size(), outputs, 1);
        auto info = result[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
            info.GetShape() != std::vector<int64_t>({1, 50, 6}))
            throw std::runtime_error("Unexpected actions output shape or dtype");
        std::copy(result[0].GetTensorData<float>(), result[0].GetTensorData<float>() + 300, output);
        return 0;
    } catch (const std::exception& error) { last_error = error.what(); return -1; }
}

void smol_destroy(void* handle) { delete static_cast<Model*>(handle); }
}
