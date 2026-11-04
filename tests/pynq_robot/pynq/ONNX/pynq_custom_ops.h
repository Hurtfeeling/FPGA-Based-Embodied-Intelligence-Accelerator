#pragma once
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

// Matches the existing GEMM_V2 Python callback ABI.
using PynqCallback = int (*)(int, int, int, const void*, const int64_t*, size_t,
                            const void*, const int64_t*, size_t,
                            void*, const int64_t*, size_t);

struct PynqKernel {
    PynqCallback callback;
    int kind;
    int constant_b;
    int cache_id;

    void Compute(OrtKernelContext* context) {
        Ort::KernelContext ctx(context);
        auto a = ctx.GetInput(0);
        auto sa = a.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int64_t> sb, output_shape;
        const void* bp = nullptr;
        if (kind == 0) {
            if (cache_id >= 0 && constant_b != 2)
                throw std::runtime_error("resident CMA cache requires pretransposed static B");
            auto b = ctx.GetInput(1);
            sb = b.GetTensorTypeAndShapeInfo().GetShape();
            bp = b.GetTensorData<int8_t>();
            if (sa.size() < 2 || sb.size() < 2 ||
                sa.back() != sb[sb.size() - (constant_b == 2 ? 1 : 2)])
                throw std::runtime_error("Unsupported GEMM rank or K mismatch");
            const size_t rank = std::max(sa.size(), sb.size());
            output_shape.assign(rank, 1);
            for (size_t i = 0; i < rank - 2; ++i) {
                const int64_t av = i + sa.size() >= rank ? sa[i + sa.size() - rank] : 1;
                const int64_t bv = i + sb.size() >= rank ? sb[i + sb.size() - rank] : 1;
                if (av != bv && av != 1 && bv != 1)
                    throw std::runtime_error("GEMM batch broadcast mismatch");
                output_shape[i] = std::max(av, bv);
            }
            output_shape[rank - 2] = sa[sa.size() - 2];
            output_shape[rank - 1] = constant_b == 2 ? sb[sb.size() - 2] : sb.back();
        } else {
            output_shape = sa;
        }
        auto output = ctx.GetOutput(0, output_shape);
        const void* ap = kind == 0 ? static_cast<const void*>(a.GetTensorData<uint8_t>())
                                  : static_cast<const void*>(a.GetTensorData<float>());
        void* op = kind == 0 ? static_cast<void*>(output.GetTensorMutableData<int32_t>())
                            : static_cast<void*>(output.GetTensorMutableData<float>());
        if (callback(kind, constant_b, cache_id, ap, sa.data(), sa.size(), bp, sb.data(), sb.size(),
                     op, output_shape.data(), output_shape.size()))
            throw std::runtime_error("PYNQ callback failed; see Python callback traceback");
    }

    OrtStatusPtr ComputeV2(OrtKernelContext* context) noexcept {
        try { Compute(context); return nullptr; }
        catch (const std::exception& error) {
            return Ort::GetApi().CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
        }
    }
};

struct PynqOp : Ort::CustomOpBase<PynqOp, PynqKernel, true> {
    int kind;
    PynqCallback callback;

    PynqOp(int kind_, PynqCallback callback_) : kind(kind_), callback(callback_) {
        // ORT 1.16 leaves these legacy callback fields unset for V2 custom ops.
        OrtCustomOp::CreateKernel = nullptr;
        OrtCustomOp::KernelCompute = nullptr;
    }

    int HasConstantB(const OrtKernelInfo* info) const {
        if (kind != 0 || !info) return 0;
        const OrtApi& api = Ort::GetApi();
        // A converter may set this attribute when B is known constant even if
        // ORT does not expose it through KernelInfoGetConstantInput_tensor.
        int64_t explicit_value = 0;
        OrtStatusPtr status = api.KernelInfoGetAttribute_int64(info, "constant_b", &explicit_value);
        if (!status) {
            if (explicit_value < 0 || explicit_value > 2)
                throw std::runtime_error("constant_b must be 0, 1 or 2");
            return static_cast<int>(explicit_value);
        }
        api.ReleaseStatus(status);
        int is_constant = 0;
        const OrtValue* value = nullptr;
        status = api.KernelInfoGetConstantInput_tensor(info, 1, &is_constant, &value);
        if (status) { api.ReleaseStatus(status); return 0; }
        return is_constant && value ? 1 : 0;
    }

    int ResidentCacheId(const OrtKernelInfo* info) const {
        if (kind != 0 || !info) return -1;
        const OrtApi& api = Ort::GetApi();
        int64_t value = -1;
        OrtStatusPtr status = api.KernelInfoGetAttribute_int64(info, "resident_cache_id", &value);
        if (!status) {
            if (value < -1 || value > 1000000)
                throw std::runtime_error("resident_cache_id outside supported range");
            return static_cast<int>(value);
        }
        api.ReleaseStatus(status);
        return -1;
    }

    const char* GetName() const { return kind == 0 ? "MatMulInteger" : "Softmax"; }
    size_t GetInputTypeCount() const { return kind == 0 ? 2 : 1; }
    size_t GetOutputTypeCount() const { return 1; }
    ONNXTensorElementDataType GetInputType(size_t index) const {
        return kind == 0 ? (index == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8
                                      : ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8)
                         : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    ONNXTensorElementDataType GetOutputType(size_t) const {
        return kind == 0 ? ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 : ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    }
    void* CreateKernel(const OrtApi&, const OrtKernelInfo* info) const {
        return new PynqKernel{callback, kind, HasConstantB(info), ResidentCacheId(info)};
    }
    OrtStatusPtr CreateKernelV2(const OrtApi& api, const OrtKernelInfo* info, void** output) const noexcept {
        try { *output = new PynqKernel{callback, kind, HasConstantB(info), ResidentCacheId(info)}; return nullptr; }
        catch (const std::exception& error) {
            return api.CreateStatus(ORT_RUNTIME_EXCEPTION, error.what());
        }
    }
};
