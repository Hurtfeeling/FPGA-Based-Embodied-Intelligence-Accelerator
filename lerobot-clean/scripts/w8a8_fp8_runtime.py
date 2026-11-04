"""Reference executor for W8A8 plus FP8-boundary ONNX graphs.

This module is independent of the strict-INT8 model and its executor. Integer
matrix products use unchanged INT8 operands and exact INT32 accumulation. FP8
operators decode to FP32, compute a reference result, then round to E4M3FN.
It is a precision simulator, not native FP8 hardware execution or an ORT plugin.
"""

import collections
from dataclasses import dataclass

import ml_dtypes
import numpy as np
import onnx
from onnx import helper, numpy_helper

DOMAIN = "org.lerobot.w8a8fp8"
FP8_DTYPE = np.dtype(ml_dtypes.float8_e4m3fn)
FP32_MASK = np.float32(-np.finfo(np.float32).max)
_torch = None


@dataclass
class Float8:
    data: np.ndarray
    exponent: np.ndarray

    def __post_init__(self):
        self.data = np.asarray(self.data)
        self.exponent = np.asarray(self.exponent, np.int32)
        if self.data.dtype != FP8_DTYPE:
            raise TypeError(f"Expected E4M3FN, got {self.data.dtype}")

    def decode(self):
        mask = self.data.view(np.uint8) == 255
        finite = np.where(mask, np.float32(0), self.data.astype(np.float32))
        result = np.ldexp(finite, self.exponent).astype(np.float32)
        return np.where(mask, FP32_MASK, result)


def encode(value, per_channel=False, per_element=False):
    value = np.asarray(value, np.float32)
    mask = np.isneginf(value) | (value < np.float32(-1e30))
    if np.any(np.isnan(value)) or np.any(np.isposinf(value)):
        raise ValueError("FP8 encoding received NaN or positive infinity")
    finite = np.where(mask, np.float32(0), value)
    if per_element:
        maximum = np.abs(finite)
    elif per_channel and finite.ndim > 1:
        maximum = np.max(np.abs(finite), axis=tuple(range(1, finite.ndim)), keepdims=True, initial=0)
    else:
        maximum = np.max(np.abs(finite), initial=0)
    safe = np.maximum(maximum, np.finfo(np.float32).tiny)
    exponent = np.where(maximum > 0, np.ceil(np.log2(safe / np.float32(448))), 0).astype(np.int32)
    scaled = np.clip(np.ldexp(finite, -exponent), -448, 448).astype(FP8_DTYPE)
    # E4M3FN has no infinity. Negative-NaN (0xff) is an explicit attention
    # mask token in this custom domain; ordinary NaN is rejected above.
    data = np.where(mask, np.uint8(255), scaled.view(np.uint8)).view(FP8_DTYPE)
    return Float8(data, exponent)


def enable_integer_backend():
    global _torch
    try:
        import torch
        torch.set_num_threads(8)
        result = torch._int_mm(torch.ones((2, 2), dtype=torch.int8), torch.ones((2, 2), dtype=torch.int8))
        if result.dtype == torch.int32:
            _torch = torch
    except (ImportError, OSError, RuntimeError, NotImplementedError):
        _torch = None


def integer_matmul(left, right, left_zero=0, right_zero=0):
    left = left.astype(np.int32) - np.asarray(left_zero, np.int32)
    right = right.astype(np.int32) - np.asarray(right_zero, np.int32)
    if left.shape[-1] * 128 * 128 > np.iinfo(np.int32).max:
        raise OverflowError("MatMulInteger reduction exceeds INT32 worst-case bound")
    if (_torch is None or np.any(left < -128) or np.any(left > 127)
            or np.any(right < -128) or np.any(right > 127)):
        return left @ right
    leading = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = np.broadcast_to(left, (*leading, *left.shape[-2:]))
    right = np.broadcast_to(right, (*leading, *right.shape[-2:]))
    result = np.empty((*leading, left.shape[-2], right.shape[-1]), np.int32)
    for index in np.ndindex(leading):
        a = _torch.from_numpy(np.array(left[index], np.int8, order="C", copy=True))
        b = _torch.from_numpy(np.array(right[index], np.int8, order="C", copy=True))
        result[index] = _torch._int_mm(a, b).numpy()
    return result


def attributes(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def restored_attributes(attrs):
    result = {k: v for k, v in attrs.items() if not k.endswith("__fp8_bits") and not k.endswith("__fp8_exponent")}
    for key, bits in attrs.items():
        if key.endswith("__fp8_bits"):
            name = key.removesuffix("__fp8_bits")
            data = np.array(bits, np.uint8).view(FP8_DTYPE)
            result[name] = Float8(data, np.array(attrs[name + "__fp8_exponent"], np.int32)).decode().item()
    return result


def resize(value, sizes, attrs):
    if attrs.get("mode") != b"linear" or attrs.get("coordinate_transformation_mode") != b"half_pixel":
        raise NotImplementedError("Only half-pixel linear Resize is supported")
    result = value
    if list(sizes[:2]) != list(value.shape[:2]):
        raise ValueError("Only spatial NCHW resize is supported")
    for axis in (2, 3):
        old, new = result.shape[axis], int(sizes[axis])
        coordinates = np.clip((np.arange(new, dtype=np.float32) + np.float32(0.5)) * old / new - np.float32(0.5), 0, old - 1)
        low = np.floor(coordinates).astype(np.int64)
        high = np.minimum(low + 1, old - 1)
        shape = [1] * result.ndim
        shape[axis] = new
        fraction = (coordinates - low).astype(np.float32).reshape(shape)
        left, right = np.take(result, low, axis=axis), np.take(result, high, axis=axis)
        result = left + (right - left) * fraction
    return result


def operation(op, values, attrs):
    x = values[0] if values else None
    binary = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Div": np.divide,
              "Pow": np.power, "Equal": np.equal, "Greater": np.greater,
              "LessOrEqual": np.less_equal, "And": np.logical_and}
    unary = {"Abs": np.abs, "Round": np.rint, "Sqrt": np.sqrt, "Reciprocal": np.reciprocal,
             "Sin": np.sin, "Cos": np.cos, "Tanh": np.tanh}
    if op in binary:
        if op == "Div" and x.dtype.kind in "iu":
            quotient = np.abs(values[0]) // np.abs(values[1])
            return [np.where((values[0] < 0) != (values[1] < 0), -quotient, quotient).astype(x.dtype)]
        return [binary[op](*values)]
    if op in unary:
        return [unary[op](x)]
    if op == "Identity":
        return [x]
    if op == "Cast":
        return [x.astype(helper.tensor_dtype_to_np_dtype(attrs["to"]))]
    if op in ("FromInteger", "ToInteger"):
        if op == "FromInteger":
            return [x.astype(np.float32)]
        return [x.astype(helper.tensor_dtype_to_np_dtype(attrs["to"]))]
    if op == "MatMulInteger":
        return [integer_matmul(values[0], values[1], values[2] if len(values) > 2 else 0,
                               values[3] if len(values) > 3 else 0)]
    if op == "RescaleMatMulInteger":
        return [values[0].astype(np.float32) * values[1] * values[2]]
    if op == "Shape":
        return [np.asarray(x.shape[attrs.get("start", 0):attrs.get("end")], np.int64)]
    if op == "Gather":
        return [np.take(x, values[1], axis=attrs.get("axis", 0))]
    if op == "Transpose":
        return [np.transpose(x, attrs.get("perm"))]
    if op == "Reshape":
        shape = values[1].copy()
        if not attrs.get("allowzero", 0):
            for i in np.flatnonzero(shape == 0):
                shape[i] = x.shape[i]
        return [x.reshape(shape)]
    if op == "Unsqueeze":
        return [np.expand_dims(x, tuple(values[1].tolist()))]
    if op == "Squeeze":
        return [np.squeeze(x, tuple(values[1].tolist()) if len(values) > 1 else None)]
    if op == "Concat":
        return [np.concatenate(values, axis=attrs["axis"])]
    if op == "Where":
        return [np.where(*values)]
    if op == "Expand":
        return [np.broadcast_to(x, np.broadcast_shapes(x.shape, tuple(values[1])))]
    if op == "Slice":
        slices = [slice(None)] * x.ndim
        axes = values[3] if len(values) > 3 else np.arange(len(values[1]))
        steps = values[4] if len(values) > 4 else np.ones(len(axes), np.int64)
        for axis, start, end, step in zip(axes, values[1], values[2], steps, strict=True):
            slices[int(axis)] = slice(int(start), int(end), int(step))
        return [x[tuple(slices)]]
    if op == "Pad":
        if attrs.get("mode", b"constant") != b"constant":
            raise NotImplementedError("Only constant Pad is supported")
        pads = values[1]
        fill = values[2].item() if len(values) > 2 and values[2] is not None else 0
        return [np.pad(x, list(zip(pads[:x.ndim], pads[x.ndim:], strict=True)), constant_values=fill)]
    if op == "ConstantOfShape":
        value = attrs.get("value")
        array = numpy_helper.to_array(value) if value is not None else np.array([0], np.int64)
        return [np.full(x.tolist(), array.item(), array.dtype)]
    if op == "Range":
        return [np.arange(*[v.item() for v in values], dtype=x.dtype)]
    if op == "CumSum":
        if attrs.get("exclusive", 0) or attrs.get("reverse", 0):
            raise NotImplementedError("Only inclusive forward CumSum is supported")
        return [np.cumsum(x, axis=int(values[1]), dtype=x.dtype)]
    if op.startswith("Reduce"):
        axes = values[1].tolist() if len(values) > 1 else attrs.get("axes")
        axes = tuple(axes) if axes is not None else None
        functions = {"ReduceMax": np.max, "ReduceMin": np.min, "ReduceSum": np.sum,
                     "ReduceMean": np.mean, "ReduceProd": np.prod}
        return [functions[op](x, axis=axes, keepdims=bool(attrs.get("keepdims", 1)))]
    if op == "Split":
        return list(np.split(x, np.cumsum(values[1])[:-1], axis=attrs.get("axis", 0)))
    if op == "ScatterND":
        if attrs.get("reduction", b"none") != b"none":
            raise NotImplementedError("ScatterND reductions are unsupported")
        result, indices = x.copy(), values[1]
        result[tuple(indices[..., i] for i in range(indices.shape[-1]))] = values[2]
        return [result]
    if op == "Clip":
        return [np.clip(*values)]
    if op == "Sigmoid":
        return [1 / (1 + np.exp(-x))]
    if op == "Softmax":
        axis = attrs.get("axis", -1)
        exp = np.exp(x - x.max(axis, keepdims=True))
        return [exp / exp.sum(axis, keepdims=True)]
    if op in ("LayerNormalization", "RMSNormalization"):
        axis = attrs.get("axis", -1) % x.ndim
        axes = tuple(range(axis, x.ndim))
        centered = x - x.mean(axes, keepdims=True) if op == "LayerNormalization" else x
        variance = np.mean(centered * centered, axis=axes, keepdims=True)
        result = (centered / np.sqrt(variance + np.float32(attrs.get("epsilon", 1e-5)))) * values[1]
        if len(values) > 2:
            result = result + values[2]
        return [result.astype(np.float32)]
    if op == "Resize":
        return [resize(x, values[3], attrs)]
    if op == "Conv":
        if _torch is not None:
            import torch.nn.functional as functional
            pads = attrs.get("pads", [0, 0, 0, 0])
            if pads[:2] != pads[2:]:
                raise ValueError("Only symmetric Conv padding is supported")
            bias = _torch.from_numpy(np.array(values[2], np.float32, copy=True)) if len(values) > 2 else None
            result = functional.conv2d(_torch.from_numpy(np.array(x, np.float32, copy=True)),
                                       _torch.from_numpy(np.array(values[1], np.float32, copy=True)), bias,
                                       stride=attrs.get("strides", [1, 1]), padding=pads[:2],
                                       dilation=attrs.get("dilations", [1, 1]), groups=attrs.get("group", 1))
            return [result.numpy()]
        if attrs.get("group", 1) != 1:
            raise NotImplementedError("NumPy Conv fallback requires group=1")
        kh, kw = values[1].shape[-2:]
        patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(2, 3))
        strides = attrs.get("strides", [1, 1])
        patches = patches[:, :, ::strides[0], ::strides[1]]
        n, _, h, w, _, _ = patches.shape
        flat = patches.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, -1)
        result = (flat @ values[1].reshape(values[1].shape[0], -1).T).reshape(n, h, w, -1).transpose(0, 3, 1, 2)
        return [result + values[2].reshape(1, -1, 1, 1) if len(values) > 2 else result]
    raise NotImplementedError(op)


class FP8Session:
    def __init__(self, path):
        enable_integer_backend()
        self.model = onnx.load(path)
        self.initializers = {t.name: numpy_helper.to_array(t) for t in self.model.graph.initializer}
        self.operations = [(n, restored_attributes(attributes(n))) for n in self.model.graph.node]

    def run(self, feed, progress=None):
        values = dict(self.initializers)
        values.update(feed)
        references = collections.Counter(i for n, _ in self.operations for i in n.input if i)
        for index, (node, attrs) in enumerate(self.operations):
            raw = [values[n] if n else None for n in node.input]
            try:
                if node.domain == DOMAIN:
                    logical, cursor = [], 0
                    for kind in attrs["input_kinds"]:
                        if kind:
                            logical.append(Float8(raw[cursor], raw[cursor + 1]))
                            cursor += 2
                        else:
                            logical.append(raw[cursor])
                            cursor += 1
                    op = node.op_type.removeprefix("F8")
                    if op == "QuantizeINT8":
                        x = logical[0].decode()
                        maximum = np.max(np.abs(x), axis=attrs["axis"], keepdims=True)
                        scale = encode(np.where(maximum > 0, maximum / np.float32(127), np.float32(1)), per_element=True)
                        decoded_scale = scale.decode()
                        if np.any(decoded_scale <= 0):
                            raise ValueError("FP8 activation scale underflowed to zero")
                        codes = np.clip(np.rint(x / decoded_scale), -127, 127).astype(np.int16)
                        codes = (codes + 128).astype(np.uint8) if attrs["unsigned"] else codes.astype(np.int8)
                        results = [codes, scale]
                    elif op == "ConstantOfShape":
                        bits = np.full(logical[0].tolist(), attrs["fill_bits"], np.uint8)
                        results = [Float8(bits.view(FP8_DTYPE), np.array(attrs["fill_exponent"], np.int32))]
                    elif op in ("Identity", "Gather", "Transpose", "Reshape", "Unsqueeze", "Squeeze", "Expand", "Slice", "Split"):
                        value = logical[0]
                        data = operation(op, [value.data, *logical[1:]], attrs)
                        if value.exponent.ndim == 0:
                            results = [Float8(v, value.exponent) for v in data]
                        else:
                            exponents = np.broadcast_to(value.exponent, value.data.shape)
                            exponents = operation(op, [exponents, *logical[1:]], attrs)
                            results = [Float8(v, e) for v, e in zip(data, exponents, strict=True)]
                    else:
                        decoded = [v.decode() if isinstance(v, Float8) else v for v in logical]
                        with np.errstate(over="ignore", under="ignore", divide="raise", invalid="raise"):
                            results = operation(op, decoded, attrs)
                        results = [encode(v) if kind else v for v, kind in zip(results, attrs["output_kinds"], strict=True)]
                    flattened = []
                    for kind, value in zip(attrs["output_kinds"], results, strict=True):
                        if kind:
                            flattened.extend([value.data, value.exponent])
                        else:
                            flattened.append(value)
                    results = flattened
                else:
                    results = operation(node.op_type, raw, attrs)
                for name, result in zip(node.output, results, strict=True):
                    array = np.asarray(result)
                    if array.dtype.kind == "f" and array.dtype != FP8_DTYPE:
                        raise TypeError(f"Non-FP8 floating tensor at boundary: {name}, {array.dtype}")
                    values[name] = array
            except Exception as error:
                raise RuntimeError(f"{node.name or node.op_type}: {error}") from error
            for name in node.input:
                references[name] -= 1
                if references[name] == 0 and name not in self.initializers and name in values:
                    del values[name]
            if progress is not None and index % 1000 == 0:
                progress(index, len(self.operations))
        return {v.name: values[v.name] for v in self.model.graph.output}
