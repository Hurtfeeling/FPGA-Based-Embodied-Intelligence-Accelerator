"""Integer reference kernels for the org.lerobot.strictint8 ONNX domain.

Activation values are signed INT8 codes plus an INT32 base-two exponent:
real_value = code * 2**exponent. Code -128 is reserved for attention -infinity.
Learned tensors are INT8. Wider integers are temporary accumulators, exponents,
indices and lookup tables. Runtime kernels never convert data to floating point.
This NumPy interpreter is a CPU correctness reference, not an ORT extension.
"""

import collections
import math
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper

DOMAIN = "org.lerobot.strictint8"
FRACTION = 20
UNIT = 1 << FRACTION
_torch_integer = None


def enable_integer_acceleration():
    """Use the optional CPU INT8 GEMM kernel; never dequantize to float."""
    global _torch_integer
    try:
        import torch
        torch.set_num_threads(8)
        probe = torch._int_mm(torch.ones((1, 1), dtype=torch.int8), torch.ones((1, 1), dtype=torch.int8))
        if probe.dtype == torch.int32:
            _torch_integer = torch
    except (ImportError, OSError, RuntimeError, NotImplementedError):
        _torch_integer = None


def integer_matmul(left, right):
    if left.shape[-1] * 127 * 127 > (1 << 31) - 1:
        raise OverflowError("Dot-product reduction exceeds the INT32 worst-case bound")
    if _torch_integer is None:
        return left.astype(np.int32) @ right.astype(np.int32)
    leading = np.broadcast_shapes(left.shape[:-2], right.shape[:-2])
    left = np.broadcast_to(left, (*leading, *left.shape[-2:]))
    right = np.broadcast_to(right, (*leading, *right.shape[-2:]))
    result = np.empty((*leading, left.shape[-2], right.shape[-1]), np.int32)
    for index in np.ndindex(leading):
        a = _torch_integer.from_numpy(np.array(left[index], dtype=np.int8, order="C", copy=True))
        b = _torch_integer.from_numpy(np.array(right[index], dtype=np.int8, order="C", copy=True))
        result[index] = _torch_integer._int_mm(a, b).numpy()
    return result


def shift_integer(value, shift):
    """Signed, bounded integer scaling; right shifts round ties away from zero."""
    value = np.asarray(value, np.int64)
    shift = np.asarray(shift, np.int64)
    left = np.clip(shift, 0, 40)
    right = np.clip(-shift, 0, 62)
    absolute = np.abs(value)
    rounded = (absolute + np.where(right > 0, np.left_shift(np.int64(1), np.maximum(right - 1, 0)), 0)) >> right
    result = np.where(value < 0, -rounded, rounded)
    return np.where(shift >= 0, np.left_shift(value, left), result)


def trunc_divide(numerator, denominator):
    numerator, denominator = np.asarray(numerator, np.int64), np.asarray(denominator, np.int64)
    if np.any(denominator == 0):
        raise ZeroDivisionError("Integer reference division by zero")
    quotient = np.abs(numerator) // np.abs(denominator)
    return np.where((numerator < 0) != (denominator < 0), -quotient, quotient)


@dataclass
class Q8:
    codes: np.ndarray
    exponent: np.ndarray

    def __post_init__(self):
        self.codes = np.asarray(self.codes)
        self.exponent = np.asarray(self.exponent, np.int32)
        if self.codes.dtype != np.int8:
            raise TypeError(f"Expected INT8 codes, got {self.codes.dtype}")

    @property
    def mask(self):
        return self.codes == -128

    def at(self, exponent):
        codes = np.where(self.mask, 0, self.codes).astype(np.int64)
        return shift_integer(codes, self.exponent.astype(np.int64) - exponent)

    def fixed(self):
        # Only transcendental kernels need a fixed grid. Clip before shifting
        # so a masked value or an extreme exponent cannot overflow INT64.
        shift = self.exponent.astype(np.int64) + FRACTION
        raw = shift_integer(np.where(self.mask, 0, self.codes).astype(np.int64), np.minimum(shift, 30))
        raw = np.clip(raw, -(1 << 40), (1 << 40) - 1)
        return np.where(self.mask, -(1 << 40), raw)


@dataclass
class Scale32:
    """Quantization metadata, not a learned weight or activation tensor."""
    codes: np.ndarray
    exponent: np.ndarray


def pack(raw, exponent=0, mask=None):
    raw = np.asarray(raw, np.int64)
    maximum = int(np.max(np.abs(raw), initial=0))
    shift = max(0, maximum.bit_length() - 7)
    rounded = shift_integer(raw, -shift)
    if np.max(np.abs(rounded), initial=0) > 127:
        shift += 1
        rounded = shift_integer(raw, -shift)
    codes = np.clip(rounded, -127, 127).astype(np.int8)
    if mask is not None:
        codes = np.where(mask, np.int8(-128), codes)
    return Q8(codes, np.asarray(int(exponent) + shift, np.int32))


def align(values):
    exponent = max(int(np.max(v.exponent)) for v in values) - 20
    return [v.at(exponent) for v in values], exponent


def add(left, right, subtract=False):
    values, exponent = align([left, right])
    raw = values[0] - values[1] if subtract else values[0] + values[1]
    return pack(raw, exponent, left.mask | right.mask)


def multiply(left, right):
    if left.exponent.ndim == 0 and right.exponent.ndim == 0:
        raw = left.codes.astype(np.int64) * right.codes.astype(np.int64)
        return pack(raw, int(left.exponent) + int(right.exponent), left.mask | right.mask)
    # Parameter exponents may be per output channel. Align the product once.
    raw = left.codes.astype(np.int64) * right.codes.astype(np.int64)
    exponents = left.exponent.astype(np.int64) + right.exponent.astype(np.int64)
    exponent = int(np.max(exponents)) - 20
    return pack(shift_integer(raw, exponents - exponent), exponent, left.mask | right.mask)


def divide(left, right):
    if left.exponent.ndim or right.exponent.ndim:
        values, exponent = align([left, right])
        return pack(trunc_divide(values[0] << 20, values[1]), -20, left.mask | right.mask)
    raw = trunc_divide(left.codes.astype(np.int64) << 24, right.codes.astype(np.int64))
    return pack(raw, int(left.exponent) - int(right.exponent) - 24, left.mask | right.mask)


def integer_sqrt(value):
    value = np.maximum(np.asarray(value, np.int64), 0)
    maximum = int(np.max(value, initial=0))
    guess = np.full_like(value, 1 << ((maximum.bit_length() + 1) // 2))
    for _ in range(32):
        guess = (guess + value // np.maximum(guess, 1)) // 2
    return np.minimum(guess, value // np.maximum(guess, 1))


def normalize(value, scale, bias, attrs):
    axis = attrs.get("axis", -1) % value.codes.ndim
    axes = tuple(range(axis, value.codes.ndim))
    count = math.prod(value.codes.shape[i] for i in axes)
    # The scale of x cancels in (x - mean) / std. Keep 16 fractional bits
    # inside this kernel, then requantize its result to INT8.
    codes = value.codes.astype(np.int64)
    center = (codes << 16) - trunc_divide(codes.sum(axes, keepdims=True) << 16, count)
    variance = trunc_divide((center * center) >> 16, 1).sum(axes, keepdims=True) // count
    epsilon = shift_integer(np.int64(attrs.get("epsilon_q20", 10)), 16 - FRACTION - 2 * int(value.exponent))
    std = integer_sqrt((variance + np.maximum(epsilon, 1)) << 16)
    normalized = trunc_divide(center << 16, np.maximum(std, 1))
    product = normalized * scale.codes.astype(np.int64)
    result = pack(product, int(scale.exponent) - 16)
    return add(result, bias) if bias is not None else result


def rms_normalize(value, scale, attrs):
    codes = value.codes.astype(np.int64)
    count = codes.shape[-1]
    variance = trunc_divide((codes * codes).sum(-1, keepdims=True) << 16, count)
    epsilon = shift_integer(np.int64(attrs["epsilon_q20"]), 16 - FRACTION - 2 * int(value.exponent))
    std = integer_sqrt((variance + np.maximum(epsilon, 1)) << 16)
    normalized = trunc_divide(codes << 32, np.maximum(std, 1))
    return pack(normalized * scale.codes.astype(np.int64), int(scale.exponent) - 16)


def resize_linear(value, sizes):
    sizes = [int(v) for v in sizes]
    if sizes[:2] != list(value.codes.shape[:2]):
        raise ValueError("Only spatial NCHW resize is supported")
    raw = value.codes.astype(np.int64)
    bits = 16
    exponent = int(value.exponent)
    for axis in (2, 3):
        size, target = raw.shape[axis], sizes[axis]
        # half_pixel coordinates, expressed as an exact integer rational.
        position = ((2 * np.arange(target, dtype=np.int64) + 1) * size * (1 << bits)) // (2 * target) - (1 << (bits - 1))
        position = np.clip(position, 0, (size - 1) << bits)
        low = position >> bits
        high = np.minimum(low + 1, size - 1)
        shape = [1] * raw.ndim
        shape[axis] = target
        fraction = (position - (low << bits)).reshape(shape)
        left, right = np.take(raw, low, axis=axis), np.take(raw, high, axis=axis)
        raw = left * (1 << bits) + (right - left) * fraction
        exponent -= bits
        if axis == 2:
            # Keep the temporary image within INT32 before the second pass.
            raw = shift_integer(raw, -8)
            exponent += 8
    return pack(raw, exponent)


def convolution(value, weights, bias, attrs):
    if attrs.get("group", 1) != 1 or any(v != 1 for v in attrs.get("dilations", [1, 1])):
        raise ValueError("Reference convolution only supports group=1, dilation=1")
    q = value.codes
    pads = attrs.get("pads", [0, 0, 0, 0])
    q = np.pad(q, [(0, 0), (0, 0), (pads[0], pads[2]), (pads[1], pads[3])])
    kh, kw = weights.codes.shape[-2:]
    strides = attrs.get("strides", [1, 1])
    patches = np.lib.stride_tricks.sliding_window_view(q, (kh, kw), axis=(2, 3))
    patches = patches[:, :, ::strides[0], ::strides[1]]
    n, _, h, w, _, _ = patches.shape
    flat = patches.transpose(0, 2, 3, 1, 4, 5).reshape(n * h * w, -1)
    raw = integer_matmul(flat, weights.codes.reshape(weights.codes.shape[0], -1).T)
    raw = raw.reshape(n, h, w, -1).transpose(0, 3, 1, 2).astype(np.int64)
    exponents = weights.exponent.reshape(1, -1, 1, 1) + int(value.exponent)
    exponent = int(np.max(exponents)) - 20
    result = pack(shift_integer(raw, exponents - exponent), exponent)
    if bias is not None:
        bias = Q8(bias.codes.reshape(1, -1, 1, 1), bias.exponent)
    return add(result, bias) if bias is not None else result


def raw_operation(op, values, attrs):
    """The source export's integer/index/boolean operations."""
    x = values[0] if values else None
    if op == "Identity":
        return [x]
    if op == "Cast":
        dtype = helper.tensor_dtype_to_np_dtype(attrs["to"])
        if np.dtype(dtype).kind == "f":
            raise TypeError("Floating Cast is forbidden in the integer runtime")
        return [x.astype(dtype)]
    binary = {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply, "Equal": np.equal,
              "Greater": np.greater, "Less": np.less, "LessOrEqual": np.less_equal,
              "And": np.logical_and, "Or": np.logical_or}
    if op in binary:
        return [binary[op](*values)]
    if op == "Div":
        return [trunc_divide(*values).astype(x.dtype)]
    if op == "Not":
        return [np.logical_not(x)]
    if op == "Where":
        return [np.where(*values)]
    if op == "Shape":
        return [np.asarray(x.shape[attrs.get("start", 0):attrs.get("end")], np.int64)]
    if op == "Gather":
        return [np.take(x, values[1], axis=attrs.get("axis", 0))]
    if op == "Reshape":
        shape = values[1].copy()
        if not attrs.get("allowzero", 0):
            for index in np.flatnonzero(shape == 0):
                shape[index] = x.shape[index]
        return [x.reshape(shape)]
    if op == "Transpose":
        return [np.transpose(x, attrs.get("perm"))]
    if op == "Unsqueeze":
        return [np.expand_dims(x, tuple(values[1].tolist()))]
    if op == "Squeeze":
        return [np.squeeze(x, tuple(values[1].tolist()) if len(values) > 1 else None)]
    if op == "Concat":
        return [np.concatenate(values, axis=attrs.get("axis", 0))]
    if op == "Expand":
        return [np.broadcast_to(x, np.broadcast_shapes(x.shape, tuple(values[1])))]
    if op == "Slice":
        result = [slice(None)] * x.ndim
        axes = values[3] if len(values) > 3 else np.arange(len(values[1]))
        steps = values[4] if len(values) > 4 else np.ones(len(axes), np.int64)
        for axis, start, end, step in zip(axes, values[1], values[2], steps, strict=True):
            result[int(axis)] = slice(int(start), int(end), int(step))
        return [x[tuple(result)]]
    if op == "ConstantOfShape":
        tensor = attrs.get("value")
        value = numpy_helper.to_array(tensor) if tensor is not None else np.array([0], np.int64)
        return [np.full(x.tolist(), value.item(), value.dtype)]
    if op == "Range":
        return [np.arange(*[int(v) for v in values], dtype=x.dtype)]
    if op == "CumSum":
        if attrs.get("exclusive", 0) or attrs.get("reverse", 0):
            raise NotImplementedError("Only forward inclusive CumSum is supported")
        return [np.cumsum(x, axis=int(values[1]), dtype=x.dtype)]
    if op.startswith("Reduce"):
        axes = values[1].tolist() if len(values) > 1 else attrs.get("axes")
        axes = tuple(axes) if axes is not None else None
        functions = {"ReduceSum": np.sum, "ReduceProd": np.prod, "ReduceMax": np.max, "ReduceMin": np.min}
        return [functions[op](x, axis=axes, keepdims=bool(attrs.get("keepdims", 1)))]
    if op == "Split":
        sizes = values[1] if len(values) > 1 else attrs["split"]
        return list(np.split(x, np.cumsum(sizes)[:-1], axis=attrs.get("axis", 0)))
    if op == "ScatterND":
        if attrs.get("reduction", b"none") != b"none":
            raise NotImplementedError("ScatterND reductions are unsupported")
        result = x.copy()
        indices = values[1]
        result[tuple(indices[..., i] for i in range(indices.shape[-1]))] = values[2]
        return [result]
    if op == "Pad":
        if attrs.get("mode", b"constant") != b"constant":
            raise NotImplementedError("Only constant Pad is supported")
        pads = values[1]
        return [np.pad(x, list(zip(pads[:x.ndim], pads[x.ndim:], strict=True)),
                       constant_values=values[2].item() if len(values) > 2 and values[2] is not None else 0)]
    raise NotImplementedError(f"Integer/index operation {op}")


def quantized_operation(op, values, attrs, tables):
    x = values[0]
    if op == "FromInteger":
        return [pack(x)]
    if op == "ToInteger":
        dtype = helper.tensor_dtype_to_np_dtype(attrs["to"])
        if attrs["to"] == onnx.TensorProto.BOOL:
            return [(x.codes != 0) | x.mask]
        return [x.at(0).astype(dtype)]
    if op == "MatMulWeights":
        left, weights, scale = values
        raw = integer_matmul(left.codes, weights)
        product = raw.astype(np.int64) * scale.codes.astype(np.int64)
        return [pack(product, int(left.exponent) + int(scale.exponent))]
    if op == "MatMul":
        right = values[1]
        raw = integer_matmul(x.codes, right.codes)
        return [pack(raw, int(x.exponent) + int(right.exponent))]
    if op in ("Add", "Sub"):
        return [add(x, values[1], op == "Sub")]
    if op == "Mul":
        return [multiply(x, values[1])]
    if op == "Div":
        return [divide(x, values[1])]
    if op in ("Equal", "Greater", "LessOrEqual", "Less"):
        aligned, _ = align(values)
        return raw_operation(op, aligned, attrs)
    if op in ("Abs", "Neg"):
        raw = np.abs(x.codes.astype(np.int64)) if op == "Abs" else -x.codes.astype(np.int64)
        return [pack(raw, int(x.exponent), x.mask)]
    if op == "Round":
        raw = x.at(0)
        # Match nearest-even at half-integers instead of ties-away.
        if int(x.exponent) < 0:
            shift = min(62, -int(x.exponent))
            absolute = np.abs(x.codes.astype(np.int64))
            whole = absolute >> shift
            remainder = absolute - (whole << shift)
            half = 1 << (shift - 1)
            rounded = whole + ((remainder > half) | ((remainder == half) & ((whole & 1) != 0)))
            raw = np.where(x.codes < 0, -rounded, rounded)
        return [pack(raw)]
    if op == "Clip":
        aligned, exponent = align(values)
        return [pack(np.clip(*aligned), exponent, x.mask)]
    if op == "Reciprocal":
        return [divide(Q8(np.array(1, np.int8), np.array(0, np.int32)), x)]
    if op == "Sqrt":
        return [pack(integer_sqrt(np.maximum(x.fixed(), 0) * UNIT), -FRACTION)]
    if op == "Pow":
        exponent = values[1]
        if exponent.codes.size == 1:
            if np.array_equal(exponent.fixed(), np.asarray(UNIT // 2)):
                return [pack(integer_sqrt(np.maximum(x.fixed(), 0) * UNIT), -FRACTION)]
            if int(exponent.at(0)) == 2 and np.array_equal(exponent.fixed(), np.asarray(2 * UNIT)):
                return [multiply(x, x)]
        # Rotary bases are constant and their exponent lies in [-2, 2]. The
        # offline-generated power table is identified in the node attributes.
        table = tables[attrs["power_table"].decode()]
        exponent_fixed = exponent.fixed()
        index = np.clip((exponent_fixed + 2 * UNIT) // (UNIT // 4096), 0, 16384)
        return [pack(table[index], -FRACTION)]
    if op in ("Sin", "Cos"):
        period = attrs["period_q20"]
        index = (x.fixed() % period) * 16384 // period
        return [Q8(tables["__lut_" + op.lower()][index], np.array(-6, np.int32))]
    if op in ("Sigmoid", "Tanh"):
        index = (np.clip(x.fixed(), -16 * UNIT, 16 * UNIT) + 16 * UNIT) // (UNIT // 256)
        return [Q8(tables["__lut_" + op.lower()][index], np.array(-7 if op == "Sigmoid" else -6, np.int32))]
    if op == "Softmax":
        axis = attrs.get("axis", -1)
        fixed = x.fixed()
        valid = np.where(x.mask, -(1 << 40), fixed)
        delta = np.clip(valid - valid.max(axis, keepdims=True), -16 * UNIT, 0)
        index = (delta + 16 * UNIT) // (UNIT // 256)
        exp = np.where(x.mask, 0, tables["__lut_exp"][index]).astype(np.int64)
        total = exp.sum(axis, keepdims=True)
        return [pack(trunc_divide(exp << 24, np.maximum(total, 1)), -24)]
    if op == "LayerNormalization":
        return [normalize(x, values[1], values[2] if len(values) > 2 else None, attrs)]
    if op == "RMSNormalization":
        return [rms_normalize(x, values[1], attrs)]
    if op in ("RotarySin", "RotaryCos"):
        positions = x.at(0) if isinstance(x, Q8) else np.asarray(x, np.int64)
        while positions.ndim > 2 and positions.shape[-1] == 1:
            positions = positions[..., 0]
        half = int(values[1].at(0)) if isinstance(values[1], Q8) else int(values[1])
        if not 1 <= half <= 256:
            raise ValueError(f"Unsupported rotary half-width {half}")
        frequencies = tables["__lut_rope_frequencies"][half, :half].astype(np.int64)
        radians = (positions[..., None] * frequencies) >> (30 - FRACTION)
        index = (radians % attrs["period_q20"]) * 16384 // attrs["period_q20"]
        table = tables["__lut_" + op.removeprefix("Rotary").lower()]
        return [Q8(np.expand_dims(table[index], -2), np.array(-6, np.int32))]
    if op == "Resize":
        if attrs.get("mode") != b"linear" or attrs.get("coordinate_transformation_mode") != b"half_pixel":
            raise NotImplementedError("Only half-pixel linear Resize is supported")
        return [resize_linear(x, values[3])]
    if op == "Conv":
        return [convolution(x, values[1], values[2] if len(values) > 2 else None, attrs)]
    if op == "Where":
        condition, left, right = values
        aligned, exponent = align([left, right])
        return [pack(np.where(condition, *aligned), exponent, np.where(condition, left.mask, right.mask))]
    if op == "Concat":
        aligned, exponent = align(values)
        return [pack(np.concatenate(aligned, axis=attrs["axis"]), exponent,
                     np.concatenate([v.mask for v in values], axis=attrs["axis"]))]
    if op == "ScatterND":
        aligned, exponent = align([x, values[2]])
        raw = raw_operation(op, [aligned[0], values[1], aligned[1]], attrs)[0]
        mask = raw_operation(op, [x.mask, values[1], values[2].mask], attrs)[0]
        return [pack(raw, exponent, mask)]
    if op == "Pad":
        padding = values[2] if len(values) > 2 and values[2] is not None else Q8(np.array(0, np.int8), np.array(0, np.int32))
        aligned, exponent = align([x, padding])
        return [pack(raw_operation(op, [aligned[0], values[1], aligned[1]], attrs)[0], exponent)]
    if op == "Range":
        aligned, exponent = align(values)
        return [pack(np.arange(*[int(v) for v in aligned], dtype=np.int64), exponent)]
    if op == "ConstantOfShape":
        # The converter stores the fill code and exponent as integer attrs.
        return [Q8(np.full(x.tolist(), attrs["fill_code"], np.int8), np.array(attrs["fill_exponent"], np.int32))]
    if op.startswith("Reduce"):
        axes = values[1].tolist() if len(values) > 1 else attrs.get("axes")
        axes = tuple(axes) if axes is not None else None
        keep = bool(attrs.get("keepdims", 1))
        if op == "ReduceMean":
            count = math.prod(x.codes.shape[a] for a in axes) if axes is not None else x.codes.size
            return [pack(trunc_divide(x.codes.astype(np.int64).sum(axes, keepdims=keep) << 16, count), int(x.exponent) - 16)]
        if op == "ReduceSum":
            return [pack(x.codes.astype(np.int64).sum(axes, keepdims=keep), int(x.exponent))]
        if op in ("ReduceMax", "ReduceMin"):
            function = np.max if op == "ReduceMax" else np.min
            return [Q8(function(x.codes, axis=axes, keepdims=keep), x.exponent)]
        raise NotImplementedError(op)
    # Structural operations preserve codes/exponent for activation tensors.
    # Per-channel parameter exponents must be broadcast and transformed too.
    if op in ("Gather", "Reshape", "Transpose", "Unsqueeze", "Squeeze", "Expand", "Slice", "Split", "Identity"):
        if x.exponent.ndim == 0:
            result = raw_operation(op, [x.codes, *values[1:]], attrs)
            return [Q8(v, x.exponent) for v in result]
        exponent = int(np.max(x.exponent)) - 20
        result = raw_operation(op, [x.at(exponent), *values[1:]], attrs)
        return [pack(v, exponent) for v in result]
    raise NotImplementedError(f"Strict INT8 kernel {op}")


class IntegerSession:
    """Execute standard integer nodes and version-1 custom INT8 nodes."""

    def __init__(self, path):
        enable_integer_acceleration()
        self.model = onnx.load(path)
        self.initializers = {t.name: numpy_helper.to_array(t) for t in self.model.graph.initializer}
        self.tables = {k: v for k, v in self.initializers.items() if k.startswith("__lut_")}
        self.operations = [(n, {a.name: helper.get_attribute_value(a) for a in n.attribute}) for n in self.model.graph.node]
        self.references = collections.Counter(i for n in self.model.graph.node for i in n.input if i)

    def run(self, feed, progress=None, checkpoint=None):
        values = dict(self.initializers)
        values.update(feed)
        start = 0
        if checkpoint is not None and checkpoint.exists():
            with np.load(checkpoint, allow_pickle=False) as saved:
                start = int(saved["__next_node"])
                values.update({k: saved[k] for k in saved.files if k != "__next_node"})
        remaining = collections.Counter(i for n, _ in self.operations[start:] for i in n.input if i)
        def save_state(index):
            if checkpoint is not None:
                state = {k: v for k, v in values.items() if k not in self.initializers}
                np.savez(checkpoint, __next_node=np.array(index, np.int64), **state)
        for index in range(start, len(self.operations)):
            node, attrs = self.operations[index]
            inputs = [values[name] if name else None for name in node.input]
            if node.domain == DOMAIN:
                kinds = attrs["input_kinds"]
                logical = []
                cursor = 0
                for kind in kinds:
                    if kind in (1, 2):
                        cls = Q8 if kind == 1 else Scale32
                        logical.append(cls(inputs[cursor], inputs[cursor + 1]))
                        cursor += 2
                    else:
                        logical.append(inputs[cursor])
                        cursor += 1
                try:
                    results = quantized_operation(node.op_type.removeprefix("I8"), logical, attrs, self.tables)
                except Exception as error:
                    save_state(index)
                    raise RuntimeError(f"{node.name}: {error}") from error
                flattened = []
                for kind, result in zip(attrs["output_kinds"], results, strict=True):
                    flattened.extend([result.codes, result.exponent] if kind == 1 else [result])
                results = flattened
            else:
                results = raw_operation(node.op_type, inputs, attrs)
            for name, value in zip(node.output, results, strict=True):
                array = np.asarray(value)
                if array.dtype.kind == "f":
                    raise TypeError(f"Floating runtime result at {node.name}")
                values[name] = array
            for name in node.input:
                remaining[name] -= 1
                if remaining[name] == 0 and name not in self.initializers and name in values:
                    del values[name]
            if progress is not None and index % 1000 == 0:
                progress(index, len(self.operations))
        if checkpoint is not None and checkpoint.exists():
            checkpoint.unlink()
        return {v.name: values[v.name] for v in self.model.graph.output}
