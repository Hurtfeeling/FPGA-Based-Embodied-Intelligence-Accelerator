"""Profile real GEMM shapes and prepare a conservative, selective PYNQ graph.

The source ONNX is immutable. Small/unmeasured GEMMs and every Softmax keep
their standard ORT implementation. Only selected MatMulInteger nodes use PL.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from math import prod
from pathlib import Path
import shutil
import time

import numpy as np
import onnx
from onnx import helper, numpy_helper


BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / 'models'
WORKSPACE = MODEL_DIR.parent.parent
EVIDENCE_FILES = [
    "dispatch_aligned32_results_20261103_202348.json",
    "dispatch_aligned32_results_20261103_202554.json",
    "dispatch_aligned32_wide_results_20261103_203457.json",
    "dispatch_aligned32_wide_results_20261103_203815.json",
]
EVIDENCE_DIR = BASE.parent / "GEMM" / "fast_gemm"
RESIDENT_CACHE_CAPACITY_BYTES = 64 * 1024**2


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def value_shape(value):
    tensor = value.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    return [dim.dim_value if dim.HasField("dim_value") else dim.dim_param or None
            for dim in tensor.shape.dim]


def dimensions(a, b):
    if not a or not b or len(a) < 2 or len(b) < 2:
        raise ValueError("Expected rank >= 2 MatMul inputs")
    if not all(isinstance(x, int) and x > 0 for x in a + b):
        raise ValueError("Non-concrete MatMul shape")
    if a[-1] != b[-2]:
        raise ValueError("MatMul K mismatch")
    batch = np.broadcast_shapes(tuple(a[:-2]), tuple(b[:-2]))
    # A static 2-D right matrix allows all A leading rows to be one PL job.
    rows = prod(a[:-1]) if len(b) == 2 else a[-2]
    return dict(M=rows, N=b[-1], K=a[-1], batch_shape=list(batch),
                matrix_count=1 if len(b) == 2 else prod(batch),
                logical_ops=2 * prod(batch) * a[-2] * b[-1] * a[-1],
                padded_M=(rows + 31) // 32 * 32,
                padded_N=(b[-1] + 31) // 32 * 32,
                padded_K=(a[-1] + 31) // 32 * 32)


def inspect_model(model):
    graph = model.graph
    shapes = {value.name: value_shape(value) for value in
              list(graph.input) + list(graph.output) + list(graph.value_info)}
    initializers = {value.name: value for value in graph.initializer}
    producers = {name: node for node in graph.node for name in node.output}
    constant_names = set(initializers)
    # Topological propagation is metadata only; folding below is restricted to
    # a single INT8 weight Transpose, never floating-point arithmetic.
    excluded = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
                "Dropout", "Multinomial", "If", "Loop", "Scan"}
    for node in graph.node:
        if node.op_type == "Constant" or (node.input and node.op_type not in excluded
                                         and all(name in constant_names for name in node.input if name)):
            constant_names.update(node.output)
    nodes = []
    for index, node in enumerate(graph.node):
        if node.op_type != "MatMulInteger":
            continue
        right = producers.get(node.input[1])
        foldable = bool(right and right.op_type == "Transpose" and len(right.input) == 1
                        and right.input[0] in initializers
                        and initializers[right.input[0]].data_type == onnx.TensorProto.INT8)
        nodes.append(dict(node_index=index, node_name=node.name,
                          input_names=list(node.input), output_names=list(node.output),
                          static_A_shape=shapes.get(node.input[0]),
                          static_B_shape=shapes.get(node.input[1]),
                          static_output_shape=shapes.get(node.output[0]),
                          constant_b=node.input[1] in constant_names,
                          foldable_int8_weight_transpose=foldable,
                          weight_initializer=right.input[0] if foldable else None,
                          weight_shape=list(initializers[right.input[0]].dims) if foldable else None))
    return nodes


def profile_input_shapes(events):
    found = {}
    for event in events:
        args = event.get("args", {})
        if args.get("op_name") != "MatMulInteger" or not event.get("name", "").endswith("_kernel_time"):
            continue
        name = event["name"][:-len("_kernel_time")]
        inputs = []
        for value in args.get("input_type_shape", []):
            if len(value) != 1:
                raise ValueError("Unexpected profile input shape: " + str(value))
            dtype, dims = next(iter(value.items()))
            inputs.append(dict(dtype=dtype, shape=dims))
        row = dict(node_name=name, node_index=args.get("node_index"), inputs=inputs,
                   outputs=args.get("output_type_shape"), duration_us=event.get("dur"))
        if name in found and found[name]["inputs"] != inputs:
            raise ValueError("A graph node executed with multiple shapes: " + name)
        found[name] = row
    return found


def run_host_profile(model_path, input_path, metadata_path):
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    options.enable_cpu_mem_arena = False
    options.enable_mem_pattern = False
    options.enable_profiling = True
    options.profile_file_prefix = str(BASE / "host_shapes")
    print("HOST_PROFILE_SESSION_START", flush=True)
    start = time.perf_counter()
    session = ort.InferenceSession(str(model_path), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    load_seconds = time.perf_counter() - start
    print("HOST_PROFILE_SESSION_READY", load_seconds, flush=True)
    with np.load(str(input_path), allow_pickle=False) as archive:
        feed = {value.name: np.ascontiguousarray(archive[value.name]) for value in session.get_inputs()}
    start = time.perf_counter()
    output = session.run(["actions"], feed)[0]
    run_seconds = time.perf_counter() - start
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite host-profile output")
    raw_path = Path(session.end_profiling())
    del session
    print("HOST_PROFILE_RUN_DONE", run_seconds, str(raw_path), flush=True)
    events = json.loads(raw_path.read_text(encoding="utf-8"))
    profiles = profile_input_shapes(events)
    # Preserve the raw profiling evidence efficiently, while metadata is easy to read.
    compressed_path = raw_path.with_suffix(raw_path.suffix + ".gz")
    with raw_path.open("rb") as source, gzip.open(str(compressed_path), "wb", compresslevel=6) as dest:
        shutil.copyfileobj(source, dest)
    raw_path.unlink()
    metadata = dict(source_sha256=digest(model_path), inputs_sha256=digest(input_path),
                    profile_is_host_only=True, host_ort_version=ort.__version__,
                    cpu_threads=4, graph_optimization="disabled", cpu_arena=False,
                    memory_pattern=False, session_load_seconds=load_seconds,
                    inference_seconds=run_seconds, output_shape=list(output.shape),
                    output_finite=True, raw_profile_file=compressed_path.name,
                    profiled_matmul_nodes=len(profiles), matmul_events=list(profiles.values()))
    write_json(metadata_path, metadata)
    return metadata


def annotate_shapes(nodes, metadata):
    profiles = {row["node_name"]: row for row in metadata["matmul_events"]}
    for node in nodes:
        event = profiles.get(node["node_name"])
        if event and len(event["inputs"]) >= 2:
            a, b = event["inputs"][:2]
            node.update(profile_A_type=a["dtype"], profile_B_type=b["dtype"],
                        actual_A_shape=a["shape"], actual_B_shape=b["shape"],
                        host_kernel_duration_us=event["duration_us"],
                        profile_resolved=True)
            node.update(dimensions(a["shape"], b["shape"]))
        else:
            node["profile_resolved"] = False
    return nodes


def load_anchors():
    anchors = {}
    for filename in EVIDENCE_FILES:
        report = json.loads((EVIDENCE_DIR / filename).read_text(encoding="utf-8"))
        if not report.get("all_outputs_verified"):
            raise ValueError("Unverified benchmark evidence: " + filename)
        for row in report["records"]:
            # Never rely on the three-slot constant-B cache surviving the graph.
            if row["constant_b"] or len(row["shapes_mnk"]) != 1:
                continue
            mnk = tuple(row["shapes_mnk"][0])
            if any(value % 32 for value in mnk):
                continue
            anchors[mnk] = dict(M=mnk[0], N=mnk[1], K=mnk[2],
                                speedup_cpu_over_pl=row["cpu_median_ms"] / row["pl_warm_median_ms"],
                                cpu_median_ms=row["cpu_median_ms"],
                                pl_median_ms=row["pl_warm_median_ms"],
                                actual_pl_cores=row["pl_core_jobs_per_call"],
                                source_file=filename, source_case=row["name"],
                                repeats=report["repeats"], b_treatment="transpose_each_call")
    return list(anchors.values())


def choose_node(node, anchors):
    if not node.get("profile_resolved"):
        return dict(target="CPU_ORT", reason="actual_shape_not_profiled")
    if node["profile_A_type"] != "uint8" or node["profile_B_type"] != "int8":
        return dict(target="CPU_ORT", reason="unsupported_integer_types")
    m, n, k = node["M"], node["N"], node["K"]
    mp, npad, kp = node["padded_M"], node["padded_N"], node["padded_K"]
    if node["constant_b"] and not node["foldable_int8_weight_transpose"]:
        return dict(target="CPU_ORT", reason="static_b_cannot_be_folded_to_stable_int8_initializer")
    if m < 32:
        return dict(target="CPU_ORT", reason="fewer_than_32_real_rows")
    if kp > 16352 or kp * npad > 16 * 1024**2 or kp * 32 > 4 * 1024**2 or npad * 32 > 2 * 1024**2:
        return dict(target="CPU_ORT", reason="outside_accelerator_or_staging_limits")
    # Copying/padding an originally unaligned input needs more margin than the
    # aligned scan; count a broadcast MatMul's per-head shape, never head sum.
    aligned = all(value % 32 == 0 for value in (m, n, k))
    margin = 1.10 if aligned else 1.20
    candidates = [anchor for anchor in anchors
                  if anchor["speedup_cpu_over_pl"] >= margin
                  and mp >= anchor["M"] and npad >= anchor["N"] and kp >= anchor["K"]]
    if not candidates:
        return dict(target="CPU_ORT", reason="no_dominated_measured_winning_shape",
                    required_anchor_speedup=margin)
    # Select the nearest dominating proof; a larger anchor has less extrapolation.
    anchor = min(candidates,
                 key=lambda value: (mp * npad * kp / (value["M"] * value["N"] * value["K"]),
                                    -value["speedup_cpu_over_pl"]))
    return dict(target="PL", reason="padded_shape_dominates_measured_uncached_winner",
                required_anchor_speedup=margin, measured_anchor=anchor,
                actual_pl_cores=min(mp // 32, 3),
                shape_dominance_is_empirical_extrapolation=True)


def scalar_constant(name, initializers, producers):
    tensor = initializers.get(name)
    if tensor is None:
        node = producers.get(name)
        if node is None or node.op_type != "Constant":
            raise ValueError("Zero point is not a scalar constant: " + name)
        attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}
        tensor = attrs.get("value")
    value = numpy_helper.to_array(tensor)
    if value.size != 1:
        raise ValueError("Zero point must be scalar")
    return value


def selection_summary(nodes):
    rows = {}
    for node in nodes:
        key = (node.get("M"), node.get("N"), node.get("K"), node["constant_b"], node["selection"]["target"])
        row = rows.setdefault(key, dict(M=key[0], N=key[1], K=key[2], constant_b=key[3],
                                       target=key[4], graph_nodes=0, matrix_count=0, logical_ops=0,
                                       reasons=Counter()))
        row["graph_nodes"] += 1
        row["matrix_count"] += node.get("matrix_count", 0)
        row["logical_ops"] += node.get("logical_ops", 0)
        row["reasons"][node["selection"]["reason"]] += 1
    return sorted(rows.values(), key=lambda value: (value["target"], str(value["M"]), str(value["N"]), str(value["K"])))


def write_selection_summary(manifest):
    lines = ["# 新版 SmolVLA：选择性 CPU / PL 分流", "",
             "模型：`../smolvla_w8a8_fp32.onnx`；选择图：`../smolvla_selective_pynq.onnx`。",
             "矩阵顺序统一为 **M,N,K**，计算 `A[M,K] × B[K,N] → C[M,N]`。", "",
             "## 规则和覆盖", "",
             "- 用新模型 `validation_0.npz` 在主机标准ORT上运行一次4线程profile，关闭图优化、CPU arena和memory pattern；1812个MatMulInteger全部取得实际形状。主机时间不是PYNQ性能数据。",
             "- 选择依据仅取既有32对齐实测中 **每次转置B** 的完整PS/PL路径，选择图不假设全图三槽BT缓存会命中。",
             "- 输入本身32对齐时，实测锚点需 `PS时间/PL时间 ≥ 1.10`；原始M/N/K任一未对齐时提高到 `≥1.20`。待处理shape补齐32后的M/N/K必须逐维不小于一个胜出锚点。该逐维外推是保守经验规则，最终收益仍以整图实测为准。",
             "- B为2维时可把A所有前导行合并为M；B为多维时按每个矩阵的M/N/K判断，不把head/batch计算量累加骗过阈值。不明shape默认CPU。",
             "- 所有199个Softmax保留原ORT；691个小型或无足够证据的GEMM保留原ORT；1121个大GEMM交给PL。", "",
             "## 逐形状选择", "",
             "`节点数`是图节点数；`矩阵数`包含广播batch/head，每个动作块执行一次完整图。", "",
             "| M,N,K | B性质 | 执行位置 | 节点数 | 矩阵数 | 逻辑GOP | 说明 |",
             "|---|---|---|---:|---:|---:|---|"]
    for row in sorted(manifest["shape_decisions"], key=lambda value: (value["target"], value["M"] or 0, value["N"] or 0, value["K"] or 0)):
        note = "达到每次转置胜出锚点" if row["target"] == "PL" else "小型/未达到保守胜出锚点"
        lines.append("| {},{},{} | {} | {} | {} | {} | {:.6f} | {} |".format(
            row["M"], row["N"], row["K"], "静态权重" if row["constant_b"] else "动态B",
            "PL" if row["target"] == "PL" else "CPU ORT", row["graph_nodes"],
            row["matrix_count"], row["logical_ops"] / 1e9, note))
    lines += ["", "## 静态权重预处理与比较口径", "",
              "选中的静态权重右输入原为 `Transpose(INT8 codes initializer)`。准备图仅对这些整数权重作精确转置折叠，并按源权重和perm去重，让B成为ORT直接持有的固定initializer，避免临时buffer释放/地址复用导致BT缓存错误。随后只移除从图输出不可达的节点、initializer及value_info，不改变保留算子的算术顺序。",
              "", "该权重预处理也是一项图优化；如果CPU基线运行原图、PL运行预处理选择图，两个结果的差异同时包含权重预处理和算子加速收益，不能全部归因PL硬件。尚未提供同样折叠预处理后的纯CPU基线。", "",
              "完整逐节点shape、锚点文件、速度比、选择理由和源/目标SHA256见 [selective_graph_manifest.json](selective_graph_manifest.json)。原始profile及实际输入shape见 [host_profile_metadata.json](host_profile_metadata.json)。"]
    (BASE / "SELECTION_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def convert_graph(model, nodes, output_path, pl_softmax=False,
                  resident_cache_bytes=RESIDENT_CACHE_CAPACITY_BYTES):
    initializers = {value.name: value for value in model.graph.initializer}
    producers = {name: node for node in model.graph.node for name in node.output}
    by_index = {node["node_index"]: node for node in nodes}
    # Keep static INT8 codes in their original [N,K] initializer layout.  The
    # custom op's constant_b=2 tells the runtime that B is already hardware
    # layout, avoiding a four-thread NEON transpose on every invocation.
    pretransposed_cache = {}
    pretransposed_nodes = 0
    resident_slots, resident_by_initializer = plan_resident_cache(nodes, resident_cache_bytes)
    for index, node in enumerate(model.graph.node):
        metadata = by_index.get(index)
        if not metadata or metadata["selection"]["target"] != "PL":
            continue
        if len(node.input) != 4:
            raise ValueError("Unexpected MatMulInteger inputs: " + node.name)
        az = scalar_constant(node.input[2], initializers, producers)
        bz = scalar_constant(node.input[3], initializers, producers)
        if az.dtype != np.uint8 or int(az.reshape(-1)[0]) != 128 or bz.dtype != np.int8 or int(bz.reshape(-1)[0]) != 0:
            raise ValueError("Unsupported MatMulInteger zero points: " + node.name)
        if metadata["constant_b"]:
            transpose = producers[node.input[1]]
            source = initializers[transpose.input[0]]
            attrs = {attribute.name: helper.get_attribute_value(attribute) for attribute in transpose.attribute}
            perm = tuple(attrs.get("perm", list(range(len(source.dims) - 1, -1, -1))))
            if len(source.dims) != 2 or perm != (1, 0):
                raise ValueError("Selected static B must be a 2-D Transpose([1,0]): " + node.name)
            key = source.name
            pretransposed_cache[key] = source.name
            node.input[1] = source.name
            metadata["pretransposed_initializer"] = source.name
            metadata["pretransposed_source_shape"] = list(source.dims)
            metadata["b_layout"] = "hardware_NK"
            cache_slot = resident_by_initializer.get(source.name)
            cache_id = int(cache_slot["cache_id"]) if cache_slot else -1
            metadata["resident_cache_id"] = cache_id
            if cache_slot:
                metadata["resident_cache_offset_bytes"] = int(cache_slot["offset_bytes"])
            pretransposed_nodes += 1
        else:
            cache_id = -1
            metadata["resident_cache_id"] = -1
        del node.input[2:]
        node.domain = "pynq"
        # Explicit metadata also makes the lifetime decision reviewable. The
        # direct initializer ensures ORT recognizes constant B without it.
        node.attribute.append(helper.make_attribute("constant_b", 2 if metadata["constant_b"] else 0))
        node.attribute.append(helper.make_attribute("resident_cache_id", cache_id))
    pl_softmax_nodes = 0
    if pl_softmax:
        value_shapes = {value.name: value_shape(value) for value in
                        list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info)}
        opsets = {value.domain: value.version for value in model.opset_import}
        for node in model.graph.node:
            if node.op_type != "Softmax":
                continue
            attrs = {attribute.name: helper.get_attribute_value(attribute)
                     for attribute in node.attribute}
            axis = int(attrs.get("axis", -1 if opsets.get("", 0) >= 13 else 1))
            shape = value_shapes.get(node.input[0])
            if not shape or len(shape) < 2:
                raise ValueError("Softmax input rank must be statically known: " + node.name)
            if axis not in (-1, len(shape) - 1):
                raise ValueError("PL Softmax only supports the last axis: " + node.name)
            node.domain = "pynq"
            pl_softmax_nodes += 1
    if not any(value.domain == "pynq" for value in model.opset_import):
        model.opset_import.append(helper.make_opsetid("pynq", 1))

    before_nodes = len(model.graph.node)
    before_initializers = len(model.graph.initializer)
    before_info = len(model.graph.value_info)
    # Backwards reachability changes no live operation or ordering. It removes
    # the now-unused Transpose(codes) nodes and duplicate old weight storage.
    required = {value.name for value in model.graph.output}
    live = []
    for node in reversed(model.graph.node):
        if any(name in required for name in node.output):
            live.append(node)
            required.update(name for name in node.input if name)
    live.reverse()
    del model.graph.node[:]
    model.graph.node.extend(live)
    kept_initializers = [value for value in model.graph.initializer if value.name in required]
    del model.graph.initializer[:]
    model.graph.initializer.extend(kept_initializers)
    kept_info = [value for value in model.graph.value_info if value.name in required]
    del model.graph.value_info[:]
    model.graph.value_info.extend(kept_info)
    onnx.save_model(model, str(output_path))
    onnx.checker.check_model(str(output_path))
    return dict(pretransposed_static_b_nodes=pretransposed_nodes,
                unique_pretransposed_int8_weights=len(pretransposed_cache),
                pretransposed_static_b_lifetime="direct_ORT_initializer",
                pl_softmax_nodes=pl_softmax_nodes,
                cpu_softmax_nodes=sum(node.op_type == "Softmax" for node in model.graph.node)-pl_softmax_nodes,
                softmax_cma_buffer_elements=(2 * 1024**2 if pl_softmax else 1024**2),
                resident_cache_capacity_bytes=resident_cache_bytes,
                resident_cache_slots=resident_slots,
                resident_cache_expected_hits_per_run=sum(max(0, slot["uses"] - 1)
                                                         for slot in resident_slots),
                static_int8_transpose_removed=True,
                dead_nodes_removed=before_nodes-len(model.graph.node),
                dead_initializers_removed=before_initializers-len(model.graph.initializer),
                dead_value_info_removed=before_info-len(model.graph.value_info),
                output_bytes=output_path.stat().st_size,
                output_sha256=digest(output_path), checker_passed=True)


def plan_resident_cache(nodes, capacity_bytes=RESIDENT_CACHE_CAPACITY_BYTES):
    """Pin frequently reused static [N,K] weights in a dedicated CMA pool.

    Cache IDs are stable within the generated graph. Every selected node using
    the same original initializer receives the same ID, so the first callback
    copies that weight to CMA and later callbacks reuse the same device address.
    """
    if not isinstance(capacity_bytes, int) or isinstance(capacity_bytes, bool) or capacity_bytes < 0:
        raise ValueError("Resident cache capacity must be a nonnegative integer byte count")
    uses = {}
    for node in nodes:
        if node.get("selection", {}).get("target") != "PL" or not node.get("constant_b"):
            continue
        name = node.get("weight_initializer")
        shape = node.get("weight_shape")
        if not name or not shape or len(shape) != 2:
            continue
        n, k = map(int, shape)
        padded_n, padded_k = (n + 31) // 32 * 32, (k + 31) // 32 * 32
        row = uses.setdefault(name, dict(initializer=name, shape=[n, k],
                                         padded_shape=[padded_n, padded_k],
                                         size_bytes=padded_n * padded_k, uses=0))
        if row["shape"] != [n, k]:
            raise ValueError("One initializer is referenced with incompatible B layouts: " + name)
        row["uses"] += 1

    # A resident copy is useful only when the same static B is consumed more
    # than once. Prioritize reuse count per CMA byte, then the bytes saved.
    candidates = [row for row in uses.values() if row["uses"] > 1]
    candidates.sort(key=lambda row: (row["uses"] - 1,
                                     (row["uses"] - 1) * row["size_bytes"],
                                     -row["size_bytes"]), reverse=True)
    slots = []
    by_initializer = {}
    offset = 0
    for row in candidates:
        size = int(row["size_bytes"])
        offset = (offset + 63) // 64 * 64
        if offset + size > capacity_bytes:
            continue
        slot = dict(cache_id=len(slots), initializer=row["initializer"],
                    shape=row["shape"], padded_shape=row["padded_shape"],
                    offset_bytes=offset, size_bytes=size, uses=row["uses"],
                    expected_hits_per_run=row["uses"] - 1)
        slots.append(slot)
        by_initializer[row["initializer"]] = slot
        offset += size
    for node in nodes:
        if node.get("selection", {}).get("target") != "PL" or not node.get("constant_b"):
            continue
        slot = by_initializer.get(node.get("weight_initializer"))
        node["resident_cache_id"] = int(slot["cache_id"]) if slot else -1
        if slot:
            node["resident_cache_offset_bytes"] = int(slot["offset_bytes"])
        else:
            node.pop("resident_cache_offset_bytes", None)
    return slots, by_initializer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DIR / "smolvla_w8a8_fp32.onnx")
    parser.add_argument("--inputs", type=Path, default=BASE / "validation_0.npz")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force-profile", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--pl-softmax", action="store_true",
                        help="Build a separate graph with last-axis Softmax nodes on PL")
    parser.add_argument("--resident-cache-mib", type=int,
                        default=RESIDENT_CACHE_CAPACITY_BYTES // 1024**2,
                        help="Resident static-B CMA cache capacity in MiB (default: 64; 0 disables it)")
    args = parser.parse_args()
    if args.resident_cache_mib < 0:
        parser.error("--resident-cache-mib must be nonnegative")
    resident_cache_bytes = args.resident_cache_mib * 1024**2
    variant_name = "smolvla_selective_pynq" + ("_softmax" if args.pl_softmax else "")
    # A cache sweep must not silently replace the established default graph.
    if resident_cache_bytes != RESIDENT_CACHE_CAPACITY_BYTES:
        variant_name += "_resident{}".format(args.resident_cache_mib)
    if args.output is None:
        args.output = MODEL_DIR / (variant_name + ".onnx")
    if args.output_manifest is None:
        args.output_manifest = args.output.with_suffix(".manifest.json")
    BASE.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = BASE / "host_profile_metadata.json"
    metadata = None
    if metadata_path.exists() and not args.force_profile:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["source_sha256"] != digest(args.model) or metadata["inputs_sha256"] != digest(args.inputs):
            raise RuntimeError("Cached host profile belongs to different source/input; use --force-profile")
    if metadata is None:
        metadata = run_host_profile(args.model, args.inputs, metadata_path)
    model = onnx.load(str(args.model))
    nodes = annotate_shapes(inspect_model(model), metadata)
    node_metadata = dict(source=str(args.model.resolve()),
                         source_sha256=metadata["source_sha256"], nodes=nodes)
    write_json(BASE / "gemm_node_metadata.json", node_metadata)
    summary = Counter((row.get("M"), row.get("N"), row.get("K"), row["constant_b"])
                      for row in nodes)
    print("SHAPES", json.dumps([dict(M=k[0], N=k[1], K=k[2], constant_b=k[3], nodes=v)
                                for k, v in summary.items()]), flush=True)
    if args.profile_only:
        return
    anchors = load_anchors()
    for node in nodes:
        node["selection"] = choose_node(node, anchors)
    counts = Counter(node["selection"]["target"] for node in nodes)
    manifest = dict(source_model=str(args.model.resolve()),
                    source_bytes=args.model.stat().st_size,
                    source_sha256=metadata["source_sha256"], source_preserved=True,
                    input_schema=[dict(name=value.name, shape=value_shape(value)) for value in model.graph.input],
                    total_matmul_nodes=len(nodes), pl_matmul_nodes=counts["PL"],
                    cpu_ort_matmul_nodes=counts["CPU_ORT"],
                    softmax_nodes_preserved=sum(node.op_type == "Softmax" for node in model.graph.node),
                    profiled_shape_nodes=sum(node["profile_resolved"] for node in nodes),
                    static_b_nodes=sum(node["constant_b"] for node in nodes),
                    dynamic_b_nodes=sum(not node["constant_b"] for node in nodes),
                    rule=dict(evidence="Only measured constant_b=False full paths; no cache-hit benefit assumed",
                              minimum_speedup_aligned=1.10, minimum_speedup_unaligned=1.20,
                              dimension_rule="pad32(M/N/K) must each be >= the measured winning anchor",
                              b_2d_rows="flatten all A leading dimensions into M",
                              b_batched_rows="per-matrix M,N,K; head/batch count never changes selection",
                              unknown_shapes="CPU_ORT", all_softmax="CPU_ORT",
                              anchor_lookup="nearest componentwise dominated measured winning shape"),
                    measured_evidence=EVIDENCE_FILES,
                    fairness_note="Hybrid graph also folds exact INT8 static weight Transposes; CPU baseline on original graph includes that preprocessing difference",
                    host_profile_metadata=metadata_path.name,
                    host_profile_inference_seconds_not_board=metadata["inference_seconds"],
                    resident_cache_capacity_bytes=resident_cache_bytes,
                    shape_decisions=selection_summary(nodes),
                    nodes=nodes, conversion_performed=False)
    print("SELECTION_PLAN", json.dumps({key:manifest[key] for key in
          ("total_matmul_nodes", "pl_matmul_nodes", "cpu_ort_matmul_nodes", "profiled_shape_nodes", "static_b_nodes", "dynamic_b_nodes")}), flush=True)
    if not args.plan_only:
        manifest.update(convert_graph(model, nodes, args.output, pl_softmax=args.pl_softmax,
                                      resident_cache_bytes=resident_cache_bytes))
        if digest(args.model) != metadata["source_sha256"]:
            raise RuntimeError("Source model changed during conversion")
        manifest.update(conversion_performed=True, output_model=str(args.output.resolve()))
    manifest["pl_softmax_enabled"] = bool(args.pl_softmax)
    write_json(args.output_manifest, manifest)
    if not args.pl_softmax and resident_cache_bytes == RESIDENT_CACHE_CAPACITY_BYTES:
        write_selection_summary(manifest)
    print("SELECTIVE_MANIFEST", str(args.output_manifest), flush=True)


if __name__ == "__main__":
    main()
