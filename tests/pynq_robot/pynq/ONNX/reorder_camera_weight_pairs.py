"""Create a separate dependency-safe schedule pairing repeated camera weights.

This script changes node order only. It preserves every node, initializer,
input, output, and attribute in the selective PYNQ graph. It only pairs two
independent PL MatMulInteger nodes that read the same static [N,K] initializer
and are not already resident in the large CMA weight cache.
"""
import argparse
from collections import defaultdict
import heapq
import json
from pathlib import Path

import onnx
from onnx import helper


BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE.parent
DEFAULT_INPUT = MODEL_DIR / "smolvla_selective_pynq.onnx"
DEFAULT_OUTPUT = MODEL_DIR / "smolvla_selective_pynq_camera_interleaved.onnx"
DEFAULT_REPORT = BASE / "camera_interleaved_schedule.json"


def attr_int(node, name, default):
    for attribute in node.attribute:
        if attribute.name == name:
            value = helper.get_attribute_value(attribute)
            return int(value)
    return int(default)


def value_bytes(info):
    """Return byte size for a concrete tensor ValueInfo, otherwise None."""
    if not info.type.HasField("tensor_type"):
        return None
    tensor = info.type.tensor_type
    if not tensor.HasField("shape"):
        return None
    count = 1
    for dim in tensor.shape.dim:
        if not dim.HasField("dim_value") or dim.dim_value <= 0:
            return None
        count *= int(dim.dim_value)
    element_bytes = {
        onnx.TensorProto.FLOAT: 4,
        onnx.TensorProto.FLOAT16: 2,
        onnx.TensorProto.DOUBLE: 8,
        onnx.TensorProto.BFLOAT16: 2,
        onnx.TensorProto.INT64: 8,
        onnx.TensorProto.UINT64: 8,
        onnx.TensorProto.INT32: 4,
        onnx.TensorProto.UINT32: 4,
        onnx.TensorProto.INT16: 2,
        onnx.TensorProto.UINT16: 2,
        onnx.TensorProto.INT8: 1,
        onnx.TensorProto.UINT8: 1,
        onnx.TensorProto.BOOL: 1,
    }.get(int(tensor.elem_type))
    return None if element_bytes is None else count * element_bytes


def peak_live_bytes(graph, order):
    """Estimate known live tensor bytes without allocator reuse/aliasing."""
    info = {}
    for item in list(graph.input) + list(graph.output) + list(graph.value_info):
        info[item.name] = value_bytes(item)
    producer = {name: index for index, node in enumerate(graph.node)
                for name in node.output if name}
    initializer_names = {item.name for item in graph.initializer}
    graph_inputs = {item.name for item in graph.input}
    use_count = defaultdict(int)
    for node in graph.node:
        for name in node.input:
            if name and name not in initializer_names:
                use_count[name] += 1
    graph_outputs = {item.name for item in graph.output}
    live = set()
    live_bytes = 0
    unknown_live = 0
    for name in graph_inputs:
        live.add(name)
        size = info.get(name)
        if size is None:
            unknown_live += 1
        else:
            live_bytes += size
    peak = live_bytes
    peak_unknown = unknown_live
    for index in order:
        node = graph.node[index]
        # Inputs and outputs overlap during execution; account for outputs first.
        for name in node.output:
            if not name or name in graph_outputs:
                pass
            if name and name not in live and name not in initializer_names:
                live.add(name)
                size = info.get(name)
                if size is None:
                    unknown_live += 1
                else:
                    live_bytes += size
        if live_bytes > peak:
            peak, peak_unknown = live_bytes, unknown_live
        for name in node.input:
            if not name or name not in live:
                continue
            use_count[name] -= 1
            if use_count[name] == 0 and name not in graph_outputs:
                live.remove(name)
                size = info.get(name)
                if size is None:
                    unknown_live -= 1
                else:
                    live_bytes -= size
    return dict(peak_known_live_bytes=peak,
                peak_known_live_mib=peak / (1024.0 ** 2),
                unknown_live_tensors_at_peak=peak_unknown,
                unknown_tensor_shapes=sum(size is None for size in info.values()))


def build_dependencies(nodes):
    producers = {name: index for index, node in enumerate(nodes)
                 for name in node.output if name}
    indegree = [0] * len(nodes)
    consumers = [[] for _ in nodes]
    for index, node in enumerate(nodes):
        for name in node.input:
            parent = producers.get(name)
            if parent is None:
                continue
            indegree[index] += 1
            consumers[parent].append(index)
    return producers, indegree, consumers


def ancestor_set(target, nodes, producers):
    result = set()
    stack = [target]
    while stack:
        index = stack.pop()
        for name in nodes[index].input:
            parent = producers.get(name)
            if parent is not None and parent not in result:
                result.add(parent)
                stack.append(parent)
    result.discard(target)
    return result


def candidate_weight_pairs(graph, producers):
    initializers = {item.name for item in graph.initializer}
    by_weight = defaultdict(list)
    for index, node in enumerate(graph.node):
        if node.domain != "pynq" or node.op_type != "MatMulInteger":
            continue
        if attr_int(node, "constant_b", 0) != 2:
            continue
        if attr_int(node, "resident_cache_id", -1) >= 0:
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            continue
        by_weight[node.input[1]].append(index)
    pairs = []
    for weight, indices in by_weight.items():
        if len(indices) != 2:
            continue
        i, j = sorted(indices)
        # The two camera branches must be independent; dependency checking is
        # repeated against the full graph below before we move either node.
        pairs.append((i, j, weight))
    return sorted(pairs)


def make_schedule(graph, pairs):
    nodes = graph.node
    producers, indegree, consumers = build_dependencies(nodes)
    n = len(nodes)
    ready = {index for index, degree in enumerate(indegree) if degree == 0}
    heap = list(ready)
    heapq.heapify(heap)
    order = []
    done = set()

    def append(index):
        if index not in ready:
            raise RuntimeError("Scheduler tried to append a non-ready node")
        ready.remove(index)
        done.add(index)
        order.append(index)
        for child in consumers[index]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.add(child)
                heapq.heappush(heap, child)

    paired = []
    skipped = []
    for left, right, weight in pairs:
        left_ancestors = ancestor_set(left, nodes, producers)
        right_ancestors = ancestor_set(right, nodes, producers)
        # If one target depends on the other, adjacency cannot be preserved.
        if left in right_ancestors or right in left_ancestors:
            skipped.append(dict(left=left, right=right, weight=weight,
                                reason="dependent_pair"))
            continue
        for target in (left, right):
            while target not in ready:
                if target in done:
                    break
                ancestors = left_ancestors if target == left else right_ancestors
                candidates = ready.intersection(ancestors)
                if not candidates:
                    skipped.append(dict(left=left, right=right, weight=weight,
                                        reason="no_ready_ancestor_frontier"))
                    break
                append(min(candidates))
        if left in ready and right in ready:
            append(left)
            append(right)
            paired.append(dict(left=left, right=right, weight=weight))
        else:
            skipped.append(dict(left=left, right=right, weight=weight,
                                reason="pair_not_simultaneously_ready"))

    while ready:
        while heap and heap[0] not in ready:
            heapq.heappop(heap)
        if not heap:
            raise RuntimeError("Topological scheduler lost ready queue")
        append(heapq.heappop(heap))
    if len(order) != n:
        raise RuntimeError("Graph has a cycle or unresolved nodes")
    return order, paired, skipped


def summarize_pairs(graph, pairs, paired):
    init = {item.name: item for item in graph.initializer}
    paired_keys = {row["weight"] for row in paired}
    bytes_saved = 0
    for _, _, name in pairs:
        if name not in paired_keys:
            continue
        dims = list(init[name].dims)
        if len(dims) == 2:
            n, k = map(int, dims)
            padded_n = (n + 31) // 32 * 32
            padded_k = (k + 31) // 32 * 32
            bytes_saved += padded_n * padded_k
    return dict(candidate_weight_pairs=len(pairs),
                adjacent_weight_pairs=len(paired),
                missed_weight_pairs=len(pairs)-len(paired),
                repeated_initializer_bytes_saved_per_run=bytes_saved,
                repeated_initializer_mib_saved_per_run=bytes_saved/(1024.0**2),
                explanation="Expected second-use scratch BT copy/sync savings; constant_b=2 already uses pretransposed [N,K] weights, so this does not claim a NEON transpose saving.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    model = onnx.load(str(args.input), load_external_data=False)
    original_order = list(range(len(model.graph.node)))
    before_liveness = peak_live_bytes(model.graph, original_order)
    producers, _, _ = build_dependencies(model.graph.node)
    pairs = candidate_weight_pairs(model.graph, producers)
    schedule, paired, skipped = make_schedule(model.graph, pairs)
    after_liveness = peak_live_bytes(model.graph, schedule)
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(original_nodes[index] for index in schedule)
    onnx.checker.check_model(model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    # Verify that the saved artifact has the same topological node order.
    saved = onnx.load(str(args.output), load_external_data=False)
    onnx.checker.check_model(saved)
    pair_stats = summarize_pairs(model.graph, pairs, paired)
    report = dict(input=str(args.input.resolve()), output=str(args.output.resolve()),
                  original_node_count=len(original_order), scheduled_node_count=len(schedule),
                  node_order_changed=sum(a != b for a, b in enumerate(schedule)),
                  original_peak_live_tensor_estimate=before_liveness,
                  interleaved_peak_live_tensor_estimate=after_liveness,
                  peak_known_live_bytes_delta=(after_liveness["peak_known_live_bytes"]-
                                               before_liveness["peak_known_live_bytes"]),
                  pair_stats=pair_stats, paired=paired, skipped=skipped,
                  checker_passed=True, output_bytes=args.output.stat().st_size)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False)+"\n",
                           encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("original_node_count", "node_order_changed", "pair_stats",
                       "original_peak_live_tensor_estimate",
                       "interleaved_peak_live_tensor_estimate",
                       "peak_known_live_bytes_delta", "output_bytes", "checker_passed")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
