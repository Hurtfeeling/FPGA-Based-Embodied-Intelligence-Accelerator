#!/usr/bin/env python
"""Initialize S8 from a frozen teacher and train FP32 distillation with optional DDP."""

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.distillation import (
    SmolVLADistiller,
    configure_student_gradients,
    create_s8_assets,
    file_sha256,
    initialize_s8,
    load_teacher,
    save_student,
    velocity_forward,
)
from lerobot.policies.smolvla.distillation_distributed import (
    DistributedRun,
    ReproducibleBatches,
    capture_rng,
    metric_statistics,
    restore_rng,
    state_sha256,
    summarize_statistics,
)
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy


def datasets(checkpoint, policy, override_root=None):
    training = json.loads((checkpoint / "train_config.json").read_text())
    info = training["dataset"]
    root = override_root or info["root"]
    if not (Path(root) / "meta" / "info.json").exists():
        raise FileNotFoundError(
            f"Dataset not found at {root}; provide its current location with --dataset-root"
        )
    full = LeRobotDataset(info["repo_id"], root=root, video_backend="pyav")
    if set(full.meta.camera_keys) != {"observation.images.wrist", "observation.images.beside"}:
        raise ValueError("S8 training requires exactly the wrist/beside dataset")
    tasks = full.meta.episodes["tasks"]
    grouped = {}
    excluded = set(info.get("exclude_episodes") or [])
    for ep in info.get("episodes") or range(full.num_episodes):
        if ep not in excluded:
            grouped.setdefault(tasks[ep][0] if tasks[ep] else "", []).append(ep)
    train, validation = [], []
    fraction = info.get("eval_split", 0.2)
    if not 0 < fraction < 1:
        raise ValueError("Need a nonempty episode holdout")
    for episodes in grouped.values():
        count = math.ceil(len(episodes) * fraction)
        if count >= len(episodes):
            raise ValueError("A task has no remaining training episodes")
        train.extend(episodes[:-count])
        validation.extend(episodes[-count:])
    if set(train) & set(validation):
        raise ValueError("Training and evaluation episodes overlap")
    delta = {"action": [i / full.fps for i in range(policy.config.chunk_size)]}
    del full
    common = {"repo_id": info["repo_id"], "root": root, "delta_timestamps": delta, "video_backend": "pyav"}
    return (
        LeRobotDataset(**common, episodes=sorted(train)),
        LeRobotDataset(**common, episodes=sorted(validation)),
        {
            "dataset": {**info, "root": str(root)},
            "train_episodes": sorted(train),
            "validation_episodes": sorted(validation),
            "eval_split": fraction,
        },
    )


def sample_flow(student, batch, generator):
    actions = student.prepare_action(batch)
    noise = torch.randn(actions.shape, dtype=torch.float32, device=actions.device, generator=generator)
    # Uniform flow times cover the complete teacher/student interpolation path.
    times = torch.rand(actions.shape[0], device=actions.device, generator=generator) * 0.998 + 0.001
    return noise, times


@torch.no_grad()
def evaluate(distiller, loader, preprocessor, run):
    distiller.eval()
    generator = torch.Generator(device=run.device).manual_seed(1701 + run.rank)
    statistics = torch.zeros(6, device=run.device, dtype=torch.float64)
    for raw in loader:
        batch = preprocessor(raw)
        noise, times = sample_flow(distiller.student, batch, generator)
        _, metrics = distiller(batch, noise, times)
        statistics += metric_statistics(metrics)
    summary = summarize_statistics(statistics, distiller.weights)
    distiller.train()
    return summary


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--vlm-assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=300)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--demo-weight", type=float, default=1.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--feature-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume", type=Path, help="Checkpoint directory containing pretrained_model and training_state.pt"
    )
    parser.add_argument("--init-only", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Two real-data updates, validation, and inference reload check",
    )
    args = parser.parse_args()
    if args.smoke_test:
        args.steps, args.batch_size = 2, 1
        args.grad_accum = 2 if int(os.environ.get("WORLD_SIZE", "1")) > 1 else 1
        args.eval_every, args.eval_batches, args.save_every = 2, 2, 2
    if (
        min(args.steps, args.batch_size, args.grad_accum, args.eval_every, args.eval_batches, args.save_every)
        <= 0
    ):
        parser.error("Step/batch counts must be positive")
    if args.workers < 0 or args.warmup_steps < 0 or args.lr <= 0:
        parser.error("Invalid worker count, warmup or learning rate")
    if args.resume and args.init_only:
        parser.error("--resume and --init-only are mutually exclusive")
    return args


def train(args, run):
    args.device = str(run.device)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    torch.set_num_threads(8)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    args.teacher, args.output = args.teacher.resolve(), args.output.resolve()

    def prepare_output():
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise FileExistsError("Output is nonempty; choose a new directory or --resume")
        args.output.mkdir(parents=True, exist_ok=True)

    run.primary_call(prepare_output)
    source_hash = run.primary_call(lambda: file_sha256(args.teacher / "model.safetensors"))
    teacher = load_teacher(args.teacher, args.vlm_assets, args.device)
    assets = args.output / "student_assets"
    run.primary_call(lambda: create_s8_assets(args.vlm_assets, assets) and None)
    state = None
    if args.resume:
        resume = args.resume.resolve()
        state = torch.load(resume / "training_state.pt", map_location="cpu", weights_only=True)
        if state["teacher_sha256"] != source_hash:
            raise ValueError("Resume teacher differs from the original teacher")
        if state.get("world_size", 1) != run.world_size:
            raise ValueError("Resume GPU count changed; use the original world size")
        for key in (
            "steps",
            "batch_size",
            "grad_accum",
            "seed",
            "lr",
            "warmup_steps",
            "demo_weight",
            "kd_weight",
            "feature_weight",
        ):
            if state["arguments"][key] != getattr(args, key):
                raise ValueError(f"Resume argument changed: {key}")
        student = SmolVLAPolicy.from_pretrained(resume / "pretrained_model", local_files_only=True).float()
        configure_student_gradients(student)
        initialization = state["initialization"]
    else:
        student, initialization = initialize_s8(teacher, assets)
    student.to(device=args.device, dtype=torch.float32)
    student.config.device = args.device
    distiller = SmolVLADistiller(student, teacher, args.demo_weight, args.kd_weight, args.feature_weight).to(
        args.device
    )
    train_dataset, eval_dataset, split = datasets(args.teacher, student, args.dataset_root)
    run.primary_call(
        lambda: (args.output / "episode_split.json").write_text(json.dumps(split, indent=2) + "\n")
    )
    pre, _ = make_pre_post_processors(
        student.config,
        str(args.teacher),
        preprocessor_overrides={"device_processor": {"device": args.device}},
    )
    parameters = [p for p in distiller.parameters() if p.requires_grad]
    teacher_ids = {id(p) for p in teacher.parameters()}
    if any(id(p) in teacher_ids for p in parameters):
        raise ValueError("Teacher parameter entered optimizer")
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in parameters if p.ndim > 1], "weight_decay": 1e-4},
            {"params": [p for p in parameters if p.ndim <= 1], "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 100 + run.rank)
    start_step, best = 0, float("inf")
    if state is not None:
        if state["episode_split"] != split:
            raise ValueError("Resume episode split changed")
        optimizer.load_state_dict(state["optimizer"])
        distiller.adapters.load_state_dict(state["adapters"])
        if "rank_rng_states" in state:
            restore_rng(state["rank_rng_states"][run.rank], generator, run.device)
        else:
            generator.set_state(state["flow_rng_state"])
        start_step, best = state["step"], state["best_eval_demo"]
        if start_step >= args.steps:
            raise ValueError("Resume checkpoint has already reached the requested step count")
    arguments_dict = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    manifest = {
        "architecture": "S8",
        "teacher_checkpoint": str(args.teacher),
        "teacher_sha256": source_hash,
        "initialization": initialization,
        "episode_split": split,
        "arguments": arguments_dict,
        "distributed": {
            "world_size": run.world_size,
            "per_gpu_batch_size": args.batch_size,
            "gradient_accumulation": args.grad_accum,
            "effective_batch_size": args.batch_size * args.grad_accum * run.world_size,
        },
        "floating_weights_dtype": "float32",
        "amp_enabled": False,
        "tf32_enabled": False,
        "training_kind": "FP32 action/feature distillation; no QAT yet",
        "deployment_size_budget_mb": [176, 186],
        "deployment_budget_requires_int8_embedding": True,
    }
    run.primary_call(
        lambda: (args.output / "run_config.json").write_text(json.dumps(manifest, indent=2) + "\n")
    )

    def write_checkpoint(step, rank_rng_states, is_best=False):
        folder = args.output / "checkpoints" / f"{step:06d}"
        if folder.exists():
            raise FileExistsError(folder)
        staging = folder.with_name(folder.name + ".incomplete")
        if staging.exists():
            staging.rename(staging.with_name(staging.name + f".abandoned_{time.time_ns()}"))
        staging.mkdir(parents=True, exist_ok=False)
        metadata = {
            **manifest,
            "optimizer_updates": step,
            "smoke_test": args.smoke_test,
            "deployment_quality_evaluated": False,
            "best_eval_demo": best,
        }
        save_student(
            student,
            staging / "pretrained_model",
            args.teacher,
            assets,
            metadata,
            published_path=folder / "pretrained_model",
        )
        provenance = json.loads((args.teacher / "train_config.json").read_text())
        provenance["policy"] = json.loads((staging / "pretrained_model" / "config.json").read_text())
        provenance["dataset"] = split["dataset"]
        (staging / "pretrained_model" / "train_config.json").write_text(
            json.dumps(provenance, indent=2) + "\n"
        )
        torch.save(
            {
                "step": step,
                "optimizer": optimizer.state_dict(),
                "adapters": distiller.adapters.state_dict(),
                "flow_rng_state": generator.get_state(),
                "rank_rng_states": rank_rng_states,
                "world_size": run.world_size,
                "teacher_sha256": source_hash,
                "best_eval_demo": best,
                "arguments": arguments_dict,
                "initialization": initialization,
                "episode_split": split,
            },
            staging / "training_state.pt",
        )
        staging.rename(folder)
        for link in ["last", "best"] if is_best else ["last"]:
            target = folder.parent / link
            temporary = target.with_name(link + ".tmp")
            temporary.unlink(missing_ok=True)
            temporary.symlink_to(folder.name, target_is_directory=True)
            temporary.replace(target)
        return folder

    def checkpoint(step, is_best=False):
        rng_states = run.gather(capture_rng(generator, run.device))
        return run.primary_call(lambda: write_checkpoint(step, rng_states, is_best))

    if run.primary:
        print(
            json.dumps(
                {
                    "student_parameters": sum(p.numel() for p in student.parameters()),
                    "trainable_parameters": sum(p.numel() for p in parameters),
                    "train_episodes": len(split["train_episodes"]),
                    "eval_episodes": len(split["validation_episodes"]),
                    "teacher_frozen": all(not p.requires_grad for p in teacher.parameters()),
                    **manifest["distributed"],
                }
            ),
            flush=True,
        )
    if args.init_only:
        checkpoint(0)
        if run.primary:
            print("S8 initialized and saved; no optimizer updates", flush=True)
        return
    train_model = (
        DistributedDataParallel(
            distiller,
            device_ids=[run.device.index],
            broadcast_buffers=False,
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
        if run.world_size > 1
        else distiller
    )
    sampler = ReproducibleBatches(
        len(train_dataset),
        args.batch_size,
        (args.steps - start_step) * args.grad_accum,
        start_step * args.grad_accum,
        args.seed,
        run.rank,
        run.world_size,
    )
    worker_options = {"multiprocessing_context": "spawn", "persistent_workers": True} if args.workers else {}
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=sampler,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(args.seed),
        **worker_options,
    )
    eval_count = min(len(eval_dataset), args.eval_batches * args.batch_size)
    eval_indices = np.linspace(0, len(eval_dataset) - 1, eval_count, dtype=int).tolist()
    eval_loader = DataLoader(
        Subset(eval_dataset, eval_indices[run.rank :: run.world_size]),
        batch_size=args.batch_size,
        num_workers=0,
    )
    iterator = iter(train_loader)
    distiller.train()
    last_batch = None
    final = None
    for step in range(start_step + 1, args.steps + 1):
        begin = time.monotonic()
        warmup = min(args.warmup_steps, args.steps)
        multiplier = (
            step / max(1, warmup)
            if step <= warmup
            else (0.1 + 0.9 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))) / 2)
        )
        for group in optimizer.param_groups:
            group["lr"] = args.lr * multiplier
        optimizer.zero_grad(set_to_none=True)
        statistics = torch.zeros(6, device=run.device, dtype=torch.float64)
        for micro in range(args.grad_accum):
            batch = pre(next(iterator))
            noise, times = sample_flow(student, batch, generator)
            sync_context = (
                train_model.no_sync() if run.world_size > 1 and micro < args.grad_accum - 1 else nullcontext()
            )
            with sync_context:
                loss, metrics = train_model(batch, noise, times, distributed_loss=run.world_size > 1)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite distillation loss")
                (loss / args.grad_accum).backward()
            statistics += metric_statistics(metrics)
            last_batch = batch
        norm = torch.nn.utils.clip_grad_norm_(parameters, 10.0, error_if_nonfinite=True)
        if any(p.grad is not None for p in teacher.parameters()):
            raise ValueError("Teacher received a gradient")
        optimizer.step()
        if step == 1 or step % 10 == 0 or args.smoke_test:
            summary = summarize_statistics(statistics, distiller.weights)
            if run.primary:
                print(
                    json.dumps(
                        {
                            "step": step,
                            **summary,
                            "gradient_norm": float(norm),
                            "lr": optimizer.param_groups[0]["lr"],
                            "seconds": time.monotonic() - begin,
                        }
                    ),
                    flush=True,
                )
        evaluation, improved = None, False
        if step % args.eval_every == 0 or step == args.steps:
            evaluation = evaluate(distiller, eval_loader, pre, run)
            improved = evaluation["demo"] < best
            best = min(best, evaluation["demo"])

            def write_evaluation(current_step=step, values=evaluation):
                with (args.output / "evaluation.jsonl").open("a") as stream:
                    stream.write(json.dumps({"step": current_step, **values}) + "\n")
                print("Evaluation", {"step": current_step, **values}, flush=True)

            run.primary_call(write_evaluation)
        if step % args.save_every == 0 or step == args.steps or improved:
            final = checkpoint(step, improved)
    if run.primary_call(lambda: file_sha256(args.teacher / "model.safetensors")) != source_hash:
        raise ValueError("Source teacher checkpoint changed")
    if args.smoke_test:
        hashes = run.gather({"student": state_sha256(student), "adapters": state_sha256(distiller.adapters)})
        if any(value != hashes[0] for value in hashes):
            raise ValueError("DDP student or feature adapters diverged across ranks")

    def smoke_inference():
        student.eval()
        fixed_noise = torch.randn(1, 50, 32, generator=generator, device=args.device)
        with torch.no_grad():
            expected = student.predict_action_chunk(last_batch, noise=fixed_noise)
        reloaded = (
            SmolVLAPolicy.from_pretrained(final / "pretrained_model", local_files_only=True)
            .to(args.device)
            .float()
            .eval()
        )
        original_state = student.state_dict()
        for name, value in reloaded.state_dict().items():
            torch.testing.assert_close(
                value, original_state[name], rtol=0, atol=0, msg=f"Reloaded weight: {name}"
            )
        fixed_time = torch.full((1,), 0.5, device=args.device)
        with torch.no_grad():
            v_before, _ = velocity_forward(student, last_batch, fixed_noise, fixed_time)
            v_after, _ = velocity_forward(reloaded, last_batch, fixed_noise, fixed_time)
        print("Reload velocity max error", float((v_before - v_after).abs().max()), flush=True)
        print(
            "Vision attention implementations",
            student.model.vlm_with_expert.vlm.model.vision_model.config._attn_implementation,
            reloaded.model.vlm_with_expert.vlm.model.vision_model.config._attn_implementation,
            flush=True,
        )
        with torch.no_grad():
            actual = reloaded.predict_action_chunk(last_batch, noise=fixed_noise)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
        assert actual.shape == (1, 50, 6) and torch.isfinite(actual).all()
        smoke = {
            "optimizer_updates": 2,
            "forward_backward_passed": True,
            "held_out_evaluation_passed": True,
            "save_reload_inference_passed": True,
            "output_shape": list(actual.shape),
            "teacher_source_unchanged": True,
            "teacher_gradients_absent": True,
            "student_dtype": str(actual.dtype),
            "checkpoint": str(final),
            "world_size": run.world_size,
            "ranks_in_sync": True,
            "gradient_accumulation": args.grad_accum,
        }
        (args.output / "smoke_test_report.json").write_text(json.dumps(smoke, indent=2) + "\n")
        print(json.dumps(smoke, indent=2), flush=True)

    if args.smoke_test:
        run.primary_call(smoke_inference)
    if run.primary:
        print("Distillation run completed", args.output, flush=True)


def main():
    args = arguments()
    run = DistributedRun.start(args.device)
    try:
        train(args, run)
    finally:
        run.close()


if __name__ == "__main__":
    main()
