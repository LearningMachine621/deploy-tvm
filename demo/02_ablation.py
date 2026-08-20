"""② 四档消融：加速从哪来（kernel 库 × 融合 × schedule 乘法链）。

  C : PyTorch fp32 eager（cuDNN/cuBLAS）
  A0: TVM opt_level=0（不融合，TOPI kernel）
  A3: TVM opt_level=3（融合，TOPI kernel）
  B : TVM opt_level=3 + AutoScheduler 快速调优（全 task，trial 数远小于主项目）

乘法链（device-only）：C/B = (C/A0) × (A0/A3) × (A3/B)
  C/A0 = kernel 库质量（cuDNN/cuBLAS vs TOPI）
  A0/A3 = 纯融合收益（同一后端内）
  A3/B = schedule 搜索收益

注意：demo 模型很小、调优预算只有 30 trials/task，数字量级与主项目（README §3）
不同——demo 演示的是**方法**：同口径测量 + 乘法链分解 + 全 task 覆盖。

用法：python demo/02_ablation.py [--trials 30]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from common import INPUT_NAME, INPUT_SHAPE, ONNX_PATH, bench_cuda_event, check_gpu_idle, log, save_json

TRIALS_DEFAULT = 30


def build_tvm(opt_level: int, mod, params, target, dev, tuning_log=None):
    import tvm
    from tvm import auto_scheduler, relay
    from tvm.contrib import graph_executor

    if tuning_log:
        with auto_scheduler.ApplyHistoryBest(str(tuning_log)):
            with tvm.transform.PassContext(opt_level=3,
                                           config={"relay.backend.use_auto_scheduler": True}):
                lib = relay.build(mod, target=target, params=params)
    else:
        with tvm.transform.PassContext(opt_level=opt_level):
            lib = relay.build(mod, target=target, params=params)
    n_kernels = len(json.loads(lib.get_graph_json())["nodes"])  # 每推理发射 kernel 数（融合证据）
    m = graph_executor.GraphModule(lib["default"](dev))
    return m, n_kernels


def bench_tvm(m, x_np, dev):
    m.set_input(INPUT_NAME, x_np)
    for _ in range(30):
        m.run()
    dev.sync()
    groups = bench_cuda_event(m.run, groups=3, iters=200)
    return groups


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS_DEFAULT, help="每 task 调优 trial 数（主项目用 200）")
    args = ap.parse_args()

    import onnx
    import tvm
    from tvm import auto_scheduler, relay

    check_gpu_idle()
    target = tvm.target.Target("cuda")
    dev = tvm.cuda(0)
    x = torch.from_numpy(np.random.RandomState(1234).randn(*INPUT_SHAPE).astype("float32"))
    x_np = x.numpy()

    mod, params = relay.frontend.from_onnx(
        onnx.load(str(ONNX_PATH)), shape={INPUT_NAME: list(INPUT_SHAPE)}, freeze_params=True)

    # ---- C: PyTorch fp32 eager ----
    from common import CKPT, TinyFormer
    model = TinyFormer().eval().cuda()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    xc = x.cuda()
    with torch.no_grad():
        c_groups = bench_cuda_event(lambda: model(xc), groups=3, iters=200)
    log("BENCH", f"C  PyTorch fp32      : {np.mean(c_groups):.3f} ms  {np.round(c_groups, 4).tolist()}")

    # ---- A0 / A3 ----
    m0, k0 = build_tvm(0, mod, params, target, dev)
    a0_groups = bench_tvm(m0, x_np, dev)
    log("BENCH", f"A0 不融合(opt=0)    : {np.mean(a0_groups):.3f} ms  kernels={k0}")

    m3, k3 = build_tvm(3, mod, params, target, dev)
    a3_groups = bench_tvm(m3, x_np, dev)
    log("BENCH", f"A3 融合(opt=3)      : {np.mean(a3_groups):.3f} ms  kernels={k3}")

    # ---- B: tuned（全 task 快速调优；铁律：必须覆盖全部 task）----
    log("TUNE", f"extract tasks + {args.trials} trials/task（全 task 覆盖，主项目为 50 task × 200）")
    tasks, weights = auto_scheduler.extract_tasks(mod["main"], params, target)
    log("TUNE", f"extracted {len(tasks)} tasks")
    tuning_log = ONNX_PATH.parent / "cache" / "tinyformer_tuning.json"
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(min_repeat_ms=100, timeout=10)
    for i, t in enumerate(tasks):
        t.tune(TuningOptions := auto_scheduler.TuningOptions(
            num_measure_trials=args.trials,
            runner=measure_ctx.runner,
            measure_callbacks=[auto_scheduler.RecordToFile(str(tuning_log))],
        ))
    del measure_ctx

    mb, kb = build_tvm(3, mod, params, target, dev, tuning_log=tuning_log)
    b_groups = bench_tvm(mb, x_np, dev)
    log("BENCH", f"B  tuned(全task)     : {np.mean(b_groups):.3f} ms  kernels={kb}")

    # ---- 乘法链 ----
    c, a0, a3, b = map(np.mean, [c_groups, a0_groups, a3_groups, b_groups])
    chain = {
        "C_pytorch_ms": round(float(c), 4),
        "A0_unfused_ms": round(float(a0), 4),
        "A3_fused_ms": round(float(a3), 4),
        "B_tuned_ms": round(float(b), 4),
        "kernel_lib_C_over_A0": round(float(c / a0), 3),
        "fusion_A0_over_A3": round(float(a0 / a3), 3),
        "schedule_A3_over_B": round(float(a3 / b), 3),
        "total_C_over_B": round(float(c / b), 3),
        "n_kernels": {"A0": k0, "A3": k3, "B": kb},
        "groups_ms": {"C": c_groups, "A0": a0_groups, "A3": a3_groups, "B": b_groups},
        "trials_per_task": args.trials,
    }
    print("\n===== 四档消融（device-only，3 组 × 200 iters）=====")
    print(f"PyTorch fp32      {c:.3f} ms")
    print(f"TVM 不融合 opt0   {a0:.3f} ms   kernels {k0}")
    print(f"TVM 融合   opt3   {a3:.3f} ms   kernels {k3}（融合收益 ×{(a0 / a3):.3f}）")
    print(f"TVM tuned         {b:.3f} ms   kernels {kb}（schedule ×{(a3 / b):.3f}）")
    print(f"乘法链: C/B = {c / b:.3f} = {c / a0:.3f}(kernel库) × {a0 / a3:.3f}(融合) × {a3 / b:.3f}(schedule)")
    save_json("02_ablation.json", chain)


if __name__ == "__main__":
    main()
