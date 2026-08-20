"""公共工具：TinyFormer 模型、CUDA-event 基准（主项目同口径）、空闲 GPU 检查。"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

DEMO_DIR = Path(__file__).resolve().parent
RESULTS = DEMO_DIR / "results"
CACHE = DEMO_DIR / "cache"
RESULTS.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
CKPT = DEMO_DIR / "tinyformer.pt"
ONNX_PATH = DEMO_DIR / "tinyformer.onnx"
INPUT_NAME = "input"
INPUT_SHAPE = (1, 3, 64, 64)


class TinyFormer(torch.nn.Module):
    """迷你 transformer：conv patch embed + 2×(LN → MLP → GELU → LN → MLP) 残差块。

    刻意保留 LayerNorm / Linear / GELU / Conv 的组合 —— 与主项目 Swin 图同类的
    "常量（LN weight/bias）参与 elementwise/MatMul" 结构，是 INT8 implicit 量化
    回退现象（README §2 贡献 4）的最小复现载体。
    """

    def __init__(self, in_ch=3, dim=64, n_blocks=2, num_classes=2):
        super().__init__()
        self.embed = torch.nn.Conv2d(in_ch, dim, kernel_size=4, stride=4)  # 64x64 -> 16x16
        self.norm0 = torch.nn.LayerNorm(dim)
        self.blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(torch.nn.ModuleDict({
                "ln1": torch.nn.LayerNorm(dim),
                "fc1": torch.nn.Linear(dim, dim * 2),
                "fc2": torch.nn.Linear(dim * 2, dim),
                "ln2": torch.nn.LayerNorm(dim),
                "fc3": torch.nn.Linear(dim, dim * 2),
                "fc4": torch.nn.Linear(dim * 2, dim),
            }))
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(dim), torch.nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.embed(x).flatten(2).transpose(1, 2)      # (B, 256, dim)
        x = self.norm0(x)
        for b in self.blocks:
            x = x + b["fc2"](torch.nn.functional.gelu(b["fc1"](b["ln1"](x))))
            x = x + b["fc4"](torch.nn.functional.gelu(b["fc3"](b["ln2"](x))))
        return self.head(x.mean(dim=1))


def fixed_input(seed: int = 1234) -> torch.Tensor:
    """per-script 相同输入（三引擎一致性，主项目 P0-2 口径）。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*INPUT_SHAPE, generator=g).cuda()


def check_gpu_idle() -> int:
    """基准纪律第一条：共享 GPU 抢占会让延迟虚高 2-3×（主项目实测）。"""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader", "-i", "0"],
        capture_output=True, text=True).stdout.strip()
    util = int(out.split(",")[0].replace("%", "").strip() or 0)
    if util > 10:
        print(f"[WARN] 当前 GPU 利用率 {util}%（共享抢占会污染延迟），"
              f"建议 CUDA_VISIBLE_DEVICES 换空闲卡后再跑基准")
    else:
        print(f"[OK] GPU 利用率 {util}%（空闲）")
    return util


def bench_cuda_event(fn, groups: int = 3, iters: int = 200, warmup: int = 50) -> list[float]:
    """device-only 延迟：torch CUDA event，3 组重复（主项目 rep3 口径）。"""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    group_ms = []
    for _ in range(groups):
        ts = []
        for _ in range(iters):
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record(); torch.cuda.synchronize()
            ts.append(s.elapsed_time(e))
        group_ms.append(float(np.mean(ts)))
    return group_ms


def save_json(name: str, obj: dict) -> Path:
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SAVE] {p}")
    return p


def log(tag: str, msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] [{tag}] {msg}", flush=True)
