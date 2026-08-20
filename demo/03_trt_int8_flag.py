"""③ TensorRT FP16 基准 + INT8 "flag ≠ 执行" 最小验证。

Part A：ORT TRT EP（FP16）vs CUDA EP 同口径基准 + 数值对齐（vs PyTorch fp32）。
Part B：原生 TRT INT8 engine（implicit 量化 + entropy calibrator）
        → IEngineInspector 逐层 tactic census → 数一数有几个 INT8 kernel。

主项目结论（README §2 贡献 4）：INT8 flag 开了、数值正确、engine 能跑，
都不代表量化 kernel 在执行——必须在 engine artifact 层验证。
本脚本把该检查压缩成 40 行可复用的最小流程。

用法：python demo/03_trt_int8_flag.py
"""
from __future__ import annotations

import ctypes
import glob
import json
import sys

import numpy as np
import torch
from common import CKPT, INPUT_NAME, INPUT_SHAPE, ONNX_PATH, TinyFormer, bench_cuda_event, check_gpu_idle, fixed_input, log, save_json

CACHE_DIR = ONNX_PATH.parent / "cache"


def preload_trt_libs() -> None:
    """把 pip 安装的 TRT .so 预载进进程（RTLD_GLOBAL），使 ORT TRT EP 的 dlopen 可解析。"""
    import tensorrt  # noqa: F401  导入即加载 libnvinfer
    import site
    for sp in site.getsitepackages():
        for pat in ("libnvinfer.so*", "libnvinfer_plugin.so*", "libnvonnxparser.so*"):
            for f in glob.glob(f"{sp}/tensorrt_libs/{pat}"):
                try:
                    ctypes.CDLL(f, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def part_a_trt_fp16(x: torch.Tensor, ref_logits: np.ndarray) -> dict:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    assert "TensorrtExecutionProvider" in providers, (
        "ORT 无 TRT EP。备选：export LD_LIBRARY_PATH=<tensorrt_libs 目录>:$LD_LIBRARY_PATH 后重跑")
    CACHE_DIR.mkdir(exist_ok=True)
    trt_sess = ort.InferenceSession(str(ONNX_PATH), providers=[
        ("TensorrtExecutionProvider", {
            "device_id": 0, "trt_fp16_enable": True,
            "trt_engine_cache_enable": True, "trt_engine_cache_path": str(CACHE_DIR)}),
        ("CUDAExecutionProvider", {})])
    cuda_sess = ort.InferenceSession(str(ONNX_PATH), providers=[("CUDAExecutionProvider", {})])
    x_np = x.cpu().numpy()

    trt_groups = bench_cuda_event(lambda: trt_sess.run(None, {INPUT_NAME: x_np}), groups=3, iters=200)
    cuda_groups = bench_cuda_event(lambda: cuda_sess.run(None, {INPUT_NAME: x_np}), groups=3, iters=200)
    out = trt_sess.run(None, {INPUT_NAME: x_np})[0]
    diff = float(np.abs(out - ref_logits).max())
    log("BENCH", f"TRT EP FP16 : {np.mean(trt_groups):.3f} ms   logits max diff={diff:.2e}")
    log("BENCH", f"CUDA EP FP32 : {np.mean(cuda_groups):.3f} ms")
    return {"trt_fp16_ms_groups": trt_groups, "cuda_fp32_ms_groups": cuda_groups,
            "trt_fp16_logits_max_abs_diff": diff}


def part_b_int8_flag_check(x: torch.Tensor, ref_logits: np.ndarray) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)   # builder 的 [TRT][W] Missing scale... 警告会打到 stderr（现场证据）
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    assert parser.parse_from_file(str(ONNX_PATH)), "ONNX parse 失败"

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)                              # ← flag 开了
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED       # ← 为 inspector 嵌入逐层 tactic

    class RandCalibrator(trt.IInt8EntropyCalibrator2):
        """随机数据校准器（demo 无业务数据；主项目用 800 真实样本）。"""

        def __init__(self, n_batches=50, batch=8):
            super().__init__()
            self.n, self.i = n_batches, 0
            self.buf = torch.zeros(batch * int(np.prod(INPUT_SHAPE)), dtype=torch.float32, device="cuda")
            self.ptr = self.buf.data_ptr()

        def get_batch_size(self):
            return 8

        def get_batch(self, names):
            if self.i >= self.n:
                return None
            self.i += 1
            self.buf.normal_()
            return [int(self.ptr)]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            pass

    config.int8_calibrator = RandCalibrator()
    log("BUILD", "building INT8 engine（implicit 量化 + entropy calibrator）...")
    engine_bytes = builder.build_serialized_network(network, config)
    assert engine_bytes is not None, "engine build 失败"
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    context = engine.create_execution_context()

    # 跑一次（拿输出 + 激活 inspector）
    x_d = x.contiguous().cuda()
    out_shape = tuple(context.get_tensor_shape("logits"))
    out_d = torch.empty(out_shape, dtype=torch.float32, device="cuda")
    stream = torch.cuda.current_stream().cuda_stream
    for name, tensor in ((INPUT_NAME, x_d), ("logits", out_d)):
        context.set_tensor_address(name, int(tensor.data_ptr()))
    context.execute_async_v3(stream)
    torch.cuda.synchronize()
    numeric_diff = float(np.abs(out_d.cpu().numpy() - ref_logits).max())
    log("NUMERIC", f"INT8 engine logits max abs diff vs fp32 = {numeric_diff:.2e}（数值层'绿灯'）")

    # ---- 核心检查：engine artifact 里到底有没有 INT8 kernel ----
    inspector = engine.create_engine_inspector()
    inspector.execution_context = context
    info = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers = info["Layers"]
    gemm, int8_gemm = 0, 0
    for layer in layers:
        tactic = str(layer.get("TacticName", ""))
        if "xmma_gemm" in tactic or "gemvx" in tactic or "cublas" in tactic.lower():
            gemm += 1
            if any(k in tactic.lower() for k in ("int8", "s8", "i8")):
                int8_gemm += 1
    verdict = "flag ≠ 执行：0 个 INT8 GEMM（builder 对缺 scale 的路径回退了）" if int8_gemm == 0 \
        else f"本模型未触发回退（{int8_gemm} 个 INT8 GEMM）——同样要用检查才能知道"
    print(f"\n===== INT8 flag 检查 =====")
    print(f"INT8 flag        : ON")
    print(f"engine 层数      : {len(layers)}，其中 GEMM {gemm} 个")
    print(f"INT8 GEMM        : {int8_gemm}")
    print(f"结论             : {verdict}")
    return {"layers": len(layers), "gemm_layers": gemm, "int8_gemm_layers": int8_gemm,
            "int8_logits_max_abs_diff": numeric_diff, "verdict": verdict}


def main() -> None:
    check_gpu_idle()
    model = TinyFormer().eval().cuda()
    model.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=True))
    x = fixed_input()
    with torch.no_grad():
        ref = model(x).cpu().numpy()

    preload_trt_libs()
    out = {"part_a": part_a_trt_fp16(x, ref)}
    try:
        out["part_b"] = part_b_int8_flag_check(x, ref)
    finally:
        sys.stderr.write("（builder 期间的 [TRT][W] Missing scale 警告即 L2 现场证据）\n")
    save_json("03_trt_int8_flag.json", out)


if __name__ == "__main__":
    main()
