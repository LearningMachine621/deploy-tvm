"""① 构建 TinyFormer → 导出 opset17 ONNX → logits 对齐校验。

图只描述结构、不保证语义：导出后必须做 PyTorch/ONNX logits 对齐（主项目 Phase A 底线）。
顺带统计原生 LayerNormalization 节点数 —— opset 语义决定下游编译器能力（主项目 620× 教训）。

用法：python demo/01_export.py
"""
from __future__ import annotations

import numpy as np
import torch
from common import CKPT, INPUT_NAME, INPUT_SHAPE, ONNX_PATH, TinyFormer, fixed_input, log, save_json


def main() -> None:
    torch.manual_seed(0)
    model = TinyFormer().eval().cuda()
    torch.save(model.state_dict(), CKPT)
    x = fixed_input()

    with torch.no_grad():
        ref = model(x).cpu().numpy()
    log("MODEL", f"params={sum(p.numel() for p in model.parameters()):,}, "
                f"ref logits={np.round(ref[0], 4).tolist()}")

    # ---- 导出 opset 17（原生 LayerNormalization）----
    torch.onnx.export(
        model, x, str(ONNX_PATH), opset_version=17,
        input_names=[INPUT_NAME], output_names=["logits"],
        do_constant_folding=True)
    import onnx
    m = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(m)
    ops = {}
    for n in m.graph.node:
        ops[n.op_type] = ops.get(n.op_type, 0) + 1
    log("ONNX", f"{str(ONNX_PATH)}  nodes={len(m.graph.node)}, "
                f"opset={m.opset_import[0].version}, LayerNormalization={ops.get('LayerNormalization', 0)}")
    note = "（GELU 在 opset17 被展开为 Erf 组合，opset20 才有原生 Gelu——图表示 vs 算子语义的又一例）" \
        if "Gelu" not in ops else ""

    # ---- logits 对齐校验（能导出 ≠ 导得对）----
    import onnxruntime as ort
    sess = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    out = sess.run(None, {INPUT_NAME: x.cpu().numpy()})[0]
    diff = float(np.abs(out - ref).max())
    assert diff < 1e-3, f"PyTorch/ONNX logits 不对齐: max diff={diff}"
    log("CHECK", f"PyTorch vs ONNX logits max abs diff = {diff:.2e}  ✅ 对齐 {note}")

    save_json("01_export.json", {
        "onnx": str(ONNX_PATH.name), "opset": 17,
        "nodes": len(m.graph.node), "op_counts": ops,
        "logits_max_abs_diff": diff,
    })


if __name__ == "__main__":
    main()
