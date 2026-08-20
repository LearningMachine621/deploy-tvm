# 推理部署选型与性能归因：TVM × TensorRT（工业缺陷检测 / RTX 4090）

> Swin-v3 双分支分类器 · ONNX opset 17 · TVM 0.11.1 · TensorRT 10.4 · 静态 shape
>
> 一句话：把"哪个推理引擎快"从跑分问题，变成**可归因、可复现、可审查**的编译问题。

---

## 1. 问题

工业缺陷检测模型（Swin-v3，输入 `(B,2,224,224)` 双通道灰度差分）部署到 RTX 4090：在线场景要求 bs=1 低延迟，离线场景要求大 batch 吞吐，且**精度不得低于 FP32 PyTorch eager 基线**。面对 TVM / ORT / TensorRT：

- 每一步加速到底来自哪一层——图优化、kernel 调度、还是硬件精度路径？
- 一个 benchmark 数字凭什么可信（GPU 抢占曾使本项目延迟虚高 2-3×）？
- 量化（INT8）值不值得上？

## 2. 贡献

1. **消融归因**：四档同进程对照（PyTorch / 不融合 / 融合 / tuned）证明 TVM 4.6× 加速 **~100% 来自 AutoScheduler schedule 搜索**——融合 +8.0% 被 TOPI-vs-cuDNN 的 −6.3% 抵消，净 ≈1.0×；3 组 × 200 iters 复测。
2. **调优铁律 + 反例**：只调 top-5 task → 45/50 fallback → 3.9× 退化（13.1ms）；确立"全 task 覆盖 + 每 bs 独立 tuning log（~10h/bs）"的部署约束。
3. **kernel 级归因**：nsys 把 TRT 3× 优势拆到**实现层**（271 vs 599/727 kernel、70% 时间在 fp16 Tensor Core、消除 PyTorch 79% 的 launch 开销），并定位天花板：**15.7% fp16 峰值 = window attention 算法结构限制，非编译器可解**。
4. **INT8 证伪（flag ≠ 执行）**：原生 INT8 engine 数值正确（2e-4）却慢 55%，六层证据链（builder log → engine inspector → nsys → ncu）证明 **0 个 INT8 kernel 实际执行**（212 tensor 缺 scale → builder 回退 TF32）。
5. **部署闭环**：TensorRT FP16 定论——115,776 样本与 FP32 基线 **100% 标签一致**、统一口径复测无系统偏差、VERBOSE 证实 0 CUDA/CPU fallback。

## 3. 核心数字

| 指标 | 值 |
|---|---|
| 延迟（bs=1，device-only，CUDA event）| PyTorch 15.35ms → TVM tuned 3.33ms（4.6×）→ **TRT FP16 1.11ms** |
| 吞吐（bs=64）| **5575 FPS**（TRT FP16）|
| 消融乘法链（rep3 同进程）| 0.937（kernel 库）× 1.080（融合）× 5.02（schedule）= **5.07×** |
| 精度 | 与 FP32 PyTorch eager 基线 100% 标签一致（TP/FP 逐项相同），mean logit diff 0.0045 |
| TRT 优势来源（nsys）| 271 vs 599 vs 727 kernel/推理；70% 时间在 fp16 Tensor Core MMA |
| 性能天花板 | 15.7% fp16 峰值（bs=64，GPU 已 97% 活跃）→ 算法层 |
| INT8 归因 | 1.72ms（比 FP16 慢 55%）；212 missing-scale / 148 GEMM 中 0 INT8 / nsys 0 个 int8 kernel |
| 工程成本 | TRT 冷 build 62.3s → cache 命中 0.27s（~230× 启动加速）；TVM 调优 ~10h/bs |

> 主表数字来自内部工业模型（含业务权重，不随本库分发）；每个数字有 raw JSON/CSV 落盘，关键实验含 3 组重复或 5000 次 × 5 组稳态测量。

## 4. 架构图

**部署与归因链路：**

```
        PyTorch checkpoint（SHA256 记录）
                    │  export opset=17 + onnxsim + logits 对齐
                    v
          ONNX（1220 节点，53 原生 LayerNorm）
                    │
        ┌───────────┴────────────┐
        v                        v
  TVM 0.11.1                 TensorRT 10.4（ORT TRT EP）
  Relay → TIR → CUDA         fp16 混合精度 + tactic 择优 + engine cache
  AutoScheduler              冷 build 62.3s → cache 0.27s
  50 task × 200 trials                │
  3.33ms (bs=1)              1.11ms (bs=1) / 5575fps (bs=64)
        │                        │
        └───────────┬────────────┘
                    v
         归因与验证层（本项目核心）
         nsys      kernel 时间线      → 271/599/727 kernel，15.7% 峰值
         inspector 逐层 tactic       → 148 GEMM census（INT8 = 0）
         ncu       硬件计数器        → underfill 分类（容器 root 方案）
         P0-1/2/3  精度/口径/覆盖    → 100% 一致 / 统一计时 / 0 fallback
```

**归因方法栈（一个问题配一个工具）：**

```
慢在哪一层   → nsys（kernel 时间线）      → TRT 赢在实现层，非硬件
为什么慢     → ncu（硬件计数器）          → underfill / 算法层天花板
flag 生效吗  → builder log + inspector   → INT8 0 kernel：flag ≠ 执行
数字可信吗   → 统一口径 CUDA event        → 5000×5 组，device-only ≈ e2e
```

## 5. 快速体验（demo/，单卡约 10 分钟）

本库附带一套**自包含最小 demo**：用纯 torch 定义的 TinyFormer（70k 参数，含 LayerNorm/Linear/Conv，无任何业务数据）跑通上述方法论三件套——同口径基准、消融乘法链、INT8 flag 检查。

```bash
pip install -r requirements.txt          # TVM 安装见 requirements.txt 尾注
git clone <本仓库> && cd <本仓库>

python demo/01_export.py                 # ① 导出 opset17 ONNX + logits 对齐校验（~10s）
python demo/02_ablation.py --trials 30   # ② 四档消融 + 全 task 快速调优（~8min）
python demo/03_trt_int8_flag.py          # ③ TRT FP16 基准 + INT8 flag≠执行检查（~1min）
```

demo 实测输出（RTX 4090，空闲卡）：

```
② 消融乘法链（TinyFormer）: C/B = 3.19× = 1.28(kernel库) × 1.49(融合) × 1.68(schedule)
   kernels 162 → 70（融合的发射次数证据）
③ INT8 flag 检查（TinyFormer）:
   INT8 flag = ON，logits diff = 1.4e-04（数值"绿灯"）
   builder log: 3 × "Missing scale and zero-point ... fall back to non-int8"
   inspector: 21 层 / 8 GEMM / INT8 GEMM = 0   ← flag ≠ 执行，最小复现
```

**demo 与主结果的数字不同**（模型、调优预算都小两个量级）——demo 复现的是**方法与现象**：INT8 的"flag 开了但 0 个 INT8 kernel"在 70k 参数的公开合成模型上同样发生；消融乘法链在任何模型上都能拆出三因子。样例输出见 `demo/results/`。

## 6. 限制条件

- **不外推**：结论限定 RTX 4090 / 本 Swin-v3 模型 / 静态 shape（bs=1-64）；其它 GPU、模型、dynamic shape 未验证（dynamic profile 未配置，每 bs 独立 engine）。
- **INT8 结论的边界**：仅证伪 implicit 量化路线（0 INT8 kernel 实际执行）；true INT8（explicit QDQ）与 FP16 的 crossover 未测，不声称"INT8 必输"。
- **归因深度**：15.7% 天花板来自 nsys kernel 级时间线 + 实测算力；FP16 engine 的 Tensor Core 单元级利用率（ncu）未测。
- **未做的下一步**：FlashAttention 是基于天花板定位的推断方向，未实测；TVM 调优成本（~10h/bs）在本项目未优化。
- **证据纪律**：每个结论有 raw 落盘；历史单次测量只作档案，主口径为 3 组 rep3 / 5000 次 × 5 组稳态；被推翻的早期归因（如 codegen 崩溃归因、26% 利用率估计）均保留推翻证据。
