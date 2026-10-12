# Anima 局部 VJP 与整网 cotangent 定位

状态：2026-09-30，Luna 执行 CMP 90HX 单步对照；FP16 候选仍未通过数值门槛。
前置证据：[分支与 scale 对照](anima_adaptive_precision_diagnostics_90hx_20260930.md)。

## 结论

固定 sigma=0.5 和初始 LoRA 状态，仅 blocks MLP Linear 使用 FP16 时，
prediction relative-L2 为 0.020736%，输入梯度为 110.742%，LoRA 梯度为 40.384%。
loss/梯度均有限，两侧各完成一次 AdamW 更新，但数值门槛不通过。
此设置不同于上一轮三步中的第二步，不能将两轮差异只归因于 sigma 或代码。

对固定参考 x、W、cotangent g 的 64-token 局部 Linear 反事实，scale 提升明显降低
FP16 VJP 误差。真实低精度 MLP1 的 scale1024 局部 dx relative-L2 为
0.02978%--0.06951%，但早期层整网 g 偏移仍超过 100%，甚至 cosine 为负。
因此已区分出局部 cast/kernel 误差与整网轨迹偏移，尚未建立两者的因果分解。
没有锁定一个足以修复的层或策略，不能直接生成 FP32 名单。

## 实验与实现

- GPU：`GPU-806debd9-d254-da57-d515-ccae15b4ac6a`，CMP 90HX 10 GiB、SM86。
- FP32 reference 从 BF16 来源权重提升；source 共 685 个 BF16 tensor。
- 单步固定 sigma=0.5、LoRA rank/alpha 4/4、AdamW lr=1e-4，full checkpoint、swap20，
  torch attention，无 compile/TF32；reference scale=1，candidate scale=1024。
- candidate 实际 56 个 FP16、321 个 FP32 Linear；只有 `blocks.*.mlp.*` 低精度，
  包括未捕获的 MLP layer2。QKV 是 FP32，下面的 QKV-FP16 是额外的局部反事实。
- latent/noise `[1,16,1,150,112]`、prompt `[1,512,1024]`，保持文本 full padding，
  latent mask 为全零。仍仅单图缓存、单 bucket，无长期质量或 Turing 认证。

新增 `bench/adaptive_runtime/anima_local_vjp.py` 在 LoRA 的 `org_forward` 外捕获
基座 x/y 与真实 y-cotangent，不使用基座加 adapter 的总输出。
返回原张量、不改变 RNG；每层均匀固定采样 64 token，检查 checkpoint 重算一致。
每侧 8 个 capture 文件实际合计 268,442,496 B，低于独立 512 MiB 限额；
FP32 权重仅在 BF16 往返逐位相同时压缩存储。

MLP eager 输入是 `[1,1,75,56,2048]`，QKV 是 `[1,4200,2048]`。
采样按 batch/feature 以外的轴 row-major flatten，记录原 shape、共同 indices、
各文件 SHA 和 dtype。分析时重新比较真实整网 capture，并核对 local signature、
replay evidence、UUID、scale 和 MLP-only 分配，不能只信任传入 comparison JSON。

## 局部结果

下表误差均已乘 100 为百分数。g drift 比较候选与参考的真实整网 base-output
cotangent；local dx 则始终固定参考 x/W/g，运行 FP16 cast-in/Linear/cast-out autograd
并除以 scale。两者不是相同实验，不可将误差相减当作因果贡献。

| Layer | FP16 local dx @1024 | 整网 g drift | g cosine | scaled g 非零转零 @1024 |
| --- | ---: | ---: | ---: | ---: |
| block0 MLP1 | 0.03078% | 147.074% | -0.7607 | 4,857 |
| block0 QKV（反事实） | 2.26029% | 114.726% | 0.0610 | 180,567 |
| block5 MLP1 | 0.02978% | 146.568% | -0.6324 | 19,516 |
| block5 QKV（反事实） | 0.03558% | 136.656% | -0.2546 | 4,351 |
| block14 MLP1 | 0.04501% | 0.56005% | 0.999984 | 3,720 |
| block14 QKV（反事实） | 0.10970% | 2.68375% | 0.999640 | 4,512 |
| block27 MLP1 | 0.06951% | 0.08805% | 0.9999996 | 6,241 |
| block27 QKV（反事实） | 0.66880% | 0.09920% | 0.9999995 | 63,350 |

八层局部 actual FP16 dx relative-L2 范围：scale1 为 10.43%--99.42%，
scale64 为 0.198%--25.61%，scale1024 为 0.02978%--2.26029%。
固定参考 x 的 FP16 forward 误差为 0.0220%--0.0263%；weight-only FP32 dx 误差约
0.00000386%--0.0000141%，表明这几个样本中单独 weight cast 的贡献很小。
原始 reference y 与抽 64 行重算 FP32 y 的误差约 0.0000695%--0.0000786%。
抽行改变 GEMM shape/kernel，因此这些局部指标不能当作原整网误差上界。

scale1024 八层非零 cotangent 转 FP16 零共 287,114 个，**真实低精度 MLP1 仅占
34,334 个，其余 252,780 属于 QKV 反事实**，不得全部归因于候选运行。
八层还有 3,179,738 个非零 FP16 次正规数、零个 overflow；这些是本次实际 cast
观测，不证明所有 CUDA kernel 都保留次正规数，也不排除其它中间边界问题。

第一层 x 在两侧相同；后续 x/y 仅小幅偏移，却伴随早期 g 大幅变化。
MLP2、GELU 和其它路径尚未捕获，不能将早期 g 漂移归因为 MLP1 局部 dx。
局部只测试真实 g 的方向，也不是完整 Jacobian 或最坏方向敏感性认证。

## 身份、失败与资源

| 身份 | SHA-256 |
| --- | --- |
| weights | `bd43b7cffe1ed1153d9c41e7beb2f18cb1273eafbaa3af3edd6a173dc90a006e` |
| inputs | `0c2d6f12569596357efd5a1b0b652150f176ef37e2f475bda0bcd533ccc17d78` |
| probe | `720e470aaec44fd2c3921a6f13c7a093effbacb64d42cacf280735539060012f` |
| local helper | `d33cbed29eea33bc83b6e4060aa5a8400a3c7dee8a352d57859cb6b8491cf047` |

两侧 initial adapter 和 snapshot/state/input/RNG replay SHA 一致。
首次目录 `output/runs/anima-adaptive-precision-90hx-20260930-local-vjp-codex-1724/`
因捕获只接受 3D 而 eager MLP 是 5D，在 reference forward 失败、无有效更新。
完整 traceback 保留于 worker.log；candidate/analysis 未运行，不能计作通过。
修复原 shape/采样契约后重新录制 reference，不使用旧参考。

成功复测目录：`output/runs/anima-adaptive-precision-90hx-20260930-local-vjp-retest/`。
reference/candidate/analysis 全部 status=ok、returncode=0；CLI exit0 仅表示诊断完成，
comparison 为拒绝、production_ready=false。

| Worker | GPU used 峰值 | RSS tree 峰值（GB） | 最低 host available（GB） | 时间 |
| --- | ---: | ---: | ---: | ---: |
| reference | 5236 MiB | 19.70 | 86.07 | 56.8s |
| candidate | 4714 MiB | 10.96 | 94.21 | 49.0s |
| analysis | 990 MiB | 2.00 | 104.29 | 15.6s |

没有 telemetry 查询错误、OOM 或资源保护停止。全机 swap_used 从 reference 首样本
2002.75 MiB 增至末样本 2026.75 MiB，candidate/analysis 保持不变；不能将全机
swap 增量独占归因于本任务。结束 90HX 17 MiB、0%，无本任务残留进程。
步时含捕获/IO，不是速度优化证据。

## 入口与后续

后续 [MLP2/GELU 与投影拆分探测](anima_mlp_path_90hx_20260930.md)已完成。
本报告记录的是先前 8 层配置；新增 profile 不改变该轮历史证据或认证范围。

```bash
.venv/bin/python -m bench.adaptive_runtime.diagnose_anima_local_vjp \
  --weights /path/to/anima-bf16.safetensors --inputs /path/to/inputs.safetensors \
  --output-dir output/runs/new-anima-local-vjp --gpu-uuid GPU-... \
  --swap 20 --worker-timeout 150
```

热测前独立 CPU 身份核验回归 641 passed、2 skipped、3 xfailed；3D/5D 修复后
定向 49 passed，Ruff 通过。真实 CPU LoRA/checkpoint/record/replay/compare/analyze
及哈希、计划、replay 篡改拒绝已覆盖。既有 3 个共享参数 xfail 不计作通过。
最终主代理 CPU 回归 `tests/test_adaptive*.py` 为 642 passed、2 skipped、3 xfailed
（退出码 0），五个相关 Python 文件 Ruff 通过；源码 SHA 与成功热测记录仍一致。

下一步优先补 MLP2 及 GELU 上下游 cotangent，再做受控的前向/反向精度反事实，
区分剩余反向边界与 nonlinear trajectory sensitivity。现有数据不足以自动认证层名单；
不应只提高 scale、增加早期 FP32、放宽门槛或宣称通用 FP16 已完成。
