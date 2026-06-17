# Research Direction After Report 25

本项目后续主线应从“让图像看起来像刺绣”转为：

```text
DST-derived supervision
  -> structure prediction
  -> executability-aware planner selection
  -> DST/PES export
  -> render-back and command-level evaluation
```

核心目标不是复刻某一个唯一 GT DST，而是生成可绣、可执行、视觉风险低的 DST/PES。

## 统一目标

报告 25 建议使用统一目标：

```text
unified_loss = ExecScore + VisualRisk
```

其中：

- `ExecScore` 关注命令层可执行性：parse success、jump、trim、jump path。
- `VisualRisk` 关注视觉副作用：off-mask stitches、visible connectors、visible connector length。

仓库中对应实现：

- `tools/eval_executability.py`
- `tools/score_unified.py`

## 当前新增工具

| 文件 | 作用 |
| --- | --- |
| `tools/score_unified.py` | 读取 executability JSON，计算统一评分 |
| `tools/run_b1_sweep.py` | 批量运行 planner 参数候选 |
| `tools/select_b1_clean.py` | 基于统一评分和 Pareto 前沿选择 B1_clean |
| `configs/sweep_b1.yaml` | B1 参数候选和评分权重 |
| `configs/b1_clean.yaml` | 当前 B1_clean 候选配置 |
| `tools/preprocess_real_image_v2.py` | 真实照片 thread/region/selected 输入预处理 |
| `docs/hitl_protocol_zh.md` | 人工筛选协议 |

## E0-E4 实验路线

| 实验 | 输入 | 选择机制 | Planner | 目的 |
| --- | --- | --- | --- | --- |
| E0 | DST-rendered preview | 无 | B1 | 当前最小基线 |
| E1 | E0 + render augmentation | 无 | B1 | 验证普通增强收益 |
| E2 | E1 + inverse pseudo-real | 无人工筛选 | B1 | 验证伪真实输入是否污染训练 |
| E3 | E2 + human input filtering | 人工筛输入 | B1_clean | 验证人工筛输入域价值 |
| E4 | E3 + unified-loss top-K | 人工 + 自动评分 | B1_clean / M1 | 目标方法 |

## 当前结论

最近一轮 holdout paired 测试显示：

- 真实照片预处理和外部 mask 对减少背景污染有效。
- `thread_mask` 干净但容易欠密度。
- `region_mask` 对填充类图案有效，例如水母样本针数接近目标。
- 单一参数无法覆盖文字、线稿、卡通填充、真实绣片照片。
- 因此下一步应该做 B1_clean sweep 和 E0-E4 实验，而不是继续盲目加大 U-Net。

## 下一步优先级

1. 用 `tools/score_unified.py` 固定统一评分。
2. 用 `tools/run_b1_sweep.py` 对验证集跑 planner 参数候选。
3. 用 `tools/select_b1_clean.py` 选择 B1_clean。
4. 构建 inverse pseudo-real 输入候选。
5. 按 `docs/hitl_protocol_zh.md` 进行人工筛选。
6. 将通过筛选的输入与原 DST-derived labels 回流训练。
