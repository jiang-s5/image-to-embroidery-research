# Report 25 后的研究方向

本项目后续主线应从“让图像看起来像刺绣”转为：

```text
DST-derived supervision
  -> structure prediction
  -> segment/polyline graph construction
  -> TSP-style path optimization
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

仓库中对应工具：

- `tools/eval_executability.py`
- `tools/score_unified.py`
- `tools/run_b1_sweep.py`
- `tools/select_b1_clean.py`

## Mean 与 Hard 选择

平均 `unified_loss` 最低不一定等于真实刺绣最稳。某些 preset 可能在一个样本上表现很好，但在其他样本上产生大量可见连接线或越界针迹。

因此 B1_clean 选择分两种：

- `mean`：从 Pareto front 中选择平均 `unified_loss` 最低的方案。
- `hard`：优先压低 `visible_connector_count`、`off_mask_stitch_length_mm`、`jump_count`、`trim_count`，更适合作为真实刺绣硬标准候选。

当前默认使用 `hard`，因为真实刺绣更怕可见连接线和越界针迹，而不是只追求单个平均分。

## 为什么主线选 TSP / 图优化

当前阶段最适合冲论文的路线是：

```text
Geometry-to-Graph-to-TSP Planner
```

原因：

- TSP / 图优化直接对应 `jump_count`、`trim_count`、`off_mask_stitch_length_mm`、`visible_connector_count`。
- 它不需要大量真实图片到 DST 的严格成对数据。
- 它可以作为 GNN 的强 baseline，也可以为未来 GNN 提供 node/edge 监督。
- RL 暂时不适合作为主线，因为 action space 大、reward 稀疏、训练不稳定。

当前实现中，`b2_graph_tsp_conservative_safe` 将每条 stitch polyline 视为图节点，用边代价惩罚距离、跳针、剪线、越界连接和可见连接线，并启用 mask-safe connector。推理时还会导出 `graph_tsp_trace.json`，记录 node、selected edge、route order 和 edge-risk 细节，为下一阶段 M1 selector / M2 edge-GNN 提供中间监督。

## 当前新增工具

| 文件 | 作用 |
| --- | --- |
| `tools/score_unified.py` | 读取 executability JSON，计算统一评分 |
| `tools/run_b1_sweep.py` | 批量运行 planner 参数候选 |
| `tools/select_b1_clean.py` | 基于统一评分和 Pareto front 选择 B1_clean |
| `configs/sweep_b1.yaml` | B1 参数候选、评分权重和选择策略 |
| `configs/b1_clean.yaml` | 当前 B1_clean 候选配置 |
| `tools/preprocess_real_image_v2.py` | 真实照片 thread/region/selected 输入预处理 |
| `planner/graph_tsp.py` | polyline graph + TSP-style edge-cost planner |
| `docs/hitl_protocol_zh.md` | 人工筛选协议 |

## E0-E4 实验路线

| 实验 | 输入 | 选择机制 | Planner | 目的 |
| --- | --- | --- | --- | --- |
| E0 | DST-rendered preview | 无 | B1 | 当前最小基线 |
| E1 | E0 + render augmentation | 无 | B1 | 验证普通增强收益 |
| E2 | E1 + inverse pseudo-real | 无人工筛选 | B1 | 验证仿真实输入是否污染训练 |
| E3 | E2 + human input filtering | 人工筛输入 | B1_clean | 验证人工筛输入域价值 |
| E4 | E3 + unified-loss top-K | 人工 + 自动评分 | B1_clean / M1 | 目标方法 |

## 当前结论

最近一轮 holdout paired 测试显示：

- 真实照片预处理和外部 mask 对减少背景污染有效。
- `thread_mask` 干净但容易欠密度。
- `region_mask` 对填充类图案有效，但也可能产生越界填充。
- 单一参数无法覆盖文字、线稿、卡通填充、真实绣片照片。
- B1 full sweep 里 `b1_low_connect` 平均分最低，但 `b1_conservative` 在 3/4 个样本上更稳，且 jump、trim、visible connector 均值更低。
- Graph-TSP sweep 里 `b2_graph_tsp_conservative_safe` 成为 hard policy 推荐：它降低了 mean unified loss、off-mask 长度和 visible connectors，但 jump count 上升。
- 所有 4 个 holdout 样本仍为 `hard_fail`，说明当前只是相对优化，还没有达到真实可用标准。

## 下一步优先级

1. 继续优化 Graph-TSP edge cost，降低 jump count 的副作用。
2. 基于 `graph_tsp_trace.json` 生成 M1 selector / M2 edge-GNN 训练样本。
3. 扩大 holdout paired set，避免 4 个样本的偶然性。
4. 优先降低 `visible_connector_count` 和 `off_mask_stitch_length_mm`。
5. 将输入预处理从全局规则升级为按图像类型路由：thread / region / selected。
6. 构建 inverse pseudo-real 输入候选。
7. 按 `docs/hitl_protocol_zh.md` 做人工筛选。
8. 只将通过筛选的输入与原始 DST-derived labels 回流训练，不把模型预测当标签。
