# Human-in-the-Loop Review Protocol

本项目中的人工反馈只负责筛选输入域，不直接判断 DST/PES 命令序列是否专业。

这样设计的原因是：同一张视觉图案可以对应多种合法刺绣走线方案，人工直接看 DST 命令容易引入主观偏差。最终是否可执行、是否有过多跳针、剪线、可见连接线，应由统一评分脚本判断。

## 人工只评三件事

| 字段 | 分值 | 含义 |
| --- | ---: | --- |
| `realism_score` | 1-5 | 候选输入是否像用户真实会提供的图片、照片或设计稿 |
| `structure_preserve_score` | 1-5 | 主体轮廓、区域和重要结构是否保持一致 |
| `stitchability_score` | 1-5 | 是否适合转成刺绣输入，是否过度摄影化、碎纹理化或背景杂乱 |

建议进入训练候选池的最低条件：

- `realism_score >= 3`
- `structure_preserve_score >= 4`
- `stitchability_score >= 4`

## 人工不评这些内容

- 不判断 `jump_count` 是否合理
- 不判断 `trim_count` 是否合理
- 不判断某个 DST 是否唯一正确
- 不直接把模型预测结果当作新标签

这些由 `tools/eval_executability.py` 和 `tools/score_unified.py` 完成。

## Manifest 字段

推荐使用以下 CSV schema：

```csv
pair_id,design_id,source_dst,label_version,candidate_image,candidate_seed,gen_method,preview_image,overlay_mask,realism_score,structure_preserve_score,stitchability_score,accept,reject_reason,reviewer,reviewed_at,planner_cfg,unified_loss,jump_count,trim_count,off_mask_stitch_length_mm,visible_connector_count,notes
```

## 拒绝原因

`reject_reason` 推荐枚举：

- `blur`
- `clutter`
- `color_noise`
- `shape_drift`
- `too_photographic`
- `low_stitchability`
- `other`

## 训练回流原则

回流训练集的是：

```text
(accepted_input_image, original_dst_derived_labels)
```

不是：

```text
(accepted_input_image, model_prediction_as_label)
```

也就是说，监督目标仍然来自原始 DST/PES 反向生成的结构标签，避免模型在闭环中不断学习自己的错误。
