# HeLa-Mem 指标文件口径

各实验目录的 `metrics.json` 用于保存**已归档的汇总**，不能代替原始数据、完整配置或独立复核。这里统一**解释口径**，不倒填或改写历史统计值。

| 字段 | 含义 | 约定 |
|---|---|---|
| `dataset` / `experiment` | 数据集规模或具体运行标签 | 延续原始历史记录 |
| `total` | 汇总覆盖的记录总数 | 整数 |
| `correct` | 符合该运行评分标准的正确条数 | 整数或 `null` |
| `incorrect` | 不正确条数 | 整数或 `null` |
| `accuracy` | 正确数 / 总数 | **0–1 小数**或 `null`，展示时乘 100% |
| `source_result_filename` | 本地原始结果文件名 | 仅作来源标识，不代表文件在仓库中 |
| `note` | 证据限制说明 | 不得省略关键未核实事项 |

部分旧文件还有 `params`、`per_type` 等特有字段，保留原始结构；其中 `per_type.accuracy` 使用百分数，应按各文件实际口径解读，不能与顶层 `accuracy` 直接混用。

1M 额外保留审计字段（`dataset_rows`、`result_rows`、`scored_rows`、`missing`、`duplicate_question_ids`、`error_rows`）。这些字段只表示记录完整性：`scored_rows` 不是 `correct`。

历史 32K、128K 和 LongMemEval 的 `metrics.json` 暂不改动，以保证已有统计及来源可追溯。未来获得可复核的 1M 正确数后，应根据同一批结果重新计算 `accuracy`，并保留核验来源、日期、评分规则和配置快照。
