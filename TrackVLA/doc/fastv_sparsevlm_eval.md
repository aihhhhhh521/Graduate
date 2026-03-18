# 在当前项目中复现 FastV / SparseVLM 风格方法并做同指标评测

> 目标：在 **不改 Benchmark 指标定义** 的前提下，把 FastV 与 SparseVLM 思路映射到当前 Uni-NaVid + TrackVLA 的推理链路，并复用现有 SR/TR/CR 与 step-level 速度统计流程。

## 1. 先对齐“方法映射”

当前仓库已经提供了可控的 token ablation 开关（`--token-ablation-mode`），可用于做第一版“方法等价实验”：

- **FastV-like**：`pool_all_2x2_to_1x1`
  - 对历史帧视觉 token 的 2x2 block 做 pooling（4 -> 1）。
  - 等价于“保留信息但降低 token 数”的稀疏/压缩路线。
- **SparseVLM-like**：`drop_history_keep_latest_nav64`
  - 丢弃历史 token，仅保留最新导航帧的 8x8（64）token。
  - 等价于“激进稀疏化 + 最新帧优先”。

这两种模式在 `uninavid_arch.py` 中直接生效，无需改评测代码。

## 2. 指标与日志（与现有评测保持一致）

推荐使用 `run_patched_stepstats.py`，它会额外输出：

- `step_stats_split*.jsonl`：每步耗时/FPS
- `speed_summary_split*.json`：速度统计汇总
- `metrics_summary_split*.json`：每 split 的 SR/TR/CR 兼容统计
- `run_meta.json`：记录 seed、commit、ablation 配置

这样可以做到“同一数据、同一指标、不同 token 策略”的公平对比。

## 3. 最小可运行命令

### 3.1 STT 配置上跑 FastV-like

```bash
PYTHONPATH="habitat-lab" python TrackVLA/run_patched_stepstats.py \
  --run-type eval \
  --exp-config TrackVLA/habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml \
  --split-num 30 \
  --split-id 0 \
  --save-path TrackVLA/exp_results/ablation/fastv_like/stt/split0 \
  --model-path Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
  --model-name uni-navid \
  --enable-step-stats \
  --log-every-n-steps 1 \
  --seed 100 \
  --token-ablation-mode pool_all_2x2_to_1x1 \
  --online-cache-prune-mode step_window
```

### 3.2 STT 配置上跑 SparseVLM-like

```bash
PYTHONPATH="habitat-lab" python TrackVLA/run_patched_stepstats.py \
  --run-type eval \
  --exp-config TrackVLA/habitat-lab/habitat/config/benchmark/nav/track/track_infer_stt.yaml \
  --split-num 30 \
  --split-id 0 \
  --save-path TrackVLA/exp_results/ablation/sparsevlm_like/stt/split0 \
  --model-path Uni-NaVid/model_zoo/uninavid-7b-full-224-video-fps-1-grid-2 \
  --model-name uni-navid \
  --enable-step-stats \
  --log-every-n-steps 1 \
  --seed 100 \
  --token-ablation-mode drop_history_keep_latest_nav64 \
  --online-cache-prune-mode step_window
```

> 建议 baseline 也跑一次（`--token-ablation-mode` 不传），用于三方对比：Baseline / FastV-like / SparseVLM-like。

## 4. 多 split 与聚合建议

1. 固定同一 `split-num`（例如 30）与同一 seed（例如 100）。
2. 每个 split 结果写入独立目录（`.../split{ID}`）。
3. 对每个方法分别聚合：
   - 任务指标：
     ```bash
     python TrackVLA/analyze_results.py --path <method_output_root>
     ```
   - 速度指标：汇总各 split 的 `speed_summary_split*.json`（平均/中位数/P95）。
4. 最终报告建议包含：
   - SR/TR/CR
   - 平均 step latency、P95 latency、平均 FPS
   - 相对 baseline 的增益（速度）与回退（任务成功率）

## 5. 如果你要“更贴近论文原始 FastV / SparseVLM”

当前实现是“方法思想映射版”。如果要更贴近原始设计，可在 `uninavid_arch.py` 继续扩展：

- 新增 mode（如 `fastv_topk_attn`、`sparsevlm_router`）
- 在 `_apply_history_token_ablation` 中实现更细粒度选择策略（例如按注意力分数 Top-K，而不是固定 pooling/drop）
- 在 `run_patched_stepstats.py` 的 `choices` 中注册新 mode

完成后仍走同一评测脚本，即可直接和现有结果横向比较。

## 6. 常见坑

- **路径问题**：`run_patched_stepstats.py` 默认相对路径，建议从 `TrackVLA/` 目录执行。
- **显存波动**：优先使用 `--online-cache-prune-mode step_window`。
- **可复现性**：固定 seed，并保留 `run_meta.json`。
- **公平性**：不同方法必须使用完全相同的 split 与 config（STT/AT 分开报告）。
