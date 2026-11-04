# 架构与最新训练方法

## 模型

音频 → FireRedAudio Encoder → 原生 Adapter → D3-Flash → 原生决策 readout。

决策主干使用 `vllm-sr/d3-flash`。采用最后一个有效 token 的表示和原生 255 码 readout 进行分类，不先生成 ASR 转写，也不通过自回归生成答案完成分类。

## 两阶段训练

| 阶段 | 初始化 | 可训练模块 | 更新数 |
|---|---|---|---:|
| Stage 1 | 预训练组件 | Adapter | 94 |
| Stage 2 | Stage 1 最佳 checkpoint | Adapter、D3 注意力投影 LoRA、原生 readout | 2,464 |

Stage 1 使用确定性打乱后的混合训练清单，仅执行 94 次全局 batch 更新，不代表完整遍历训练集。

两个阶段均冻结音频 Encoder；Stage 2 除 LoRA 和 readout 外的 D3 主干参数冻结。LoRA 应用于全部注意力层的 q/k/v/o 投影，rank 为 16，alpha 为 32，dropout 为 0.05。

Stage 2 完成 2 个 epoch，共 78,824 次样本曝光。本次报告第 2,250 次更新保存的 checkpoint，而非第 2,464 次更新的训练结束权重。

## 优化与精度

- 优化器：AdamW，weight decay 为 0。
- Adapter 基础学习率：`5e-6`。
- LoRA 与 readout 基础学习率：`2e-5`。
- 学习率调度：3% warmup，之后 cosine decay。
- 梯度裁剪：最大范数 1。
- 全局 batch：32；Stage 2 micro-batch 为 4，梯度累计 8 次，尾批按实际样本数归一化。
- 随机种子：20261022。
- 不使用量化；冻结主干采用 BF16，可训练 Adapter、LoRA、readout 采用 FP32。

## 训练数据与音频

训练集 39,412 条，验证集 2,773 条。两个阶段使用同一混合数据划分，来源见 [数据说明](training-data.md)。

训练音频为 16 kHz 单声道，仅保留 0.1–30 秒范围，超长训练音频过滤而非截断。本轮评测保留长音频的完整输入，使用支持完整长度的适配实现；训练过滤与评测覆盖不可混为一谈。

## checkpoint 与结果范围

第 2,250 步 checkpoint 为训练中按 dev 指标保存的已有候选。本轮后续模型选择允许参考测试集结果，因此公开结果仍标明 **test-selected; no independent test**，不据此声称新的独立盲测。

第 2,464 步训练结束权重尚未完成本轮评测，未进行两个 checkpoint 的完整结果比较。已完成的五项指标见 [主页](../README.md)，评分与未完成范围见 [评测协议](evaluation-protocol.md)。

本仓库当前只提供方法文档；尚未发布实现、精确依赖版本、模型权重和完整训练 manifest，不能视为可直接运行的复现包。
