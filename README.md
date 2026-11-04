# Jev-Audio-Adapter

**复用 FireRed 音频编码器，通过 Adapter 将 D3 决策模型拓展到音频。**

音频 → FireRed Encoder → Adapter → D3 → 决策输出

本仓库提供最新训练方法、评测结果与**待 GPU 验证的参考推理代码**；权重单独发布在 [ModelScope](https://modelscope.cn/models/MewTiger/Jev-Audio-Adapter)。不发布数据集、音频或逐样本输出。

## 下载与参考推理

- **代码**：[inference.py](inference.py)、[依赖](requirements-inference.txt)、[推理说明](docs/inference.md)。
- **权重**：ModelScope 的 `model.safetensors`，固定 revision `858cff0254c976a615f05f2d4746b8584c811c3d`；[发布清单](release_manifest.json)记录 SHA-256 和基础模型版本。
- **能力**：音频 + 问题 + 候选答案 → 分类选择与 logits，不是开放式文本生成。
- **验证边界**：已完成 CPU 契约测试、真实权重结构检查及小型 Qwen3.5 前向一致性检查；**完整 GPU 加载和真实音频端到端推理尚未验证，不保证开箱即用**。原有评测结果不等同于这份新 CLI 的复测结果。
- **离线加载**：适配器强制 SHA-256 校验；本地冻结基座仅检查必需文件和结构，未强制逐文件哈希校验，请使用文档指定 revision。
- **许可**：原创推理代码目前没有单独授予复用许可；公开可见不代表可自由复用或商用。适配器保持已发布的非商用许可；第三方源码保留原许可，见[声明](THIRD_PARTY_NOTICES.md)。

## 模型对比 · Accuracy ↑（%）

以下按模型横向展示公开报告值与本项目最新结果，**不是相同数据版本、预处理和评分协议下的统一复测排名**。外部结果未在本仓库重跑；“—”表示未报告或不适用，不表示零分。

| 模型 / 结果来源 | CochlScene | NSynth | NonSpeech7k | MELD | MMAU | MMSU | MMAR |
|---|---:|---:|---:|---:|---:|---:|---:|
| [Qwen-Audio-Jev · 论文 v2](https://arxiv.org/html/2610.21614v2#S4) | 78.96 | 74.24 | 90.48 | 58.15 | 78.40 | 83.14 | 70.00 |
| [AudioJev（shlv）· 论文三 seed 均值](https://arxiv.org/html/2610.01293v1#A4) | — | 67.90 ± 0.92† | — | — | 70.90 ± 0.35‡ | 61.70 ± 0.68† | 55.33 ± 0.84 |
| [Toddler-Jev · 模型卡](https://huggingface.co/TeenageWorks/Toddler-Jev#evaluation-results) | — | — | — | — | 68.70 | 61.30 | 56.90 |
| [Jev-Omni · 模型卡](https://huggingface.co/akhilaaa3/Jev-Omni#results) | — | — | — | — | 63.10 | — | — |
| **Jev-Audio-Adapter · FireRed + D3-Flash** | **77.39** | **66.99** | **92.69** | **60.19** | 尚未运行 | **76.16** | 未完成 |

### 来源与口径

- **Qwen-Audio-Jev**：锁定论文 v2 表 2；论文将 78.40% 标为 MMAU，未据此认定与本项目 1,000 题划分完全相同。
- **AudioJev（shlv）**：独立外部项目，不是本仓库。采用论文附录 D 的 native-order 三 seed 均值 ± 样本标准差。† NSynth 为 512 题 note-property 测试、MMSU 为 3,931 题短音频子集，**与本项目 4,096 / 5,000 题不是同一评测集合**。‡ MMAU 列采用 public 1,000 题的 70.90%，不混用 full 10,000 题的 68.88%。
- **Toddler-Jev**：模型卡报告 MMAU-mini 1,000 题、MMSU 5,000 题、MMAR 1,000 题；成绩测于最终续训之前，不等同当前最终权重重测。
- **Jev-Omni**：模型卡报告 MMAU 1,000 题的 micro accuracy；未明确与本项目相同的数据 revision。
- **本项目**：使用训练第 2,250 步 checkpoint，仅发布五个已完整完成测试集的结果。不同来源的 split、任务定义、音频处理、候选顺序及评分方式未全部对齐，不据本表直接宣称领先。

## 本项目最新评测明细

Accuracy 由正确数除以完整测试集题数计算；上表保留两位小数，下表保留四位小数。

| 数据集 | 正确数 / 总题数 | Accuracy（%） |
|---|---:|---:|
| CochlScene | 5,949 / 7,687 | 77.3904 |
| NSynth | 2,744 / 4,096 | 66.9922 |
| NonSpeech7k | 672 / 725 | 92.6897 |
| MELD | 1,571 / 2,610 | 60.1916 |
| MMSU | 3,808 / 5,000 | 76.1600 |

全套计划为七个测试集、22,118 题。本轮完成 20,186 题，执行错误为 0；MMAR 仅完成 68 / 1,000，MMAU 尚未运行。未完成测试集不发布完整集成绩，不计作零分，也不计算七项平均或全套合并准确率。

本轮不作为独立盲测：checkpoint 的选择允许参考测试集结果。以上仅代表当前 checkpoint 的实际结果，尚未完成多 checkpoint 的完整比较。

## 方法与协议

- [架构与最新训练方法](docs/architecture.md)
- [数据使用与范围](docs/training-data.md)
- [评测协议](docs/evaluation-protocol.md)

结果基于本轮公共音频预处理和完整题目集合，不主张与未对齐协议的外部结果严格可比。
