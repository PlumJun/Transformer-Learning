# Transformer Learning

Transformer 课程学习笔记与带中文注释的 PyTorch 实现。笔记按概念、公式、模块及训练推理流程整理，代码实现一个用于**英语到中文翻译**的 Encoder–Decoder Transformer。

## 从这里开始

- [完整学习笔记](<Transformer学习笔记/Transformer学习笔记（整理版）.md>)：13 个核心主题，包含公式推导、维度表、17 张课程图示、易错点速查与自检问题。
- [模型实现](Transformer_code/model/tf_model.py)：Embedding、正弦位置编码、注意力、Encoder、Decoder、FFN 与输出层。
- [训练入口](Transformer_code/main.py)：训练、验证、损失与 BLEU 评估。
- [翻译入口](Transformer_code/translate.py)：加载权重进行交互式单句翻译。
- [配置文件](Transformer_code/config.py)：模型规模、设备、数据路径与权重路径。

## 目录结构

```text
Transformer-Learning/
├── README.md
├── .gitignore
├── .gitattributes
├── Transformer学习笔记/
│   ├── Transformer学习笔记（整理版）.md
│   └── 所用图片/                 # 笔记引用的原始课程图片
└── Transformer_code/
    ├── main.py                   # 训练、验证与测试入口
    ├── translate.py              # 交互式翻译
    ├── beam_decoder.py           # Beam Search 解码
    ├── config.py                 # 模型、设备及路径配置
    ├── requirements.txt
    ├── model/
    │   ├── tf_model.py           # Transformer 各模块
    │   └── train_utils.py        # 分块损失计算、Noam 学习率调度
    ├── tools/
    │   ├── data_loader.py        # 数据集、批次整理与 Mask
    │   ├── tokenizer_utils.py    # SentencePiece 分词器加载
    │   └── create_exp_folder.py  # 实验结果目录管理
    ├── data/                     # 语料分析与转换脚本
    └── tokenizer/                # 分词器脚本、模型与词表
```

## 笔记与代码对照

| 学习主题 | 主要代码位置 |
| --- | --- |
| 词嵌入 | `model/tf_model.py` → `Embeddings` |
| 正弦位置编码 | `model/tf_model.py` → `PositionalEncoding` |
| 缩放点积注意力 | `model/tf_model.py` → `attention` |
| 多头注意力与交叉注意力 | `model/tf_model.py` → `MultiHeadedAttention`、`DecoderLayer` |
| 填充掩码、因果掩码、目标右移 | `tools/data_loader.py` → `subsequent_mask`、`Batch` |
| 残差与归一化 | `model/tf_model.py` → `SublayerConnection`、`LayerNorm` |
| 前馈网络 | `model/tf_model.py` → `PositionwiseFeedForward` |
| 编码器与解码器堆叠 | `model/tf_model.py` → `Encoder`、`Decoder`、`Transformer` |
| 输出概率 | `model/tf_model.py` → `Generator` |
| 训练与学习率调度 | `main.py`、`model/train_utils.py` |
| 自回归解码与束搜索 | `translate.py`、`beam_decoder.py` |

笔记主要沿课程的经典 Post-LN 架构讲解，同时说明 Pre-LN。当前代码中 `SublayerConnection.forward` 使用 `x + dropout(sublayer(norm(x)))`，属于 **Pre-LN**，Encoder 与 Decoder 堆叠后还有最终归一化。对照阅读时，应以具体实现的计算顺序为准。

## 环境与运行

### 1. 准备依赖

原依赖文件记录的 Python 环境为 **Python 3.10**。请先安装适合本机设备的 PyTorch，再安装其余依赖；`requirements.txt` 未直接安装 PyTorch。

```bash
cd Transformer_code
python -m pip install -r requirements.txt
```

依赖包括 `sentencepiece`、`sacrebleu`、`numpy` 与 `tqdm`。PyTorch 与 CUDA 的历史环境记录保留在 `requirements.txt` 注释中。

### 2. 准备训练数据

把英中平行语料放到以下位置：

```text
Transformer_code/data/json/train.json
Transformer_code/data/json/dev.json
Transformer_code/data/json/test.json
```

每个文件是 JSON 数组，每项为 `[英文句子, 中文句子]`，例如：

```json
[
  ["I am a student.", "我是一名学生。"],
  ["The robot picks up a red cup.", "机器人抓取一个红色杯子。"]
]
```

仓库附带 `tokenizer/eng.model`、`tokenizer/chn.model` 及对应词表，可以直接供现有代码加载。若要重新训练分词器，先在 `data` 目录运行 `get_corpus.py` 生成语料，再在 `tokenizer` 目录运行 `tokenize.py`，并确保词表大小、特殊 Token ID 与 `config.py` 一致。

### 3. 检查配置后训练

当前训练入口使用 CUDA / DataParallel 相关实现，默认设备为 GPU 0。运行前请检查 `config.py` 中的设备、词表大小、数据路径与训练参数。

```bash
# 在 Transformer_code 目录中执行
python main.py
```

训练权重写入 `run/train/exp*/weights/`。默认模型配置为 `d_model=512`、8 个注意力头、Encoder / Decoder 各 6 层、`d_ff=2048`。

### 4. 加载权重翻译

把训练好的权重放回本地，并将 `config.py` 中的 `test_model_path` 指向对应文件，然后运行：

```bash
# 在 Transformer_code 目录中执行
python translate.py
```

程序会提示输入英文句子，并输出中文翻译。当前上传保留原有源码，未重新运行完整 GPU 训练或验证 BLEU 分数。

## 仓库包含哪些资料？

仓库包含整理版 Markdown、其引用的全部 17 张原始图片、Python 源码、依赖文件与 SentencePiece 分词器文件。

原始训练语料与模型检查点保留在本地：语料及派生数据约 **159 MiB**，两个训练检查点合计约 **751 MiB**。这些大文件、训练输出、Python 缓存、IDE 配置和本地环境文件已由 `.gitignore` 排除。克隆仓库后，需要自行准备语料或放回已有检查点才能训练或加载翻译模型。

课程截图作为个人学习资料保留，图内来源信息未改动；笔记末尾列有核对概念使用的论文与官方文档。后续整理时请继续保留来源信息。
