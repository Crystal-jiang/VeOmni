# BAGEL-7B-MoT NPU 训练指南

本文档覆盖从零开始拉起 BAGEL-7B-MoT SeedOmni V2 训练的全部步骤，包括环境准备、数据下载与预处理、模型权重转换、配置修改和训练启动。

---

## 一、环境准备

### 1.1 依赖安装

```bash
cd /path/to/VeOmni
uv sync --extra npu --dev
source .venv/bin/activate
```

需要 `transformers >= 5.9.0`（代码中 `qwen3vl` 模块依赖 `accepts_precomputed_kwargs`，低版本会 import 失败）。

### 1.2 CANN 环境

```bash
source /home/cann/CANN8.5.2/ascend-toolkit/set_env.sh
```

### 1.3 验证

```bash
python -c "import transformers; print(transformers.__version__)"  # 需 >= 5.9.0
npu-smi info -l  # 确认 NPU 卡可见
```

---

## 二、模型权重

### 2.1 下载上游 checkpoint

从 HuggingFace 下载 BAGEL-7B-MoT 原始权重：

```bash
python scripts/download_hf_data.py \
  --repo_id BAAI/BAGEL-7B-MoT \
  --local_dir /path/to/BAGEL-7B-MoT
```

下载后目录结构：

```
BAGEL-7B-MoT/
├── config.json          # model_type: "bagel"
├── llm_config.json      # Qwen2 LLM 配置
├── vit_config.json      # SigLIP ViT 配置
├── ema.safetensors      # EMA 权重（text encoder + Qwen2 MoT + SigLIP + flow connector）
├── ae.safetensors       # VAE 权重
├── tokenizer.json
├── tokenizer_config.json
├── vocab.json
├── merges.txt
└── ...
```

### 2.2 转换为 SeedOmni V2 格式

将单体 checkpoint 拆分为 5 个子模块：

```bash
python scripts/convert_model.py \
  --model_path /path/to/BAGEL-7B-MoT \
  --output_dir /path/to/BAGEL-7B-MoT-seedomni
```

输出目录结构：

```
BAGEL-7B-MoT-seedomni/
├── bagel_text_encoder/      # tokenizer + embedding + lm_head
├── bagel_siglip_navit/      # SigLIP ViT + connector + vit_pos_embed
├── bagel_vae/               # VAE encoder/decoder
├── bagel_flow_connector/    # VAE↔LLM 投影 + timestep embedding
└── bagel_qwen2_mot/         # Qwen2-MoT backbone
```

> **注意**：`vit_pos_embed.pos_embed` 和 `latent_pos_embed.pos_embed` 是确定性 sin-cos 位置编码，转换时允许 missing 并从 config 重新生成。如果输出目录已存在，需先删除或加 `--force`。

### 2.3 修改配置

编辑 `configs/seed_omni/Bagel/bagel_7b_mot/base.yaml`，将 `model.model_path` 和 `infer.model_path` 改为转换后的路径：

```yaml
model:
  model_path: /path/to/BAGEL-7B-MoT-seedomni

infer:
  model_path: /path/to/BAGEL-7B-MoT-seedomni
```

---

## 三、数据集

BAGEL 训练使用三个数据源，按权重 `[0.5, 0.2, 0.3]` 混合采样。每个数据源需要转为 **多分片 parquet** 格式（单 JSON 文件不支持流式 DataLoader reset）。

### 3.1 ImageNet-1K（T2I 文本生成图像，权重 0.5）

**下载**：

```bash
# 方式一：HuggingFace（需签署协议）
huggingface-cli download ILSVRC/imagenet-1k --repo-type dataset --local-dir /path/to/imagenet-1k

# 方式二：官网下载 tar 包后解压
# https://image-net.org/ → 登录 → 下载 ILSVRC2012_img_train.tar
```

**转换**：

```bash
python scripts/prepare_imagenet1k.py \
  --input /path/to/imagenet-1k/data \
  --classes /path/to/imagenet-1k/classes.py \
  --output /home/usr/data/imagenet1k_train \
  --num_shards 10
```

- 输入：HF parquet（`image: struct{bytes, path}`, `label: int64`）
- 输出：多分片 parquet（`conversations: str` 逗号分隔类别标签, `images: list[str]` 图片路径）
- 图片文件保存到 `/home/usr/data/imagenet1k_images/`（与 parquet 目录平级，避免 `get_data_files` 扫到子目录）
- 约 101 万条样本，10 个分片

**数据语义**：user 给文本描述（如 `"golden retriever"`），assistant 生成对应图片。图片路由到 `bagel_vae_context`，产生 `decode_velocity` loss。

### 3.2 Tulu-3-SFT-Mixture（纯文本 SFT，权重 0.2）

**下载**：

```bash
python scripts/download_hf_data.py \
  --repo_id allenai/tulu-3-sft-mixture \
  --local_dir /home/dataset/tulu-3-sft-mixture
```

**无需转换**。原始 parquet 格式（`id`, `messages`, `source`）直接可用。

**数据语义**：纯文本多轮对话，无图片，只产生 `bagel_text_encoder.decode` CE loss。

### 3.3 ShareGPT4V（I2T 图像理解，权重 0.3）

**下载**：

```bash
# 标注 JSON
wget https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/resolve/main/sharegpt4v_instruct_gpt4-vision_cap100k.json

# COCO2017 图片
wget https://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip -d /home/dataset/coco/
```

**转换**：

```bash
python scripts/prepare_sharegpt4v.py \
  --input /path/to/sharegpt4v_instruct_gpt4-vision_cap100k.json \
  --image_root /home/dataset \
  --output /home/usr/data/sharegpt4v_cap_100k \
  --num_shards 6 \
  --filter coco
```

- `--image_root`：图片路径前缀（JSON 中 `image` 字段是相对路径如 `coco/train2017/xxx.jpg`）
- `--filter coco`：只保留 COCO 图片子集（约 5 万条）
- 输出：多分片 parquet（`conversations: list[struct]` ShareGPT 格式, `images: list[str]` 绝对路径）

**数据语义**：user 给图片 + 文本问题，assistant 回复文本描述。图片路由到 `bagel_siglip_context`（SigLIP 理解分支）。

### 3.4 配置数据源

编辑 `configs/seed_omni/Bagel/bagel_7b_mot/data.yaml`：

```yaml
sources:
  - /home/usr/data/imagenet1k_train
  - /home/dataset/tulu-3-sft-mixture
  - /home/usr/data/sharegpt4v_cap_100k
names:
  - imagenet1k
  - tulu-3-sft-mixture
  - sharegpt4v_cap_100k
schedule:
  - schedule_type: const
    weights: [0.5, 0.2, 0.3]
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

> **重要**：`sources` 路径下只能包含 `.parquet` 文件，不能有子目录（如 `images/`），否则 `get_data_files` 会报 "files are not supported" 错误。图片文件应存放在与 parquet 目录平级的位置。

---

## 四、启动训练

### 4.1 训练脚本

使用 `run_bagel.sh`：

```bash
#!/bin/bash
source /home/cann/CANN8.5.2/ascend-toolkit/set_env.sh
source .venv/bin/activate

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MASTER_PORT=$((40000 + RANDOM % 10000))

bash train.sh tasks/omni/train_omni.py configs/seed_omni/Bagel/bagel_7b_mot/base.yaml \
  --train.max_steps 20 \
  2>&1 | tee logs_v2/bagel_npu_train_$(date +%Y%m%d_%H%M%S).log
```

启动：

```bash
./run_bagel.sh
```

### 4.2 关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--train.max_steps` | 20 | 训练步数（覆盖 yaml 中的值） |
| `--train.global_batch_size` | 8 | 全局 batch size |
| `--train.micro_batch_size` | 1 | 每卡 micro batch size |
| `ASCEND_RT_VISIBLE_DEVICES` | `0,1,2,3,4,5,6,7` | 使用的 NPU 卡号 |
| `MASTER_PORT` | 随机 40000-49999 | 分布式通信端口 |

### 4.3 日志

训练日志自动保存到 `logs_v2/bagel_npu_train_<时间戳>.log`。

### 4.4 预期输出

三数据集混合训练时，每个 step 的 loss 包含两部分：

- `bagel_text_encoder.decode`：文本 CE loss（所有数据源都有）
- `bagel_flow_connector.decode_velocity`：图像生成 flow matching loss（仅 imagenet1k 样本产生，其他数据源为 0）

正常训练时 loss 范围约 0.5 ~ 1.1，grad_norm 约 1.5 ~ 5.5。
