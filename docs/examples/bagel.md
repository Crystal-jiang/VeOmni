# BAGEL training guide (NPU)

## Environment setup

```shell
# Source CANN environment
source /usr/local/cann/ascend-toolkit/set_env.sh

# Activate VeOmni venv (NPU extra)
source .venv/bin/activate

# Install additional dependencies
pip install diffusers pyarrow
```

## Download dataset

Download the BAGEL example dataset (contains T2I and VLM samples):

```shell
wget -O bagel_example.zip \
    https://lf3-static.bytednsdoc.com/obj/eden-cn/nuhojubrps/bagel_example.zip
unzip bagel_example.zip -d /path/to/data
```

The dataset structure:

```
bagel_example/
├── t2i/                    # Text-to-image generation data (parquet)
│   ├── chunk_0.parquet     # columns: image (binary), captions (JSON string)
│   └── ...
├── editing/                # Image editing data (parquet)
│   └── seedxedit_multi/
└── vlm/                    # VLM understanding data (jsonl)
    ├── images/             # JPEG/PNG images
    └── llava_ov_si.jsonl   # columns: id, image (filename), conversations (list)
```

The `BagelUnifiedDataset` adapter automatically reads both `vlm/` and `t2i/` subdirectories, converts them to a unified format, and interleaves understanding and generation samples.

### Data format

Each sample is converted to one of two formats:

**Understanding** (from `vlm/`):
```python
{"task": "understanding", "prompt": "Describe this image.", "response": "A cat sitting on a sofa.", "image": PIL.Image}
```

**Generation** (from `t2i/`):
```python
{"task": "generation", "prompt": "A green towel with brushes and soap.", "image": PIL.Image}
```

## Download and convert model weights

### Download BAGEL-7B-MoT

```python
from huggingface_hub import snapshot_download

save_dir = "/path/to/BAGEL-7B-MoT"
snapshot_download(
    cache_dir=save_dir + "/cache",
    local_dir=save_dir,
    repo_id="ByteDance-Seed/BAGEL-7B-MoT",
    local_dir_use_symlinks=False,
    resume_download=True,
    allow_patterns=["*.json", "*.safetensors", "*.bin", "*.py", "*.md", "*.txt"],
)
```

### Convert weights

The upstream BAGEL checkpoint is monolithic. VeOmni requires it split into three submodules:

```shell
python scripts/multimodal/convert_bagel.py \
    --model_path /path/to/BAGEL-7B-MoT \
    --output_dir ./bagel_veomni \
    --vit_select_layer -2 \
    --vit_rope
```

Output structure:

```
bagel_veomni/
├── bagel_foundation/       # MoT LLM backbone + tokenizer
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
├── bagel_vision_encoder/   # SigLIP NaViT (understanding)
│   ├── config.json
│   └── model.safetensors
└── bagel_vision_decoder/   # FLUX VAE + rectified-flow (generation)
    ├── config.json
    └── model.safetensors
```

## Start training on NPU

### 8-card NPU (recommended)

```shell
source /usr/local/cann/ascend-toolkit/set_env.sh
source .venv/bin/activate

export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

bash train.sh tasks/omni/train_omni_model.py configs/multimodal/omni/bagel.yaml \
  2>&1 | tee logs/bagel_train_$(date +%Y%m%d_%H%M%S).log

```

### Training output

Training produces two types of loss:

```
loss: X.XX  foundation_ce_loss: X.XX  image_decoder_mse_loss: X.XX  grad_norm: X.XX  lr: X.XX
```

- `foundation_ce_loss`: Cross-entropy loss on understanding tokens (next-token prediction)
- `image_decoder_mse_loss`: Rectified-flow MSE loss on generation tokens (velocity prediction)

Since `micro_batch_size=1`, each step processes one sample, so only one loss type appears per step depending on whether the sample is understanding or generation.

### Checkpoint

Checkpoints are saved in DCP (Distributed Checkpoint) format under `bagel_sft/checkpoints/`:

```
bagel_sft/
├── checkpoints/
│   ├── global_step_10/
│   ├── global_step_20/
│   └── ...
└── model_assets/
    ├── config.json
    ├── tokenizer.json
    └── ...
```

## Known limitations

1. **Memory**: BAGEL-7B-MoT (~14B total parameters) is tight on 64GB NPU. 8 cards with CPU offload + gradient checkpointing + frozen foundation/encoder is the minimum viable configuration.
2. **Inference**: Only the training path is implemented. The denoising loop for image generation inference is not yet available.
3. **EMA**: The upstream BAGEL uses EMA weights (decay=0.9999) during training. VeOmni does not currently support EMA.
4. **CFG dropout**: The upstream BAGEL applies classifier-free guidance dropout at the data level. VeOmni's `BagelSampleTransform` does not implement this.
