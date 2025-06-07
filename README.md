# Pun Generating Models: Aligning Your Model with Humor Preferences

## 简介

在本次研究中，我们深入探索了如何通过结构化与分阶段的训练方式，使大语言模型能更好地贴合人类对双关语的偏好。我们先实施了 PGCL - DPO 方法，而后将其优化为 PGCL - DPOP，这一方法有效应对了灾难性遗忘等难题，提升了模型生成有意义双关语的能力。实验数据表明，PGCL - DPOP 在双关结构的精准度与幽默效果的呈现上，均显著超越了基线方法。本研究充分展现了强化学习与偏好优化在幽默生成这类创造性语言任务中的潜力，为后续相关研究与应用拓展了思路、奠定了基础。

本文档仅介绍环境配置、数据集处理、模型训练、代码实现等相关内容，具体实验结果和分析请参考 [实验报告](report.pdf)。

## 实验环境

本次实验在以下环境中进行：

- 操作系统：Ubuntu 22.04
- Python 版本：3.10
- CUDA 版本：12.4
- GPU：NVIDIA A800 80GB

## 安装及配置

### 安装LLaMA-Factory

```bash
cd PGM
conda create -n pgm python=3.10
pip install -e ".[torch,metrics]"
```

使用 `llamafactory-cli version` 来快速校验安装是否成功

同时，为了满足模型批量推理的需求，我们还需要安装 `vllm`：

```bash
pip install vllm==0.5.4
```

### Checkpoint 下载

本次实验选择了 Qwen2.5-3B 作为基础模型，您可以通过以下命令下载模型权重：

```bash
bash ckpt_download.sh
```

也可以选择手动下载模型权重，下载地址为 [Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B/tree/main)。

如果您希望使用其他模型进行实验，请将下载的模型权重放置在 `LLM/` 目录下，并在配置文件中指定相应的模型路径。

## 训练流程

+ 数据准备
  + 收集和清洗数据集
  + 数据增强
+ 模型训练
  + 超参数调优
  + 训练监控
+ 评估与优化
  + 性能评估
  + 模型优化

## 数据集

[ChinesePun](ChinesePun.json) 数据集包含了 1000 多条中文双关语句子，涵盖了同形异义和同音异义双关语。每条双关语句子都包含了双关结构和对应的双关词语，为模型的训练提供了丰富的语料基础。

## 评价指标

+ 平均长度
+ 语料/句子多样性
  + Dist1: 语料/句子中独特单词的比例
  + Dist2: 语料/句子中独特双词组的比例
+ 结构成功率: 模型自动判断双关语是否在句子中使用
+ 双关语成功率: 人工判断双关语是否正确使用

## 实验过程

### 前置参数

在开始实验之前，我们需要设置一些前置参数，包括模型名称、模板、数据集类型等。以下是一个示例配置：

```bash
type=homophonic
model=Qwen2.5-3B
template=qwen
```

### 数据预处理

在正式训练之前，我们对数据集进行了清洗和格式化处理。我们将双关语数据集拆分为同形异义双关语和同音异义双关语两部分，分别存储在 `homographic.json`和 `homophonic.json`两个文件中，并将其输出到 `data/`目录下，以便后续针对模型在不同类型双关语生成的能力进行优化和评估。

```bash
python process_dataset.py --output_dir ./data --repeat_times 10
```

### 模型推理

DPO 方法的核心在于通过对比学习和动态偏好优化来提升模型生成双关语的能力。因此，我们首先需要对模型进行推理，以生成初步的双关语句子。我们使用 `vllm` 进行批量推理，命令如下：

```bash
python vllm_infer.py \
  --model_name_or_path ./LLM/${model} \
  --template ${template} \
  --dataset ${type}_cn_generate \
  --save_name ./${model}-${type}-cn.json \
  --top_p 0.95 \
  --top_k 5 \
  --temperature 0.95
```

对于已经存在LoRA适配器的模型，我们可以使用以下命令进行推理：

```bash
python vllm_infer.py \
  --model_name_or_path ./LLM/${model} \
  --adapter_name_or_path ./saves/${model}/${type}/stage1 \
  --template ${template} \
  --dataset ${type}_cn_generate \
  --save_name ./${model}-${type}-cn.json \
  --top_p 0.95 \
  --top_k 5 \
  --temperature 0.95
```

### DPO数据集构建

在LLama-Factory中，使用DPO方法需要构建一个特定格式的数据集（prompt + chosen/rejected），例如[dpo_zh_demo.json](/home/amax/liujingyuan/PGCL/data/dpo_zh_demo.json)。因此，我们需要将生成的双关语句子转换为DPO格式。我们可以使用以下命令来完成这一任务：

```bash
python reconstruct.py \
  --raw_file ./data/${type}_cn.json \
  --generate_file ./${model}-${type}-cn.json \
  --output_file ./data/dpo_${type}_cn.json
```

### 第一阶段：结构偏好对齐

在第一阶段，我们使用 DPOP 方法对模型进行微调，以对齐模型的双关语结构偏好。我们筛选出生成的双关语句子中不满足双关结构的句子，并将其作为拒绝样本。然后，我们使用以下命令进行训练：

```bash
llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path ./LLM/${model} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dpo_${type}_cn \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir ./saves/${model}/${type}/stage1 \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta 0.5 \
    --pref_ftx 0 \
    --pref_loss dpop \
    --dpop_lambda 0.5 \
    --top_p 0.95 \
    --top_k 5 \
    --temperature 0.95
```

### 第二阶段：幽默偏好对齐

在第二阶段，我们进一步优化模型，使其能够生成更具幽默感的双关语。我们使用自定义的Humor DPO方法，构建出了一个三元组数据集，其中包含了符合双关语结构且具有幽默感的句子$y^{+}$，符合双关语结构但不具幽默感的句子$y^{*}$，以及不符合双关语结构的句子$y_{h^*}^{-}$。

具体公式如下：

$$
r_{\phi}^{+}(\theta) = \beta \log \frac{\pi_{\theta}(y^{+}|x)}{\pi_{\phi}(y^{+}|x)}
$$

$$
r_{h^*}^{-}(\theta) = \beta \log \frac{\pi_{\theta}(y_{h^*}^{-}|x)}{\pi_{\phi}(y_{h^*}^{-}|x)}
$$

$$
r^{*}(\theta) = \beta \log \frac{\pi_{\theta}(y^{*}|x)}{\pi_{\phi}(y^{*}|x)}
$$

$$
\mathcal{L}_{I - humor - dpo}(\theta) = -\log\sigma(r_{\phi}^{+}(\theta) - r^{*}(\theta)) - \gamma \log\sigma(r^{*}(\theta) - r_{h^*}^{-}(\theta))
$$

其中，$\gamma$ 是一个超参数，用于控制结构偏好和幽默偏好的平衡。

我们使用以下命令进行训练：

```bash
llamafactory-cli train \
    --stage dpo \
    --do_train True \
    --model_name_or_path ./LLM/${model} \
    --adapter_name_or_path ./saves/${model}/${type}/stage1 \
    --model_name_or_path ./LLM/${model} \
    --preprocessing_num_workers 16 \
    --finetuning_type lora \
    --template qwen \
    --flash_attn auto \
    --dataset_dir data \
    --dataset dpo_${type}_cn \
    --cutoff_len 2048 \
    --learning_rate 5e-05 \
    --num_train_epochs 3.0 \
    --max_samples 100000 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --packing False \
    --report_to none \
    --output_dir ./saves/${model}/${type}/stage2 \
    --bf16 True \
    --plot_loss True \
    --overwrite_output_dir True \
    --trust_remote_code True \
    --ddp_timeout 180000000 \
    --include_num_input_tokens_seen True \
    --optim adamw_torch \
    --lora_rank 8 \
    --lora_alpha 16 \
    --lora_dropout 0 \
    --lora_target all \
    --pref_beta 0.5 \
    --pref_ftx 0 \
    --pref_loss humor \
    --humor_gamma 0.5
    --top_p 0.95 \
    --top_k 5 \
    --temperature 0.95
```

### 自动化脚本

为了简化训练流程，我们提供了一个自动化脚本 `train.sh`，您可以直接运行该脚本来完成模型的训练和评估。脚本会自动处理数据预处理、模型推理、DPO数据集构建以及模型训练等步骤。

```bash
bash train.sh
```

## 评价与结果

在模型训练完成后，我们使用以下命令对模型进行评估：

```bash
python evaluate.py ${filename}.json
```

同时，我们还可以使用以下命令调用模型进行推理，更加直观地查看模型生成的双关语句子：

```bash
llamafactory-cli chat \
    --model_name_or_path ./LLM/${model} \
    --adapter_name_or_path ./saves/${model}/${type}/stage1 \
    --template qwen \
    --infer_backend huggingface \
    --trust_remote_code True
```

也可以使用LLaMA-Factory提供的可视化界面进行交互式推理：

```bash
llamafactory-cli webui
```

## 代码修改

除了上述提到的[`process_dataset.py`](process_dataset.py),[`reconstruct.py`](reconstruct.py),[`vllm_infer.py`](vllm_infer.py)等数据集预处理和模型推理脚本，以及用于计算评价指标的[`evaluate.py`](evaluate.py)之外，我们还对 LLaMA-Factory 的核心代码进行了修改，以支持 DPOP 和 Humor DPO 方法的训练和评估。这些修改主要集中在数据集配置、模型训练、损失函数计算等方面。

以下仅为重点修改部分，一些细节修改请参考代码仓库，具体可以参考 [`src/llamafactory/train/dpo/workflow.py`](src/llamafactory/train/dpo/workflow.py) 中的运行流程，找到对应的函数和类。

首先，需要在 [`src/llamafactory/data/dataset.py`](src/llamafactory/data/dataset.py) 中添加新的数据集配置，以支持同形异义双关语和同音异义双关语的生成。

```json
"homographic_cn_generate":{
"file_name":"homographic_cn.json",
"columns":{
    "prompt":"prompt",
    "response":"chosen"
}
},
"homophonic_cn_generate":{
"file_name":"homophonic_cn.json",
"columns":{
    "prompt":"prompt",
    "response":"chosen"
}
},
"dpo_homographic_cn_1": {
"file_name": "dpo_homographic_cn_1.json",
"ranking": true,
"formatting": "sharegpt",
"columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "generate"
}
},
"dpo_homographic_cn_2": {
"file_name": "dpo_homographic_cn_2.json",
"ranking": true,
"formatting": "sharegpt",
"columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "generate",
    "gold": "gold"
}
},
"dpo_homophonic_cn_1": {
"file_name": "dpo_homophonic_cn_1.json",
"ranking": true,
"formatting": "sharegpt",
"columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "generate"
}
},
"dpo_homophonic_cn_2": {
"file_name": "dpo_homophonic_cn_2.json",
"ranking": true,
"formatting": "sharegpt",
"columns": {
    "messages": "conversations",
    "chosen": "chosen",
    "rejected": "generate",
    "gold": "gold"
}
}
```

在 [`src/llamafactory/train/dpo/trainer.py`](src/llamafactory/train/dpo/trainer.py) 中添加了 `humor_loss` 和 `dpop_loss` 方法，用于计算幽默偏好和双关结构偏好的损失函数。

```python
def dpop_loss(self, policy_chosen_logps: "torch.Tensor", policy_rejected_logps: "torch.Tensor", reference_chosen_logps: "torch.Tensor", reference_rejected_logps: "torch.Tensor") -> "torch.Tensor":
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    logits = pi_logratios - ref_logratios - self.dpop_lambda * torch.clamp_min(-pi_logratios, 0.0)
    dpop_loss = -F.logsigmoid(self.beta * logits)
    return dpop_loss
```

```python
def humor_loss(self, policy_chosen_logps: "torch.Tensor", policy_rejected_logps: "torch.Tensor", reference_chosen_logps: "torch.Tensor", reference_rejected_logps: "torch.Tensor", policy_gold_logps: "torch.Tensor", reference_gold_logps: "torch.Tensor") -> "torch.Tensor":
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps
  
    logits = pi_logratios - ref_logratios

    pi_gold_logratios = policy_gold_logps - policy_chosen_logps
    ref_gold_logratios = reference_gold_logps - reference_chosen_logps
    gold_logits = pi_gold_logratios - ref_gold_logratios
    humor_loss = -F.logsigmoid(self.beta * logits) + self.humor_gamma * (-F.logsigmoid(self.beta * gold_logits))
    return humor_loss
```

为了以上两个损失函数的计算，我们在 [`src/llamafactory/train/dpo/trainer.py`](src/llamafactory/train/dpo/trainer.py) 中添加了 `dpop_lambda` 和 `humor_gamma` 两个超参数，并在初始化时进行设置。

```python
def __init__(self, ...):
    ...
    self.beta = beta
    self.dpop_lambda = dpop_lambda
    self.humor_gamma = humor_gamma
    ...
```

在 [`src/llamafactory/train/dpo/trainer.py`](src/llamafactory/train/dpo/trainer.py) 中，我们还修改了 `compute_preference_loss` 方法，用于计算偏好学习的损失函数。该方法根据不同的损失类型（如 DPO、DPOP、Humor）来计算损失，并返回相应的奖励。

```python
def compute_preference_loss(
    self,
    policy_chosen_logps: "torch.Tensor",
    policy_rejected_logps: "torch.Tensor",
    policy_gold_logps: Optional[torch.FloatTensor],
    reference_chosen_logps: Optional["torch.Tensor"],
    reference_rejected_logps: Optional["torch.Tensor"],
    reference_gold_logps: Optional[torch.FloatTensor],
) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", Optional["torch.Tensor"]]:
    r"""Compute loss for preference learning."""
    if not self.finetuning_args.use_ref_model:
        if self.loss_type == "orpo":
            losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
        elif self.loss_type == "simpo":
            losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}.")

        chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
        rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
    else:
        if self.loss_type == "humor":
            losses = self.humor_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, policy_gold_logps, reference_gold_logps
            )
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
            gold_rewards = self.beta * (policy_gold_logps - reference_gold_logps).detach()
        elif self.loss_type == "dpop":
            losses = self.dpop_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
            rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        else:
            losses, chosen_rewards, rejected_rewards = self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )

    return losses, chosen_rewards, rejected_rewards, gold_rewards if self.loss_type == "humor" else None
```

由于 `compute_preference_loss`返回值个数的变化，我们还需要在 [`src/llamafactory/train/dpo/trainer.py`](src/llamafactory/train/dpo/trainer.py) 中修改 `get_batch_loss_metrics` 方法，以适应新的返回值。

```python
def get_batch_loss_metrics(
    self,
    model: "PreTrainedModel",
    batch: dict[str, "torch.Tensor"],
    train_eval: Literal["train", "eval"] = "train",
) -> tuple["torch.Tensor", dict[str, "torch.Tensor"]]:
    r"""Compute the DPO loss and other metrics for the given batch of inputs for train or test."""
    metrics = {}
    (
        policy_chosen_logps,
        policy_rejected_logps,
        policy_gold_logps,
        policy_chosen_logits,
        policy_rejected_logits,
        policy_gold_logits,
        policy_chosen_logps_avg,
    ) = self.concatenated_forward(model, batch)

    reference_chosen_logps, reference_rejected_logps, reference_gold_logps = self.compute_reference_log_probs(model, batch)
    losses, chosen_rewards, rejected_rewards, gold_rewards = self.compute_preference_loss(
        policy_chosen_logps,
        policy_rejected_logps,
        policy_gold_logps,
        reference_chosen_logps,
        reference_rejected_logps,
        reference_gold_logps,
    )
    sft_loss = -policy_chosen_logps_avg
    if self.ftx_gamma > 1e-6:
        losses += self.ftx_gamma * sft_loss

    prefix = "eval_" if train_eval == "eval" else ""
    metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().item()
    metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().item()
    if self.loss_type == "humor":
        metrics[f"{prefix}rewards/gold"] = gold_rewards.mean().item()
        metrics[f"{prefix}rewards/gold_vs_chosen"] = (gold_rewards - chosen_rewards).mean().item()
        metrics[f"{prefix}rewards/gold_vs_rejected"] = (gold_rewards - rejected_rewards).mean().item()
    metrics[f"{prefix}rewards/accuracies"] = (chosen_rewards > rejected_rewards).float().mean().item()
    metrics[f"{prefix}rewards/margins"] = (chosen_rewards - rejected_rewards).mean().item()
    metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.mean().item()
    metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.mean().item()
    if self.loss_type == "humor":
        metrics[f"{prefix}logps/gold"] = policy_gold_logps.detach().mean().item()
    metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.mean().item()
    metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.mean().item()
    if self.loss_type == "humor":
        metrics[f"{prefix}logits/gold"] = policy_gold_logits.mean().item()
    if self.loss_type == "orpo":
        metrics[f"{prefix}sft_loss"] = sft_loss.mean().item()
        metrics[f"{prefix}odds_ratio_loss"] = ((losses - sft_loss) / self.beta).mean().item()
  
    return losses.mean(), metrics
```

同时，由于原有的 DPO 方法只支持二元组的输入，即chosen 和 rejected，我们在 [`src/llamafactory/train/dpo/trainer.py`](src/llamafactory/train/dpo/trainer.py) 中添加了对三元组输入的支持，主要体现在 `tokenize_row` 和 `concatenated_inputs` 中。

```python
def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
    ···
    else:
        chosen_tokens = self.tokenizer(
            chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
        )
        rejected_tokens = self.tokenizer(
            rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
        )
        prompt_tokens = self.tokenizer(
            prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
        )
        if self.loss_type == "humor":
            gold_tokens = self.tokenizer(
                feature["gold"], truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )

        batch["chosen_labels"] = chosen_tokens["input_ids"]
        batch["rejected_labels"] = rejected_tokens["input_ids"]
        if self.loss_type == "humor":
            batch["gold_labels"] = gold_tokens["input_ids"]
        batch["prompt_input_ids"] = prompt_tokens["input_ids"]
        batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

        if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
            batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["rejected_labels"])
            )
            batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                labels=torch.tensor(batch["chosen_labels"])
            )
            if self.loss_type == "humor":
                batch["gold_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["gold_labels"])
                )
    return batch
```

```python
def concatenated_inputs(
    self,
    batch: Dict[str, Union[List, torch.LongTensor]],
    is_encoder_decoder: bool = False,
    is_vision_model: bool = False,
    label_pad_token_id: int = -100,
    padding_value: int = 0,
    device: Optional[torch.device] = None,
) -> Dict[str, torch.LongTensor]:
    ···
    if self.loss_type == "humor":
        for k in batch:
            if k.startswith("gold") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("gold", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)
  
    if is_encoder_decoder:
    ···
    return concatenated_batch
```

在以sharegpt格式读取数据时，我们也需要对三元组数据进行处理，因此需要对[`src/llamafactory/data`](src/llamafactory/data)目录下的一系列文件进行修改。

在[`src/llamafactory/data/parser.py`](src/llamafactory/data/parser.py)中，对 `DatasetAttr`类进行了修改，以支持三元组数据的读取。

```python
class DatasetAttr:
    ···
    # dpo columns
    chosen: Optional[str] = None
    rejected: Optional[str] = None
    gold:Optional[str] = None
    kto_tag: Optional[str] = None
    ···
```

在 [`src/llamafactory/data/converter.py`](src/llamafactory/data/converter.py) 中添加了对三元组数据处理的支持。

```python
class SharegptDatasetConverter(DatasetConverter):
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        ···

        if (not self.dataset_attr.ranking and len(aligned_messages) % 2 != 0) or (
            self.dataset_attr.ranking and len(aligned_messages) % 2 == 0
        ):
            logger.warning_rank0(f"Invalid message count in {messages}.")
            broken_data = True

        if broken_data:
            logger.warning_rank0("Skipping this abnormal example.")
            prompt, response = [], []
        elif self.dataset_attr.kto_tag and isinstance(example[self.dataset_attr.kto_tag], bool):  # kto example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]
            if example[self.dataset_attr.kto_tag]:
                response = response + [{"role": Role.ASSISTANT.value, "content": ""}]
            else:
                response = [{"role": Role.ASSISTANT.value, "content": ""}] + response
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
            and self.dataset_attr.gold is None
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
            ]
        # add gold for humor dpo
        elif (
            self.dataset_attr.ranking
            and isinstance(example[self.dataset_attr.chosen], dict)
            and isinstance(example[self.dataset_attr.rejected], dict)
            and isinstance(example[self.dataset_attr.gold], dict)
        ):  # pairwise example
            chosen = example[self.dataset_attr.chosen]
            rejected = example[self.dataset_attr.rejected]
            gold = example[self.dataset_attr.gold]
            if (
                chosen[self.dataset_attr.role_tag] not in accept_tags[-1]
                or rejected[self.dataset_attr.role_tag] not in accept_tags[-1]
                or gold[self.dataset_attr.role_tag] not in accept_tags[-1]
            ):
                logger.warning_rank0(f"Invalid role tag in {[chosen, rejected, gold]}.")
                broken_data = True

            prompt = aligned_messages
            response = [
                {
                    "role": tag_mapping[chosen[self.dataset_attr.role_tag]],
                    "content": chosen[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[rejected[self.dataset_attr.role_tag]],
                    "content": rejected[self.dataset_attr.content_tag],
                },
                {
                    "role": tag_mapping[gold[self.dataset_attr.role_tag]],
                    "content": gold[self.dataset_attr.content_tag],
                },
            ]
        else:  # normal example
            prompt = aligned_messages[:-1]
            response = aligned_messages[-1:]

        output = {
            "_prompt": prompt,
            "_response": response,
            "_system": system,
            "_tools": example[self.dataset_attr.tools] if self.dataset_attr.tools else "",
            "_images": self._find_medias(example[self.dataset_attr.images]) if self.dataset_attr.images else None,
            "_videos": self._find_medias(example[self.dataset_attr.videos]) if self.dataset_attr.videos else None,
            "_audios": self._find_medias(example[self.dataset_attr.audios]) if self.dataset_attr.audios else None,
        }
        return output
```

在 [`src/llamafactory/data/collator.py`](src/llamafactory/data/collator.py) 中，我们也需要对数据集的加载进行相应的修改，以支持三元组数据的读取。

```python
class PairwiseDataCollatorWithPadding(MultiModalDataCollatorForSeq2Seq):
    r"""Data collator for pairwise data."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, "torch.Tensor"]:
        r"""Pad batched data to the longest sequence in the batch.

        We generate 2 * n examples where the first n examples represent chosen examples and
        the last n examples represent rejected examples.
        """
        concatenated_features = []
        # if gold in features
        if "gold_input_ids" in features[0]:
            for key in ("chosen", "rejected", "gold"):
                for feature in features:
                    target_feature = {
                        "input_ids": feature[f"{key}_input_ids"],
                        "attention_mask": feature[f"{key}_attention_mask"],
                        "labels": feature[f"{key}_labels"],
                        "images": feature["images"],
                        "videos": feature["videos"],
                        "audios": feature["audios"],
                    }
                    concatenated_features.append(target_feature)
        else:
            for key in ("chosen", "rejected"):
                for feature in features:
                    target_feature = {
                        "input_ids": feature[f"{key}_input_ids"],
                        "attention_mask": feature[f"{key}_attention_mask"],
                        "labels": feature[f"{key}_labels"],
                        "images": feature["images"],
                        "videos": feature["videos"],
                        "audios": feature["audios"],
                    }
                    concatenated_features.append(target_feature)

        return super().__call__(concatenated_features)
```

在 [`src/llamafactory/data/processor/pairwise.py`](src/llamafactory/data/processor/pairwise.py) 中，我们修改了 `PairwiseDatasetProcessor` 类，以支持三元组数据的处理。

```python
class PairwiseDatasetProcessor(DatasetProcessor):
    def _encode_data_example(
        self,
        prompt: list[dict[str, str]],
        response: list[dict[str, str]],
        system: Optional[str],
        tools: Optional[str],
        images: list["ImageInput"],
        videos: list["VideoInput"],
        audios: list["AudioInput"],
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        chosen_messages = self.template.mm_plugin.process_messages(
            prompt + [response[0]], images, videos, audios, self.processor
        )
        rejected_messages = self.template.mm_plugin.process_messages(
            prompt + [response[1]], images, videos, audios, self.processor
        )
        has_gold = len(response) == 3
        if has_gold:
            gold_messages = self.template.mm_plugin.process_messages(
                prompt + [response[2]], images, videos, audios, self.processor
            )
        prompt_ids, chosen_ids = self.template.encode_oneturn(self.tokenizer, chosen_messages, system, tools)
        _, rejected_ids = self.template.encode_oneturn(self.tokenizer, rejected_messages, system, tools)
        if has_gold:
            _, gold_ids = self.template.encode_oneturn(self.tokenizer, gold_messages, system, tools)

        if self.template.efficient_eos:
            chosen_ids += [self.tokenizer.eos_token_id]
            rejected_ids += [self.tokenizer.eos_token_id]
            if has_gold:
                gold_ids += [self.tokenizer.eos_token_id]

        prompt_ids, _ = self.template.mm_plugin.process_token_ids(
            prompt_ids, None, images, videos, audios, self.tokenizer, self.processor
        )
        # consider the response is more important
        source_len, target_len = infer_seqlen(
            len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), self.data_args.cutoff_len
        )
        prompt_ids = prompt_ids[:source_len]
        chosen_ids = chosen_ids[:target_len]
        rejected_ids = rejected_ids[:target_len]
        if has_gold:
            gold_ids = gold_ids[:target_len]
        else:
            gold_ids = []

        chosen_input_ids = prompt_ids + chosen_ids
        chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
        rejected_input_ids = prompt_ids + rejected_ids
        rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
        if has_gold:
            gold_input_ids = prompt_ids + gold_ids
            gold_labels = [IGNORE_INDEX] * source_len + gold_ids

        return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, gold_input_ids if has_gold else None, gold_labels if has_gold else None

    def preprocess_dataset(self, examples: dict[str, list[Any]]) -> dict[str, list[Any]]:
        # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
        model_inputs = defaultdict(list)
        for i in range(len(examples["_prompt"])):
            if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
                logger.warning_rank0(
                    "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
                )
                continue

            chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, gold_input_ids, gold_labels = self._encode_data_example(
                prompt=examples["_prompt"][i],
                response=examples["_response"][i],
                system=examples["_system"][i],
                tools=examples["_tools"][i],
                images=examples["_images"][i] or [],
                videos=examples["_videos"][i] or [],
                audios=examples["_audios"][i] or [],
            )
            model_inputs["chosen_input_ids"].append(chosen_input_ids)
            model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
            model_inputs["chosen_labels"].append(chosen_labels)
            model_inputs["rejected_input_ids"].append(rejected_input_ids)
            model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
            model_inputs["rejected_labels"].append(rejected_labels)
            if gold_input_ids is not None and gold_labels is not None:
                model_inputs["gold_input_ids"].append(gold_input_ids)
                model_inputs["gold_attention_mask"].append([1] * len(gold_input_ids))
                model_inputs["gold_labels"].append(gold_labels)
            model_inputs["images"].append(examples["_images"][i])
            model_inputs["videos"].append(examples["_videos"][i])
            model_inputs["audios"].append(examples["_audios"][i])

        return model_inputs

    def print_data_example(self, example: dict[str, list[int]]) -> None:
        valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
        valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
        if "gold_labels" in example:
            valid_gold_labels = list(filter(lambda x: x != IGNORE_INDEX, example["gold_labels"]))
        print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
        print(
            "chosen_inputs:\n{}".format(self.tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False))
        )
        print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
        print(f"chosen_labels:\n{self.tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
        print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
        print(
            "rejected_inputs:\n{}".format(
                self.tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)
            )
        )
        print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
        print(f"rejected_labels:\n{self.tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
        if "gold_input_ids" in example:
            print("gold_input_ids:\n{}".format(example["gold_input_ids"]))
            print(
                "gold_inputs:\n{}".format(self.tokenizer.decode(example["gold_input_ids"], skip_special_tokens=False))
            )
            print("gold_label_ids:\n{}".format(example["gold_labels"]))
            print(f"gold_labels:\n{self.tokenizer.decode(valid_gold_labels, skip_special_tokens=False)}")
```

最后，在 [`src/llamafactory/hparams/finetuning_args.py`](src/llamafactory/hparams/finetuning_args.py) 中添加了新的超参数，以支持直接通过 `llamafactory-cli train`终端命令行进行调用。

```python
class RLHFArguments:
    ···

    pref_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair", "orpo", "simpo", "humor", "dpop"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )

    ···

    humor_gamma: float = field(
        default=0.1,
        metadata={"help": "The target reward margin term in Humor loss."},
    )
    dpop_lambda: float = field(
        default=5.0,
        metadata={"help": "The lambda parameter in the DPO-P loss."},
    )
```
