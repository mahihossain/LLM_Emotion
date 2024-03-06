# %%
# It is based on LoRA paper: https://arxiv.org/abs/2106.09685

import json
import os
from pprint import pprint
import bitsandbytes as bnb
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    IntervalStrategy,
)

from trl import SFTTrainer

# %%
MODEL_NAME = "google/gemma-7b"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, cache_dir="../models", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, cache_dir="../models")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# %%
def print_trainable_parameters(model):
  """
  Prints the number of trainable parameters in the model.
  """
  trainable_params = 0
  all_param = 0
  for _, param in model.named_parameters():
    all_param += param.numel()
    if param.requires_grad:
      trainable_params += param.numel()
  print(
      f"trainable params: {trainable_params} || all params: {all_param} || trainables%: {100 * trainable_params / all_param}"
  )

# %%
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

# %%
config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=8,
    bias="none",
    task_type="CAUSAL_LM"
)

model.modules()

model = get_peft_model(model, config)
print_trainable_parameters(model)

# %%
data = load_dataset("csv", data_files="../data/train_VPN10.csv")

# %%
data

# %%
data["train"][0]

# %%
def generate_prompt(data_point):
  return f"""
<human>: {data_point["User"]}
<context>: {data_point["context"]}
<assistant>: {data_point["EmotionRegulation1"]}
""".strip()

def generate_and_tokenize_prompt(data_point):
  full_prompt = generate_prompt(data_point)
  tokenized_full_prompt = tokenizer(full_prompt, padding=True, truncation=True)
  return tokenized_full_prompt

# %%
data = data["train"].shuffle().map(generate_and_tokenize_prompt)

# %%
data

# %%
import warnings

# Proper regular expression to match the warning message
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly")


# %%
training_args = transformers.TrainingArguments(
    output_dir="./saves",
    num_train_epochs=5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    #save_strategy="epoch",
    save_strategy=IntervalStrategy.STEPS,  # change this line
    save_total_limit=1,  # add this line
    logging_strategy="epoch",
    logging_steps=1,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,
    args=training_args,
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
)
model.config.use_cache = False
trainer.train()

# %%
model.save_pretrained("./models/gemma-7b-llm-emo-person-10-finetuned-peft/")


