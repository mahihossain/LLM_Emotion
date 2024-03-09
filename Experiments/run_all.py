# It is based on LoRA paper: https://arxiv.org/abs/2106.09685

import os
os.environ['HF_TOKEN'] = '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'

import subprocess
import bitsandbytes as bnb
import torch
import pandas as pd
from collections import Counter
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    IntervalStrategy
)
from datasets import load_dataset
from peft import (
    PeftConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
    LoraConfig
)
import transformers

MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto", cache_dir="../models")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="../models", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

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

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

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

import warnings

# Proper regular expression to match the warning message
warnings.filterwarnings("ignore", message="torch.utils.checkpoint: please pass in use_reentrant=True or use_reentrant=False explicitly")

training_args = transformers.TrainingArguments(
    output_dir="./saves",
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    optim="paged_adamw_32bit",
    save_strategy=IntervalStrategy.STEPS,
    save_total_limit=1,
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

def train(person_number, model):
    data = load_dataset("csv", data_files=f"../data/train_VPN{person_number:02d}.csv")
    data = data["train"].shuffle().map(generate_and_tokenize_prompt)

    trainer = transformers.Trainer(
        model=model,
        train_dataset=data,
        args=training_args,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    model.config.use_cache = False
    trainer.train()
    
    model.cpu()
    model.save_pretrained(f"./models/llama-2-7b-chat-hf-llm-emo-person-{person_number:02d}-finetuned-peft/")



# Define the emotions
emotions = ['REST', 'DEPRECIATION', 'AVOIDANCE', 'STABILIZE_SELF', 'ATTACK_OTHER', 'WITHDRAWAL', 'ATTACK_SELF']

# Define the function to process each person
def predict(person_number):
    # Load the test data
    df = pd.read_csv(f'../data/test_VPN{person_number:02d}.csv')

    # Load the model
    PEFT_MODEL = f"./models/llama-2-7b-chat-hf-llm-emo-person-{person_number:02d}-finetuned-peft/"
    config = PeftConfig.from_pretrained(PEFT_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model_name_or_path,
        return_dict=True,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer=AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = PeftModel.from_pretrained(model, PEFT_MODEL)

    # Set the generation config
    generation_config = model.generation_config
    generation_config.max_new_tokens = 200
    generation_config.temperature = 0.0
    generation_config.top_p = 0.7
    generation_config.num_return_sequences = 1
    generation_config.pad_token_id = tokenizer.eos_token_id
    generation_config.eos_token_id = tokenizer.eos_token_id

    # If cuda is available, use it
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Iterate over the dataset
    df_copy = df.copy()
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        # Generate the prompt
        prompt = f"""
        <human>: {row['User']}
        <context>: {row['context']}
        <assistant>:
        """.strip()

        # Generate the model's response
        encoding = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                input_ids = encoding.input_ids,
                attention_mask = encoding.attention_mask,
                generation_config = generation_config
            )

        # Process the model's response
        assistant_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        assistant_response = assistant_response.split('<assistant>')[-1].strip().lower()
        emotion_counter = Counter()
        for emotion in emotions:
            emotion_counter[emotion] = assistant_response.count(emotion.lower())
        if emotion_counter:
            most_common_emotion = emotion_counter.most_common(1)[0][0]
            if row['EmotionRegulation1'].lower() in [emotion for emotion, count in emotion_counter.items() if count > 0]:
                df_copy.loc[index, 'EmotionRegulation1'] = row['EmotionRegulation1']
            else:
                df_copy.loc[index, 'EmotionRegulation1'] = most_common_emotion
        else:
            df_copy.loc[index, 'EmotionRegulation1'] = 'REST'

    # Save the results
    df_copy = df_copy[['EmotionRegulation1']]
    df_copy = df_copy.rename(columns={'EmotionRegulation1': 'Predicted'})
    df_copy['GroundTruth'] = df['EmotionRegulation1']
    csv_file = f'../data/predictions/llama_prediction_person_{person_number:02d}.csv'
    df_copy.to_csv(csv_file, index=False)

    # Commit and push the new CSV file to Git
    try:
        subprocess.run(['git', 'add', csv_file], check=True)
        subprocess.run(['git', 'commit', '-m', f'Add prediction for person {person_number:02d}'], check=True)
        subprocess.run(['git', 'push'], check=True)
    except subprocess.CalledProcessError as e:
        print(f'Error occurred while pushing to Git: {e}')
        print('Continuing with the next person...')

# Fine-tune the model for all 10 people
for i in tqdm(range(1, 10), desc="Processing people", unit="person"):
    print(f"Processing person {i:02d}...\n")
   # train(i, model)
    predict(i)
    # model = model.cpu()
    # del model
    # torch.cuda.empty_cache()
