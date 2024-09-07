# %%
import torch
from peft import (
    PeftConfig,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

import pandas as pd

from collections import Counter
from tqdm import tqdm

import os
os.environ['HF_TOKEN'] = 'put your token here!'

# %%
# load the test data
df = pd.read_csv('../data/test_VPN01.csv')

# %%
PEFT_MODEL = "./models/gemma-7b-llm-emo-person-01-finetuned-peft/"

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

# %%
generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.01
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eos_token_id = tokenizer.eos_token_id

# if cuda is available, use it
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# prompt = """
# <human>: midjourney prompt for a boy running in the snow
# <context>: 
# <assistant>:
# """.strip()

# encoding = tokenizer(prompt, return_tensors="pt").to(device)
# with torch.inference_mode():
#   outputs = model.generate(
#       input_ids = encoding.input_ids,
#       attention_mask = encoding.attention_mask,
#       generation_config = generation_config
#   )

# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# %%
emotions = ['REST', 'DEPRECIATION', 'AVOIDANCE', 'STABILIZE_SELF', 'ATTACK_OTHER', 'WITHDRAWAL', 'ATTACK_SELF']
df_copy = df.copy()

# %%
emotions

# %%
# iterate over the dataset, User column is <human> and <context> is the context and <assistant> is the model response
for index, row in tqdm(df.iterrows(), total=df.shape[0]):
    prompt = f"""
    <human>: {row['User']}
    <context>: {row['context']}
    <assistant>:
    """.strip()

    encoding = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids = encoding.input_ids,
            attention_mask = encoding.attention_mask,
            generation_config = generation_config
        )

    # Decode the assistant's response
    assistant_response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the part of the assistant's response that comes after '<assistant>'
    assistant_response = assistant_response.split('<assistant>')[-1].strip().lower()

    # Check if the assistant's response contains a valid emotion
    emotion_counter = Counter()
    for emotion in emotions:
        emotion_counter[emotion] = assistant_response.count(emotion.lower())

    if emotion_counter:
        # If a valid emotion is found and it's the same as the ground truth, keep the ground truth in df_copy
        most_common_emotion = emotion_counter.most_common(1)[0][0]
        if row['EmotionRegulation1'].lower() in [emotion for emotion, count in emotion_counter.items() if count > 0]:
            df_copy.loc[index, 'EmotionRegulation1'] = row['EmotionRegulation1']
        else:
            # If a valid emotion is found but it's not the same as the ground truth, replace the ground truth with the most common emotion in df_copy
            df_copy.loc[index, 'EmotionRegulation1'] = most_common_emotion
    else:
        # If no valid emotion is found, put 'REST' in df_copy
        df_copy.loc[index, 'EmotionRegulation1'] = 'REST'

# %%
# Keep only the 'EmotionRegulation1' column and convert it to a DataFrame
df_copy = df_copy[['EmotionRegulation1']]

# Rename the 'EmotionRegulation1' column to 'Predicted'
df_copy = df_copy.rename(columns={'EmotionRegulation1': 'Predicted'})

# Add a column 'GroundTruth' with the original 'EmotionRegulation1' column from df
df_copy['GroundTruth'] = df['EmotionRegulation1']

# Save the df_copy to a csv file as prediction_person_10.csv
df_copy.to_csv('../data/predictions/gemma/gemma_prediction_person_01.csv', index=False)


