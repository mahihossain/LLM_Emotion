import os
import torch
import pandas as pd
from collections import Counter
from tqdm import tqdm
from peft import (
    PeftConfig,
    PeftModel
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)

# Define the emotions
emotions = ['REST', 'DEPRECIATION', 'AVOIDANCE', 'STABILIZE_SELF', 'ATTACK_OTHER', 'WITHDRAWAL', 'ATTACK_SELF']

# Define the function to process each person
def process_person(person_number):
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
    df_copy.to_csv(f'../data/predictions/llama_prediction_person_{person_number:02d}.csv', index=False)

# Process all 10 people
for i in range(1, 11):
    process_person(i)