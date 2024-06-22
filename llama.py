import json
import random
import pickle


from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)

from trl import SFTTrainer, SFTConfig, DataCollatorForCompletionOnlyLM
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

path = '../data/noise_ft.jsonl'
# model = LlamaForCausalLM.from_pretrained(model)
# tokenizer = LlamaTokenizer.from_pretrained(model)

def create_dataset(path, ds, system, out=None):
    examples = []
    
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            obj = json.loads(line)
            traj = obj['messages'][-1]['content'].split('\n\n')
            traj = list(map(lambda x: ' '.join(x.split(' ')[1:]), traj))
            traj = traj[::ds]
            user = ' '.join(obj['messages'][1]['content'].split(' ')[1:])
            
            examples.append({
                'messages': [
                    {'role': 'system', 'content': system},
                    {'role': 'user', 'content': user},
                    {'role': 'assistant', 'content': '\n\n'.join(traj)}
                ]
            })
            
    return examples

def load_dataset(path):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if not line: continue
            examples.append(json.loads(line))
    return examples

def main():
    data_path = ''
    model = 'meta-llama/Meta-Llama-3-8B-Instruct'
    
    dataset = load_dataset(data_path)
    train_data, eval_data = train_test_split(dataset, test_size=0.1, seed=69420) 
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    collator = DataCollatorForCompletionOnlyLM(response_template='<|start_header_id|>system<|end_header_id|>', tokenizer=tokenizer)
    
    config = SFTConfig(
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=1,
        max_steps=300,
        warmup_steps=30,
        output_dir="./results",
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_seq_length=1024,
        weight_decay=0.01,
    )
    
    trainer = SFTTrainer(
        model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=collator,
        args=config,
    )
    
    trainer.train()
