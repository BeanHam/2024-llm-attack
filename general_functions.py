import os
import re
import json
import torch
import wandb
import evaluate
import argparse
import numpy as np
import bitsandbytes as bnb

from tqdm import tqdm
from trl import SFTTrainer
from datasets import load_dataset
from typing import Mapping, Iterable
from os import path, makedirs, getenv
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, DataCollatorForLanguageModeling, AutoModel

QUANZATION_MAP = {
    '4bit': BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    '8bit': BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_skip_modules=["lm_head"],
        torch_dtype=torch.bfloat16,
    ),
}

DEFAULT_TRAINING_ARGS = TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=10,
        max_steps=50,
        learning_rate=2e-4,
        fp16=True if torch.cuda.is_available() else False,
        logging_steps=1,
        output_dir='outputs',
        optim='paged_adamw_8bit' if torch.cuda.is_available() else 'adamw_torch',
        use_mps_device=False,
        log_level='info',
        logging_first_step=True,
        evaluation_strategy='steps',
        eval_steps=25
    )

def get_dataset_slices(dataset: str) -> dict:
    """
    Returns a dictionary of subsets of the training, validation, and test splits of a dataset.
    """

    # Download the dataset splits, including the dataset version if specified
    train_data = load_dataset(dataset, split='train')
    val_data = load_dataset(dataset, split='validation')
    test_data = load_dataset(dataset, split='test')
    
    # Return the dictionary of dataset splits
    return {'train': train_data, 'val': val_data, 'test': test_data}

def get_model_and_tokenizer(model_id: str, 
                            quantization_type: str='', 
                            gradient_checkpointing: bool=True, 
                            device: str='auto') -> tuple[AutoModel, AutoTokenizer]:
    """
    Returns a Transformers model and tokenizer for fine-tuning. If quantization_type is provided, the model will be quantized and prepared for training.
    """

    # Download the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Set the pad token (needed for trainer class, no value by default for most causal models)
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Download the model, quantize if requested
    if quantization_type:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     quantization_config=QUANZATION_MAP[quantization_type], 
                                                     device_map=device)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                     device_map=device)

    # Enable gradient checkpointing if requested
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    # Prepare the model for training if quantization is requested
    if quantization_type:
        model = prepare_model_for_kbit_training(model)

    return model, tokenizer

def find_lora_modules(model: AutoModel, 
                      include_modules: Iterable=(bnb.nn.Linear4bit), 
                      exclude_names: Iterable=('lm_head')) -> list[str]:
    """
    Returns a list of the modules to be tuned using LoRA.
    """

    # Create a set to store the names of the modules to be tuned
    lora_module_names = set()

    # Iterate over the model and find the modules to be tuned
    for name, module in model.named_modules():

        # Check if the module is in the list of modules to be tuned
        if any(isinstance(module, include_module) for include_module in include_modules):

            # Split the name of the module and add it to the set
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    # Return the list of module names to be tuned, excluding any names in the exclude list
    return [name for name in list(lora_module_names) if name not in exclude_names]

def get_lora_model(model: AutoModel,
                   matrix_rank: int=8,
                   scaling_factor: int=32,
                   dropout: float=0.05,
                   bias: str='none',
                   task_type: str='CAUSAL_LM',
                   include_modules: Iterable=(bnb.nn.Linear4bit),
                   exclude_names: Iterable=('lm_head')) -> AutoModel:
    """
    Returns a model with LoRA applied to the specified modules.
    """

    config = LoraConfig(
        r=matrix_rank,
        lora_alpha=scaling_factor,
        target_modules=find_lora_modules(model, include_modules, exclude_names),
        lora_dropout=dropout,
        bias=bias,
        task_type=task_type,
    )

    return get_peft_model(model, config)

def format_data_as_instructions(data: Mapping, 
                                evidence: str, 
                                tokenizer: AutoTokenizer) -> list[str]:
    """
    Formats text data as instructions for the model. Can be used as a formatting function for the trainer class.
    """

    output_texts = []

    if evidence == 'yes':
        system="""## TASK: 
        You are a helpful multiple choice question-answering assistant! 

        I will provide you with a QUESTION, an EVIDENCE associated with the question, and multiple CHOICES. 

        Please use the provided evidence to answer the multiple-choice question. Only one choice is the correct answer.""" 
    else:
        system="""## TASK: 
        You are a helpful multiple choice question-answering assistant! 

        I will provide you with a QUESTION and multiple CHOICES. Please answer the question by selecting one correct choice.""" 
        
    # Iterate over the data and format the text
    for i in tqdm(range(len(data['question_sentence'])), desc='Formatting data'):
        question=f"\n\n## QUESTION:\n{data['question_sentence'][i]}"
        evidence=f"\n\n## EVIDENCE:\n{data['evidence'][i]}"
        choices=f"\n\n## CHOICES:\n{[str(j)+': '+data['choices'][i][j] for j in range(len(data['choices'][i]))]}"
        if evidence == 'yes': user_input=system+question+evidence+choices+"\n\n## ANSWER:"
        else: user_input=system+question+choices+"\n\n## ANSWER:"
        chat = [
          {"role": "user", "content": user_input},
          {"role": "assistant", "content": data['answer'][i]},
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)

    return output_texts
    
def get_default_trainer(model: AutoModel,
                tokenizer: AutoTokenizer,
                train_dataset: Mapping,
                eval_dataset: Mapping=None,
                formatting_func: callable=format_data_as_instructions,                
                max_seq_length: int=974,
                training_args: TrainingArguments=None) -> SFTTrainer:
    """
    Returns the default trainer for fine-tuning a summarization model based on the specified training config.
    """

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args if training_args else DEFAULT_TRAINING_ARGS,
        formatting_func=formatting_func,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        max_seq_length=max_seq_length,
        packing=False
    )

    return trainer

def evaluate_model(model: AutoModelForCausalLM, 
                   tokenizer: AutoTokenizer, 
                   data: Iterable,
                   max_tokens: int=1024,
                   min_new_tokens: int=1,
                   max_new_tokens: int=32,
                   num_return_sequences: int=10,
                   remove_suffix: str=None,
                   evidence: str='yes') -> dict:
    """
    Evaluate a Hugging Face model on a dataset using three text summarization metrics.
    """
    
    accuracy = []
    confidence = []
    if evidence == 'yes':
        system="""## TASK: 
        You are a helpful multiple choice question-answering assistant! 

        I will provide you with a QUESTION, an EVIDENCE associated with the question, and multiple CHOICES. 

        Please use the provided evidence to answer the multiple-choice question. Only one choice is the correct answer.
    
        Please use the following output format example: ## ANSWER: {1}. ## CONFIDENCE: {80%}. 
    
        No explaination is needed."""
    else:
        system="""## TASK: 
        You are a helpful multiple choice question-answering assistant! 

        I will provide you with a QUESTION and multiple CHOICES. Please answer the question by selecting one correct choice.
        
        Please use the following output format example: ## ANSWER: {1}. ## CONFIDENCE: {80%}. 
    
        No explaination is needed."""
                        
    # Iterate over the test set
    for idx in tqdm(range(len(data))):

        question=f"\n\n## QUESTION:\n{data['question_sentence'][idx]}"
        evidence=f"\n\n## EVIDENCE:\n{data['evidence'][idx]}"
        choices=f"\n\n## CHOICES:\n{[str(j)+': '+data['choices'][idx][j] for j in range(len(data['choices'][idx]))]}"
        if evidence == 'yes': user_input=system+question+evidence+choices
        else: user_input=system+question+choices
        chat = [{"role": "user", "content": user_input}]
        input_data = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        
        # Calculate the position of the start of the output string
        start_decode = len(tokenizer.encode(input_data, truncation=True, max_length=max_tokens))
        input_ids = tokenizer(input_data, return_tensors='pt', truncation=True, max_length=max_tokens).to(model.device)
        with torch.no_grad():
            output = model.generate(**input_ids, 
                                    max_new_tokens=max_new_tokens, 
                                    min_new_tokens=min_new_tokens, 
                                    num_return_sequences=num_return_sequences,
                                    pad_token_id=tokenizer.eos_token_id)
        output = [tokenizer.decode(i[start_decode:]).replace(remove_suffix, '').replace('</a>', '') for i in output]
        decoded = []
        for d in output:
            try:
                d = re.sub(r'[^\w\s]', ' ', d)
                if d.split()[0].isnumeric(): answer = float(d.split()[0])
                else: answer = float(d.split('ANSWER')[1].split()[0])
                confid = float(d.split('CONFIDENCE')[1].split()[0])
                if ( (answer <= 4.0) & (confid <= 100.0) ):
                    decoded.append([answer, confid])
            ## sometimes, the model does not return confidence
            except:
                next
        
        if len(decoded) == 0:
            next
        else:
            summary={}
            total=0
            for k, v in decoded:
                total+=v
                if k in summary: summary[k]+=v
                else:summary[k]=v
            for k in summary:
                summary[k]/=total
            #print(summary)
            pred=max(summary, key=summary.get)
            conf=summary[pred]
            gt=float(data['answer'][idx])
        
            # metric calculation
            accuracy.append(gt == pred)
            confidence.append(conf)

    metrics = {
        'accuracy':np.mean(accuracy),
        'confidence':np.mean(confidence),
    }
    
    return metrics, confidence
