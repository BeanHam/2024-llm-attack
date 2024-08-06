#!/usr/bin/env python3

import os
import sys
import json
import torch
import logging
import datasets
import argparse
import numpy as np
import transformers
import bitsandbytes as bnb

from utils import *
from general_functions import *
from os import path, makedirs, getenv
from transformers import TrainingArguments
from huggingface_hub import login as hf_login

## disable wandb
os.environ["WANDB_DISABLED"] = "true"

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Fine-tune a summarization model.')
    
    # Model ID
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct', help='The model ID to fine-tune.')
    parser.add_argument('--device', type=str, default='auto', help='The device to mount the model on.')
    
    # Model arguments
    parser.add_argument('--use_mps_device', type=str, default='False', help='Whether to use an MPS device.')
    parser.add_argument('--gradient_checkpointing', type=str, default='True', help='Whether to use gradient checkpointing.')
    parser.add_argument('--quantization_type', type=str, default='4bit', help='The quantization type to use for fine-tuning.')
    parser.add_argument('--lora', type=str, default='True', help='Whether to use LoRA.')
    parser.add_argument('--tune_modules', type=str, default='linear4bit', help='The modules to tune using LoRA.')
    parser.add_argument('--exclude_names', type=str, default='lm_head', help='The names of the modules to exclude from tuning.')
    parser.add_argument('--resume_from_checkpoint', type=str, default='False', help='Whether to resume from a checkpoint.')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, default='beanham/realtime_qa', help='The dataset to use for fine-tuning.')    
    parser.add_argument('--max_seq_length', type=int, default=1024, help='The maximum sequence length to use for fine-tuning.')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='llama3', help='Whether to use the default prompts for a model')

    # Training arguments
    parser.add_argument('--batch_size', type=int, default=1, help='The batch size to use for fine-tuning.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4, help='The number of gradient accumulation steps to use for fine-tuning.')
    parser.add_argument('--warmup_ratio', type=int, default=0.03, help='The number of warmup steps to use for fine-tuning.')
    parser.add_argument('--max_steps', type=int, default=-1, help='The maximum number of steps to use for fine-tuning.')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='The learning rate to use for fine-tuning.')
    parser.add_argument('--fp16', type=str, default='True', help='Whether to use fp16.')
    parser.add_argument('--output_dir', type=str, default='outputs', help='The directory to save the fine-tuned model.')
    parser.add_argument('--optim', type=str, default='paged_adamw_8bit', help='The optimizer to use for fine-tuning.')

    # Logging arguments
    parser.add_argument('--evaluation_strategy', type=str, default='steps', help='The evaluation strategy to use for fine-tuning.')
    parser.add_argument('--eval_steps', type=int, default=0.1, help='The number of steps between evaluations.')
    parser.add_argument('--save_strategy', type=str, default='steps', help='The number of steps between logging.')
    parser.add_argument('--save_steps', type=int, default=0.1, help='The number of steps between saving the model to the hub.')
    parser.add_argument('--logging_strategy', type=str, default='steps', help='The number of steps between logging.')
    parser.add_argument('--logging_steps', type=int, default=0.1, help='The number of steps between logging.')
    parser.add_argument('--epoch', type=int, default=1, help='The length split of the dataset.')
    parser.add_argument('--evidence', type=str, default='yes', help='The length split of the dataset.')
    
    # Parse arguments
    args = parser.parse_args()
    
    # change saving directory
    args.output_dir = 'outputs_'+args.use_model_prompt_defaults+'_'+args.evidence
    args.save_dir = 'outputs_'+args.use_model_prompt_defaults+'_'+args.evidence+'/final_model/'
    args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    # make first level directories
    if not path.exists(args.output_dir):
        makedirs(args.output_dir)
    if not path.exists(args.save_dir):
        makedirs(args.save_dir)    

    # HF Login
    hf_login()
    
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    data = get_dataset_slices(args.dataset)
    train_data = data['train']
    val_data = data['val']
    test_data = data['test']
    train_data.set_format(type='torch', device='cuda')
    val_data.set_format(type='torch', device='cuda')
    test_data.set_format(type='torch', device='cuda')
    
    # ----------------------
    # Load Model & Tokenizer
    # ----------------------
    print('Getting model and tokenizer...')
    model, tokenizer = get_model_and_tokenizer(args.model_id,
                                               quantization_type=args.quantization_type,
                                               gradient_checkpointing=bool(args.gradient_checkpointing),
                                               device=args.device)

    # ----------------------
    # Get LoRA Model
    # ----------------------
    if args.lora == 'True':
        print('Getting LoRA model...')
        if args.tune_modules == 'linear':
            lora_modules = [torch.nn.Linear]
        elif args.tune_modules == 'linear4bit':
            lora_modules = [bnb.nn.Linear4bit]
        elif args.tune_modules == 'linear8bit':
            lora_modules = [bnb.nn.Linear8bit]
        else:
            raise ValueError(f'Invalid tune_modules argument: {args.tune_modules}, must be linear, linear4bit, or linear8bit')
        model = get_lora_model(model,
                               include_modules=lora_modules,
                               exclude_names=args.exclude_names)    
    
    # -----------------------
    # Set Training Parameters
    # -----------------------
    print('Instantiating trainer...')
    training_args = TrainingArguments(
            per_device_train_batch_size=args.batch_size, ## 1
            gradient_accumulation_steps=args.gradient_accumulation_steps, ## 4
            warmup_ratio=args.warmup_ratio,
            learning_rate=args.learning_rate,
            fp16=args.fp16 == 'True',
            output_dir=args.output_dir,
            optim=args.optim,
            use_mps_device=args.use_mps_device == 'True',
            evaluation_strategy=args.evaluation_strategy,
            logging_strategy=args.logging_strategy,
            save_strategy=args.save_strategy,
            eval_steps=args.eval_steps,
            save_steps=args.save_steps,
            logging_steps=args.logging_steps,
            resume_from_checkpoint=args.resume_from_checkpoint == 'True',
            max_steps=args.max_steps,
            num_train_epochs=args.epoch,
        )
        
    # -----------------------
    # data formatter
    # -----------------------   
    def data_formatter(data: Mapping) -> list[str]:
        """
        Wraps the format_data_as_instructions function with the specified arguments.
        """
        return format_data_as_instructions(data, args.evidence, tokenizer)
        
    trainer = get_default_trainer(model, 
                                  tokenizer, 
                                  train_data, 
                                  eval_dataset=val_data,
                                  formatting_func=data_formatter,
                                  max_seq_length=args.max_seq_length,
                                  training_args=training_args)
    
    model.config.use_cache = False

    # Fine-tune model
    print('Fine-tuning model...')
    trainer.train()

    print('Saving model and tokenizer...')
    trainer.model.save_pretrained(args.save_dir)
    tokenizer.save_pretrained(args.save_dir)
