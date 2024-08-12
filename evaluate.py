import os
import gc
import json
import torch
import argparse
import numpy as np
import pandas as pd
import transformers

from utils import *
from tqdm import tqdm
from general_functions import *
from datasets import load_dataset
from os import path, makedirs, getenv
from huggingface_hub import login as hf_login


#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument('--dataset', type=str, default='beanham/realtime_qa')
    parser.add_argument('--device', type=str, default='cuda', help='The device to mount the model on.')
    parser.add_argument('--hf_token_var', type=str, default='[your token]', help='hf login token')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='llama3', help='Whether to use the default prompts for a model')
    parser.add_argument('--challenging', type=str, default='yes', help='Whether to use the default prompts for a model')
    args = parser.parse_args()
 
    args.max_new_tokens=16
    args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    if args.challenging=='yes':
        args.save_path=f'inference_results_challenging/'
    else:
        args.save_path=f'inference_results/'
    if args.hf_token_var:
        hf_login(token=getenv(args.hf_token_var))
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    data = get_dataset_slices(args.dataset)
    train_data = data['train']
    train_data.set_format(type='torch', device='cuda')
    
    # ----------------------
    # Checkpoints
    # ----------------------
    checkpoints = os.listdir(f'outputs_llama3/')
    if '.ipynb_checkpoints' in checkpoints:
        checkpoints.remove('.ipynb_checkpoints')
    if 'runs' in checkpoints:
        checkpoints.remove('runs')

    for checkpoint in checkpoints:
        
        #-----------------------
        # load model & tokenizer
        #-----------------------
        print(f'{checkpoint}...')
        print('Getting model and tokenizer...')
        model, tokenizer = get_model_and_tokenizer(args.model_id,
                                                   gradient_checkpointing=False,
                                                   quantization_type='4bit',
                                                   device='auto')
        model = PeftModel.from_pretrained(model, f'outputs_llama3/{checkpoint}/')

        #------------
        # inference
        #------------
        model.eval()
        model_outputs, metrics  = evaluate_model(model=model,
                                                 tokenizer=tokenizer,
                                                 data=train_data,
                                                 max_new_tokens=args.max_new_tokens,
                                                 remove_suffix=args.suffix,
                                                 challenging=args.challenging)

        for k, v in metrics.items(): print(f'   {k}: {v}')
        with open(args.save_path+f"{checkpoint}.json", 'w') as f: json.dump(metrics, f)
        np.save(args.save_path+f"{checkpoint}.npy", model_outputs)
        
        ## clear cache
        model.cpu()
        del model, checkpoint
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
