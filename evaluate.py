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
    parser.add_argument('--finetuned', type=str, default='no', help='hf login token')
    parser.add_argument('--use_model_prompt_defaults', type=str, default='llama3', help='Whether to use the default prompts for a model')
    parser.add_argument('--evidence', type=str, default='yes', help='hf login token')
    args = parser.parse_args()
    args.suffix = MODEL_SUFFIXES[args.use_model_prompt_defaults]
    args.save_path=f'inference_results_{args.evidence}/'
    if args.hf_token_var:
        hf_login(token=getenv(args.hf_token_var))
    if not path.exists(args.save_path):
        makedirs(args.save_path)
        
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    data = get_dataset_slices(args.dataset)
    test_data = data['test'].select(range(100))
    test_data.set_format(type='torch', device='cuda')
    
    # ----------------------
    # Checkpoints
    # ----------------------
    checkpoints = os.listdir(f'outputs_llama3_{args.evidence}/')
    if '.ipynb_checkpoints' in checkpoints:
        checkpoints.remove('.ipynb_checkpoints')
    if 'runs' in checkpoints:
        checkpoints.remove('runs')

    # ----------------------
    # Baseline: no finetune
    # ----------------------
    if args.finetuned=='no':
        #-----------------------
        # load model & tokenizer
        #-----------------------
        print('Getting model and tokenizer...')
        model, tokenizer = get_model_and_tokenizer(args.model_id,
                                                   gradient_checkpointing=False,
                                                   quantization_type='4bit',
                                                   device='auto')
        #------------
        # inference
        #------------
        model.eval()
        metrics, confidence  = evaluate_model(model=model,
                                              tokenizer=tokenizer,
                                              data=test_data,
                                              max_new_tokens=32,
                                              remove_suffix=args.suffix,
                                              evidence=args.evidence)

        for k, v in metrics.items(): print(f'   {k}: {v}')
        with open(args.save_path+f"baseline.json", 'w') as f: json.dump(metrics, f)
        np.save(args.save_path+f"baseline.npy", confidence)
            
    # ----------------------
    # Finetuned
    # ----------------------            
    else:
        for checkpoint in checkpoints:
        
            #-----------------------
            # load model & tokenizer
            #-----------------------
            print('Getting model and tokenizer...')
            model, tokenizer = get_model_and_tokenizer(args.model_id,
                                                       gradient_checkpointing=False,
                                                       quantization_type='4bit',
                                                       device='auto')
            model = PeftModel.from_pretrained(model, f'outputs_llama3_{args.evidence}/{checkpoint}/')

            #------------
            # inference
            #------------
            model.eval()
            metrics, confidence  = evaluate_model(model=model,
                                                  tokenizer=tokenizer,
                                                  data=test_data,
                                                  max_new_tokens=32,
                                                  remove_suffix=args.suffix,
                                                  evidence=args.evidence)

            for k, v in metrics.items(): print(f'   {k}: {v}')
            with open(args.save_path+f"{checkpoint}.json", 'w') as f: json.dump(metrics, f)
            np.save(args.save_path+f"{checkpoint}.npy", confidence)
            
            ## clear cache
            model.cpu()
            del model, checkpoint
            gc.collect()
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
