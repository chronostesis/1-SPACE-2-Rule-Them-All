# Pyvene method of getting activations
import os
import torch



from tqdm import tqdm
import numpy as np
import sys
sys.path.append('../')

#import llama
import pickle
import argparse
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

# Specific pyvene imports
from utils import get_llama_activations_pyvene, tokenized_tqa, tokenized_tqa_gen, tokenized_tqa_gen_end_q
from interveners import wrapper, Collector, ITI_Intervener
import pyvene as pv
from evaluation import *
from datasets import load_from_disk

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='facebook/opt-1.3b')
    parser.add_argument('--dataset_name', type=str, default='truthful_qa')
    parser.add_argument('--CoT_activations', type=int, default=False)
    parser.add_argument('--random_token', type=bool, default=False)
    parser.add_argument(--output_dir, type=str, default='../features/')
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    #os.environ['HF_ENDPOINT'] = 'https: // hf - mirror.com'
    model_name_or_path = args.model_name# or directory path to model

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16,device_map='cuda', trust_remote_code = True)
    device = "cuda"
    dataset=None
    if args.dataset_name in["ted","ted2",'pdtb']:
        print('loading dataset'+args.dataset_name)
        dataset=load_faithful_dataset(dataset_name=args.dataset_name)
        formatter=tokenized_ted
    elif args.dataset_name == 'truthful_qa':
        load_directory = '../truthfulqa_cache'
        dataset = load_from_disk(load_directory)
        formatter = tokenized_tqa
    elif args.dataset_name == "tqa_gen":
        dataset = load_dataset("../truthfulqa_cache/generation", 'default')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthfulqa/truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else: 
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q":
        prompts, labels, categories = formatter(dataset, tokenizer,args.CoT_activations)
        with open(f'../features/{args.model_name}_{args.dataset_name}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels = formatter(dataset, tokenizer)

    collectors = []
    pv_config = []
    for layer in range(model.config.num_hidden_layers): 
        collector = Collector(multiplier=0, head=-1) #head=-1 to collect all head activations, multiplier doens't matter
        collectors.append(collector)
        pv_config.append({
            "component": f"model.layers[{layer}].self_attn.o_proj.input",
            "intervention": wrapper(collector),
        })
    collected_model = pv.IntervenableModel(pv_config, model)

    all_layer_wise_activations = []
    all_head_wise_activations = []
    all_mlp_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, mlp_wise_hidden_states = get_llama_activations_pyvene(collected_model, collectors, prompt, device,args.random_token)
        # layer_wise_activations, head_wise_activations, mlp_wise_hidden_states = get_llama_activations_bau(collected_model, collectors, prompt, device)
        all_layer_wise_activations.append(layer_wise_activations[:,-1,:].copy())
        all_head_wise_activations.append(head_wise_activations.copy())
        all_mlp_wise_activations.append(mlp_wise_hidden_states.copy())

    name=model_name_or_path.split('/')[-1]if '/' in model_name_or_path else model_name_or_path
    print("Saving labels")
    s='_random'if args.random_token else ''
    prefix=args.output_dir
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    np.save(f'{prefix}/{name}_{args.dataset_name}_labels{s}.npy', labels)

    print("Saving layer wise activations")
    np.save(f'{prefix}/{name}_{args.dataset_name}_layer_wise{s}.npy', all_layer_wise_activations)

    print("Saving head wise activations")
    np.save(f'{prefix}/{name}_{args.dataset_name}_head_wise{s}.npy', all_head_wise_activations)


if __name__ == '__main__':
    main()
