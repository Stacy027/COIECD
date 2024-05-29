from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import GenerationConfig
import json
from transformers import  TopKLogitsWarper, TopPLogitsWarper
from tqdm import tqdm
import torch
import numpy as np

PATH_TO_CONVERTED_WEIGHTS = '/model_path'
PATH_TO_CONVERTED_TOKENIZER = '/model_path'

def load_json_data(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        # data = json.load(f)
    return data

def load_json(filename):
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        # data = f.readlines()
        data = json.load(f)
    return data


def entropy_from_scores(logits):
    logits = logits - logits.logsumexp(dim=-1, keepdims=True)
    entropy = (-1 * logits.exp() * logits).sum(-1)
    return entropy

def top_k_top_p_filtering(
    logits: torch.FloatTensor,
    top_k: int = 0,
    top_p: float = 0.9,
    filter_value: float = -float("Inf"),
    min_tokens_to_keep: int = 1,
) -> torch.FloatTensor:
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering

    Args:
        logits: logits distribution shape (batch size, vocabulary size)
        top_k (`int`, *optional*, defaults to 0):
            If > 0, only keep the top k tokens with highest probability (top-k filtering)
        top_p (`float`, *optional*, defaults to 1.0):
            If < 1.0, only keep the top tokens with cumulative probability >= top_p (nucleus filtering). Nucleus
            filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimumber of tokens we keep per batch example in the output.

    From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    if top_k > 0:
        logits = TopKLogitsWarper(top_k=top_k, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    if 0 <= top_p <= 1.0:
        logits = TopPLogitsWarper(top_p=top_p, filter_value=filter_value, min_tokens_to_keep=min_tokens_to_keep)(
            None, logits
        )

    return logits


def cal_constraint_bounds(scores, logits, mass=0.9, threshold_ratio=4):
    # calculate entropy
    normalized = torch.nn.functional.log_softmax(logits, dim=-1)
    p = torch.exp(normalized)
    ent = -(normalized * p).nansum(-1, keepdim=True)

    normalized = torch.nn.functional.log_softmax(scores, dim=-1)
    shifted_scores = (-normalized) - ent

    scores_normalized = shifted_scores.log_softmax(dim=-1) 
    probs_min = torch.min(scores_normalized, dim=-1).values
    probs_thresh = probs_min + np.log(threshold_ratio)
    probs_max = torch.max(scores_normalized, dim=-1).values
    probs_filter = probs_max - np.log(threshold_ratio)
    probs_filter = probs_filter.unsqueeze(-1)
    mask_filter = [scores_normalized > probs_filter]
    
    probs_thresh = probs_thresh.unsqueeze(-1)
    mask = [scores_normalized >= probs_thresh]
    count_mask = [scores_normalized < probs_thresh]
    if count_mask[0].sum() == 1:
        mask = torch.ones(scores.shape[-1], dtype=torch.bool).unsqueeze(0)
    
    return mask, mask_filter

def coiecd_constraint(logits_cond, logits_uncond, alpha=1.0):
    
    logits_diff = logits_cond - logits_uncond
    
    typical_mask, mask_filter = cal_constraint_bounds(logits_cond, logits_uncond)
    constraint_list = torch.ones_like(logits_diff)

    alpha_list = torch.ones_like(logits_diff) * alpha

    constraint_list[typical_mask] = float(0)
    constraint_list[mask_filter] = float(1)
    _constraint_list = 1- constraint_list
    
    logits_merged = constraint_list * logits_cond + _constraint_list * logits_uncond + logits_diff * alpha_list
    
    return logits_merged



def model_generate(model, input_ids, attention_mask, tgt_len, past_key_values=None):
    ans = torch.tensor([], dtype=torch.int64, device=device)
    n = input_ids.shape[0]
    for i in range(tgt_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values)
            past_key_values = outputs.past_key_values
        
        logits = outputs.logits[:, -1, :] 
        logits = logits - logits.logsumexp(dim=-1, keepdims=True) 
        
        probs = torch.nn.functional.softmax(logits, dim=-1)

        next_tokens = torch.argmax(probs, dim=-1)
        ans = torch.cat([ans, next_tokens], dim=-1)
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)
    answer = tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    return answer#, ave_logits, total_entropy

def model_answer(model, tokenizer, question, facts, tgt_len):
    # Generate
    context = f'Given the following information:{facts}\nAnswer the following question based on the given information with one or few words: {question}\nAnswer:'
    prompt = f'Answer the following question based on your internal knowledge with one or few words: {question}\nAnswer:'

    batch = [context, prompt]
    inputs = tokenizer(batch, padding=True, return_tensors='pt', truncation=True, max_length=2048).to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask 
    

    #------------------context output--------------------------#
    cond_inputs = input_ids[0].unsqueeze(0)
    cond_answer = model_generate(model, cond_inputs, attention_mask[0].unsqueeze(0), tgt_len)
    # print(auto_cond_answer, cond_answer)

    #------------------w/o context output----------------------#
    uncond_inputs = input_ids[1].unsqueeze(0)
    uncond_answer = model_generate(model, uncond_inputs, attention_mask[1].unsqueeze(0), tgt_len)
    # print(auto_uncond_answer, uncond_answer)
    #------------------cad output--------------------------#
    past_key_values = None    
    ans = torch.tensor([], dtype=torch.int64, device=device)
    beta = 1.0
    n = input_ids.shape[0]
    alpha = 0.1
    for i in range(tgt_len):
    # for i in range(tgt_len):
        with torch.no_grad():
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict=True,
                            use_cache=True,
                            past_key_values=past_key_values
                        )
            past_key_values = outputs.past_key_values
        
        
        logits = outputs.logits[:, -1, :] 
        logits = logits - logits.logsumexp(dim=-1, keepdims=True) # scale logits for numerical stability of exp(logits) operation and keep the value of softmax(logits) unchanged

        logits_cond = logits[0].unsqueeze(0)
        logits_uncond = logits[1].unsqueeze(0)
        
        logits_merged = coiecd_constraint(logits_cond, logits_uncond, beta)
        
        # logits_merged = top_k_top_p_filtering(logits_merged, top_k=0, top_p=0.9)
        probs = torch.nn.functional.softmax(logits_merged, dim=-1)
        next_tokens = torch.argmax(probs, dim=-1)
        ans = torch.cat([ans, next_tokens], dim=-1)
        
        # print(ret, end='')
        if next_tokens[0] == tokenizer.eos_token_id:
            break
        # prepare for next iteration
        input_ids = next_tokens.unsqueeze(-1).tile(n, 1)
        attention_mask = torch.cat([attention_mask, torch.ones(n, 1, dtype=torch.long, device=device)], dim=-1)   
    coiecd_answer = tokenizer.decode(ans, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    return coiecd_answer, cond_answer, uncond_answer

if __name__ == '__main__':
    generation_config = GenerationConfig.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS, device_map="auto", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
    
    # tokenizer.pad_token_id = 0
    path = '/data_path'
    output_path = '/output_path'
    id = 0
    
    filter_value = -float("Inf")
    data = load_json_data(path)

    for line in tqdm(data):
        line = json.loads(line)
        question = line['question']
        ground_truth = line['answer']
        context = line['context']
    
        if len(context) > 2000:
            continue
        tgt_len = len(tokenizer.encode(ground_truth, add_special_tokens=False)) 
        coiecd_answer, cond_answer, uncond_answer = model_answer(model, tokenizer, question, context, tgt_len)
        output_data = {'id': id,
                'Question': question,
                'Context': context,
                'True Answer': ground_truth,
                'coiecd_answer': coiecd_answer,
                'cond_answer': cond_answer,
                'uncond_answer': uncond_answer,
                }
        id += 1
        with open(output_path, 'a+') as f:
            json.dump(output_data, f, indent=4)
            f.write(',')
            f.write('\n')
