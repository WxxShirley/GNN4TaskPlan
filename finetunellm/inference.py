import json 
import argparse
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
from tqdm import tqdm
from user_prompt import LLM_INFER_PROMPT
import os


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='huggingface', choices=['huggingface', 'multimedia', 'dailylife'])
    parser.add_argument('--llm', type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--load_lora', type=bool, default=True)
    # TODO: adjust the folder name based on actually saved LLM's checkpoint folder
    parser.add_argument('--lora_ckpt', type=str, default="CodeLlama-7b_seed0/checkpoint-5000")
    args = parser.parse_args()
    print(args, "\n")

    # Load fine-tuned model 
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'left'
    
    print(f'Load LoRA fine-tuned LLM {args.llm} from {args.dataset}_{args.lora_ckpt}')
    model = AutoPeftModelForCausalLM.from_pretrained(f"output/{args.dataset}_{args.lora_ckpt}")
    model = model.to(device)

    tool_str = ", ".join([tool["id"] for tool in json.load(open(f"../data/{args.dataset}/tool_desc.json", 'r'))["nodes"]])
    
    alignment_ids = json.load(open(f"../data/{args.dataset}/split_ids.json", 'r'))["test_ids"]["chain"]

    request_file = f"../data/{args.dataset}/user_requests.json"
    
    prompt = "## Task List:\n" + tool_str + "\n" + LLM_INFER_PROMPT

    inputs = []
    for line in open(request_file, 'r'):
        content = json.loads(line)
        if content["id"] in alignment_ids:
            inputs.append({
                "id": content["id"],
                "user_request": content["user_request"]
            })
    
    llm_dict = {
        "codellama/CodeLlama-13b-Instruct-hf": 'CodeLlama-13b',
        "codellama/CodeLlama-7b-Instruct-hf": 'CodeLlama-7b',
        "lmsys/vicuna-13b-v1.5": 'vicuna-13b'
    }
    write_dir = f"../prediction/ft_{llm_dict[args.llm]}"
    if not os.path.exists(write_dir):
        os.mkdir(write_dir)
    
    write_file = open(f"{write_dir}/{args.dataset}_direct.json", 'a')
    
    has_inferenced = []
    for line in open(f"{write_dir}/{args.dataset}_direct.json", 'r'):
        content = json.loads(line)
        has_inferenced.append(content["id"])

    for cur_input in tqdm(inputs):
        if cur_input["id"] in has_inferenced:
            continue 
            
        cur_prompt = prompt.replace("{{user_request}}", cur_input["user_request"])

        tokens = tokenizer(cur_prompt, return_tensors='pt')
        input_ids = tokens['input_ids'].to(device)

        output = model.generate(input_ids, max_length=1024, temperature=0.9)
        decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        decode_output = decode_output.replace(cur_prompt, "")
        if decode_output[-1] == ']':
            decode_output += '}'

        try:
            output_dict = json.loads(decode_output)
            print("SUCC!", output_dict, "\n")
        except Exception as e:
            print("ERROR!", e, "\n")
            
            reformat_prompt = """Please reformat your response into a strict JSON format.""" \
                            + """Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads().""" \
                            + """## Response: {{illegal_response}}\n ## Strict JSON Format:"""
            reformat_prompt = reformat_prompt.replace("{{illegal_response}}", decode_output)
            tokens = tokenizer(reformat_prompt, return_tensors='pt')
            input_ids = tokens['input_ids'].to(device)
            output = model.generate(input_ids, max_length=1024, temperature=0.9)
            decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            decode_output = decode_output.replace(reformat_prompt, "")
            if decode_output[-1] == ']':
                decode_output += '}'
            
            try:
                output_dict = json.loads(decode_output)
            except Exception as e:
                print("Second ERROR!", decode_output)
                continue
        
        write_content = {
            "id": cur_input["id"],
            "user_request": cur_input["user_request"],
            "task_steps": output_dict["task_steps"],
            "task_nodes": output_dict.get("task_nodes", []),
            "task_links": output_dict.get("task_links", []),
        }
        write_file.write(json.dumps(write_content) + "\n")
        write_file.flush()
