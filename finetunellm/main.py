import json 
import argparse
from peft import LoraConfig, get_peft_model 
from transformers import TrainingArguments, Trainer, IntervalStrategy, AutoModelForCausalLM, AutoTokenizer
import random
import torch
import numpy as np
from user_prompt import LLM_INFER_PROMPT
import sys 
sys.path.append("../")
from utils import TextDataset, prepare_training_ids, init_random_state


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100


def prepare_llm_training_data(dataset_name="huggingface", train_ids=None):
    raw_data = f"../data/{dataset_name}/data.json"

    data_contents = []

    tool_file = f"../data/{dataset_name}/tool_desc.json"
    tool_str = ", ".join([tool['id'] for tool in json.load(open(tool_file, 'r'))['nodes']])

    for line in open(raw_data, 'r'):
        content = json.loads(line)

        if train_ids and content["id"] not in train_ids:
            continue 

        input = content['user_request']

        prompt = "## Task List:\n" + tool_str + "\n" + LLM_INFER_PROMPT
        
        output = {
                "task_steps": content["task_steps"],
                "task_nodes": content["task_nodes"],
                "task_links": content["task_links"]
        }

        data_contents.append({
            'id': content['id'],
            'prompt': prompt, 
            "input": input, 
            "output": json.dumps(output)
        })
    
    random.shuffle(data_contents)
    return data_contents


def tokenizer_dataset(raw_data, max_ans_len=64, max_txt_len=256):
    bos_tokens = tokenizer(BOS, add_special_tokens=False)
    eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
    eos_tokens = tokenizer(EOS, add_special_tokens=False)
    
    full_input_ids, full_attention_masks, full_labels = [], [], []
    for sample in raw_data:
        input = sample["prompt"].replace("{{user_request}}", sample["input"])
        label = sample["output"]

        # print(input, label, "\n\n\n")
        
        tokenized_input = tokenizer(input, add_special_tokens=False)
        tokenized_label = tokenizer(label, add_special_tokens=False)
        # tmp_input_len.append(len(tokenized_input.input_ids))
        # tmp_ans_len.append(len(tokenized_label.input_ids))
        
        label_ids = tokenized_label.input_ids[:max_ans_len] + eos_tokens.input_ids
        input_ids = bos_tokens.input_ids + tokenized_input.input_ids[:max_txt_len] + eos_user_tokens.input_ids + label_ids 
        label_ids = [IGNORE_INDEX] * (len(input_ids) - len(label_ids)) + label_ids

        full_input_ids.append(input_ids)
        full_attention_masks.append([1] * len(input_ids))
        full_labels.append(label_ids)
    
    max_length = max([len(x) for x in full_input_ids])
    # print(f"Max Length of Input {np.max(np.array(tmp_input_len))}, Avg Length {np.mean(np.array(tmp_input_len)):.3f}, 95% Percentitle {np.percentile(np.array(tmp_input_len), 95):.3f}")
    # print(f"Max Length of Ans {np.max(np.array(tmp_ans_len))}, Avg Length {np.mean(np.array(tmp_ans_len)):.3f}, 95% Percentitle {np.percentile(np.array(tmp_ans_len), 95):.3f}")
    for i in range(len(full_input_ids)):
        pad_length = max_length - len(full_input_ids[i])
        full_input_ids[i] =  [0] * pad_length + full_input_ids[i]
        full_attention_masks[i] = [0] * pad_length + full_attention_masks[i] 
        full_labels[i] = [IGNORE_INDEX] * pad_length + full_labels[i]
    
    input_ids = torch.tensor(full_input_ids).to(device)
    attention_mask = torch.tensor(full_attention_masks).to(device)
    label_input_ids = torch.tensor(full_labels).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_input_ids
    }
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='dailylife', choices=['huggingface', 'multimedia', 'dailylife'])
    parser.add_argument('--llm', type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--max_txt_length', type=int, default=512)
    parser.add_argument('--max_ans_length', type=int, default=300)
    parser.add_argument('--num_epoch', type=int, default=2)
    args = parser.parse_args()
    print(args, "\n")
    init_random_state(args.seed)
    
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.llm)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = 'left'

    alignment_ids = json.load(open(f"../data/{args.dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
    train_ids = prepare_training_ids(args.dataset, alignment_ids=alignment_ids)
    data_contents = prepare_llm_training_data(args.dataset, train_ids=train_ids)
    encodings = tokenizer_dataset(data_contents, max_ans_len=args.max_ans_length, max_txt_len=args.max_txt_length)
    dataset = TextDataset(encodings)

    train_num = int(0.85 * len(dataset))
    
    train_dataset = torch.utils.data.Subset(dataset, list(range(train_num)))
    val_dataset = torch.utils.data.Subset(dataset, list(range(train_num, len(dataset))))
    
    print(len(dataset), len(train_dataset), len(val_dataset))
    # print(train_dataset[0])
    llm_dict = {
        "codellama/CodeLlama-13b-Instruct-hf": 'CodeLlama-13b',
        "codellama/CodeLlama-7b-Instruct-hf": 'CodeLlama-7b',
        "lmsys/vicuna-13b-v1.5": "vicuna-13b"
    }

    save_dir = f"output/{args.dataset}_{llm_dict[args.llm]}_seed{args.seed}"

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    kwargs = {
        'max_memory': {0: '80GiB'},
        # if have 2 A100
        # 'max_memory': {0: '80GiB', 1: '80GiB'}
        'device_map': "auto",
    }

    model = AutoModelForCausalLM.from_pretrained(args.llm, **kwargs)
    ft_model = get_peft_model(model, peft_config)
    ft_model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=save_dir,
        learning_rate=1e-5, 
        per_device_eval_batch_size=2, # 1 for multimedia / dailylife
        per_device_train_batch_size=2, # 1 for multimedia / dailylife
        num_train_epochs=args.num_epoch,
        weight_decay=0.01,
        eval_steps=200,
        evaluation_strategy=IntervalStrategy.STEPS,
        save_steps=200,
        save_total_limit=1,
        load_best_model_at_end=True
    )

    trainer = Trainer(
        model=ft_model,
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # trainer.train()
    # if the training have error
    # try `pip install markupsafe==2.0.1`
    # trainer.train(resume_from_checkpoint=True)
    trainer.train()
