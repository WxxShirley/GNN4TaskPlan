import json 
from torch.utils.data import Dataset
import numpy as np
import sys
sys.path.append("../")
from utils import prepare_training_ids, init_random_state
from transformers import AutoTokenizer


PROMPT = """\n\n# GOAL #\nPlease understand the user's request and generate task steps and task invocation graph to solve it.""" \
       + """\n\n# REQUIREMENT #\n1. The format must in a strict JSON format as {"task_steps": [ concrete step descriptions ], "task_nodes": [ a list of tasks to be executed in sequence to fulfill user's request ], "task_links": [{"source": "task name i", "target": "task name j"}]}\n""" \
       + """2. The generated task steps and task nodes can resolve the given user request perfectly. Task name must be selected from TASK LIST.\n""" \
       + """3. Task steps should strictly aligned with task nodes, and the number of task steps should be same with the task nodes.\n""" \
       + """4. The task links should reflect the dependencies among task nodes, i.e. the order in which the APIs are invoked.\n""" 


class TaskPlanningDataset(Dataset):
    def __init__(self, dataset_name):
        super().__init__()
        
        self.dataset = dataset_name
        self.idxes_split = self.get_idx_split()
        self.raw_data_dictionary = self.load_raw_data()
        self.prepare_prompt()

    def prepare_prompt(self):
        tool_list = json.load(open(f"../data/{self.dataset}/tool_desc.json", "r"))["nodes"]
        tool_string = "# TASK LIST #:\n" + ", ".join([task["id"] for task in tool_list]) 

        self.prompt = tool_string + PROMPT +  """\n\n# USER REQUEST #: {{user_request}}\nNow please generate your result in a strict JSON format:\n# RESULT #:"""

    def __len__(self):
        return len(self.id_mapping)
    
    def __getitem__(self, index):
        origin_id = self.id_mapping[index]
        origin_data = self.raw_data_dictionary[origin_id]
        
        cur_request = self.prompt.replace("{{user_request}}", origin_data["request"])
        return {
            "id": index, 
            "origin_id": origin_id,
            "request": cur_request,
            "label": json.dumps(origin_data["label"])
        }

    def load_raw_data(self):
        data_file = f"../data/{self.dataset}/data.json"
        data_dict = {}
        for line in open(data_file, 'r'):
            content = json.loads(line)
            data_dict[content["id"]] = {
                "id": content["id"], # origin ID
                "request": content["user_request"],
                "label": {
                    "task_steps": content["task_steps"],
                    "task_nodes": [node["task"] for node in content["task_nodes"]],
                    "task_links": content["task_links"],
                }
            }
        return data_dict

    def get_idx_split(self):
        split_id_file = f"../data/{self.dataset}/split_ids.json"
        test_ids = json.load(open(split_id_file, 'r'))["test_ids"]["chain"]

        train_ids = prepare_training_ids(self.dataset, train_num=3000, alignment_ids=test_ids)
        
        # rename test_ids / train_ids 
        assert len(list(set(train_ids) & set(test_ids))) == 0
        full_ids = train_ids + test_ids 
        self.id_mapping = {idx: origin_id for idx, origin_id in enumerate(full_ids)}
        self.reverse_id_mapping = {origin_id: idx for idx, origin_id in enumerate(full_ids)}

        formatted_train_ids = [self.reverse_id_mapping[origin_id] for origin_id in train_ids]
        formatted_test_ids = [self.reverse_id_mapping[origin_id] for origin_id in test_ids]
        assert len(list(set(formatted_train_ids) & set(formatted_test_ids))) == 0
        print(f"[Data Split] # Train {len(formatted_train_ids)}  # Test {len(formatted_test_ids)}")

        return {
            "train": formatted_train_ids, 
            # the split of train/val will be provided in `main.py`
            "val": [],
            "test": formatted_test_ids
        }


if __name__ == "__main__":
    init_random_state(0)
    plan_dataset = TaskPlanningDataset("huggingface")
    print(plan_dataset[1], "\n")
    
    path_mapping = {
        "CodeLlama-13B": "codellama/CodeLlama-13b-Instruct-hf",
        "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
        "CodeLlama-7B": "codellama/CodeLlama-7b-Instruct-hf",
        "Vicuna-13B": "lmsys/vicuna-13b-v1.5"
    }
    llm = "Mistral-7B"
    tokenizer = AutoTokenizer.from_pretrained(path_mapping[llm])
    
    input_lengths, output_lengths = [], []

    for i in range(3000):
        data = plan_dataset[i]
        input_lengths.append(len(tokenizer(data["request"]).input_ids))
        output_lengths.append(len(tokenizer(data["label"]).input_ids))
    
    input_lengths, output_lengths = np.array(input_lengths), np.array(output_lengths)
    print("[Input] MAX {:.0f} AVG {:.3f} 95% {:.3f}".format(np.max(input_lengths), np.mean(input_lengths), np.percentile(input_lengths, q=95)))
    print("[Output] MAX {:.0f} AVG {:.3f} 95% {:.3f}".format(np.max(output_lengths), np.mean(output_lengths), np.percentile(output_lengths, q=95)))
