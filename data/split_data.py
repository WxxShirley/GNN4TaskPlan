"""For each dataset, we prepare test sets where the sequential mode has 500 samples"""
import json
import random


def split(dataset_name):
    data_file = open(f"{dataset_name}/data.json", 'r')

    chains = []
    for line in data_file:
        content = json.loads(line)
        if content["type"] == "chain":
            chains.append(content)

    # split
    random.shuffle(chains)
    chain_ids = [one_sample["id"] for one_sample in chains[:500]]
    
    # write files
    write_content = {
        "test_ids": {
            "chain": chain_ids,
        }
    }
    with open(f"{dataset_name}/split_ids.json", 'w') as file:
        json.dump(write_content, file, indent=4, sort_keys=True)


if __name__ == "__main__":
    random.seed(0)

    for dataset in ["huggingface", "multimedia", "dailylife"]:
        split(dataset)
