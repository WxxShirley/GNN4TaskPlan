import json
import torch 
from scipy.sparse import csr_matrix
import numpy as np 
import scipy.sparse as sp
import os
import random
import copy
import prettytable as pt
from collections import defaultdict


def convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))


def compute_normalize_adj(adj):
    row_sum = np.array(adj.sum(1))
    d_inv = np.power(row_sum, -0.5).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)
    return norm_adj


def load_tool(dataset_name, tool_feature="n+d"):
    """Load tools' textual information, mapping (from `idx` to `name`), and the graph"""
    tool_data = json.load(open(f"../data/{dataset_name}/tool_desc.json", 'r'))["nodes"]

    tool_text, name2idx = [], {}
    idx = 0
    for tool in tool_data:
        tool_str = "{{tool_name}} # Description # {{tool_desc}}"
        tool_str = tool_str.replace("{{tool_name}}", tool["id"])
        tool_str = tool_str.replace("{{tool_desc}}", tool["desc"])
        if tool_feature == "n+d+a":
            tool_str += "Input type: " + str(tool.get("input-type", "")) + " Output-type: " + str(tool.get("output-type", ""))
        
        tool_text.append(tool_str)
        name2idx[tool["id"]] = idx
        idx += 1

    link_data = json.load(open(f"../data/{dataset_name}/graph_desc.json", 'r'))['links']
    # print(len(link_data))
    link_source, link_target = [name2idx[link["source"]] for link in link_data], [name2idx[link["target"]] for link in link_data]
    edge_index = torch.LongTensor([link_source, link_target])
    link_graph = csr_matrix((np.ones(len(link_source)), (np.array(link_source), np.array(link_target))),
                             shape=(len(tool_text), len(tool_text)))
    link_graph = convert_sp_mat_to_sp_tensor(compute_normalize_adj(link_graph))

    idx2name = {v: k for k, v in name2idx.items()}

    adj_graph = {tool: [] for tool in name2idx.keys()}
    for link in link_data:
        adj_graph[link["source"]].append(link["target"])

    return tool_text, name2idx, idx2name, edge_index, link_graph, adj_graph


def reformat_steps(content):
    steps = content["task_steps"]
    if len(steps) and isinstance(steps[0], dict):
        if 'description' in steps[0].keys():
            steps = [step['description'] for step in steps if 'description' in step.keys()]
        else:
            steps = [". ".join([str(s) for s in list(step.values())]) for step in steps]
    elif len(steps) and isinstance(steps[0], list):
        steps = [step[0] for step in steps if len(step) > 0]
    elif len(steps) == 0:
        steps = []
    
    steps = [step.replace("Step ", "") for step in steps if isinstance(step, str)]
    
    return steps


def reformat_task_nodes(content):
    raw_nodes = content["task_nodes"]
    nodes = []
    for node in raw_nodes:
        if isinstance(node, dict) and "task" in node.keys():
            if isinstance(node["task"], list):
                nodes.extend(node["task"])
            elif isinstance(node["task"], str):
                nodes.append(node["task"])
        elif isinstance(node, dict) and 'name' in node.keys():
             nodes.append(node["name"])
        elif isinstance(node, str):
            nodes.append(node)
        # else:
        #      print("error node format", node)
    return nodes


def reformat_task_links(content):
    raw_links = content["task_links"]
    links = []
    for link in raw_links:
        if isinstance(link, dict) and "source" in link.keys() and "target" in link.keys():
            if link["source"] and link["target"]:
                if isinstance(link["source"], str) and isinstance(link["target"], str):
                    links.append(", ".join([link["source"], link["target"]]))
            
        else:
            # vicuna-13b's prediction on multimedia
            if isinstance(link, list) and len(link) == 2:
                nodes = reformat_task_nodes(content)
                link = [e.replace("Step ", "") for e in link]
                try:
                    source_idx, target_idx = eval(link[0]) - 1, eval(link[1]) - 1
                    links.append(", ".join([nodes[source_idx], nodes[target_idx]]))
                except Exception as e:
                    print(f"Reformat link error", e)
    return links


def load_test_data(dataset_name="huggingface", llm_name="CodeLlama-13b", init_alignment_ids=[], method="direct"):
    llm_pred_file = f"../prediction/{dataset_name}/{llm_name}/{method}.json"
    gt_file = f"../data/{dataset_name}/data.json"

    if not os.path.exists(llm_pred_file):
        raise NotImplementedError('LLM Prediction file not exists!')
    
    pred_contents_dict, alignment_ids = {}, []

    for line in open(llm_pred_file, 'r'):
        content = json.loads(line)
        if content.get("task_steps") is None or content["task_steps"] == [] or content["id"] not in init_alignment_ids:
            continue 
        
        alignment_ids.append(content["id"])
        pred_contents_dict[content["id"]] = {
            "user_request": content["user_request"], 
            "steps": reformat_steps(content), 
            "pred_task_nodes": reformat_task_nodes(content), 
            "pred_task_links": reformat_task_links(content),
            "gt_task_nodes": [],
            "gt_task_links": []
        }
    
    for line in open(gt_file, 'r'):
        content = json.loads(line)
        if content["id"] in alignment_ids:
            pred_contents_dict[content["id"]]["gt_task_nodes"] = reformat_task_nodes(content)
            pred_contents_dict[content["id"]]["gt_task_links"] = reformat_task_links(content)

    # print(len(alignment_ids))
    return alignment_ids, pred_contents_dict


def prepare_lm_gnn_training_data(dataset_name="huggingface", tmp_print=False, train_ids=None, maximum=''):
    """Return training data for LM+GNN, each of which in the format `<step, tool>`"""
    raw_data_file = f"../data/{dataset_name}/data.json"

    data_x, data_y = [], []

    for line in open(raw_data_file, 'r'):
        content = json.loads(line)
        
        # this data sample is already in testing data
        if (train_ids and content["id"] not in train_ids):
            continue

        if content['n_tools'] == 1:
            if len(content["task_steps"]) != 1 or len(content["task_nodes"]) != 1:
                continue 

            texts = content['task_steps']
            labels = [node["task"] for node in content['task_nodes']]
        elif content['n_tools'] > 1 and content["type"] == "chain":
            texts = content['task_steps']
            nodes = [node['task'] for node in content['task_nodes']]
            non_dup_nodes = list(set(nodes))

            if len(texts) != len(nodes) or len(nodes) != len(non_dup_nodes):
                continue 

            in_degrees, out_degrees = [0 for _ in nodes], [0 for _ in nodes]
            for link in content['task_links']:
                in_degrees[nodes.index(link['target'])] += 1
                out_degrees[nodes.index(link['source'])] += 1
            
            stop_flg = False 
            start_nodes, end_nodes = [], []

            for deg, node in zip(in_degrees, nodes):
                if deg == 0:
                    start_nodes.append(node)
                if deg > 1:
                    stop_flg = True 

            for deg, node in zip(out_degrees, nodes):
                if deg == 0:
                    end_nodes.append(node)
                if deg > 1:
                    stop_flg = True
            
            if stop_flg or len(start_nodes) != 1 or len(end_nodes) != 1:
                continue 

            cur_node = start_nodes[0]

            while True:
                for link in content['task_links']:
                    if link['source'] == cur_node:
                        cur_node = link['target']
                        start_nodes.append(cur_node)
                        break 
            
                if len(start_nodes) == len(nodes) or cur_node == end_nodes[0]:
                    break 

            labels = copy.deepcopy(start_nodes)
    
        if len(texts) == len(labels):
            data_x.extend(texts)
            data_y.extend(labels)
        
    if maximum:
        maximum = int(maximum)
        mask = list(range(len(data_x)))
        random.shuffle(mask)
        mask = sorted(mask[:maximum])

        data_x = [data_x[i] for i in mask]
        data_y = [data_y[i] for i in mask]
    
    if tmp_print:
        tb = pt.PrettyTable()
        tb.field_names = ["Step", "Tool"]
        for text, label in zip(data_x[:20], data_y[:20]):
            tb.add_row([text[:65], label])
        print(tb)

        # counter = defaultdict(int)
        # tb = pt.PrettyTable()
        # tb.field_names = ["Tool", "Frequency"]
        # for invoked_tool in data_y:
        #     counter[invoked_tool] += 1
        # for k, v in counter.items():
        #     tb.add_row([k, str(v) + f"({v/len(data_y):.3f})"])
        # print(tb)
        # print(len(counter))
        
    
    print(f"[Training Data] {dataset_name.upper()} # Samples {len(data_x)}")
    # print(len(data_x), len(data_y))
    return [[text, label] for text, label in zip(data_x, data_y)]


def prepare_training_ids(dataset_name, train_num=3000, modes=['single', 'chain'], alignment_ids=None):
    gt_data = f"../data/{dataset_name}/data.json"
    candidate_ids, id2type = [], {}
    for line in open(gt_data, 'r'):
        content = json.loads(line)

        if alignment_ids and content["id"] in alignment_ids:
            continue

        if content["type"] in modes:
            candidate_ids.append(content["id"])
            id2type[content["id"]] = content["type"]
    
    random.shuffle(candidate_ids)
    type_counter = [1 if id2type[data_id] == 'chain' else 0 for data_id in candidate_ids[:train_num]]
    print(f"[Training Data] # Chain Samples {sum(type_counter)} ({sum(type_counter) / train_num * 100:.2f})")
    return candidate_ids[:train_num]
