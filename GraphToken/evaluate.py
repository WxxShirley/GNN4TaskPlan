import numpy as np 
import json 
import prettytable as pt
import os


def reformat_task_nodes(content):
    raw_nodes = content["task_nodes"]
    nodes = [node["task"] for node in raw_nodes]

    if len(nodes) and not isinstance(nodes[0], str):
        nodes = [ node.get("name", "") for node in nodes]
    return nodes


def reformat_task_links(content):
    raw_links = content["task_links"]
    links = [", ".join([link["source"], link["target"]]) for link in raw_links if isinstance(link, dict) and isinstance(link.get("source", ""), str) and isinstance(link.get("target", ""), str)]
    return links


def f1_score(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0
    
    intersect = set(pred) & set(gt)
    precision = len(intersect) / len(pred)
    recall = len(intersect) / len(gt)
    f = 2 * precision * recall / (precision + recall + 1e-9)
    return f 


def batch_f1_score(pred_list, gt_list):
    f1_score_list = [f1_score(pred, gt) for pred, gt in zip(pred_list, gt_list) ]
    return round(np.mean(np.array(f1_score_list)), 4)


def batch_task_succ(pred_list, gt_list):
    scores = [float(f1_score(pred, gt) >= 0.99) for pred, gt in zip(pred_list, gt_list)]
    succ_rate = round(sum(scores) / len(scores) * 100, 2)
    return succ_rate


def node_hallucination_rate(solution, valid_tools):
    if len(solution) == 0:
        return [0.0, 0.0]
    
    hall_list = [1.0 if node not in valid_tools else 0.0 for node in solution ]
    micro_hall = sum(hall_list) / len(solution)
    macro_hall = 1.0 if sum(hall_list) >= 1 else 0.0

    return [micro_hall, macro_hall]


def batch_node_hallucination(solutions, valid_tools):
    hall_scores = [node_hallucination_rate(sol, valid_tools) for sol in solutions]
    avg_score = np.round(np.mean(np.array(hall_scores), axis=0), 4)
    # avg_score[0] - micro_hallucination
    # avg_score[1] - macro_hallucination
    return avg_score


def prediction_loader(filename, content_type):
    readfile = open(filename, 'r')

    return_data = {}

    for line in readfile:
        data = json.loads(line)
        data_id = data["id"]

        if content_type == 'id':
            retrieve_data = data_id 
        elif content_type == "graph":
            nodes, links = reformat_task_nodes(data), reformat_task_links(data)
            retrieve_data = {"nodes": nodes, "links": links}
        return_data[data_id] = retrieve_data
    
    return return_data


def evaluate(dataset, llm_name, method):
    alignment = json.load(open(f"../data/{dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
    
    gt_filename = f"../data/{dataset}/data.json"
    pred_filename = f"prediction/{dataset}/{llm_name}/{method}.json"

    if not os.path.exists(pred_filename):
        print(f"{pred_filename} does not exists!")
        return 

    gt_tool_nodes = json.load(open(f"../data/{dataset}/tool_desc.json", 'r'))["nodes"]
    gt_tool_links = json.load(open(f"../data/{dataset}/graph_desc.json", 'r'))["links"]
    gt_tool_nodes = [tool["id"] for tool in gt_tool_nodes]
    gt_tool_links = [", ".join([link["source"], link["target"]]) for link in gt_tool_links]
    gt_graph_dict = prediction_loader(gt_filename, content_type="graph")
    pred_graph_dict = prediction_loader(pred_filename, content_type="graph")

    pred_align = prediction_loader(pred_filename, "id")
    alignment_ids = [data_id for data_id in alignment if data_id in pred_align]

    print(f"{pred_filename} # Valid Predictions {len(pred_align)}")
        
    pred_graphs = [pred_graph_dict.get(data_id, {"nodes": [], "links": []}) for data_id in alignment_ids]
    gt_graphs = [gt_graph_dict[data_id] for data_id in alignment_ids]

    node_f1 = batch_f1_score([pred_g["nodes"] for pred_g in pred_graphs], [gt_g["nodes"] for gt_g in gt_graphs])
    link_f1 = batch_f1_score([pred_g["links"] for pred_g in pred_graphs], [gt_g["links"] for gt_g in gt_graphs]) 

    node_hr = batch_node_hallucination([pred_g["nodes"] for pred_g in pred_graphs], gt_tool_nodes)
    link_hr = batch_node_hallucination([pred_g["links"] for pred_g in pred_graphs], gt_tool_links)
    succ_rate = batch_task_succ([pred_g["nodes"] for pred_g in pred_graphs], [gt_g["nodes"] for gt_g in gt_graphs])
            
    table.add_row([dataset, llm_name, method, node_f1, link_f1, succ_rate, node_hr[0], node_hr[1], link_hr[0], link_hr[1]])


if __name__ == "__main__":
    table = pt.PrettyTable()
    table.field_names = ['Dataset', 'LLM', 'Method', 'NF', 'LF', "Succ", 'NH-1', 'NH-2', 'LH-1', 'LH-2']

    for dataset in ["huggingface"]:
        for llm in ["Mistral-7B"]:
            for gnn_type in ["GCN"]:
                method_name = f"GraphToken_{gnn_type}"
                evaluate(dataset, llm, method_name)
     
    print(table)
