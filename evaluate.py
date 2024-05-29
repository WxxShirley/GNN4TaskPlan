import numpy as np 
from datasets import load_metric 
import json 
import prettytable as pt
import click
from utils import reformat_steps, reformat_task_links, reformat_task_nodes


def f1_score(pred, gt):
    if len(pred) == 0 or len(gt) == 0:
        return 0
    
    intersect = set(pred) & set(gt)
    precision = len(intersect) / len(pred)
    recall = len(intersect) / len(gt)
    f = 2 * precision * recall / (precision + recall + 1e-9)
    return f 


def batch_f1_score(pred_list, gt_list):
    f1_score_list = [f1_score(pred, gt) for pred, gt in zip(pred_list, gt_list)]
    return round(np.mean(np.array(f1_score_list)), 4)


def node_hallucination_rate(solution, valid_tools, data_id=None):
    if len(solution) == 0:
        return [0.0, 0.0]
    
    hall_list = [1.0 if node not in valid_tools else 0.0 for node in solution ]
    micro_hall = sum(hall_list) / len(solution)
    macro_hall = 1.0 if sum(hall_list) >= 1 else 0.0

    return [micro_hall, macro_hall]


def batch_node_hallucination(solutions, valid_tools, print_ids=None):
    if print_ids:
        hall_scores = [node_hallucination_rate(sol, valid_tools, id) for sol, id in zip(solutions, print_ids)]
    else:
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
        elif content_type == "steps":
            steps = reformat_steps(data)
            retrieve_data = ", ".join(steps)
            
        elif content_type == "graph":
            nodes, links = reformat_task_nodes(data), reformat_task_links(data)
            retrieve_data = {"nodes": nodes, "links": links}
        
        elif content_type == "efficiency":
            retrieve_data = {
                "cost_time": data.get("cost_time", 0),
                "llm_query_times": data.get("llm_query_times", 1)
            }
        
        return_data[data_id] = retrieve_data
    
    return return_data


def evaluate(dataset, llm_name, method, metrics=["graph"], modes=["chain"], compute_all=True, remove_non_pred=False):
    alignment = json.load(open(f"data/{dataset}/split_ids.json", 'r'))["test_ids"]
    
    gt_filename = f"data/{dataset}/data.json"
    pred_filename = f"prediction/{dataset}/{llm_name}/{method}.json"

    table = pt.PrettyTable()
    if "step" in metrics:
        table.field_names = ['Dataset', 'LLM', 'Mode', 'Step-R1', 'Step-R2', 'NF', 'LF', 'NH-1', 'NH-2', 'LH-1', 'LH-2']
    else:
        table.field_names = ['Dataset', 'LLM', 'Mode', 'NF', 'LF', 'NH-1', 'NH-2', 'LH-1', 'LH-2']

    gt_tool_nodes = json.load(open(f"data/{dataset}/tool_desc.json", 'r'))["nodes"]
    gt_tool_links = json.load(open(f"data/{dataset}/graph_desc.json", 'r'))["links"]
    gt_tool_nodes = [tool["id"] for tool in gt_tool_nodes]
    gt_tool_links = [", ".join([link["source"], link["target"]]) for link in gt_tool_links]
    gt_graph_dict = prediction_loader(gt_filename, content_type="graph")
    pred_graph_dict = prediction_loader(pred_filename, content_type="graph")

    pred_align = prediction_loader(pred_filename, "id")
    print(f"{pred_filename} # Valid Predictions {len(pred_align)}")
    for mode in modes:
        alignment_ids = alignment[mode]
        if remove_non_pred:
            alignment_ids = [data_id for data_id in alignment[mode] if data_id in pred_align]
        
        if not len(alignment_ids):
            continue

        metrics_dict = {}

        if "step" in metrics:
            # step metrics: ['roug1', 'rouge2']
            gt_steps_dict = prediction_loader(gt_filename, content_type="steps")
            pred_steps_dict = prediction_loader(pred_filename, content_type="steps")

            pred_content = [pred_steps_dict.get(data_id, "...") for data_id in alignment_ids]
            gt_content = [gt_steps_dict[data_id] for data_id in alignment_ids]

            rouge = load_metric("rouge")
            rouge_scores = rouge.compute(predictions=pred_content, references=gt_content, use_aggregator=True)

            for key in ["rouge1", "rouge2"]:
                metrics_dict[f"step_{key}"] = round(rouge_scores[key].mid.fmeasure, 4)

        if "graph" in metrics:
            pred_graphs = [pred_graph_dict.get(data_id, {"nodes": [], "links": []}) for data_id in alignment_ids]
            gt_graphs = [gt_graph_dict[data_id] for data_id in alignment_ids]

            node_f1 = batch_f1_score([pred_g["nodes"] for pred_g in pred_graphs], [gt_g["nodes"] for gt_g in gt_graphs])
            link_f1 = batch_f1_score([pred_g["links"] for pred_g in pred_graphs], [gt_g["links"] for gt_g in gt_graphs]) if mode != "single" else 'N/A'

            node_hr = batch_node_hallucination([pred_g["nodes"] for pred_g in pred_graphs], gt_tool_nodes, alignment_ids)
            link_hr = batch_node_hallucination([pred_g["links"] for pred_g in pred_graphs], gt_tool_links)
            
            if 'step' not in metrics:
                table.add_row([dataset, llm_name, mode, node_f1, link_f1, node_hr[0], node_hr[1], link_hr[0], link_hr[1]])
            else:
                table.add_row([dataset, llm_name, mode, metrics_dict['step_rouge1'], metrics_dict['step_rouge2'], node_f1, link_f1, node_hr[0], node_hr[1], link_hr[0], link_hr[1]])
    
    print(table)


@click.command()
@click.option("--dataset", default="huggingface", help="The directory of the data")
@click.option("--llm", type=str, default="CodeLlama-13b") 
@click.option("--remove_non_pred", type=int, default=1)
@click.option("--method", type=str, default="direct")
def main(dataset, llm, remove_non_pred, method):
    # TODO: you need to specify the LLM's short name, the dataset, and the method's name
    # e.g. evaluate CodeLlama-7b's direct inference on HuggingFace
    #  python evaluate.py --llm=CodeLlama-7b --dataset=huggingface --method=direct
    evaluate(dataset, llm, method, remove_non_pred=remove_non_pred)


if __name__ == "__main__":
    main()
