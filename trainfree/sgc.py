import sys 
import argparse
import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import numpy as np
import json
import prettytable as pt
import time
sys.path.append("../")
from utils import TextDataset, load_tool, load_test_data, sequence_greedy_tool_selection
from evaluate import f1_score
import warnings
warnings.filterwarnings('ignore')


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] 
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def bi_evaluate(id_list, pred_dict, tmp_print=False):
    init_scores, searched_scores, cost_times = [], [], []

    for data_id in id_list:
        content = pred_dict[data_id] 
        
        gt_node, gt_link = content["gt_nodes"], content["gt_links"]
        pred_node, pred_link = content["pred_nodes"], content["pred_links"]
            
        search_node = content['search_nodes']
        search_link = [", ".join(link) for link in content["search_links"]]

        node_f1, link_f1 = f1_score(pred_node, gt_node), f1_score(pred_link, gt_link)
        search_node_f1, search_link_f1 = f1_score(search_node, gt_node), f1_score(search_link, gt_link)
        
        init_scores.append([node_f1, link_f1])

        searched_scores.append([search_node_f1, search_link_f1])
        cost_times.append(content["cost_time"])

    avg_pred_score = np.round(np.mean(np.array(init_scores), axis=0), 4)
    avg_searched_score = np.round(np.mean(np.array(searched_scores), axis=0), 4)
    if tmp_print:
        print(f"Init   [Node-F1] {avg_pred_score[0]:.4f} [Link-F1] {avg_pred_score[1]:.4f}")
        print(f"Search [Node-F1] {avg_searched_score[0]:.4f} [Link-F1] {avg_searched_score[1]:.4f}")
    
    return {
        "base-node-f1": avg_pred_score[0],
        "base-link-f1": avg_pred_score[1],
        "search-node-f1": avg_searched_score[0],
        "search-link-f1": avg_searched_score[1],
        "cost_time": np.round(np.mean(np.array(cost_times)), 4)
    }


def text_lm_forward(text, max_length=256, batch_size=256, pool="cls"):
    x = tokenizer(text, padding=True, truncation=True, max_length=max_length)
    format_data = TextDataset(x)

    text_emb = None 
    dataloader = DataLoader(format_data, shuffle=False, batch_size=batch_size)

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        output = lm_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            output_hidden_states=True,
        )
        
        if pool == "cls":
            emb = output['hidden_states'][-1]
            cls_token_emb = emb.permute(1, 0, 2)[0]
        else:
            cls_token_emb = mean_pooling(output, batch["attention_mask"])
        
        token_emb = cls_token_emb.cpu().detach().numpy()
        if text_emb is None:
            text_emb = token_emb
        else:
            text_emb = np.vstack((text_emb, token_emb))
    return text_emb


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='huggingface', choices=['huggingface', 'multimedia', 'dailylife', 'tmdb'])
    parser.add_argument('--lm_name', type=str, default='intfloat/e5-large', choices=['intfloat/e5-large', 'sentence-transformers/all-roberta-large-v1', 'intfloat/e5-large-v2'])
    parser.add_argument('--lm_pool', type=str, default="cls", choices=['mean', 'cls'])
    # LLM Choices ['CodeLlama-13b', 'mistral-7b', 'Baichuan-13b', 'CodeLlama-7b', 'vicuna-13b']
    parser.add_argument('--llm_name', type=str, default='CodeLlama-13b')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--use_graph', type=int, default=1)
    parser.add_argument('--alpha_list', type=list, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]) 
    parser.add_argument('--measure', type=str, default='distance', choices=['distance', 'dot'])

    args = parser.parse_args()
    print(args, "\n")
    device = torch.device(args.device)

    tool_texts, tool2index, index2tool, _, link_g, adj_g = load_tool(dataset_name=args.dataset)

    tokenizer = AutoTokenizer.from_pretrained(args.lm_name)
    lm_model = AutoModel.from_pretrained(args.lm_name).to(device)
    trainable_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
    print(f"[Model] Number of parameters: {trainable_params}")

    tool_emb = text_lm_forward(tool_texts, max_length=128, batch_size=64)

    if args.use_graph:
        tool_emb = torch.FloatTensor(tool_emb)
        tool_emb_neigh = torch.sparse.mm(link_g, tool_emb)

        tool_emb_list = []
        for alpha in args.alpha_list:
            cur_tool_emb = alpha * tool_emb + (1 - alpha) * tool_emb_neigh
            cur_tool_emb = cur_tool_emb.cpu().detach().numpy()
            tool_emb_list.append(cur_tool_emb)

    alignment_ids = json.load(open(f"../data/{args.dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
    
    # for "method", you can also replace with other prompting methods
    new_alignment_ids, pred_dict = load_test_data(args.dataset, args.llm_name, init_alignment_ids=alignment_ids, method="direct")
    
    # for computing Hallucination
    valid_nodes = [node["id"] for node in json.load(open(f"../data/{args.dataset}/tool_desc.json", 'r'))["nodes"]]
    valid_links = [", ".join([link["source"], link["target"]]) for link in json.load(open(f"../data/{args.dataset}/graph_desc.json", 'r'))["links"]]

    if args.use_graph:
        tb = pt.PrettyTable()
        tb.field_names = ["Dataset", "LLM", "LM", "Alpha", "Node-F1", "Link-F1"]
        lm_name = args.lm_name.split('/')[-1]

        for idx in range(len(tool_emb_list)):
            final_pred_dict = {}

            for data_id in new_alignment_ids:
                st_time = time.time()
                steps = pred_dict[data_id]["steps"]
          
                steps_emb = text_lm_forward(steps, max_length=64, batch_size=len(steps)+1, pool=args.lm_pool)
                ans = sequence_greedy_tool_selection(steps_emb, tool_emb_list[idx], index2tool, adj_g, measure=args.measure)
                
                final_pred_dict[data_id] = {
                    "steps": steps,
                    "pred_nodes": pred_dict[data_id]["pred_task_nodes"],
                    "pred_links": pred_dict[data_id]["pred_task_links"],
                    "search_nodes": ans["task_nodes"],
                    "search_links": ans["task_links"],
                    "gt_nodes": pred_dict[data_id]["gt_task_nodes"],
                    "gt_links": pred_dict[data_id]["gt_task_links"],
                    "cost_time": time.time() - st_time
                }

            score_dict = bi_evaluate(new_alignment_ids, final_pred_dict)
            if idx == 0:
                tb.add_row([args.dataset, args.llm_name, lm_name, 'Direct', score_dict['base-node-f1'], score_dict['base-link-f1']])
            alpha_name = args.alpha_list[idx] if idx != len(args.alpha_list) - 1 else "No Graph"
            tb.add_row([args.dataset, args.llm_name, lm_name, alpha_name, score_dict['search-node-f1'], score_dict['search-link-f1']])
        
        print(tb)
    else:
        final_pred_dict = {}
        for data_id in new_alignment_ids:
            st_time = time.time()
            steps = pred_dict[data_id]["steps"]

            steps_emb = text_lm_forward(steps, max_length=64, batch_size=len(steps)+1, pool=args.lm_pool)
            ans = sequence_greedy_tool_selection(steps_emb, tool_emb, index2tool, adj_g, measure=args.measure)

            final_pred_dict[data_id] = {
                "steps": steps,
                "pred_nodes": pred_dict[data_id]["pred_task_nodes"],
                "pred_links": pred_dict[data_id]["pred_task_links"],
                "search_nodes": ans["task_nodes"],
                "search_links": ans["task_links"],
                "gt_nodes": pred_dict[data_id]["gt_task_nodes"],
                "gt_links": pred_dict[data_id]["gt_task_links"],
                "cost_time": time.time() - st_time
            }
    
        bi_evaluate(new_alignment_ids, final_pred_dict, tmp_print=True)
