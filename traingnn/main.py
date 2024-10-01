import argparse 
import torch 
from model import ModelTrainer
import prettytable as pt
import json 
import os
from sklearn.neighbors import kneighbors_graph
import sys
sys.path.append("../")
from utils import init_random_state, sequence_greedy_tool_selection, prepare_lm_gnn_training_data, load_test_data, get_cur_time, prepare_training_ids, load_tool, save_checkpoint
from sampler import TrainSampler     
from evaluate import batch_f1_score, f1_score
import numpy as np


def build_tool_textual_sim_graph(tool_emb, k=5, metric='cosine'):
    # metric: str = "minkowski",
    text_adj = kneighbors_graph(tool_emb, k, mode='connectivity', metric=metric, include_self=False).toarray()
    text_adj_g = {tool: [] for tool in tool2index.keys()}

    for i in range(text_adj.shape[0]):
        for j in range(text_adj.shape[1]):
            if text_adj[i, j]:
                text_adj_g[index2tool[i]].append(index2tool[j])
    
    # for k, v in text_adj_g.items():
    #     print(k, v)
    return text_adj_g


def bi_evaluate(id_list, pred_dict, tmp_print=False):
    init_scores, searched_scores = [], []
    
    high_fix_examples = []
    succ_cnt, fail_cnt = 0, 0
    for data_id in id_list:
        content = pred_dict[data_id] 
        
        gt_node, gt_link = content["gt_nodes"], content["gt_links"]
        pred_node, pred_link = content["pred_nodes"], content["pred_links"]
            
        search_node = content['search_nodes']
        search_link = [", ".join(link) for link in content["search_links"]]

        node_f1, link_f1 = f1_score(pred_node, gt_node), f1_score(pred_link, gt_link)
        search_node_f1, search_link_f1 = f1_score(search_node, gt_node), f1_score(search_link, gt_link)
        init_succ, search_succ = float(node_f1 >= 0.99), float(search_node_f1 >= 0.99)

        if search_node_f1 > node_f1 and search_link_f1 > link_f1:
            succ_cnt += 1
            high_fix_examples.append([data_id, round(search_node_f1 - node_f1), round(search_link_f1 - link_f1)])
        elif search_node_f1 < node_f1 and search_link_f1 < link_f1:
            fail_cnt += 1

        init_scores.append([node_f1, link_f1, init_succ])
        searched_scores.append([search_node_f1, search_link_f1, search_succ])

    avg_pred_score = np.round(np.mean(np.array(init_scores), axis=0), 4)
    avg_searched_score = np.round(np.mean(np.array(searched_scores), axis=0), 4)
    if tmp_print:
        print(f"Init   [Node-F1] {avg_pred_score[0]:.4f} [Link-F1] {avg_pred_score[1]:.4f}")
        print(f"Search [Node-F1] {avg_searched_score[0]:.4f} [Link-F1] {avg_searched_score[1]:.4f}")
    # print(f"# Succ {succ_cnt} # Fail {fail_cnt}")

    high_fix_examples = sorted(high_fix_examples, key=lambda x: x[1], reverse=True)[:50]
    
    return {
        "base-node-f1": avg_pred_score[0],
        "base-link-f1": avg_pred_score[1],
        "base-acc": avg_pred_score[2],
        "search-node-f1": avg_searched_score[0],
        "search-link-f1": avg_searched_score[1],
        "search-acc": avg_searched_score[2]
    }, [example[0] for example in high_fix_examples]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataset and training related
    parser.add_argument('--dataset', type=str, default='huggingface', choices=['huggingface', 'multimedia', 'dailylife'])
    parser.add_argument('--load_alignment', type=bool, default=True)
    parser.add_argument('--maximum', type=str, default='')

    # LM related 
    parser.add_argument('--lm_name', type=str, default='intfloat/e5-large', choices=['intfloat/e5-large', 'sentence-transformers/all-roberta-large-v1', 'sentence-transformers/all-MiniLM-L6-v2', "intfloat/e5-large-v2"])
    
    # GNN related
    parser.add_argument('--gnn_name', type=str, default='GCN', choices=['SGC', 'GCN', 'GAT', 'SAGE', 'GIN', 'TransformerConv'])
    parser.add_argument('--gnn_hidden_dim', type=int, default=1024)
    parser.add_argument('--gnn_layer', type=int, default=1)

    # Training models related 
    parser.add_argument('--train_num', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_negatives', type=int, default=2)
    parser.add_argument('--lm_frozen', type=int, default=0)
    parser.add_argument('--text_negative', type=int, default=0) # negative samples are neighbors in Textual-Sim Tool Graph

    # Test related
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--measure', type=str, default='dot', choices=['dot', 'distance'])
    parser.add_argument('--save_model', type=int, default=0, choices=[0, 1])
    parser.add_argument('--load_model', type=int, default=0)

    args = parser.parse_args()
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    init_random_state(args.seed)
    device = torch.device(args.device)
   
    if args.dataset == 'multimedia':
        args.batch_size = 256
    
    ####################################
    #### Prepare Trainset and Tool #####
    ####################################
    alignment_ids = json.load(open(f"../data/{args.dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
    train_ids = prepare_training_ids(args.dataset, train_num=args.train_num, alignment_ids=alignment_ids)

    train_data = prepare_lm_gnn_training_data(dataset_name=args.dataset,
                                              train_ids=train_ids)
    tool_texts, tool2index, index2tool, edge_index, sgc_edge_index, adj_g = load_tool(dataset_name=args.dataset)

    ####################################
    ##### Prepare and Load model #######
    ####################################
    controller = ModelTrainer(args, device=device)
    if args.text_negative:
        tool_emb = controller.model.tool_forward(tool_texts).detach().cpu().numpy()
        text_adj_g = build_tool_textual_sim_graph(tool_emb)
        sample_obj = TrainSampler(train_data, args.num_negatives, text_adj_g, tool2index, hard_negative=True, batch_size=args.batch_size)
    else:
        sample_obj = TrainSampler(train_data, args.num_negatives, adj_g, tool2index, hard_negative=True, batch_size=args.batch_size)
    tool_adj = edge_index if args.gnn_name != 'SGC' else sgc_edge_index
    tool_adj = tool_adj.to(device)

    save_path = f"ckpts/{args.dataset}_lm{'_tune' if not args.lm_frozen else '_frozen'}_{args.gnn_name}_epoch{args.epoch}_batch{args.batch_size}_{'text' if args.text_negative else 'raw'}_neg.pt"
    if os.path.exists(save_path):
        ckpt = torch.load(save_path)
        controller.model.load_state_dict(ckpt, strict=False)
        print(f"Load Pre-trained Model from {save_path}")
    else:
        best_model, total_time = controller.train(tool_texts, tool_adj, sample_obj)
        controller.model = best_model 
        if args.save_model:
            save_checkpoint(best_model, save_path)
        print(f"\nFinish Training, Overall time {total_time:.3f}s")
    
    controller.model.eval()

    if args.gnn_name == 'SGC':
        alpha = controller.model.gnn_model.alpha 
        print(alpha)
    
    ####################################
    ############# Model Test ###########
    ####################################
    tool_emb = controller.model.tool_forward(tool_texts, tool_adj).detach().cpu().numpy()
    
    table = pt.PrettyTable()
    table.field_names = ["Dataset", "LLM", "LM", "GNN", "N-F1", "L-F1", "Accuracy"]
    lm_name = args.lm_name.split('/')[-1]
    candidate_example_ids = set()

    base_llms = ['CodeLlama-13b', 'mistral-7b']
    # you can add any other you have inferenced LLMs
    # base_llms = ['Baichuan-13b', 'vicuna-13b', 'CodeLlama-7b', 'mistral-7b', 'CodeLlama-13b', 'gpt-35-turbo']

    for base_llm in base_llms:
        methods = ['direct']
        # you can add any other you have inferenced methods, like other prompt templates 
        # methods = ['direct', 'direct_demo1_graph']

        for method in methods:
            new_alignment_ids, pred_content_dict = load_test_data(args.dataset, base_llm, alignment_ids, method=method)
        
            final_pred_dict = {}

            for data_id in new_alignment_ids:
                steps = pred_content_dict[data_id]["steps"]
            
                steps_emb = controller.model.lm_forward(steps, max_length=64, batch_size=len(steps)+1).detach().cpu().numpy()
                ans = sequence_greedy_tool_selection(steps_emb, tool_emb, index2tool, adj_g, measure=args.measure)
                
                final_pred_dict[data_id] = {
                    "steps": steps,
                    "pred_nodes": pred_content_dict[data_id]["pred_task_nodes"],
                    "pred_links": pred_content_dict[data_id]["pred_task_links"],
                    "search_nodes": ans["task_nodes"],
                    "search_links": ans["task_links"],
                    "gt_nodes": pred_content_dict[data_id]["gt_task_nodes"],
                    "gt_links": pred_content_dict[data_id]["gt_task_links"]
                }
            
            score_dict, cur_candidates = bi_evaluate(new_alignment_ids, final_pred_dict)
            if len(list(candidate_example_ids)) == 0:
                candidate_example_ids = set(cur_candidates)
            else:
                candidate_example_ids = candidate_example_ids & set(cur_candidates)
                # print(candidate_example_ids)
            table.add_row([args.dataset, base_llm, lm_name, method, score_dict['base-node-f1'], score_dict['base-link-f1'], score_dict["base-acc"]])
            table.add_row([args.dataset, base_llm, lm_name, "+" + args.gnn_name, score_dict['search-node-f1'], score_dict['search-link-f1'], score_dict["search-acc"]])
    
    print(table)
    # print(candidate_example_ids)

    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
