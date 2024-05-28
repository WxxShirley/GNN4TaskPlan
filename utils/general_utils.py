import torch 
import random 
import numpy as np
import datetime
import time
import pytz


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def init_random_state(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model, save_path):
    param_grad_dic = {
        k: v.requires_grad for (k, v) in model.named_parameters()
    }
    state_dict = model.state_dict()
    for k in list(state_dict.keys()):
        if k in param_grad_dic.keys() and not param_grad_dic[k]:
            del state_dict[k]

    torch.save(state_dict, save_path)


def parallel_greedy_tool_selection(steps_emb, support, id2tool, adj_graph, steps=None, measure="dot"):
    answers = []
    prev_tool = None 
    for i in range(steps_emb.shape[0]):
        cur_step_emb = steps_emb[i, :]
        
        if measure == "distance":
            distance = np.sqrt(np.sum(np.asarray(cur_step_emb - support) ** 2, axis=1))
            candidate_idxes = list(np.argsort(distance))
        elif measure == "dot":
            score = np.dot(cur_step_emb, support.T)
            candidate_idxes = list(np.argsort(-score))
            
        if i == 0:
            # First Step: select the tool with highest score from All tools
            candidate_idx = candidate_idxes[0]
            prev_tool = id2tool[candidate_idx]
        else:
            
            for idx in candidate_idxes:
                if id2tool[idx] != prev_tool and id2tool[idx] not in answers:
                    prev_tool = id2tool[idx]
                    break 
        
        answers.append(prev_tool)
        if steps:
            print(steps[i], prev_tool)
    
    search_tool_list = answers 
    search_tool_link = [[answers[idx], answers[idx+1]] for idx in range(len(answers)-1)]
    return {
        "task_nodes": search_tool_list, 
        "task_links": search_tool_link
    }


def sequence_greedy_tool_selection(steps_emb, support, id2tool, adj_graph, steps=None, measure="dot"):
    """Greedy Strategy: Each time select the tool with highest score, and also consider the neighboring relationship"""
    prev_tool = None
    answers = []
    for i in range(steps_emb.shape[0]):
        cur_step_emb = steps_emb[i, :]
        
        if measure == "distance":
            distance = np.sqrt(np.sum(np.asarray(cur_step_emb - support) ** 2, axis=1))
            candidate_idxes = list(np.argsort(distance))
        elif measure == "dot":
            score = np.dot(cur_step_emb, support.T)
            candidate_idxes = list(np.argsort(-score))
            
        if i == 0:
            # First Step: select the tool with highest score from All tools
            candidate_idx = candidate_idxes[0]
            prev_tool = id2tool[candidate_idx]
        else:
            # Remaining Steps: select the tool with highest score from Neighboring tools
            neighs = [neigh_name for neigh_name in adj_graph[prev_tool]]
            
            for idx in candidate_idxes:
                if id2tool[idx] != prev_tool and id2tool[idx] in neighs:
                    prev_tool = id2tool[idx]
                    break 
        
        # no suitable tools, skip
        if i > 0 and prev_tool == answers[-1]:
            continue 

        answers.append(prev_tool)
        if steps:
            print(steps[i], prev_tool)
    
    search_tool_list = answers 
    search_tool_link = [[answers[idx], answers[idx+1]] for idx in range(len(answers)-1)]
    return {
        "task_nodes": search_tool_list, 
        "task_links": search_tool_link
    }
