import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModel
from gnn import SGC, GNNEncoder
import sys 
sys.path.append("../")
from utils import TextDataset, init_random_state
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader
import time
import copy
from utils import init_random_state
from sampler import TrainSampler


class LMGNNModel(nn.Module):
    def __init__(self, args, emb_dim=1024):
        super().__init__()
        self.lm_name = args.lm_name 
        # Co-train structure with two core components - LM / GNN
        self.tokenizer = AutoTokenizer.from_pretrained(self.lm_name)
        self.lm_model = AutoModel.from_pretrained(self.lm_name)
        
        self.gnn_name = args.gnn_name
        if args.gnn_name == "SGC":
            gnn = SGC()
        else:
            gnn = GNNEncoder(emb_dim, hidden_dim=args.gnn_hidden_dim,
                             output_dim=emb_dim,
                             gnn_type=args.gnn_name, n_layers=args.gnn_layer)
        self.gnn_model = gnn 

        if args.lm_frozen:
            for name, param in self.lm_model.named_parameters():
                param.requires_grad = False

    def tool_forward(self, tool_text, tool_adj=None):
        """Return tool embeddings, first tool's LM embedding, then GNN update"""
        init_tool_emb = self.lm_forward(tool_text, max_length=128, batch_size=64)
        if tool_adj is None:
            return init_tool_emb
        
        tool_emb = self.gnn_model(init_tool_emb, tool_adj)
        return tool_emb

    def lm_forward(self, plain_text, max_length=128, batch_size=256, device="cuda:0"):
        """Return LM encoded representations of input plain text"""
        x = self.tokenizer(plain_text, padding=True, truncation=True, max_length=max_length)
        format_data = TextDataset(x)

        text_emb = None 
        dataloader = DataLoader(format_data, shuffle=False, batch_size=batch_size)

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = self.lm_model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                output_hidden_states=True
            )

            emb = output['hidden_states'][-1]
            cls_token_emb = emb.permute(1, 0, 2)[0]
             
            if text_emb is None:
                text_emb = cls_token_emb
            else:
                text_emb = torch.vstack((text_emb, cls_token_emb))
        
        return text_emb

    def inference(self, step_text, tool_text, tool_adj, static=False):
        """Movel Evaluation"""
        tool_x = self.lm_forward(tool_text, max_length=128, batch_size=64)
        if not static:
            tool_x = self.gnn_model(tool_x, tool_adj)
        
        step_x = self.lm_forward(step_text, max_length=64, batch_size=1024)
        
        return tool_x, step_x

    def forward(self, step_text, tool_text, tool_adj, pos_tools, neg_tools):
        # step 1 tool graph forward propagation
        updated_tool_x = self.tool_forward(tool_text, tool_adj)

        # step 2 obtain steps' representations
        step_x = self.lm_forward(step_text, max_length=64, batch_size=1024)
        
        # step 3 retrieve both pos and neg tools' embeddings to compute scores
        pos_tool_emb = updated_tool_x[pos_tools, :]
        neg_tool_emb = updated_tool_x[neg_tools, :]

        pos_score = torch.sum(step_x * pos_tool_emb, dim=-1)
        neg_score = torch.sum(step_x * neg_tool_emb, dim=-1)

        bpr_loss = torch.mean(torch.nn.functional.softplus(neg_score - pos_score))

        return bpr_loss 


class ModelTrainer():
    def __init__(self, args, device):
        self.seed = args.seed 
        self.device = device 
        self.epoch = args.epoch 
        self.patience = args.patience 

        init_random_state(args.seed)

        model = LMGNNModel(args)
        self.model = model.to(device)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0.0)
        self.eval_steps, self.eval_loader = None, None
        self.text_negative = args.text_negative
        print(f"[Model] LM_GNN Number of parameters: {trainable_params}")
        
    def train_one_epoch(self, tool_text, tool_adj, train_loader, train_step_texts):
        total_loss = 0.0 

        self.model.train()
        for batch_data in train_loader:
            self.optimizer.zero_grad()

            batch_data = batch_data[0].to(self.device)

            step_idx = batch_data[:, 0].detach().cpu().numpy().tolist()
            step_texts = [train_step_texts[i] for i in step_idx]

            one_batch_loss = self.model(step_texts, tool_text, tool_adj, batch_data[:, 1], batch_data[:, 2])

            one_batch_loss.backward()
            self.optimizer.step()
        
            total_loss += one_batch_loss
        
        return total_loss / len(train_loader)

    def train(self, tool_text, tool_adj, sample_obj: TrainSampler):
        best_evaluate_acc, stop_cnt, best_model = 0, 0, None 
        
        self.eval_loader, self.eval_steps = sample_obj.sample(maximum=1000, tmp_print=False)

        init_acc = self.evaluate(tool_text, tool_adj, static_mode=True)
        print(f"Static Evaluation {init_acc:.4f}")
        
        init_timer = time.time()
        for epoch in range(self.epoch):
            start_time = time.time()
            
            train_loader, train_step_texts = sample_obj.sample(shuffle=True)
            bpr_loss = self.train_one_epoch(tool_text, tool_adj, train_loader, train_step_texts)

            evaluate_acc = self.evaluate(tool_text, tool_adj, static_mode=False)
  
            if evaluate_acc >= best_evaluate_acc:
                best_evaluate_acc = evaluate_acc
                stop_cnt = 0
                best_model = copy.deepcopy(self.model)
            else:
                stop_cnt += 1 
            
            if stop_cnt >= self.patience:
                break 

            print(f"Epoch: {epoch:3d}, Time {time.time() - start_time:.4f}s, Loss: {bpr_loss.item():.4f}, Eval Acc: {evaluate_acc:.4f}")

        return best_model, time.time() - init_timer
        
    @torch.no_grad()
    def evaluate(self, tool_text, tool_adj, static_mode=False, tmp_prinpt=False):
        self.model.eval()

        total_sample, true_sample = 0, 0 

        for batch_data in self.eval_loader:
            batch_data = batch_data[0].to(self.device)

            step_idx = batch_data[:, 0].detach().cpu().numpy().tolist()
            step_texts = [self.eval_steps[i] for i in step_idx]

            tool_emb, step_emb = self.model.inference(step_texts, tool_text, tool_adj, static_mode)

            score_matrix = torch.mm(step_emb, tool_emb.t())
            pred_tool = torch.argmax(score_matrix, dim=1).detach().cpu().numpy().tolist()

            gt_tool = batch_data[:, 1].detach().cpu().numpy().tolist()

            total_sample += len(gt_tool)
            true_sample += sum([1 if a == b else 0 for a, b in zip(pred_tool, gt_tool)])
        
        return true_sample / total_sample
