import torch 
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer
from gnn import GNNEncoder
import contextlib


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'

IGNORE_INDEX = -100 


class GraphToken(torch.nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        
        self.max_txt_len = args.max_txt_length
        self.max_new_tokens = args.max_ans_length

        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.padding_side = 'left'
        self.device = torch.device(args.device)
        
        # TODO: reset the kwargs based on your device
        kwargs = {
            "max_memory": {0: '48GiB'}, 
            "device_map": "auto",
        }
        model = AutoModelForCausalLM.from_pretrained(args.llm_model_path, torch_dtype=torch.float16, **kwargs)

        # Freeze Llama's parameters
        for _, param in model.named_parameters():
            param.requires_grad = False 
        self.model = model 
        self.word_embedding = self.model.model.get_input_embeddings()
        print(f"Finish loading pre-trained {args.llm} model!")
        
        self.graph_tokenizer = GNNEncoder(
            input_dim=args.gnn_in_dim, 
            hidden_dim=args.gnn_hidden_dim, 
            output_dim=args.gnn_output_dim, 
            n_layers=args.n_layers, 
            gnn_type=args.gnn_type).to(self.device)
        
    def encode_task_graph(self, task_graph, batch_size):
        node_embeds = self.graph_tokenizer(task_graph.x, task_graph.edge_index)
        
        # apply a mean-pooling to obtain graph's embedding
        graph_embed = torch.mean(node_embeds, dim=0, keepdim=True)
        graph_embed = graph_embed.repeat(batch_size, 1)
        return graph_embed
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, samples, task_graph):
        # encode prompts, user requests, and labels 
        requests = self.tokenizer(samples["request"], add_special_tokens=False)
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        
        # encode special tokens
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(samples['id'])
        # encode graphs 
        graph_embeds = self.encode_task_graph(task_graph, batch_size)

        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids 
            input_ids = requests.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids

            input_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            # print(bos_embeds.shape, graph_embeds.shape, input_embeds.shape)
            input_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), input_embeds], dim=0)

            batch_inputs_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (input_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)

        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]

            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
        
        input_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids
            )
        return outputs.loss 
    
    def inference(self, samples, task_graph):
        requests = self.tokenizer(samples["request"], add_special_tokens=False)

        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(samples["id"])
        graph_embeds = self.encode_task_graph(task_graph, batch_size)

        batch_inputs_embeds = []
        batch_attention_mask = []
        for i in range(batch_size):
            input_ids = requests.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            input_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            input_embeds = torch.cat([bos_embeds, graph_embeds[i].unsqueeze(0), input_embeds], dim=0)
            batch_inputs_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
        
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
        
        input_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                # 300 - HuggingFace, Multimedia
                # 600 - DailyLife
                max_new_tokens=400,
                attention_mask=attention_mask,
                # use_cache=True
                use_cache=False
            )
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return samples["id"], pred, samples["request"]
   
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0 

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return trainable_params, all_param
