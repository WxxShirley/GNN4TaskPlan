# This is our implementation of baseline method: GraphToken

#  For 13B LLMs, training configuration is as follows:
python3 -u main.py  --llm=CodeLlama-13B --dataset=huggingface --num_epochs=4 --batch_size=6  --eval_batch_size=6   --max_ans_length=256  
python3 -u main.py  --llm=CodeLlama-13B --dataset=multimedia  --num_epochs=4 --batch_size=6  --eval_batch_size=6  --max_txt_length=512  --max_ans_length=256 
python3 -u main.py  --llm=CodeLlama-13B --dataset=dailylife --num_epochs=4 --batch_size=4 --eval_batch_size=4  --max_txt_length=580 --max_ans_length=400 


# For 7B LLMs, training configuration is as follows:
WANDB_DISABLED=True python3 -u main.py --llm=CodeLlama-7B  --dataset=huggingface  --gnn_type=GCN  --num_epochs=4 --batch_size=16 --eval_batch_size=32   --max_ans_length=256  
WANDB_DISABLED=True python3 -u main.py --llm=CodeLlama-7B  --dataset=multimedia  --gnn_type=GCN  --num_epochs=4 --batch_size=16 --eval_batch_size=32   --max_txt_length=512  --max_ans_length=256 
WANDB_DISABLED=True python3 -u main.py --llm=CodeLlama-7B  --dataset=dailylife  --gnn_type=GCN  --num_epochs=4 --batch_size=16 --eval_batch_size=32   --max_txt_length=600  --max_ans_length=350
