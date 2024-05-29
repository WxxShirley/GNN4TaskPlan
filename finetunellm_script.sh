# Fine-tune LLMs

## HuggingFace 
TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1 python -u main.py --max_txt_length=800 --max_ans_length=400 >codellama7b_huggingface.log
## Multimedia
TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1 python -u main.py --max_txt_length=1000 --max_ans_length=500 --dataset=multimedia >codellama7b_multimedia.log
## Daily Life
TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1 python -u main.py --max_txt_length=1000 --max_ans_length=500 --llm=lmsys/vicuna-13b-v1.5 --dataset=dailylife >vicuna13b_dailylife.log 



# Inference with Fine-tuned LLMs
## You have to specify model's checkpoints and the dataset
python inference.py  --lora_ckpt=CodeLlama-7b_seed0/checkpoint-5000 --dataset=huggingface 
