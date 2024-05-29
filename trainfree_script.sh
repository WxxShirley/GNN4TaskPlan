cd trainfree

# LLM's Direct Inference

## Enviromental Setup
pip3 install "fschat[model_worker,webui]"

## Deploy LLM as local API services
python3 -m fastchat.serve.controller
python3 -m fastchat.serve.vllm_worker --model-path codellama/CodeLlama-13b-Instruct-hf
python3 -m fastchat.serve.openai_api_server --host localhost --port 8008

## Direct inference `llm` - llm_type `use_demos` - number of in-context learning examples (1-shot as default setting)
python3 direct.py --llm=CodeLlama-13b-Instruct-hf --dataset=huggingface --use_demos=1



# GraphSearch
## First deploy LLM as local API services
## You have to run direct-inference to prepare LLM's decomposed task steps, then run GraphSearch based on the steps
## Run GraphSearch  `mode` - either run for all samples or just a single case   `strategy` - search strategy
python3 -u graphsearch.py --dataset=multimedia --llm=CodeLlama-13b-Instruct-hf  --mode=full --strategy=greedy
python3 -u graphsearch.py --dataset=multimedia --llm=CodeLlama-13b-Instruct-hf  --mode=full --strategy=beam --width=2
python3 -u graphsearch.py --dataset=multimedia --llm=CodeLlama-13b-Instruct-hf  --mode=full --strategy=adaptive --threshold=4



# SGC
## You have to run direct inference to prepare LLM's decomposed task steps, then run SGC based on the steps
## `llm` refers to concrete LLMs with short names
## `lm_name` refers to LM backbones, choices ['intfloat/e5-large', 'sentence-transformers/all-roberta-large-v1', 'intfloat/e5-large-v2']
## `use_graph` refers to whether performs SGC's forward propagation
python3 sgc.py --dataset=multimedia --llm=CodeLlama-13b  --lm_name=intfloat/e5-large --use_graph=1 
python3 sgc.py --dataset=dailylife --llm=CodeLlama-13b  --lm_name=intfloat/e5-large --use_graph=1 
python3 sgc.py --dataset=huggingface --llm=CodeLlama-13b  --lm_name=intfloat/e5-large --use_graph=1 
