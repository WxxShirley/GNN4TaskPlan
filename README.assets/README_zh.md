# Can Graph Learning Improve Planning in LLM-based Agents?

[![Paper](https://img.shields.io/badge/Paper-arXiv%20Link-blue)](https://arxiv.org/abs/2405.19119)

这是我们刚被NeurIPS'2024录用的论文的官方代码实现。

###  


![task](task.jpg)

不同于已有的提示工程和大模型微调方法，我们研究了正交于这两种主流方法的新方向：**利用图学习技术提升大模型智能体的规划能力**。
在任务规划中，不同的子任务之间存在着依赖关系，这些关系可以用任务图（Task Graph）来表示。每个节点代表一个子任务，边则表示子任务之间的依赖。
在任务图构建完成，任务规划可以很自然地转化为: 在Task Graph中选择一条连通的路径或者子图，来满足用户的请求。
基于此，我们的研究主要做出了三点贡献：
* 任务规划的图建模: 我们第一次把任务规划问题转化为图上的决策问题，探索了图学习技术在这一领域的潜力。同时，这也为图学习带来了一个新的应用场景——任务规划。
* 理论研究: 我们深入分析了LLMs在处理图任务时的局限性，发现虽然Transformers具备一定的表达能力，但自注意力机制的偏差和自回归损失可能限制了它们在图决策任务中的表现。
* 算法与实验: 我们引入了GNN来增强LLMs的任务规划能力，它具有Training-free和Training-required两种变体。结果表明，我们的方法不仅性能优于现有的解决方案，而且计算效率更高。

> 引用格式参考 
```
@article{wu2024graph,
  title={Can Graph Learning Improve Task Planning?},
  author={Xixi Wu and Yifei Shen and Caihua Shan and Kaitao Song and Siwei Wang and Bohang Zhang and Jiarui Feng and Hong Cheng and Wei Chen and Yun Xiong and Dongsheng Li},
  journal={arXiv preprint arXiv:2405.19119},
  year={2024}
}
```


## 环境配置

运行本仓库所需要的环境在`requirements.txt`中已有声明，可以通过运行下面的指令安装所需的环境:

```shell
pip install -r requirements.txt
```

### 部署开源的大模型

对于Training-free的方法，即LLM直接推理(Direct Inference)和LLM进行图搜索(GraphSearch)，我们的代码实现是利用`FastChat`(https://github.com/lm-sys/FastChat) 把LLM部署成本地可调用的API服务实现的，具体来说部署在了本地的`8008`端口。

* 安装FastChat  
  ```shell
  pip3 install "fschat[model_worker,webui]"
  pip3 install vllm
  ```

* 安装完成后，通过依次执行下面的三个指令来部署某个指定的大模型
  ```shell
  # 首先需要启动Controller
  python3 -m fastchat.serve.controller --host 127.0.0.1 

  # 然后是指明需要部署的LLM名称，当第一次执行这个指令的时候，会自动下载这个LLM的Checkpoints
  python3 -m fastchat.serve.vllm_worker --model-path codellama/CodeLlama-13b-Instruct-hf --host 127.0.0.1
  # 部署其他LLM的示例，只要更换`model-path`就可以了
  # python3 -m fastchat.serve.vllm_worker --model-path lmsys/vicuna-13b-v1.5 --host 127.0.0.1
  # python3 -m fastchat.serve.vllm_worker --model-path codellama/CodeLlama-7b-Instruct-hf --host 127.0.0.1
  
  # 最后是开启本地的API服务，需要指明对应的端口
  python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 8008
  ```
  
  故障排查
  <details show=hide>
  
  1. 如果启动服务后仍无法收到请求LLM的响应，可以在Controller的指令中切换host，如下所示
     ```shell
     python3 -m fastchat.serve.controller --host 0.0.0.0 
     ```
  2. 如果启动`vllm_worker`遇到如下的报错:
       ```log
       2024-07-22 17:30:50 | ERROR | stderr |   File "/root/miniconda3/lib/python3.8/site-packages/vllm/engine/async_llm_engine.py", line 228, in _run_workers_async
       2024-07-22 17:30:50 | ERROR | stderr |     assert output == other_output
       2024-07-22 17:30:50 | ERROR | stderr | AssertionError
       ```
     可以 (i) 进入到`site-packages/vllm/engine/async_llm_engine.py`文件中，将上述断言的部分注释掉(227-228行)，或者
     (ii) 用`model_worker`代替`vllm_worker`，即
     ```shell 
     # model-path我填写了将LLM存储在本地的一个路径，请根据实际填写
     python3 -m fastchat.serve.model_worker --model-path /root/autodl-tmp/models/AI-ModelScope/Mistral-7B-Instruct-v0.2 --host 127.0.0.1 

     python3 -m fastchat.serve.openai_api_server --host 127.0.0.1  --port 8008 
     ```
  </details>


## 仓库内容预览 

```
.
├── GraphToken                     --> 对比模型GraphToken的复现
├── README.assets    
├── README.md       
├── data                           --> 提供了五个实验的数据集和处理的代码
│   ├── dailylife
│   ├── huggingface
│   ├── multimedia
│   ├── raw                        --> RestBench和UltraTool的原始数据
│   │   └── RestBench                    `https://github.com/Yifan-Song793/RestGPT`
│   │   └── ultratool                    `https://github.com/JoeYing1019/UltraTool`
│   ├── raw_process_restgpt.py     --> 处理RestBench的代码
│   ├── raw_process_ultratool.py   --> 处理UltraTool的代码
│   ├── split_data.py              --> 测试集划分
│   ├── tmdb
│   └── ultratool
├── evaluate.py                    --> 验证的代码
├── finetunellm                    --> LLM微调和微调后的LLM推理的代码
├── finetunellm_script.sh          --> LLM微调指令
├── prediction                     --> LLM直接推理的内容
├── requirements.txt     
├── trainfree                      --> Training-free方法的代码(Direct, GraphSearch, and SGC)
├── trainfree_script.sh            --> Training-free方法的运行指令
├── traingnn                       --> Training-required方法的代码
├── traingnn_reproduce.sh          --> Training-required方法的运行和复现的指令
└── utils                 
```

这个仓库里，我们同时提供了Baseline的实现、所有实验数据集的数据和处理代码、Training-free和Training-required代码、以及LLM在规划任务上微调的代码，后面我们会逐一介绍每一部分。


## 数据集

所有数据集的内容和处理过程都在`data`文件夹下。
我们一共使用了五个数据集，其中HuggingFace，Multimedia，DailyLife来自于[TaskBench](https://github.com/microsoft/JARVIS/blob/main/taskbench/README.md)，TMDB来自于[RestBench](https://github.com/Yifan-Song793/RestGPT)，另外补充了[UltraTool](https://github.com/JoeYing1019/UltraTool)。

每个数据集都由以下文件构成：
* `data.json` 数据集的具体信息，每一条样本由用户请求、对应分解的步骤、和任务调用链构成
* `graph_desc.json` 该数据集的任务图，包括节点和边
* `tool_desc.json` 该数据集的所有任务，即上一个文件中的节点信息
* `user_requests.json` 数据集中的用户请求
* `split_ids.json` 划分的训练集ID


TaskBench我们使用了该Benchmark处理好的数据集。

RestBench的处理过程在`data/raw_process_restgpt.py`中，原始的数据在`data/raw/RestBench`中。由于RestBench中每个任务(一个API)只有API对应的请求地址，我们首先赋予了每个API一个独特的名称和具体的功能描述。这之后，基于任务之间的参数依赖和类型构建了任务图。最后，对原始的数据进行重构成上述的标准格式，并提示GPT-4来生成每个请求对应的分解步骤，恰好和调用的任务对齐。

为了演示在大的任务图上的拓展性，我们引入了新的数据集UltraTool，处理过程在`data/raw_process_ultratool.py`中，原始的数据在`data/raw/ultratool`中。由于原始的UltraTool数据集规模较大、且很多Task只出现在1-2个样本中，我们首先对数据集进行了过滤：只考虑调用的任务数>=2的样本、且只考虑出现的频率>=5的任务。我们基于过滤后的样本和任务构建了任务图，并用相似的方法提示GPT生成步骤描述，完成了数据集的重构。


## Training-free的代码
```
├── trainfree
│   ├── direct.py               --> LLM直接推理的代码(Direct Inference)
│   ├── direct_diffprompt.py    --> LLM直接推理使用了不同提示模版的代码，包括(i)更多的上下文学习例子和(ii)Plan like a Graph
│   ├── graphsearch.py          --> LLM进行图搜索的代码(GraphSearch)
│   └── sgc.py                  --> 我们的SGC
```

Training-free的方法有：

* LLM直接推理 **Direct Inference**  我们默认使用了1-shot上下文学习的设置，使用的提示参考了[TaskBench](https://github.com/microsoft/JARVIS/blob/main/taskbench/inference.py)
  > 在`direct_diffprompt.py`中，我们提供了额外的提示模版，相比于默认的1-shot能实现更好的结果。其中Plan like a Graph(PlaG)是受到该论文["Graph-enhanced Large Language Models in Asynchronous Plan Reasoning"](https://arxiv.org/abs/2402.02805)的启发。

* LLM进行图搜索 **GraphSearch**  LLM在任务图上进行启发式搜索，在多条遍历到的路径中选择一条最合适的作为规划结果。根据搜索策略的不同，有BeamSearch, GreedySearch, 和AdaptiveSearch这三个变体。我们的实现参考了[ControlLLM](https://github.com/OpenGVLab/ControlLLM/blob/main/cllm/services/tog/tool.py).
* **SGC**  我们提出的无需训练的GNN方法，使用了一个没有参数的GNN模型，即SGC，进行任务的检索来完成规划



所有运行的指令和参数设置在脚本`trainfree_script.sh`中。

⚠️ **注意**：在运行GraphSearch/SGC之前，需要先有该LLM下的直接推理文件获得的步骤信息。



## Training-required GNN代码
```
├── traingnn
│   ├── gnn.py              --> 不同GNN编码器的实现，包括SGC, GCN, GAT, SAGE, GIN, and TransformerConv
│   ├── main.py             --> 启动文件
│   ├── model.py            --> LM+GNN模型
│   └── sampler.py          --> 采样器，形成 `<步骤，正样本任务，负样本任务>`三元组来训练模型
├── traingnn_reproduce.sh   --> 复现所有实验结果的指令
```

运行`main.py`文件可以完成(LM+)GNN的训练和测试，特别地，我们解释以下的参数使用:
* `lm_name`: LM编码器的名称，默认为`intfloat/e5-large`
* `gnn_name`: GNN编码器的名称
* `gnn_hidden_dim`: GNN的隐藏层维度
* `num_negatives`: 每个正样本所采集的负样本数量，默认为`2`
* `text_negative`: 负样本是否和正样本编码表征接近
* `lm_frozen`: LM的参数是否冻结，如果是`1`，则LM冻结、只训练GNN；否则`0`，意味着LM与GNN共同训练

运行的指令示例，更多地在`traingnn_reproduce.sh`中。
```shell
# HuggingFace数据集，只训练GNN
python main.py --lm_frozen=1 --epoch=10 --text_negative=1 --gnn_name=SAGE --lr=0.001

# HuggingFace数据集，LM+GNN共同训练
python main.py --lm_frozen=0 --epoch=20 --text_negative=1 --gnn_name=SAGE
```



## LLMs微调
```
├── finetunellm
│   ├── inference.py         --> Direct inference of fine-tuned LLMs
│   ├── main.py              --> Fine-tuning LLM
│   └── user_prompt.py       --> Instruction Template
```

我们在论文中探究了方法对于微调过的LLM(即获得了指定数据集上的规划能力)的有效性，具体来说，我们对[CodeLlama-7B](https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf)和[Vicuna-13B](https://huggingface.co/lmsys/vicuna-13b-v1.5), 使用LoRA技术进行了微调。

`main.py`文件中提供了LLM微调的代码，`inference.py`是模型微调完成后进行推理的代码(运行时需要指定LLM的名称和checkpoint的路径)。

所有的运行指令在`finetunellm_script.sh`中，在实验中我们使用了2张A100进行微调，并在该设备条件下进行的训练参数配置。读者可以根据设备条件修改训练的参数。



## 结果验证

`evaluate.py`提供了对规划结果的统一验证过程，评估的指标包括: `Node-F1`, `Link-F1`, `Node-Hallucination`, `Link-Hallucination`.

为了方便读者复现，我们已经在`prediction`文件夹下提供了两个LLMs在HuggingFace数据集上直接推理的结果，可以通过运行如下指令，查看不同LLM直接推理结果的得分，其中`llm`为要评估的LLM的名称 `method`为对应的方法名称：

```shell 
python evaluate.py --llm=CodeLlama-13b --dataset=huggingface --method=direct
```

对应的结果如下，`NF`指Node-F1得分，`LF`指Link-F1得分，`NH`指节点的幻觉率（1/2分别代表微观和宏观），`LH`指边的幻觉率
```
+-------------+---------------+-------+--------+--------+--------+--------+--------+--------+
|   Dataset   |      LLM      |  Mode |   NF   |   LF   |  NH-1  |  NH-2  |  LH-1  |  LH-2  |
+-------------+---------------+-------+--------+--------+--------+--------+--------+--------+
| huggingface | CodeLlama-13b | chain | 0.5755 | 0.2888 | 0.1656 | 0.4306 | 0.4228 | 0.6338 |
+-------------+---------------+-------+--------+--------+--------+--------+--------+--------+
```


## Baseline复现

我们提供了对Training-required方法，GraphToken，在规划任务上的复现代码。

运行的指令在`GraphToken/run.sh`中，读者可以根据自身使用的实验设备调整训练参数。

> Perozzi, Bryan, et al. "Let Your Graph Do the Talking: Encoding Structured Data for LLMs." arXiv preprint, 2024.


---

📮 如果在运行时还遇到其他问题，欢迎在Issue区展开讨论或邮件联系作者: xxwu@se.cuhk.edu.hk 

