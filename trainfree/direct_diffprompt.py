"""Upgrade prompt to make LLM generate better task planning results"""
import json
import click
import os
import aiohttp
import asyncio
import time
import sys
sys.path.append("../")
from utils import get_cur_time


async def inference_one_case(input, url, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type=False):
    user_request = input["user_request"]
    
    if resource_type:
        prompt = """\n# GOAL #:\nBased on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}"""
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. The task links (task_links) should reflect the temporal dependencies among task nodes, i.e. the order in which the APIs are invoked;"""
    else:
        prompt = """\n# GOAL #: Based on the above tools, I want you generate task steps and task nodes to solve the # USER REQUEST #. The format must in a strict JSON format, like: {"task_steps": [ 'concrete task steps format as Step x: Use xxx tool to do xxx' ], "task_nodes": [{"task": "tool name must be from # TASK LIST #", "arguments": [ a concise list of arguments for the tool. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]} """
        prompt += """\n\n# REQUIREMENTS #: \n1. the generated task steps and task nodes can resolve the given user request # USER REQUEST # perfectly. Task name must be selected from # TASK LIST #; \n2. the task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes; \n3. the dependencies among task steps should align with the argument dependencies of the task nodes; \n4. the tool arguments should be align with the input-type field of # TASK LIST #;"""
    
    prompt += demo_string
    prompt += """\n\n# USER REQUEST #: {{user_request}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""
    final_prompt = tool_string + prompt.replace("{{user_request}}", user_request)
    # print(llm)
    payload = json.dumps({
        "model": f"{llm}",
        "messages": [
            {
                "role": "user",
                "content": final_prompt
            }
        ],
        "temperature": temperature,
        "top_p": top_p, 
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 2000,
        "stream": False,
        "stop": None
    })
    st_time = time.time()

    try:
        returned_content = await get_response(url, payload, resource_type)
    except Exception as e:
        print(f"Failed #id {input['id']}: {type(e)} {e}")
        raise e 
    
    res = {"id": input["id"], "user_request": input["user_request"]}
    res["task_steps"] = returned_content["task_steps"]
    res["task_nodes"] = returned_content["task_nodes"]
    res["task_links"] = returned_content["task_links"]
    res["cost_time"] = round(time.time() - st_time, 4)
        
    write_file.write(json.dumps(res) + "\n")
    write_file.flush()


async def get_response(url, payload, resource_type=False):
    headers = {
        'Content-Type': 'application/json'
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, data=payload, timeout=300) as response:
            resp = await response.json()

    if response.status == 429:
        raise Exception(f"Rate Limit Error {resp}")
    if response.status != 200:
        raise Exception(f"{resp}")

    origin_content = resp["choices"][0]["message"]["content"]
    origin_content = origin_content.replace("\n", "")
    origin_content = origin_content.replace("\_", "_")
    content = origin_content.replace("\\", "")
    
    try:
        content = json.loads(content)
        if isinstance(content, list) and len(content):
            merge_content = {}
            for c in content:
                for k, v in c.items():
                    merge_content[k].extend(v) if k in merge_content else merge_content.update({k: v})
        return content 
    except json.JSONDecodeError as e:
        # encounter JSON decoder error
        # prompt LLM to reformat the response into strict JSON format
        if resource_type:
            prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"task_steps": [ "concrete steps, format as Step x: Call xxx tool with xxx: 'xxx' and xxx: 'xxx'" ], "task_nodes": [{"task": "task name must be from # TASK LIST #", "arguments": [ {"name": "parameter name", "value": "parameter value, either user-specified text or the specific name of the tool whose result is required by this node"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
        else:
            prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of task steps, task nodes, and task links;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. You must output the result in this schema: {"task_steps": [ "concrete steps format as 'Step x, Use xxx tool to do xxx'" ], "task_nodes": [{"task": "tool name must be from # TOOL LIST #", "arguments": [ a concise list of arguments for the tool. ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
            
        prompt = prompt.replace("{{illegal_result}}", origin_content)
        payload = json.loads(payload)

        payload["messages"][0]["content"] = prompt
        payload = json.dumps(payload)
            
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=payload, timeout=120) as response:
                resp = await response.json()

        if response.status == 429:
            raise Exception(f"Rate Limit Error {resp}")
        if response.status != 200:
            raise Exception(f"{resp}")
        
        content = resp["choices"][0]["message"]["content"]
        content = content.replace("\n", "")
        content = content.replace("\_", "_")
        start_pos = content.find("STRICT JSON FORMAT #:")
        if start_pos!=-1:
            content = content[start_pos+len("STRICT JSON FORMAT #:"):]

        content = content[content.find("{"):content.rfind("}")+1]
        try:
            # print(content)
            content = json.loads(content)
            return content
        except json.JSONDecodeError as e:
            raise Exception(f"JSON Decoding Error {e}")


@click.command()
@click.option("--dataset", default="huggingface", help="The directory of the data")
@click.option("--temperature", type=float, default=0.2)
@click.option("--top_p", type=float, default=0.1)
@click.option("--api_addr", type=str, default="localhost")
@click.option("--api_port", type=int, default=8008)
@click.option("--llm", type=str, default="CodeLlama-13b-Instruct-hf") 
@click.option("--use_demos", type=int, default=1)
@click.option("--multiworker", type=int, default=4)
@click.option("--graph", type=int, default=1)
def main(dataset, temperature, top_p, api_addr, api_port, llm, use_demos, multiworker, graph):
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)

    url = f"http://{api_addr}:{api_port}/v1/chat/completions"

    llm_short_names = {
        "CodeLlama-13b-Instruct-hf": "CodeLlama-13b",
        "vicuna-13b-v1.5": "vicuna-13b",
        "Mistral-7B-Instruct-v0.2": "mistral-7b",
        "Llama-2-13b-chat-hf": "llama2-13b",
        "CodeLlama-7b-Instruct-hf": "CodeLlama-7b",
        "Baichuan2-13B-Chat": "Baichuan-13b"
    }
    llm_short = llm_short_names[llm]
    prediction_dir = f"../prediction/{dataset}/{llm_short}"

    infer_step_file = f"{prediction_dir}/direct_demo{use_demos}{'' if graph == 0 else '_graph'}.json"

    if not os.path.exists(prediction_dir):
        os.makedirs(prediction_dir, exist_ok=True)
    
    alignment = json.load(open(f"../data/{dataset}/split_ids.json", 'r'))["test_ids"]
    alignment_ids = alignment["chain"]
    
    has_inferenced = []
    if os.path.exists(infer_step_file):
        rf = open(infer_step_file, 'r')
        for line in rf:
            data = json.loads(line)
            has_inferenced.append(data["id"])
        rf.close()
    
    user_request_file = open(f"../data/{dataset}/user_requests.json", 'r')
    inputs = []
    for line in user_request_file:
        input = json.loads(line)
        if input["id"] not in has_inferenced and input["id"] in alignment_ids:
            inputs.append(input)
    user_request_file.close()

    write_file = open(infer_step_file, "a") 
    print(infer_step_file)
    
    # Prepare Tool String to prompt LLM
    tool_list = json.load(open(f"../data/{dataset}/tool_desc.json", "r"))["nodes"]
    tool_string = "# TASK LIST #:\n"
    for k, tool in enumerate(tool_list):
        tool_string += json.dumps(tool) + "\n"
   
    # Plan like a Graph (PlaG)
    # Graph-enhanced Large Language Models in Asynchronous Plan Reasoning (https://arxiv.org/abs/2402.02805)
    if graph:
        if dataset == 'dailylife':
            tool_string += "\nAbove TASK forms a COMPLETE GRAPH where each pair of tasks form a link\n"
        else:
            link_list = json.load(open(f"../data/{dataset}/graph_desc.json", "r"))["links"]
            tool_string += "\nLinks within the TASK GRAPH that describe their dependencies:\n"
            for link in link_list:
                tool_string += json.dumps({"source": link["source"], "target": link["target"]}) + "\n"

        tool_string += "\n# HINT # Please carefully understand the above whole Task Graph's relations, which indicates their input-output dependencies. Since you have to generate a task invocation path for the given request, you can first think about this graph and then produce your output task_steps, task_nodes, and task_links.\n"
    
    # Prepare Demo(s) String to prompt LLM
    demo_string = ""
    if use_demos:
        demos_id_list = {
            "huggingface": ["10523150", "11922608", "22067492"],
            "multimedia": ["20566230", "16273916", "19003517"],
            "dailylife": ["27267145", "91005535", "38563456"],
        }

        demos_id = demos_id_list[dataset][:use_demos]

        demos_rf = open(f"../data/{dataset}/data.json", "r")
        demos = []
        for line in demos_rf:
            data = json.loads(line)
            if data["id"] in demos_id:
                demo = {
                    "user_request": data["user_request"],
                    "result": {
                        "task_steps": data["task_steps"],
                        "task_nodes": data["task_nodes"],
                        "task_links": data["task_links"]
                    }
                }
                demos.append(demo)
        demos_rf.close()
    
        if len(demos) > 0:
            demo_string += "\nHere are provided examples for your reference.\n"
            for demo in demos:
                demo_string += f"""\n# EXAMPLE #:\n# USER REQUEST #: {demo["user_request"]}\n# RESULT #: {json.dumps(demo["result"])}"""

    # Set up multi-worker
    sem = asyncio.Semaphore(multiworker)
    
    resp_type = dataset == "dailylife"
    async def inference_wrapper(input, url, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type):
        async with sem:
            await inference_one_case(input, url, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type)
    
    if len(inputs) == 0:
        print("All Completed!")
        return 
    else:
        print(f"Detected {len(has_inferenced)} has been inferenced, ")
        print(f"Start inferencing {len(inputs)} tasks ... ")
    
    loop = asyncio.get_event_loop()

    tasks = []
    for input in inputs:
        tasks.append(inference_wrapper(input, url, temperature, top_p, tool_string, write_file, llm, demo_string, resource_type=resp_type))
       
    done, failed = [], []
    results = loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    for result in results:
        if isinstance(result, Exception):
            print(result)
            failed.append(result)
        else:
            done.append(result)

    print(f"Completed {len(done)} Failed {len(failed)}")
    loop.close()

    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
    
        
if __name__ == "__main__":
    main()   
