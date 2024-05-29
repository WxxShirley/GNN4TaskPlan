import click
import os
import json
import copy
import requests
import time
import sys
sys.path.append("../")
from utils import reformat_steps, get_cur_time


def get_llm_response(llm, url, prompt, prompt_answer_type="default"):
    global counter
    payload = json.dumps({
        "model": llm,
        "messages": [
            {
                "role": "user",
                "content":  prompt
            }
        ],
        "temperature": 0.2, 
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 2000,
        "stream": False,
        "stop": None
    })

    response = requests.post(url, data=payload)  
    counter += 1  
    resp = response.json()
    
    if response.status_code == 429:
        raise Exception(f"Rate Limit Error {resp}")
    if response.status_code != 200:
        print('Error Content')

    origin_content = resp["choices"][0]["message"]["content"]
    origin_content = origin_content.replace("\n", "")
    origin_content = origin_content.replace("\_", "_")
    content = origin_content.replace("\\", "")

    try:
        content = json.loads(content)
        return content
    except json.JSONDecodeError as e:
        if prompt_answer_type == "solution":
            prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of each candidate tool;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"best_solution": [the best solution in a list form]}\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
        elif prompt_answer_type == "default":
            prompt = """Please format the result # RESULT # to a strict JSON format # STRICT JSON FORMAT #. \nRequirements:\n1. Do not change the meaning of each candidate tool;\n2. Don't tolerate any possible irregular formatting to ensure that the generated content can be converted by json.loads();\n3. Pay attention to the matching of brackets. Write in a compact format and avoid using too many space formatting controls;\n4. You must output the result in this schema: {"candidate tool name": "score" , ... }\n# RESULT #:{{illegal_result}}\n# STRICT JSON FORMAT #:"""
        
        prompt = prompt.replace("{{illegal_result}}", origin_content)
        payload = json.loads(payload)

        payload["messages"][0]["content"] = prompt 
        payload = json.dumps(payload)

        response = requests.post(url, data=payload)    
        resp = response.json()

        if response.status_code == 429:
            raise Exception(f"Rate Limit Error {resp}")
        if response.status_code != 200:
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
            # raise Exception(f"JSON Decoding Error {e}")
            return {'best_solution': []}


def prompt_llm_final_solutions(llm, url, user_request, parsed_steps, solution_list, tmp_print=False):
    if len(solution_list) == 1:
        return {"best_solution": solution_list[0]}
    if len(solution_list) == 0:
        return {"best_solution": []}

    prompt = """\n# GOAL #\nBased on the provided USER REQUEST and the initially inferred STEPS (to be performed in sequence to solve the user's request), select the best tool solution list from the SOLUTION LIST. The selected solution should be the one that can perfectly solve the user's request. The format must be in strict JSON format, like: {"best_solution": [list of invoked tools]}"""
    prompt += """\n\n# REQUIREMENTS #\n1. Your goal is to select the best solution that can perfectly solve user's request and follow initial inferred steps! Only return the best solution strictly from the provided SOLUTION LIST. Do not change their corresponding sequences and strictly align with the content;"""
    prompt += """\n2. Carefully analyze both the user's request and the previously inferred task steps. """
    prompt += f"""\n3. Make sure that each tool in the final solution list exists in the valid # TOOL LIST #: {tool_name_list}."""
     
    prompt += """\n\n# USER REQUEST #: {{user_request}} \n# STEPS #: {{steps}} \n# SOLUTION LIST #: {{solution_list}} \nnow please generate your result in a strict JSON format:\n# RESULT #:"""
    
    prompt = prompt.replace("{{user_request}}", user_request)
    prompt = prompt.replace("{{steps}}", str(parsed_steps))
    prompt = prompt.replace("{{solution_list}}", str(solution_list))
    
    response = get_llm_response(llm, url, prompt, prompt_answer_type="solution")
    
    if tmp_print:
        print(response)
    
    return response


def prompt_llm_candidate_tool_scores(llm, url, step_description: str, provided_demo: int, tool_candidates: list, tmp_print=False):
    prompt = """\n# GOAL #: Based on the provided CANDIDATE TOOL LIST and the user's request described in the STEP, generate a score dictionary to assess each tool's problem-solving abilities for the given request. The output format must in a strict JSON format, like: {"candidate tool name 1": score, "candidate tool name 2": score, "candidate tool name 3": score, ...} """
    prompt += """\n\n# REQUIREMENTS #: \n1. the keys of the generated score dictionary must align with the provided candidate tools, and you should output scores for ALL candidate tools;\n2. the 'score' field denotes a concrete score that assesses whether each tool can solve the given step's demand. The score should be in the range of [1, 2, 3, 4, 5], where a higher score indicates better task-solving and matching abilities;"""
    prompt += """'score=1' means the tool is totally not related to the task and does not provide any useful output for solving the task."""
    prompt += """'score=2' means tool is somewhat not related to the task and may not provide any useful output for solving the task."""
    prompt += """'score=3' means the tool is probably related to the task and provides some intermediate output that is partially helpful for solving the task, but it may not be the optimal one."""
    prompt += """'score>3' The tool is closely or directly related to the task and provides an output that is mostly helpful for solving the task or that matches the returns of the task with regard to the type."""
    prompt += """\n3. carefully consider the user's intention described in the STEP to assess the score for each tool."""
    prompt += """\n4. if the STEP contains a candidate tool, its score should be >= 3 (this tool may not be perfect, but other alternatives may be more suitable, so pay attention to the input and output requirements)."""

    if provided_demo:
        prompt += "\n"
        demos_dict = {
            "huggingface": [
                {
                    "step": "Answer a question related to the depth information from the document",
                    "candidates": ["Document Question Answering", "Question Answering", "Visual Question Answering", "Text Generation"],
                    "result": {
                        "Document Question Answering": 5,
                        "Question Answering": 2,
                        "Visual Question Answering": 3,
                        "Text Generation": 1,
                    }
                },
                {
                    "step": "Generate a new text based on the translated French text.",
                    "candidates": ["Text Generation", "Text-to-Image", "Text-to-Video", "Translation"],
                    "result": {
                        "Text Generation": 5,
                        "Text-to-Image": 1, 
                        "Text-to-Video": 1,
                        "Translation": 2,
                    }
                }
            ],
            "multimedia": [
                {
                    "step": "Use Image Stitcher to stitch together two images.",
                    "candidates": ["Image Search", "Image Stitcher", "Image Colorizer", "Image Style Transfer"],
                    "result": {
                        "Image Search": 1, 
                        "Image Stitcher": 5, 
                        "Image Colorizer": 1,
                        "Image Style Transfer": 1,
                    }
                },
                {
                    "step": "Extract a still image from the video 'example.mp4'",
                    "candidates": ["Video-to-Image", "Video Search", "Video-to-Text"],
                    "result": {
                        "Video-to-Image": 5,
                        "Video Search": 1,
                        "Video-to-Text": 2,
                    }
                },
                {
                    "step": "Use Text Downloader to download the text content from https://www.example.com/article.",
                    "candidates": ["Text Search", "Text Summarizer", "Text Downloader"],
                    "result": {
                        "Text Search": 2,
                        "Text Summarizer": 2,
                        "Text Downloader": 5,
                    }
                },
                {
                    "step": "Extract text from the image obtained in Step 1 using OCR",
                    "candidates": ["Image-to-Text", "Text Summarizer"],
                    "result": {
                        "Image-to-Text": 5,
                        "Text Summarizer": 1,
                    }
                }
            ],
            "dailylife": [
                {
                    "step": "Call search_by_engine API with query: 'How to use Microsoft Word' and engine: 'Google'",
                    "candidates": ["search_by_engine", "apply_for_job"],
                    "result": {
                        "search_by_engine": 5,
                        "apply_for_job": 1
                    }
                },
                {
                    "step": "Call buy_insurance API with insurance: 'Health Insurance' and company: 'ABC Insurances'",
                    "candidates": ["buy_insurance", "stock_operation", "online_shopping"],
                    "result": {
                        "buy_insurance": 5,
                        "stock_operation": 2,
                        "online_shopping": 1
                    }
                },
                {
                    "step": "Call organize_meeting_online API with topic: 'Data Privacy and Security'",
                    "candidates": ["organize_meeting_online", "attend_meeting_online", "make_video_call"],
                    "result": {
                        "organize_meeting_online": 5,
                        "attend_meeting_online": 3,
                        "make_video_call": 2
                    }
                },
            ]
        }
        
        demo_string = ""
        for demo in demos_dict[dataset_name]:
            demo_string += f"""\n# EXAMPLE #:\n# STEP #: {demo["step"]}\n# CANDIDATE TOOL LIST #: {demo["candidates"]}\n# RESULT #: {json.dumps(demo["result"])}"""
        prompt += demo_string
    
    candidate_tool_string = "# CANDIDATE TOOL LIST #:\n"
    candidate_tool_names =[]

    assert len(tool_candidates) <= len(tool_dict)

    for tool in tool_candidates:
        candidate_tool_names.append(tool["id"])
        candidate_tool_string += json.dumps(tool) + "\n"
    
    prompt += """\n\n # STEP #: {{step_description}}\nnow please generate your result in a strict JSON format:\n# RESULT #:"""
    
    if isinstance(step_description, dict):
        step_description = step_description["description"]
    final_prompt = candidate_tool_string + prompt.replace("{{step_description}}", step_description)
    # print(final_prompt)
    response = get_llm_response(llm, url, final_prompt, prompt_answer_type='default')
    
    if tmp_print:
        # print(candidate_tool_string)
        print(step_description, response, end=' ')
    
    return response


def generate_candidates(score_dict, tool_candidates):
    tool_score_list = []

    score_dict_copy = copy.deepcopy(score_dict)

    for i, tool in enumerate(tool_candidates):
        tool_name = tool["id"]
        score = score_dict.get(tool_name, None)
        if not isinstance(score, int):
            # if this candidate is not returned in the response dict, score is 1 as LLM thinks it is not important
            score = 1
            score_dict[tool_name] = score 
        else:
            # we remove the tools which also in candidates to filter those suggested by LLMs
            del score_dict[tool_name]
        
        tool_score_list.append((score, i, tool_name))
        tool_score_list = sorted(tool_score_list, reverse=True)

    # We also introduce the suggested TOOLs from LLM
    for tool, score in score_dict.items():
        if tool in tool_dict.keys():
            if isinstance(score, str) or isinstance(score, int):
                tool_score_list.append((int(score), len(tool_candidates), tool))
                tool_candidates.append(tool_dict[tool])
    tool_score_list = sorted(tool_score_list, reverse=True)

    if search_strategy == "greedy":
        return [ tool_candidates[tool_score_list[0][1]] ]
    elif search_strategy == "beam":
        new_tool_candidates = [
            tool_candidates[item[1]]
            for item in tool_score_list[: min(beam_width, len(tool_candidates))] # if item[0] >= score_threshold
        ]
        return new_tool_candidates
    elif search_strategy == "adaptive":
        new_tool_candidates = []
        for tool in tool_candidates:
            tool_name = tool["id"]
            score = score_dict_copy.get(tool_name, 1)
            if isinstance(score, int) and score >= score_threshold:
                new_tool_candidates.append(tool)
        
        if len(new_tool_candidates) == 0:
            new_tool_candidates = [ tool_candidates[tool_score_list[0][1]] ]
        return new_tool_candidates
    else:
        raise NotImplementedError()


def dfs(llm, url, current_idx, steps, solutions, current_tools, tmp_print=False):
    if current_idx == len(steps) or counter >= 30:
        solutions.append(copy.deepcopy(current_tools))
        return 
    
    if current_idx == 0:
        candidate_tools = copy.deepcopy(tool_nodes)
    else:
        if len(current_tools) == 0:
            return 
        last_tool = current_tools[-1]
        candidate_tools = copy.deepcopy(tool_graph[last_tool]["children"])
        
        if len(candidate_tools) == 0:
            return 
    
    candidate_score = prompt_llm_candidate_tool_scores(llm, url, steps[current_idx], 1, candidate_tools, tmp_print)
    candidate_list = generate_candidates(candidate_score, candidate_tools)
    
    if tmp_print:
        print([candidate["id"] for candidate in candidate_list], end='\n')
    
    for tool in candidate_list:
        tool_name = tool["id"]
        if tool_name in current_tools and search_strategy != 'greedy':
            continue 

        current_tools.append(tool_name)
        dfs(llm, url, current_idx+1, steps, solutions, current_tools, tmp_print)
        current_tools.pop(-1)
    
    return 
 
    
def graph_search_one_case(input, llm, url, tmp_print=False, maximum_solutions=20):
    user_request = input["user_request"]
    inferred_steps = reformat_steps(input)
    
    if tmp_print:
        print(f"# User Request {input['id']} #\n", user_request)
        print("# Inferred Steps #\n", inferred_steps)
    
    solutions, current_sol = [], []

    dfs(llm, url, 0, inferred_steps, solutions, current_sol, tmp_print)

    if len(solutions) == 0:
        return None
    
    if tmp_print:
        for one_sol in solutions:
            print(one_sol)

    solutions = solutions[:maximum_solutions]
    solution_resp = prompt_llm_final_solutions(llm, url, user_request, inferred_steps, solutions, tmp_print)

    return solution_resp


@click.command()
@click.option("--dataset", default="huggingface", help="The directory of the data")
@click.option("--api_addr", type=str, default="localhost")
@click.option("--api_port", type=int, default=8008)
@click.option("--llm", type=str, default="CodeLlama-13b-Instruct-hf") 
@click.option("--strategy", type=str, default="beam") # ['beam', 'greedy', 'adaptive']
@click.option("--width", type=int, default=2)
@click.option("--threshold", type=int, default=3)
@click.option("--mode", type=str, default="single")
@click.option("--request_id", type=str, default="")
def main(dataset, api_addr, api_port, llm, strategy, width, threshold, mode, request_id):
    global tool_graph, tool_nodes, tool_dict, tool_name_list
    global search_strategy, beam_width, score_threshold
    global dataset_name, counter

    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)

    # GraphSearch Parameters
    search_strategy = strategy
    beam_width = width
    score_threshold = threshold
    dataset_name = dataset

    url = f"http://{api_addr}:{api_port}/v1/chat/completions"
    
    llm_short_names = {
        "CodeLlama-13b-Instruct-hf": "CodeLlama-13b",
        "vicuna-13b-v1.5": "vicuna-13b",
        "Mistral-7B-Instruct-v0.2": "mistral-7b",
        "CodeLlama-7b-Instruct-hf": "CodeLlama-7b",
        "Baichuan2-13B-Chat": "Baichuan-13b"
    }
    
    llm_short = llm_short_names[llm]
    pred_filename = f"../prediction/{dataset}/{llm_short}/direct.json"

    if not os.path.exists(pred_filename):
        raise Exception("Prediction file does not exsists!")
    
    pred_data = []
    pred_rf = open(pred_filename, "r")
    for line in pred_rf:
        data = json.loads(line)
        pred_data.append(data)
    pred_rf.close()

    # prepare API graph data
    graph_data = json.load(open(f"../data/{dataset}/graph_desc.json"))
    tool_nodes, tool_links = graph_data["nodes"], graph_data["links"]
    
    # `tool_graph` saves the mapping between nodes and its neighbors
    # `tool_dict` saves the mapping between tools and its details
    tool_graph, tool_dict = {}, {}

    for tool in tool_nodes:
        if 'input-type' in tool.keys() and 'output-type' in tool.keys():
            tool.pop('input-type')
            tool.pop('output-type')

        tool_graph[tool["id"]] = {"id": tool["id"], "children": []}
        tool_dict[tool["id"]] = copy.deepcopy(tool)
        tool_name_list.append(tool["id"])
    
    for link in tool_links:
        neighbor_tool = tool_dict[link["target"]]
        tool_graph[link["source"]]["children"].append(neighbor_tool)
    
    if mode == "single":
        input_data = [data for data in pred_data if data["id"] == request_id]
        if len(input_data):
            graph_search_one_case(input_data[0], llm, url, tmp_print=True)

    elif mode == "full":
        write_filename = f"../prediction/{dataset}/{llm_short}/graphsearch_{strategy}{'_'+str(beam_width) if strategy=='beam' else ''}.json"
        write_file = open(write_filename, "a")

        has_inferenced = []

        if os.path.exists(write_filename):
            rf = open(write_filename, "r")
            for line in rf:
                data = json.loads(line)
                has_inferenced.append(data["id"])
            rf.close()
        
        alignment_ids = json.load(open(f"../data/{dataset}/split_ids.json", 'r'))["test_ids"]["chain"]
        for input_data in pred_data:
            counter = 0 # LLM query times
            if input_data["id"] in has_inferenced or input_data["id"] not in alignment_ids:
                continue
            
            start_time = time.time()
            search_response = graph_search_one_case(input_data, llm, url, tmp_print=True)
            
            task_nodes, task_links = input_data["task_nodes"], input_data["task_links"]
            status = "fail"
            if search_response and len(search_response.get("best_solution", [])) > 0:
                solution = search_response["best_solution"]
                if isinstance(solution[0], list):
                    # Sometimes, LLM's output is in the format like {"best_solution": [[ ... ]]}
                    solution = solution[0]

                task_nodes = [{"task": node} for node in solution]
                task_links = [{"source": node, "target": solution[i+1]} for i, node in enumerate(solution[:-1])]
                status = "succ"
                print(f"Search Success {input_data['id']}\n")
            else:
                print(f"Search Failed {input_data['id']}\n") 
            
            write_content = {
                "id": input_data["id"],
                "task_steps": input_data["task_steps"],
                "task_nodes": task_nodes,
                "task_links": task_links,
                "cost_time": round(time.time() - start_time, 4),
                "llm_query_times": counter,
                "status": status
            }
            write_file.write(json.dumps(write_content) + "\n")
            write_file.flush()
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
    

if __name__ == "__main__":
    tool_graph, tool_nodes, tool_dict, tool_name_list = None, None, {}, []
    beam_width, search_strategy, score_threshold = 2, "beam", 3
    dataset_name = "huggingface"

    counter = 0

    main()    
    