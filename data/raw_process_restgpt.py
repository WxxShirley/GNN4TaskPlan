"""Reformat raw RestBench dataset to align with our experimental setups"""
import json
from openai import AzureOpenAI


def reformat_one_api(raw_tool):
    for method in ["GET", "POST", "PUT", "DELETE"]:
       raw_tool = raw_tool.replace(method, "")
    tool = raw_tool.replace(" ", "")
    return tool


def load_valid_api(dataset_name):
    api_file = json.load(open(f"{dataset_name}/tool.json", 'r'))
    apis_info = api_file["nodes"]
    # key: "id", "desc", "parameters", "raw"
    
    id2tool = {api["id"]: api for api in apis_info}
    raw2toolname = {api["raw"]: api["id"] for api in apis_info}
    toolname2raw = {api["id"]: api["raw"] for api in apis_info}

    tool_string = "# TASK LIST #\n"
    for tool in apis_info:
        tool.pop("raw")
        tool.pop("raw_desc")
        tool_string += json.dumps(tool) + "\n"
    
    return id2tool, raw2toolname, toolname2raw, tool_string


def load_raw_request(dataset_name, write_raw_file=False, write_data_file=False):
    req_file = json.load(open(f"raw/RestGPT/{dataset_name}.json", 'r'))
    write_contents = []
    idx = 0
    for one_req in req_file:
        req_text, solutions = one_req["query"], one_req["solution"]
        raw_task_nodes = [raw2toolname[reformat_one_api(node)] for node in solutions]
        task_nodes = [{"task": node} for node in raw_task_nodes]
        task_links = [{"source": node, "target": raw_task_nodes[i+1]} for i, node in enumerate(raw_task_nodes[:-1])] 
        write_contents.append({
            "id": idx,
            "user_request": req_text,
            "task_nodes": task_nodes,
            "task_links": task_links
        })
        idx += 1
    
    if write_raw_file:
        with open(f"{dataset_name}/data_raw.json", 'w') as wf:
            for write_dict in write_contents:
                wf.write(json.dumps(write_dict) + "\n")
        
        with open(f"{dataset_name}/user_requests.json", 'w') as wf:
            for write_dict in write_contents:
                tmp_dict = {"id": write_dict["id"], "user_request": write_dict["user_request"]}
                wf.write(json.dumps(tmp_dict) + "\n")
    

    if write_data_file:
        with open(f"{dataset_name}/data.json", 'a+') as file:
            for write_dict in write_contents:
                res = prompt_llm_generate_steps(write_dict, full_tool_str)
                write_dict["task_steps"] = res['task_steps']
                file.write(json.dumps(write_dict) + '\n')
                file.flush()
                print(write_dict['user_request'], res)


def prompt_llm_generate_steps(one_sample, tool_string):
    prompt = "# GOAL #\nGiven the # USER REQUEST # and corresponding ground-truth # SOLUTION #, I need you to generate the chain of thoughts that can decompose user's request into managable sub-tasks, where each of them can be solved by a specific TASK."
    prompt += "\n# REQUIREMENTS #\n1. The decomposed task steps can work in sequence to perfectly solve the user's request, and each one corresponds to call one TASK in the # SOLUTION #."
    prompt += "\n2. Each of the step's calling TASK (i.e., API) should in the # TOOL LIST #"
    prompt += "\n3. Your output should be in a strict JSON format which can be DIRECTLY parsed by `json.loads()`, like `{'task_steps': ['Step 1 do xxx', 'Step 2 do xxx', ,,,]}`"
    prompt += "\n4. The number of generated steps should STRICTLY align with the length of solution."
    
    demo_string = "# USER REQUEST #\n What is the logo of the Walt Disney?. # SOLUTION # ['SearchCompany', 'GetCompanyLogo']. # RESULT # {'task_steps': ['Step 1 Call SearchCompany to retrieve the Compan's information named Walt Disney', 'Step 2 Call GetCompanyLogo to obtain this company's logo']}"
    prompt = tool_string + prompt + "\nExample:\n" + demo_string
    prompt += "\n# USER REQUEST #\n {{user_request}}. \n# SOLUTION #\n {{solution}}.\n\nNow please generate the task steps in a strict JSON format:# RESULT #\n"
    prompt = prompt.replace("{{user_request}}", one_sample["user_request"])
    solution = [node["task"] for node in one_sample['task_nodes']]
    prompt = prompt.replace("{{solution}}", str(solution))

    response = client.chat.completions.create(
        model = "gpt-4",
        messages = [
            {
                "role": "user",
                "content": prompt
            }
        ]
    )
    
    origin_content = response.choices[0].message.content
    origin_content = origin_content.replace("\n", "")
    origin_content = origin_content.replace("\_", "_")
    content = origin_content.replace("\\", "")
    content = content[content.find("{"):content.rfind("}")+1]
    
    content = json.loads(content)
    return content


def format_tool_graph_files(dataset_name):
    # add links 
    tool_ids = list(name2tool.keys())
    links = []
    for tool_a in tool_ids:
        for tool_b in tool_ids:
            if tool_a == tool_b:
                continue 

            words1 = [ word for word in toolname2raw[tool_a].split('/') if len(word)]
            words2 = [ word for word in toolname2raw[tool_b].split('/') if len(word)]
            if words1[-1] == words2[0]:
                links.append({"source": tool_a, "target": tool_b})
            elif len(list(set(words1) & set(words2))):
                union_list = set(words1) & set(words2)
                if dataset_name == 'tmdb' and len(list(set(['movie', 'person', 'tv', 'collection', 'network', 'season', 'episode', 'search']) & union_list)):
                    links.append({"source": tool_a, "target": tool_b})
                    links.append({"source": tool_b, "target": tool_a})
    print(f"Add Links {len(links)}")
    
    tool_desc_wr, graph_desc_wr = f"{dataset_name}/tool_desc.json", f"{dataset_name}/graph_desc.json"
    nodes = {"nodes": [tool for tool in name2tool.values()]}
    with open(tool_desc_wr, 'w') as wf:
        json.dump(nodes, wf, indent=4, sort_keys=True)

    nodes["links"] = links 
    with open(graph_desc_wr, 'w') as wf:
        json.dump(nodes, wf, indent=4, sort_keys=True)
  

def split_ids(dataset_name):
    write_dict = {"single": [], "chain": [], "dag": []}
    raw_file = json.load(open(f"raw/RestGPT/{dataset_name}.json", 'r'))
    for idx, content in enumerate(raw_file):
        if len(content["solution"]) == 1:
            write_dict["single"].append(idx)
        elif len(content["solution"]) >= 2:
            write_dict["chain"].append(idx)
    
    with open(f"{dataset_name}/split_ids.json", 'w') as wf:
        json.dump({"test_ids": write_dict}, wf, indent=4, sort_keys=True)


if __name__ == "__main__":
    dataset = "tmdb"
    
    # Step 1 - Reformatting original APIs by assigning unique task names and descriptions 
    name2tool, raw2toolname, toolname2raw, full_tool_str = load_valid_api(dataset)
    
    # TODO: replace with your own OpenAI endpoint and API-Key
    client = AzureOpenAI(
        azure_endpoint = "END_POINT", 
        api_key = "YOUR_API_KEY",  
        api_version = "2024-02-15-preview"  
    )
    
    load_raw_request(dataset, write_raw_file=False, write_data_file=False)
    # Step 2 - Constructing a Task Graph
    # Step 3 - Reformatting Raw Data Examples
    format_tool_graph_files(dataset)
    split_ids(dataset)
    