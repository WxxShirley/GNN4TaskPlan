"""Original UltraTool dataset:https://github.com/JoeYing1019/UltraTool"""
"""Our reformatting steps: i) filter samples and tasks ii) use back-instruct to prompt GPT-4-turbo to fill in steps given user request and ground-truth tasks"""
import json 
from collections import defaultdict
import prettytable as pt
import random
import numpy as np
import copy
from openai import AzureOpenAI
import os


MINIMUM_FREQ = 5
PATH_TO_ULTRATOOL = "myultratool" 
DEL_TASK = ['balance_query', 'hotel_search', 'restaurant_review_tool', 'view_schedule', 'postal_code_lookup', 'set_schedule_location', 'book_hotel', 'add_calendar_note', 'verify_and_save_file', 'restaurant_review_analyzer', 'hotel_room_availability', 'search_and_select_hotel', 'filter_hotels']
REPLACE_TASK = {
    'balance_query': 'account_balance_query',
    'hotel_search': 'search_hotels',
    'restaurant_review_tool': 'restaurant_review',
    'view_schedule': 'view_agenda',
    'postal_code_lookup': 'postal_code_search',
    'set_schedule_location': 'set_agenda_location',
    'book_hotel': 'hotel_booking',
    'add_calendar_note': 'calendar_note',
    'verify_and_save_file': 'verify_file_content',
    'restaurant_review_analyzer': 'restaurant_review',
    'hotel_room_availability': 'check_room_availability',
    'search_and_select_hotel': 'search_hotels',
    'filter_hotels': 'search_hotels'
}

# hand-crafted in-context learning examples to teach GPT-4 how to write suitable steps that align with tasks 
demos = [
    {
        "user_request": "As Zhang Wei, I need to book a deluxe suite and a standard room in the hotel with the ID H12345, both for the stay from September 2, 2023, to September 3, 2023, and to check the reservation details of both rooms to confirm the order status.",
        "solution": ["hotel_booking", "hotel_booking_query"],
        "task_steps": [
                "Step 1 Call hotel_booking to book the rooms", 
                "Step 2 Call hotel_booking_query to query the reservation details"
        ],        
    },
    {
        "user_request": "I want to know the current exchange rates of the US dollar and the British pound against the Chinese yuan, and I would like to calculate how much I can get by exchanging 10,000 Chinese yuan into US dollars and 2,000 British pounds into Chinese yuan respectively.",
        "solution": ["currency_exchange_rate", "foreign_currency_exchange"],
        'task_steps': [
            "Step 1 Call currency_exchange_rate to query US dollar and British pound to Chinese yuan exchange rate",
            "Step 2 Call foreign_currency_exchange to query use USD and British pound to calculate corresponding Chinese yuan "
        ],
    },
    {
        "user_request": "Please reschedule the originally arranged repair service for my Haier refrigerator model BCD-200, which needs repair due to not cooling, from 9 am on May 5th to 10 am on next Monday (May 8th) at No. 10 Chaowai Street, Chaoyang District, Beijing, and contact number 13800138000. Also, inquire about the current repair status.",
        "solution": ["get_repair_request_id", "appliance_repair_rescheduling", "appliance_repair_status_query"],
        "task_steps": [
            "Step 1 Call get_repair_request_id to obtain the original repair request information",
            "Step 2 Call appliance_repair_rescheduling to change the repair time",
            "Step 3 Call appliance_repair_status_query to inquire about the repair status based on the reservation's information"
        ],
   }
]



def load_raw_file(filename="raw/ultratool/dev.json"):
    """Only preserving samples where invoked tasks >= 2"""
    valid_samples, total_length = [], 0

    read_file = open(filename, 'r')
    for line in read_file:
        content = json.loads(line)
        total_length += 1
        if len(content["tools"]) >= 2:
            valid_samples.append(content)
    
    print(f"Filtering raw samples ... ")
    print(f"# Valid {len(valid_samples)} # Total {total_length}\n")
    return valid_samples


def filtering_tasks_and_sampels(write_file=True):
    task_dict, task_counter, task2domain = defaultdict(dict), defaultdict(int), defaultdict(set) 
   
    for sample in all_data:
        cur_domain = sample["domain"]

        for tool in sample["tools"]:
            task_counter[tool["name"]] += 1
            task_dict[tool["name"]] = {
                "id": tool["name"],
                "desc": tool["description"],
                "domain": cur_domain
            }

            task2domain[tool["name"]].add(cur_domain)

    # filter tasks via frequency (valid tasks should appear more than 5 times)
    valid_tasks = [task for task in task_counter.keys() if task_counter[task] >= MINIMUM_FREQ]
    sum_task = sum([task_counter[task] for task in valid_tasks])
    
    # print valid tasks
    table = pt.PrettyTable()
    table.field_names = ["Domain", "Task", "Description", "Frequency (%)"]
    for task in valid_tasks:
        table.add_row([str(task2domain[task])[:20], task, task_dict[task]["desc"][:30], str(task_counter[task]) + " ("+str(round(task_counter[task] / sum_task * 100, 2))+ ") " ])
    print(table)

    final_samples, final_task_list = [], []
    for sample in all_data:
        flg = True 
        for tool in sample["tools"]:
            if tool["name"] not in valid_tasks:
                flg = False

        if flg is True: 
            final_samples.append(sample)
            final_task_list.extend([tool["name"] for tool in sample["tools"]])
    
    print(f'Final # Sample {len(final_samples)}, # Valid-Task {len(valid_tasks)}, # Used-Task {len(set(final_task_list))}')
    final_task_contents = [task_dict[task] for task in list(set(final_task_list)) if task not in DEL_TASK]
    for task in final_task_contents:
        task["domain"] = list(task2domain[task["id"]])
    return final_samples, final_task_contents


def construct_task_graph():
    task_links, task_nodes, task_names = [], copy.deepcopy(final_tasks), [task["id"] for task in final_tasks]
    
    # add links via trajectory
    for one_sample in final_data:
        for idx, task in enumerate(one_sample["tools"][:-1]):
            source_name, target_name = task["name"], one_sample["tools"][idx+1]["name"]
            source_name = source_name if source_name not in REPLACE_TASK.keys() else REPLACE_TASK[source_name]
            target_name = target_name if target_name not in REPLACE_TASK.keys() else REPLACE_TASK[target_name]

            assert source_name in task_names and target_name in task_names

            link = source_name + ", " + target_name
            task_links.append(link)
 
    task_links.extend(["create_document" + ", " + "file_write", "read_agenda" + ", " + "check_room_booking_status", "search_agenda" + ", " + "edit_agenda"])
    task_links.extend(['create_reminder, clock_alarm_change', 'book_restaurant, get_dish_id', 'select_hotel, check_room_availability'])
    task_links = list(set(task_links))
    task_links = [{"source": link.split(", ")[0], "target": link.split(", ")[1]} for link in task_links]
    
    print(f"Task Graph # Node {len(task_nodes)} # Link {len(task_links)}")

    # IMPORTANT: write file
    with open(f"{PATH_TO_ULTRATOOL}/tool_desc.json", 'w') as file:
        json.dump({"nodes": task_nodes}, file, indent=4)

    with open(f"{PATH_TO_ULTRATOOL}/graph_desc.json", 'w') as file:
        json.dump({"nodes": task_nodes, "links": task_links}, file, indent=4)


def reformat_data_samples():
    for idx, one_sample in enumerate(final_data):
        one_sample["id"] = str(idx)
    
    # split test-set 
    random.shuffle(final_data)
    test_ids = [one_sample["id"] for one_sample in final_data[:500]]

    with open(f"{PATH_TO_ULTRATOOL}/split_ids.json", 'w') as file:
        json.dump({"test_ids": {"chain": test_ids}}, file, indent=4)

    for one_sample in final_data:
        cur_id, query = one_sample["id"], one_sample["question"]
        task_list = [REPLACE_TASK.get(task["name"], task["name"])  for task in one_sample["tools"]]
        
        task_steps = [step_dict["step"] for step_dict in one_sample["plan"] if step_dict["tool"] not in ['null', 'No tool required'] and '=' in step_dict["tool"]]
        
        task_links = [{"source": node, "target": task_list[idx+1]} for idx, node in enumerate(task_list[:-1])]
        task_nodes = [{"task": node} for node in task_list]

        write_dict = {
            "id": cur_id,
            "user_request": query,
            "task_steps": task_steps,
            "task_nodes": task_nodes,
            "task_links": task_links,
            "domain": one_sample["domain"],
            "n_tools": len(task_list)
        }
        
        with open(f"{PATH_TO_ULTRATOOL}/data.json", 'a+') as write_file:
            write_file.write(json.dumps(write_dict) + "\n")
            write_file.flush()

        with open(f"{PATH_TO_ULTRATOOL}/user_requests.json", 'a+') as write_file:
            write_file.write(json.dumps({"id": cur_id, "user_request": query}) + "\n")
            write_file.flush()


def prompt_llm_generate_steps(one_sample):
    prompt = "# GOAL #\nGiven the # USER REQUEST # and corresponding ground-truth # SOLUTION #, I need you to generate the chain of thoughts that can decompose user's request into managable sub-tasks, where each of them can be solved by a specific TASK."
    prompt += "\n# REQUIREMENTS #\n1. The decomposed task steps can work in sequence to perfectly solve the user's request, and each one corresponds to call one TASK in the # SOLUTION #."
    prompt += "\n2. Each of the step's calling TASK (i.e., API) should in the # SOLUTION # and you should follow the original sequence for generating steps."
    prompt += "\n3. Your output should be in a strict JSON format which can be DIRECTLY parsed by `json.loads()`, like `{'task_steps': ['Step 1 Call xx to do xxx', 'Step 2 Call xx to do xxx', ,,,]}`"
    prompt += "\n4. The number of generated steps should STRICTLY align with the length of solution."
    prompt += "\n5. The generated steps should be reasonable and contain necessary information."
 
    demo_string = "# EXAMPLES #\n"
    for demo in demos:
        current_demo = "# USER REQUEST #\n{{user_request}}. # SOLUTION #{{solution}}. # RESULT #{{result}}"
        current_demo = current_demo.replace("{{user_request}}", demo["user_request"])
        current_demo = current_demo.replace("{{solution}}", str(demo["solution"]))
        current_demo = current_demo.replace("{{result}}", json.dumps({"task_steps": demo['task_steps']}))
        demo_string += current_demo + "\n"

    prompt = prompt + demo_string
    prompt += "\n# USER REQUEST #\n {{user_request}}. \n# SOLUTION #\n {{solution}}.\n\nNow please generate the task steps in a strict JSON format:# RESULT #\n"
    prompt = prompt.replace("{{user_request}}", one_sample["user_request"])
    solution = [node["task"] for node in one_sample['task_nodes']]
    prompt = prompt.replace("{{solution}}", str(solution))

    response = client.chat.completions.create(
        model = "gpt-35-turbo", 
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


def generate_and_write_files():
    origin_data_filename = f"{PATH_TO_ULTRATOOL}/data.json"
    rewrite_filename = f"{PATH_TO_ULTRATOOL}/prompt_data.json"

    has_inferenced_ids = []
    if os.path.exists(rewrite_filename):
        for line in open(rewrite_filename, 'r'):
            sample = json.loads(line)
            has_inferenced_ids.append(sample["id"])
    
    succ_cnt, fail_cnt = 0, 0
    for line in open(origin_data_filename, 'r'):
        sample = json.loads(line)
        if sample["id"] in has_inferenced_ids:
            continue 

        try:
            generated_steps = prompt_llm_generate_steps(sample)
        except Exception as e:
            print(e)
            fail_cnt += 1
            continue 

        write_content = {
            "id": sample["id"],
            "user_request": sample["user_request"],
            # replace original steps with LLM's generated ones
            "task_steps": generated_steps["task_steps"],
            "task_nodes": sample["task_nodes"],
            "task_links": sample["task_links"]
        }

        with open(rewrite_filename, 'a+') as write_file:
            write_file.write(json.dumps(write_content) + "\n")
            write_file.flush()
            print(write_content['user_request'], generated_steps, "\n")
            succ_cnt += 1
    
    print(f"\n\nFinish!\n# Prev {len(has_inferenced_ids)}  # Succ {succ_cnt}  # Fail {fail_cnt}")


if __name__ == "__main__":
    random.seed(0)
    # Step 1 - filtering data sampels
    # all_data contains filtered samples where # invoked tasks >= 2
    all_data = load_raw_file()
    
    # Step 2 - filtering valid tasks and only consider data samples that contain valid tasks
    # final_tasks have appeared more than 5 times
    # we further filter the data samples based on valid tasks
    final_data, final_tasks = filtering_tasks_and_sampels()
    
    # Step 3 - construct task graph based on valid tasks, and reformat data samples like TaskBench
    construct_task_graph()
    reformat_data_samples()
    
    # Step 4 - fill in steps
    # prompt GPT-4 to fill in more suitable steps (original steps are too coarse-grained)
    client = AzureOpenAI(
        azure_endpoint = "VALID ENDPOINT", 
        api_key = "OPENAI-API-KEY",  
        api_version = "2024-02-15-preview"  
    )
    generate_and_write_files()
