LLM_INFER_PROMPT = """Please understand the user's request and generate task steps and task invocation graph to solve the request.\n""" \
                  + """## Requirements:\n1. The format must in a strict JSON format, like: {"task_steps": [ "concrete steps, format as Step x: Call xxx to do xxx" ], "task_nodes": [{"task": "task name must be from TASK LIST", "arguments": [ {"name": "parameter name", "value": "parameter value"} ]}], "task_links": [{"source": "task name i", "target": "task name j"}]}\n""" \
                  + """2. The generated task steps and task nodes can resolve the given user request perfectly. Task name must be selected from TASK LIST.\n""" \
                  + """3. The task steps should strictly aligned with the task nodes, and the number of task steps should be same with the task nodes.\n""" \
                  + """4. The task links (task_links) should reflect the dependencies among task nodes, i.e. the order in which the APIs are invoked, you should also understand each tool's input and output demands.\n""" \
                  + """## User Request:\n{{user_request}}\nNow please generate your response in a strict JSON format:\n## Response:\n"""
        