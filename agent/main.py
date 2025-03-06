import typing
from pprint import pprint
import json

from ollama import Client
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langchain_community.tools import DuckDuckGoSearchRun
from semantic_router.utils.function_call import FunctionSchema
from pydantic import BaseModel, Field

class AgentRes(BaseModel):
    tool_name: str = Field(description="must be a string = 'tool_browser' or 'final_answer'")
    tool_input: dict = Field(description="must be a dictionary, e.g. {'q': 'who died on September 9 2024'}")
    tool_output: str | None = Field(description="can be a string or None", default=None)
    
    @classmethod
    def from_llm(cls, res: dict):
        try:
            out = json.loads(res['message']['content'])
            return cls(tool_name=out['name'], tool_input=out['parameters'])
        except Exception as e:
            print(f'Error from Ollama: \n{res}\n')
            raise e

class State(typing.TypedDict):
    user_q: str
    chat_history: list
    lst_res: list[AgentRes]
    output: dict

def node_agent(state: State):
    print("--- node_agent ---")
    dic_tools = {
        "tool_browser": tool_browser,
        "final_answer": final_answer
    }
    agent_res = run_agent(prompt=get_prompt(), 
        dic_tools=dic_tools,
        user_q=state['user_q'], 
        chat_history=state['chat_history'], 
        lst_res=state['lst_res'], 
        )

    print(agent_res)
    return {'lst_res': [agent_res]}

def ollama_run():
    client = Client(host="http://172.24.64.1:11434")
    # model = "deepseek-r1:14b"
    model = "llama3.2:latest"
    model = "llama3.1"
    # system, user, assistant
    response = client.chat(
        model=model,
        messages=[{"role": "user", "content": "who died on Septemober 9 2024"}],
        format="json",
    )
    print(response)

def ollama_chat(messages: list):
    client = Client(host="http://172.24.64.1:11434")
    # model = "deepseek-r1:14b"
    model = "llama3.2:latest"
    model = "mistral-nemo:latest"
    # system, user, assistant
    response = client.chat(
        model=model,
        messages=messages,
        format="json",
    )
    return(response)


# way 1: use langchain tool
@tool("tool_browser")
def tool_browser(q: str) -> str:
    """Search the web using DuckDuckGoSearchRun tool by the input `q`"""
    search = DuckDuckGoSearchRun()
    return search.run(q)


# way 2: use FunctionSchema
def browser(q: str) -> str:
    """Search the web using DuckDuckGoSearchRun tool by the input `q`"""
    search = DuckDuckGoSearchRun()
    return search.run(q)

def test_browser():
    tool_browser = FunctionSchema(browser).to_ollama()
    return tool_browser

@tool("final_answer")
def final_answer(text: str) -> str:
    """Return a natural language response to the user by passing the input `text`.
    You should provide as much context as possible and specify the source of the information."""
    return text


def test_connection():
    import requests

    try:
        response = requests.get("http://172.24.64.1:11434/api/version")
        print(f"連接測試結果: {response.status_code}")
        print(response.text)
        return True
    except Exception as e:
        print(f"連接測試失敗: {e}")
        return False

def get_prompt():
    prompt = """
You know everything, you must answer every question from the user, you can use the list of tools provided to you.
Your goal is to provide the user with the best possible answer, including key information about the sources and tools used.

Note, when using a tool, you provide the tool name and the arguments to use in JSON format. 
For each call, you MUST ONLY use one tool AND the response format must ALWAYS be in the pattern:
```json
{"name":"<tool_name>", "parameters": {"<tool_input_key>":<tool_input_value>}}
```
Remember, do NOT use any tool with the same query more than once.
Remember, if the user doesn't ask a specific question, you MUST use the `final_answer` tool directly.

Every time the user asks a question, you take note of some keywords in the memory.
Every time you find some information related to the user's question, you take note of some keywords in the memory.

You should aim to collect information from a diverse range of sources before providing the answer to the user. 
Once you have collected plenty of information to answer the user's question use the `final_answer` tool.
"""
    return prompt

def get_prompt_tools():
    dic_tools = {
        "tool_browser": tool_browser,
        "final_answer": final_answer
    }
    str_tools = "\n".join([str(n+1)+". `"+str(v.name)+"`: "+str(v.description) for n, v in enumerate(dic_tools.values())])
    prompt_tools = f"You can use the following tools:\n{str_tools}"
    return prompt_tools

def test_final_answer():
    messages = [
        {"role": "system", "content": get_prompt()+"\n"+get_prompt_tools()},
        {"role": "user", "content": "hello"}
    ]
    response = ollama_chat(messages)
    return response

def test_tool_browser(q: str):
    messages = [
        {"role": "system", "content": get_prompt()+"\n"+get_prompt_tools()},
        {"role": "user", "content": q}
    ]
    response = ollama_chat(messages)
    return response

def test_browser_agent(q: str, silent: bool = True):
    res = test_tool_browser(q)
    tool_input = json.loads(res['message']['content'])['parameters']['q']
    print(f'{tool_input=}')
    if not silent:
        print('try to use browser tool...')
    context = tool_browser.invoke(input={"q": tool_input})
    print(f'{context=}')
    if not silent:
        print('thinking the final answer...')
    messages = [
        {"role": "system", "content": "Give the most accurate answer using the following information:\n"+context},
        {"role": "user", "content": q}
    ]
    response = ollama_chat(messages)
    return response

def test_browser_agent_pydantic(q: str):
    # q = 'who died on Septemober 9 2024'
    agent_res = AgentRes(
        tool_name='tool_browser',
        tool_input={'q': q},
        tool_output=str(tool_browser.invoke(input={"q": q}))
    )
    return agent_res

def save_memory(lst_res: list[AgentRes], user_q: str):
    memory = []
    for res in [res for res in lst_res if res.tool_output is not None]:
        memory.extend([
            # assistant message
            {"role": "assistant", "content": json.dumps({"name": res.tool_name, "parameters": res.tool_input})},

            # user message
            {"role": "user", "content": res.tool_output}
        ])
    if memory:
        memory += [{"role":"user", "content":(f'''
                This is just a reminder that my original query was `{user_q}`.
                Only answer to the original query, and nothing else, but use the information I gave you. 
                Provide as much information as possible when you use the `final_answer` tool.
                ''')}]
    return memory

def run_agent(prompt:str, dic_tools:dict, user_q:str, chat_history:list[dict], lst_res:list[AgentRes]) -> AgentRes:
    ## start memory
    memory = save_memory(lst_res=lst_res, user_q=user_q)
    
    ## track used tools
    if memory:
        tools_used = [res.tool_name for res in lst_res]
        if len(tools_used) >= len(dic_tools.keys()):
            memory[-1]["content"] = "You must now use the `final_answer` tool."

    ## prompt tools
    str_tools = "\n".join([str(n+1)+". `"+str(v.name)+"`: "+str(v.description) for n,v in enumerate(dic_tools.values())])
    prompt_tools = f"You can use the following tools:\n{str_tools}"
        
    ## messages
    messages = [{"role":"system", "content":prompt+"\n"+prompt_tools},
                *chat_history,
                {"role":"user", "content":user_q},
                *memory]
    #pprint(messages) #<--print to see prompt + tools + chat_history
    
    ## output
    llm_res = ollama_chat(messages)
    return AgentRes.from_llm(llm_res)


def test_run_agent():
    q = "who died on Septemober 9 2024"
    # q = "hows the weather in taiwan taipei in 20250305?"
    # q = "tsmc 100bilion invest in america is real?"
    # res = test_browser_agent(q, silent=False)
    # print(AgentRes.from_llm(res))
    # print(test_browser_agent_pydantic(q))
    chat_history = [{"role": "user", "content": "hi there, how are you?"},
         {"role": "assistant", "content": "I'm good, thanks!"},
         {"role": "user", "content": "I have a question"},
         {"role": "assistant", "content": "tell me"}]
    dic_tools = {
        "tool_browser": tool_browser,
        "final_answer": final_answer
    }
    agent_res = run_agent(user_q=q, chat_history=chat_history, lst_res=[], lst_tools=dic_tools.keys())
    print("\nagent_res:", agent_res)


def node_tool(state):
    dic_tools = {
        "tool_browser": tool_browser,
        "final_answer": final_answer
    }
    print('--- node_tool ---')
    res = state['lst_res'][-1]
    print(f"{res.tool_name}(input={res.tool_input})")
    agent_res = AgentRes(
        tool_name=res.tool_name,
        tool_input=res.tool_input,
        tool_output=str(dic_tools[res.tool_name].invoke(input=res.tool_input))
    )
    return {"output": agent_res} if res.tool_name == 'final_answer' else {"lst_res": [agent_res]}

def conditional_edges(state):
    print('--- conditional_edges ---')
    last_res = state['lst_res'][-1]
    next_node = last_res.tool_name if isinstance(state['lst_res'], list) else 'final_answer'
    print(f"{next_node=}")
    return next_node

def langgraph_main():
    dic_tools = {
        "tool_browser": tool_browser,
        "final_answer": final_answer
    }

    ## start the graph
    workflow = StateGraph(State)

    ## add Agent node
    workflow.add_node(node="Agent", action=node_agent) 
    workflow.set_entry_point(key="Agent")  #<--user query

    ## add Tools nodes
    for k in dic_tools.keys():
        workflow.add_node(node=k, action=node_tool)

    ## conditional_edges from Agent
    workflow.add_conditional_edges(source="Agent", path=conditional_edges)

    ## normal_edges to Agent - tools return to Agent except final_answer
    for k in dic_tools.keys():
        if k != "final_answer":
            workflow.add_edge(start_key=k, end_key="Agent")

    ## end the graph
    workflow.add_edge(start_key="final_answer", end_key=END)
    g = workflow.compile()
    
    # For non-IPython environments, save the graph visualization to a file
    try:
        # Use the correct method to generate the Mermaid diagram
        from langchain_core.runnables.graph import MermaidDrawMethod
        
        # Get the PNG data using the API draw method
        png_data = g.get_graph().draw_mermaid_png(draw_method=MermaidDrawMethod.API)
        
        # Save the PNG to a file
        with open("graph.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved as 'graph.png'")
    except Exception as e:
        print(f"Could not save graph visualization: {e}")
    
    return g

def main():
    print("Hello from agent!")
    # test_connection()  # 先測試連接
    # ollama_run()
    # print(tool_browser.invoke(input={"q": "who died on Septemober 9 2024"}))
    # result = test_browser()
    # print(result)
    # prompt_tools = get_prompt_tools()
    # print(prompt_tools)
    # test_final_answer()
    # print(test_tool_browser("who died on Septemober 9 2024"))
    q = "who died on Septemober 9 2024"
    # q = "hows the weather in taiwan taipei in 20250305?"
    # q = "tsmc 100bilion invest in america is real?"
    # res = test_browser_agent(q, silent=False)
    # print(AgentRes.from_llm(res))
    # print(test_browser_agent_pydantic(q))
    chat_history = [{"role": "user", "content": "hi there, how are you?"},
         {"role": "assistant", "content": "I'm good, thanks!"},
         {"role": "user", "content": "I have a question"},
         {"role": "assistant", "content": "tell me"}]
    dic_tools = {
        "tool_browser": tool_browser,
        "final_answer": final_answer
    }
    agent_res = run_agent(prompt=get_prompt(), dic_tools=dic_tools, user_q=q, chat_history=chat_history, lst_res=[])
    state = State(user_q=q, chat_history=chat_history, lst_res=[agent_res], output={})
    # print(state)
    # print(node_agent(state))
    # print(node_tool(state))
    # print(conditional_edges(state))
    g = langgraph_main()
    state = {'user_q':q,
         'chat_history':chat_history, 
         'lst_res':[], 
         'output':{} }

    ## 1) invoke function
    out = g.invoke(input=state)
    agent_out = out['output'].tool_output

    ## 2) stream function
    steps = g.stream(input=state) 
    for n,step in enumerate(steps):
        print("--- step", n, "---")
        print(step)

if __name__ == "__main__":
    main()
