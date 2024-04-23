from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import BraveSearch
from langchain_community.chat_models import ChatOllama
from settings import BRAVE_API_KEY

tools = [BraveSearch.from_api_key(api_key=BRAVE_API_KEY, search_kwargs={"count": 1})]
prompt = hub.pull("hwchase17/react")

llm = ChatOllama(model="llama3")
agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
agent_executor.invoke({"input": "Did China benefit or suffer from Emperor Qin？"})