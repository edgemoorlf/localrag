from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.chat_models import ChatOllama


class Agent:
    def __init__(self, 
                 model_name: str, 
                 name: str,                  
                 retriever: VectorStoreRetriever=None,
                 temperature: float=0,
                 ) -> None:
        """Create an agent

        Args:
            model_name(str): model name
            name (str): agent name 
            temperature (float): higher values make the output more random, while lower values make it more focused and deterministic
            sleep_time (float): sleep because of rate limits
        """
        self.model_name = model_name
        self.name = name
        self.temperature = temperature
        self.retriever = retriever
        self.memory_lst = []
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def query(self, question: str, retriever: VectorStoreRetriever=None) -> tuple:
        retre = retriever if retriever is not None else self.retriever
        rag_template = """请仅依据以下内容回答问题:
        {context}
        问题: {question}
        """
        print(question)
        rag_prompt = ChatPromptTemplate.from_template(rag_template)
        rag_chain = (
            {"context": retre, "question": RunnablePassthrough()}
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain.invoke(question), question