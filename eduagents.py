from agent import Agent
from langchain_core.vectorstores import VectorStoreRetriever
from tqdm import tqdm
import argparse, util

class TeachingAssistantAgent(Agent):
    def __init__(self, 
                 model_name: str, 
                 name: str, 
                 retriever: VectorStoreRetriever=None,
                 temperature: float=0,
                 ) -> None:
        super().__init__(model_name, 
                         '[TeachingAssistantAgent]{}'.format(name), 
                         retriever,
                         temperature)
        self.template = '''您是一位优秀的老师助理，帮助老师备课，请仅就相关内容给出{}的课程安排，
按几节课划分一下主题。这门课会分{}节上，每堂课大约{}分钟，对象是{}。
请尽量用中文表达。'''
            
    def query(self, 
              topic: str,
              sections: str,
              duration: str,
              audience: str,
              retriever: VectorStoreRetriever=None) -> str:
        question = self.template.format(topic, sections, duration, audience)
        return super().query(question, retriever)
        

class VerifierAgent(Agent):
    def __init__(self, 
                 model_name: str, 
                 name: str, 
                 retriever: VectorStoreRetriever=None,
                 temperature: float=0,
                 ) -> None:
        super().__init__(model_name, 
                         '[VerifierAgent]{}'.format(name), 
                         retriever,
                         temperature)
        self.template = '''您是一位优秀的内容校验者，我们对课程的要求是：{}
请判断一下以下的内容是否符合仅依据提供的材料安排课程的要求：{} 
请尽量用中文表达。'''
        
    def query(self, 
              requirements: str, 
              content: str, 
              retriever: VectorStoreRetriever=None) -> str:
        question = self.template.format(requirements, content)
        return super().query(question, retriever)

class TeacherAgent(Agent):
    def __init__(self, 
                 model_name: str, 
                 name: str, 
                 retriever: VectorStoreRetriever=None,
                 temperature: float=0,
                 ) -> None:
        super().__init__(model_name, 
                         '[TeacherAgent]{}'.format(name), 
                         retriever,
                         temperature)
        self.template = '''您是一位优秀的高中老师，在上一门关于{0}的课，今天是第{1}堂，题目是{2},
请仅仅根据提供的内容讲解这堂课，内容不超过{3}分钟。
请尽量用中文表达。'''

    def query(self, 
              topic: str,
              numofk: str,
              klass: str,
              retriever: VectorStoreRetriever=None) -> str:
        question = self.template.format(topic, numofk, klass, '5') # 5 mins due to the limit of LLM output size
        return super().query(question, retriever)

class QuestionairAgent(Agent):
    def __init__(self, 
                 model_name: str, 
                 name: str, 
                 retriever: VectorStoreRetriever=None,
                 temperature: float=0,
                 ) -> None:
        super().__init__(model_name, 
                         '[QuestionairAgent]{}'.format(name), 
                         retriever,
                         temperature)
        self.template = '''您是一位高级诊断性测验创建者。您将为一门对象是中国高中生的课{0}，
创建出一些优秀的低成本测验和诊断。您将构建多个多选题来检验内容主题{1}。
这些题目应当非常相关，并超越仅仅的事实。多选题应当包括合理的、竞争性的备选答案，
而且不应该包括一个全部为以上选项。请在最后提供这些问题的答案。请尽量用中文表达。'''

    def query(self, 
              topic: str,
              klass: str,
              retriever: VectorStoreRetriever=None) -> str:
        question = self.template.format(topic, klass)
        return super().query(question, retriever)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Teaching Agents with Online RAG.')
    parser.add_argument('-m', '--model', help='LLM model to be used', default="phi3")
    parser.add_argument('-l', '--lists', nargs='+', help='<Required> urls to be vectorized', required=True)
    parser.add_argument('-t', '--topic', help='Topic', required=True)
    parser.add_argument('-s', '--sections', help='Sections', default="4")
    parser.add_argument('-d', '--duration', help='Duration', default="45")
    parser.add_argument('-a', '--audience', help='Audience', default="高中生")
    parser.add_argument('-k', '--klass', help='Class', default="深度学习")
    args = parser.parse_args()

    retriever = util.prepareRag(args.lists)
    model = args.model

    from eduagents import TeachingAssistantAgent, VerifierAgent, TeacherAgent
    taa = TeachingAssistantAgent(model, "Lily", retriever)
    output, input = taa.query(args.topic, args.sections, args.duration, args.audience)
    print(output)

    va = VerifierAgent(model, "Cici", retriever)
    output, input = va.query(input, output)
    print(output)

    ta = TeacherAgent(model, "Laura", retriever)
    output, input = ta.query(args.topic, '2', args.klass)
    print(output)

    output, input = va.query(input, output)
    print(output)

    qa = QuestionairAgent(model, "Queen", retriever)
    output, input = qa.query(args.topic, args.klass)
    print(output)

    output, input = va.query(input, output)
    print(output)
