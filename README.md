如何用RAG+AI Agent快速实现企业本地化大模型

【准备工作】

0. Python虚拟环境，建议3.10, Windows下建议使用miniconda:

> conda create -n rag python=3.10
生成完毕以后

> conda activate rag

1. 直接到官网下载并安装Ollama
2. 启动PowerShell/Terminal，先pull需要的Embedding模型
> ollama pull nomic-embed-text

3. 然后直接运行phi3或llama3（Ollama会自动下载）如果只想拿个本地phi3或llama3玩玩，只需要做第2、4步，就可以了。不过Phi3似乎比llama3对中文更加友好，所以我优先采用。

> ollama run phi3

4. 最后需要安装所需python库​

> pip install -r requirement.txt

5. 运行

> python main.py

[说明]

* main.py是llama3+RAG的一个演示

* eduagents.py是通过各种Agents来提供一堂课程从备课提纲、讲课内容、测试题目及答案，以及校验。这里使用了Wiki上的深度学习作为例子，但实际上可以更换为任何课题。
> python eduagents.py -l https://zh.wikipedia.org/wiki/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0 -t 深度学习

比如 https://zh.wikipedia.org/wiki/%E7%89%9B%E9%A1%BF%E8%BF%90%E5%8A%A8%E5%AE%9A%E5%BE%8B 及“牛顿第二定律”