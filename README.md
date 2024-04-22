如何用Llama3+RAG快速实现企业本地化大模型

【准备工作】

0. Python虚拟环境，建议3.10, Windows下建议使用miniconda:

> conda create -n rag python=3.10
生成完毕以后

> conda activate rag

1. 直接到官网下载并安装Ollama
2. 启动PowerShell/Terminal，先pull需要的Embedding模型
> ollama pull nomic-embed-text

3. 然后直接运行llama3（Ollama会自动下载）如果只想拿个本地llama3玩玩，只需要做第2、4步，就可以了：
> ollama run llama3

4. 最后需要安装所需python库​

> pip install -r requirement.txt

5. 运行

> python main.py