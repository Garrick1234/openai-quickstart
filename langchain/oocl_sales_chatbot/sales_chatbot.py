import gradio as gr

from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
import os


def initialize_sales_bot(vector_store_dir: str="oocl_sale"):
    os.environ['OPENAI_API_KEY'] = 'sk-cxaJqd8WiUff6tui995521441b814dA79585124bCd531460'
    os.environ['OPENAI_BASE_URL'] = 'https://api.xiaoai.plus/v1'

    db = FAISS.load_local(vector_store_dir, OpenAIEmbeddings(),  allow_dangerous_deserialization=True)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    
    global SALES_BOT    
    SALES_BOT = RetrievalQA.from_chain_type(llm,
                                           retriever=db.as_retriever(search_type="similarity_score_threshold",
                                                                     search_kwargs={"score_threshold": 0.8}))
    # 返回向量数据库的检索结果
    SALES_BOT.return_source_documents = True

    return SALES_BOT

def sales_chat(message, history):
    print(f"[message]{message}")
    print(f"[history]{history}")
    # TODO: 从命令行参数中获取
    enable_chat = True

    ans = SALES_BOT({"query": message})
    # 如果检索出结果，或者开了大模型聊天模式
    # 返回 RetrievalQA combine_documents_chain 整合的结果
    if ans["source_documents"] or enable_chat:
        print(f"[result]{ans['result']}")
        print(f"[source_documents]{ans['source_documents']}")
        return ans["result"]
    # 否则输出套路话术
    else:
        return ("感谢您的提问！目前我没有找到一个直接的答案，但我会帮您记录下这个问题并转交给我们的专业团队。"
                "您也可以通过邮件（support@example.com）或电话（123-456-7890）联系我们获得进一步的帮助。")
    

def launch_gradio():
    demo = gr.ChatInterface(
        fn=sales_chat,
        title="船运公司客服",
        # retry_btn=None,
        # undo_btn=None,
        chatbot=gr.Chatbot(height=600),
    )

    demo.launch(share=True, server_name="localhost")

if __name__ == "__main__":
    # 初始化房产销售机器人
    initialize_sales_bot()
    # 启动 Gradio 服务
    launch_gradio()
