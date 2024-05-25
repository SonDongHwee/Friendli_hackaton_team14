import os
from friendli import Friendli
import gradio as gr
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models.friendli import ChatFriendli
from pymongo.operations import SearchIndexModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough


MONGODB_ATLAS_CLUSTER_URI='mongodb+srv://team-14:HT6WL4dKGA6FFY5h@cluster0.tyqdayd.mongodb.net/'
FRIENDLI_TOKEN=''
OPENAI_API_KEY=''

llm = ChatFriendli(model="meta-llama-3-70b-instruct", friendli_token=FRIENDLI_TOKEN, max_tokens=4096)
llm_ = Friendli(token=FRIENDLI_TOKEN)
db = MongoClient(MONGODB_ATLAS_CLUSTER_URI)

DB_NAME = "team-14"
QUERY_COLLECTION_NAME = 'Query_only'
QNA_COLLECTION_NAME = 'qna_comment_db'
ANN_COLLECTION_NAME = 'policy'
REF_COLLECTION_NAME = 'reference_db'
ATLAS_VECTOR_SEARCH_INDEX_NAME = 'ref'


REF_COLLECTION = db[DB_NAME][REF_COLLECTION_NAME]
QUERY_COLLECTION = db[DB_NAME][QUERY_COLLECTION_NAME]
QNA_COLLECTION = db[DB_NAME][QNA_COLLECTION_NAME]



search_model = SearchIndexModel(
    definition={
        "fields": [
            {
                "numDimensions": 1536,
                "path": "embedding",
                "similarity": "cosine",
                "type": "vector"
            }
        ]
    },
    name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    type="vectorSearch",
)

vector_store = MongoDBAtlasVectorSearch(
    embedding=OpenAIEmbeddings(disallowed_special=(),api_key=OPENAI_API_KEY),
    collection=QUERY_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
vector_store_announcement = MongoDBAtlasVectorSearch(
    embedding=OpenAIEmbeddings(disallowed_special=(),api_key=OPENAI_API_KEY),
    collection=ANN_COLLECTION_NAME,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
)
vector_store_ref = MongoDBAtlasVectorSearch(
            collection=REF_COLLECTION,
            embedding=OpenAIEmbeddings(disallowed_special=(), api_key=OPENAI_API_KEY),
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
        )
retriever_ref = vector_store_ref.as_retriever(search_kwargs={'k': 2})
retriever = vector_store.as_retriever(search_type="similarity_score_threshold",search_kwargs={"score_threshold": 0.05, "k":2})
retriever_ann = vector_store_announcement.as_retriever(search_kwargs={"k":1})

def retrieve_documents(query_text: str, retriever):
    results = retriever.get_relevant_documents(query_text)
    print(f"Retrieved {len(results)} documents.")
    print("===========================================================================================================")

    return results

def generate_response(query_text, retrieved_docs, LLM):
    # 검색된 문서의 텍스트를 결합
    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    # sources = "\n\n".join(str(doc.metadata['idx']) for doc in retrieved_docs)
    # print(f"Source index: {sources}")
    print("Query : ", template.format(context=context, question=query_text))

    return LLM.call_as_llm(message=template.format(context=context, question=query_text))

template = """Use the following pieces of context to answer the question at the end.
If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.

{context}

Question: {question}

Helpful Answer:"""

prompt = PromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever_ann | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def chat_function(message, history):
    new_messages = []
    for user, chatbot in history:
        new_messages.append({"role" : "user", "content": user})
        new_messages.append({"role" : "assistant", "content": chatbot})
    new_messages.append({"role": "user", "content": message})

    stream = llm_.chat.completions.create(
        model="meta-llama-3-70b-instruct",
        messages=new_messages,
        stream=True
    )
    res = ""
    for chunk in stream:
        res += chunk.choices[0].delta.content or ""
    return res

def chat_function_rag(message, history):
    print(message)
    new_messages = []
    for user, chatbot in history:
        new_messages.append({"role" : "user", "content": user})
        new_messages.append({"role" : "assistant", "content": chatbot})
    new_messages.append({"role": "user", "content": message})

    retrieved_docs = retrieve_documents(message, retriever) 

    from_ref = False
    if (len(retrieved_docs)==0):
        print("No relevant prior questions found.")
        retrieved_docs = retrieve_documents(message, retriever_ref)
        from_ref = True

    response = generate_response(message, retrieved_docs, llm)
    return response, retrieved_docs, from_ref

def chat_function_ann(message, history):
    print(message)
    new_messages = []
    new_messages.append({"role": "user", "content": message})

    retrieved_docs = retrieve_documents(message, retriever_ann)    
    response = generate_response(message, retrieved_docs, llm)
    return response, retrieved_docs

def vote(data: gr.LikeData):
    if data.liked:
        print("You upvoted this response: " + data.value)
    else:
        print("You downvoted this response: " + data.value)

# gr.ChatInterface(chat_function_rag).launch()
with gr.Blocks(theme=gr.themes.Default()) as demo:
    gr.Markdown("# QueryXstudent")
    gr.Markdown("Hi!, I'm QueryBook! Happy to support you")
    
    def respond_1(message, chat_history, ):
        response, retrieved_docs, from_ref = chat_function_rag(message, chat_history)
        chat_history.append((message, response))
        
        if not from_ref:
            idx = retrieved_docs[0].metadata['idx']
            answer = QNA_COLLECTION.find({"idx": idx})[0]['answer']
            like = QNA_COLLECTION.find({"idx": idx})[0]['metadata']['priority_weight']
            question = QNA_COLLECTION.find({"idx": idx})[0]['question']
            comment = QNA_COLLECTION.find({"idx": idx})[0]['comment']

            answer = "🧐Helpfulness: " + "⭐️"*like + "\n\n" + " 👶🏻 Question:" + question + "\n\n" +  " 👱🏻‍♂️ Answer: "+ answer
            answer = answer + "\n\n 👨‍🏫 Comment: " + comment
            return gr.update(value=chat_history), "", answer
        else:
            answer = "# 앗! 이런 질문을 한 사람은 학생이 처음입니다! DB Candidate 에 등록하시겠습니까?"
            return gr.update(value=chat_history), "", answer
        
    def respond_2(message, chat_history_2):
        response_2 = rag_chain.invoke(message)
        chat_history_2.append((message, response_2))

        return  gr.update(value=chat_history_2), ""

    with gr.Column(scale=1):
        gr.Markdown("## 🔥 실시간 인기 글 🔥")
        top_5_documents = QNA_COLLECTION.find().sort("metadata.view", -1).limit(5)
        question_str = f" - {top_5_documents[0].get('question', '')}\n  - {top_5_documents[1].get('question', '')} \n   - {top_5_documents[2].get('question', '')} \n   - {top_5_documents[3].get('question', '')} \n  - {top_5_documents[4].get('question', '')} \n"
        post_list = gr.Markdown(value=question_str, elem_id="post_list")

    with gr.Column(scale=2):
            with gr.Column():
                chatbot = gr.Chatbot(label='Course Announcements',
                                    height=400, 
                                    avatar_images=((os.path.join(os.path.dirname(__file__), "student.png")),
                                                    (os.path.join(os.path.dirname(__file__), "qb.png"))),
                                    show_share_button=True)
                
                msg = gr.Textbox(label="Your message")

                with gr.Row():
                    ret_btn = gr.Button("See Comment")
                    submit_btn = gr.Button("Submit")
                with gr.Row():
                    clear_btn = gr.Button("Clear")
                    undo_btn = gr.Button("Delete previous")

                gr.Markdown("## 이런 질문을 찾으셨나요? (다른 수강생의 질문) 😋")
                retrieved_str = ''
                comment = gr.Markdown(value=retrieved_str, elem_id="ret_doc")

                submit_btn.click(respond_1, inputs=[msg, chatbot], outputs=[chatbot, msg, comment])
                clear_btn.click(lambda: None, None, chatbot, queue=False)
                undo_btn.click(lambda history: history[:-1], chatbot, chatbot, queue=False)
                

            with gr.Column():
                chatbot_2 = gr.Chatbot(label='Course Contents',
                                    height=400, 
                                    avatar_images=((os.path.join(os.path.dirname(__file__), "student.png")),
                                                    (os.path.join(os.path.dirname(__file__), "qb.png"))),
                                    show_share_button=True)
                msg_2 = gr.Textbox(label="Your message")
                submit_btn_2 = gr.Button("Submit")
                with gr.Row():
                    clear_btn_2 = gr.Button("Clear")
                    undo_btn_2 = gr.Button("Delete previous")

                submit_btn_2.click(respond_2, inputs=[msg_2, chatbot_2], outputs=[chatbot_2, msg_2])
                clear_btn_2.click(lambda: None, None, chatbot_2, queue=False)
                undo_btn_2.click(lambda history: history[:-1], chatbot_2, chatbot_2, queue=False)

    chatbot.like(vote, None)
    chatbot_2.like(vote, None, None)
    
demo.css = """
#post_list {
    height: 140px;
    overflow-y: scroll;
}
"""

# demo = gr.ChatInterface(fn=chat_function_rag, 
#                         title='QueryXstudent', 
#                         chatbot=chatbot,
#                         theme="soft",
#                         description="Hi!, I'm QueryBook! Happy to support you",
#                         undo_btn="Delete previous",
#                         clear_btn="Clear",
#                         )
demo.launch(share=True, server_name='0.0.0.0')