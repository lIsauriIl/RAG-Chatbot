from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.embeddings import OllamaEmbeddings
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp, Ollama
from langchain_community.chat_models import ChatLlamaCpp, ChatOllama
from llama_cpp import Llama


# Function to load the vectorstore and retriever
def load_vectorstore_retriever(vecdb_path: str = './vectordb', embed_model: str = 'mxbai-embed-large') -> Chroma:
    """
    Load the vectorstore retriever using the specified vector database path and embedding model.
    
    Args:
        vecdb_path (str): The path to the vector database.
        embed_model (str): The embedding model to use.
        
    Returns:
        Chroma: The retriever object. (there is a mistake in these docs but it works so leave it)
    """
    vecdb = Chroma(persist_directory=vecdb_path, embedding_function=OllamaEmbeddings(model=embed_model))
    retriever = vecdb.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    return retriever

# Function to create the LlamaCpp model
def create_model() -> ChatLlamaCpp:
    """
    Create the LlamaCpp model with specified parameters.
    
    Returns:
        ChatLlamaCpp: The created model object.
    """
    callback = CallbackManager([StreamingStdOutCallbackHandler()])
    model = ChatLlamaCpp(
        model_path='llama-2-7b-chat.Q8_0.gguf',
        n_ctx=5000,
        max_tokens=2000,
        verbose=False,
        f16_kv=True,
        #callback_manager=callback,
        temperature=0.1
    )

    llm = LlamaCpp(
        model_path='llama-2-7b-chat.Q8_0.gguf',
        n_ctx=5000,
        max_tokens=2000,
        verbose=False,
        f16_kv=True,
        callback_manager=callback
    )
    ollama_llm = Ollama(model='llama2', callback_manager=callback)
    ollama_model = ChatOllama(model='llama2', callback_manager=callback)

    llama_model = Llama(
        model_path='llama-2-7b-chat.Q8_0.gguf',
        n_ctx=5000,
        #max_tokens=2000,
        chat_format='llama-2',
        #callback_manager=callback
        verbose=False
    )
    return model

# Function to create a system prompt template
def create_sys_prompt(text):
    """
    Create a system prompt template using the provided text.
    
    Args:
        text (str): The system prompt text.
        
    Returns:
        ChatPromptTemplate: The created prompt template.
    """
    prompt = ChatPromptTemplate.from_messages([
        ('system', text),
        MessagesPlaceholder(variable_name='chat_history'),
        ('human', '{input}')
    ])
    return prompt

# Function to create the RAG (Retrieval-Augmented Generation) chain
def create_rag_chain(retriever, model, sys_prompt, context_sys_prompt):
    """
    Create the RAG chain using the specified retriever, model, system prompt, and contextualised system prompt.
    
    Args:
        retriever (Chroma): The retriever object.
        model (ChatLlamaCpp): The model object.
        sys_prompt (ChatPromptTemplate): The system prompt template.
        context_sys_prompt (ChatPromptTemplate): The contextualised system prompt template.
        
    Returns:
        object: The created RAG chain.
    """
    qa_chain = create_stuff_documents_chain(model, sys_prompt)
    history_retriever = create_history_aware_retriever(model, retriever, context_sys_prompt)
    history_rag_chain = create_retrieval_chain(history_retriever, qa_chain)
    return history_rag_chain

# Function to create a runnable with message history
def create_runnable(input_runnable, get_history_func, with_rag_history=False) -> RunnableWithMessageHistory:
    """
    Create a runnable with message history using the specified input runnable and history function.
    
    Args:
        input_runnable (object): The input runnable object.
        get_history_func (function): The function to get the message history.
        with_rag_history (bool): Whether to include RAG history.
        
    Returns:
        RunnableWithMessageHistory: The created runnable object.
    """
    if with_rag_history:
        runnable = RunnableWithMessageHistory(
            input_runnable, get_history_func,
            input_messages_key='input',
            history_messages_key='chat_history',
            output_messages_key='answer'
        )
    else:
        runnable = RunnableWithMessageHistory(
            input_runnable, get_history_func,
            input_messages_key='input',
            output_messages_key='answer'
        )
    return runnable

# Function to run the runnable
def run_runnable(runnable, query, config, chat_history):
    """
    Run the runnable with the specified query, configuration, and chat history.
    
    Args:
        runnable (RunnableWithMessageHistory): The runnable object.
        query (str): The user query.
        config (dict): The configuration dictionary.
        chat_history (list): The chat history.
        
    Returns:
        dict: The response from the runnable.
    """
    response = runnable.invoke({'input': query}, config)
    return response

# Main chatbot function
def run_chatbot():
    """
    Main function to run the chatbot. Continuously prompts the user for queries until 'exit' is entered.
    """
    session_history = {}
    chat_history = []

    def get_session_id(session_id: str) -> BaseChatMessageHistory:
        """
        Get the session ID from the session history or create a new one if it doesn't exist.
        
        Args:
            session_id (str): The session ID.
            
        Returns:
            BaseChatMessageHistory: The chat message history for the session.
        """
        if session_id not in session_history:
            session_history[session_id] = ChatMessageHistory()
        return session_history[session_id]

    sys_prompt_text = (
        "You are an AI chatbot that helps users in question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. Use only the context. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise. If possible, use one to three word answers. "
        "If asked to list things, just list them, and do NOT give explanations unless asked. "
        "Do NOT say things like 'based on the provided context', or things like that, when answering a question. "
        "Just give the answer directly. Also, don't include any citations in your answer. "
        "Citations such as '(Dolan and Brockett, 2005)'. Don't include things like this. "
        "Give the answer with no citations."
        "\n\n"
        "Context: {context}"
    )

    contextualised_sys_prompt_text = (
        "Given the chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
        "\n\n"
        #"{chat_history}"
    )

    retriever = load_vectorstore_retriever()
    model = create_model()

    sys_prompt = create_sys_prompt(text=sys_prompt_text)
    contextualised_sys_prompt = create_sys_prompt(text=contextualised_sys_prompt_text)

    rag_chain = create_rag_chain(retriever=retriever, model=model, sys_prompt=sys_prompt, context_sys_prompt=contextualised_sys_prompt)
    runnable = create_runnable(input_runnable=rag_chain, get_history_func=get_session_id, with_rag_history=True)
    config = {'configurable': {'session_id': 'test_id'}}

    while True:
        query = str(input('Enter query, or "exit" to exit: '))
        if query == 'exit':
            break
        db = Chroma(persist_directory='vectordb', embedding_function=OllamaEmbeddings(model='mxbai-embed-large'))
        results = db.similarity_search_with_score(query, k=4)
        #for context in results:
         #  print('\n')
          # print(context[0].page_content)
           #print('Distance: ', context[1])
        #print(results)
        #print('\n')
        response = run_runnable(runnable=runnable, query=query, config=config, chat_history=chat_history)
        print(response['answer'])
        print('\n')

# Run the chatbot
run_chatbot()
