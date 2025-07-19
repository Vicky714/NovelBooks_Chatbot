def book_bot(user_input):
    from langchain_openai import ChatOpenAI
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from dotenv import load_dotenv
    import os
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_USE_LEGACY_KERAS'] = '1'

    # Load environment variables
    load_dotenv()

    # Load LLM model from Groq API (LLaMA 3)
    model = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.environ['OPENAI_API_KEY'],
        model_name='llama3-70b-8192',
        temperature=0.5
    )

    # Load embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Load PDF documents from BOOKS folder
    loader = DirectoryLoader(
        path='BOOKS',
        glob='*.pdf',
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500
    )

    def format_docs(documents):
        text = '\n'.join(doc.page_content for doc in documents)
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]

    # Convert docs to chunks and index with FAISS
    vector_store = FAISS.from_documents(format_docs(documents), embedding_model)

    # Create retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 5})

    # Prompt for the model
    prompt = PromptTemplate(
        template="""
        Answer the question based only on the following context. 
        If the answer is not present in the context, say "I don't know" instead of guessing.
        Context:
        {context}
        Question:
        {question}
        Answer:""",
        input_variables=['context', 'question']
    )

    # Output parser
    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': retriever,
        'question': RunnablePassthrough()
    })

    final_chain = parallel_chain | prompt | model | parser
    return final_chain.invoke(user_input)
