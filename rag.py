
import streamlit as st
import os
import tempfile
from pinecone import Pinecone,ServerlessSpec
from langchain_cohere import CohereEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.llms import Cohere
from tqdm import tqdm


os.environ["PINECONE_API_KEY"] =os.getenv("PINECONE_API_KEY")
os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
embeddings = CohereEmbeddings(model="embed-english-v3.0")

st.title("Interactive QA Bot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(tmp_file_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)


    index_name = "rag"
    index = pc.Index(index_name)

    # Insert documents into Pinecone
    batch_size = 100
    with st.spinner("Processing document..."):
        for i in tqdm(range(0, len(texts), batch_size)):
            i_end = min(i+batch_size, len(texts))
            metadatas = [{"text": t.page_content} for t in texts[i:i_end]]
            ids = [f"id_{j}" for j in range(i, i_end)]
            embeds = embeddings.embed_documents([t.page_content for t in texts[i:i_end]])
            index.upsert(vectors=zip(ids, embeds, metadatas))

    # Initialize Pinecone vector store
    vectorstore = Pinecone(index, embeddings.embed_query, "text")

    # Initialize Cohere LLM
    llm = Cohere(temperature=0.7, cohere_api_key="4f98475a-f42b-4c6e-9840-fb6994cde1e6")

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

    st.success("Document processed successfully!")

    # User input for questions
    user_question = st.text_input("Ask a question about the document:")

    if user_question:
        with st.spinner("Generating answer..."):
            answer = qa.run(user_question)

        st.subheader("Answer:")
        st.write(answer)

        st.subheader("Retrieved Document Segments:")
        retrieved_docs = vectorstore.similarity_search(user_question, k=2)
        for i, doc in enumerate(retrieved_docs):
            st.write(f"Segment {i+1}:")
            st.write(doc.page_content)

    # Clean up temporary file
    os.unlink(tmp_file_path)

else:
    st.info("Please upload a PDF document to start.")