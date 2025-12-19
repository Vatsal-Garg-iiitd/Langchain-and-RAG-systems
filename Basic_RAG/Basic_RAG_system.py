from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound, VideoUnavailable

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint,HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from dotenv import load_dotenv

load_dotenv()

#Youtube Transcript loader
video_id="wjZofJX0v4M"
try:
    transcript_list = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
    #Join all the text into one chunk
    transcript=" ".join(chunk.text for chunk in transcript_list)
except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
    print(f"Transcript not available: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")

#Splitter to split the text
splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
chunks=splitter.create_documents([transcript])


#Embedding model to create vector embeddings
embedding=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Using FAISS as vector store
vector_store=FAISS.from_documents(chunks,embedding=embedding)

retriver= vector_store.as_retriever(search_type="mmr",search_kwargs={"k":3})

llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation"
)
model=ChatHuggingFace(llm=llm)

prompt=PromptTemplate(
    template="""      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: {question}""",
      input_variables=["context","question"]
)

parser=StrOutputParser()

def fromat_docs(retrieved_docs):
    context_text="\n\n".join(doc.page_content for doc in retrieved_docs)
    return context_text


parallel_chain=RunnableParallel({
    "context": retriver | RunnableLambda(fromat_docs),
    "question":RunnablePassthrough()
})

chain= parallel_chain|prompt|model|parser

result= chain.invoke("Give me 5 Interesting facts about GPT from the context i you retrieved")

print(result)
