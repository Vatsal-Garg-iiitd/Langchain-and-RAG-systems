from qdrant_client import QdrantClient, models

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace, HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
load_dotenv()

client = QdrantClient(url="http://localhost:6333")

dense_vector_name="Embedding_Vector"
sparse_vector_name="TIDF_Vector"
dense_vector_model="sentence-transformers/all-MiniLM-L6-v2"
sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")


client.create_collection(
    collection_name="Qdrant_collection_2",
    vectors_config={dense_vector_name:models.VectorParams(size=384, distance=models.Distance.COSINE)},
    sparse_vectors_config={sparse_vector_name:models.SparseVectorParams(index=models.SparseIndexParams(on_disk=False))}
)

print("Qdrant collection created successfully.")


llm=HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-V3.2",
    task="text-generation",
)

embedding=HuggingFaceEmbeddings(model_name=dense_vector_model)

model= ChatHuggingFace(llm=llm)


docs = [
    Document(
        metadata={
            "title": "Beyond Horizons: AI Chronicles",
            "author": "Dr. Cassandra Mitchell",
        },
        page_content="An in-depth exploration of the fascinating journey of artificial intelligence, narrated by Dr. Mitchell. This captivating account spans the historical roots, current advancements, and speculative futures of AI, offering a gripping narrative that intertwines technology, ethics, and societal implications.",
    ),
    Document(
        metadata={
            "title": "Synergy Nexus: Merging Minds with Machines",
            "author": "Prof. Benjamin S. Anderson",
        },
        page_content="Professor Anderson delves into the synergistic possibilities of human-machine collaboration in 'Synergy Nexus.' The book articulates a vision where humans and AI seamlessly coalesce, creating new dimensions of productivity, creativity, and shared intelligence.",
    ),
    Document(
        metadata={
            "title": "AI Dilemmas: Navigating the Unknown",
            "author": "Dr. Elena Rodriguez",
        },
        page_content="Dr. Rodriguez pens an intriguing narrative in 'AI Dilemmas,' probing the uncharted territories of ethical quandaries arising from AI advancements. The book serves as a compass, guiding readers through the complex terrain of moral decisions confronting developers, policymakers, and society as AI evolves.",
    ),
    Document(
        metadata={
            "title": "Sentient Threads: Weaving AI Consciousness",
            "author": "Prof. Alexander J. Bennett",
        },
        page_content="In 'Sentient Threads,' Professor Bennett unravels the enigma of AI consciousness, presenting a tapestry of arguments that scrutinize the very essence of machine sentience. The book ignites contemplation on the ethical and philosophical dimensions surrounding the quest for true AI awareness.",
    ),
    Document(
        metadata={
            "title": "Silent Alchemy: Unseen AI Alleviations",
            "author": "Dr. Emily Foster",
        },
        page_content="Building upon her previous work, Dr. Foster unveils 'Silent Alchemy,' a profound examination of the covert presence of AI in our daily lives. This illuminating piece reveals the subtle yet impactful ways in which AI invisibly shapes our routines, emphasizing the need for heightened awareness in our technology-driven world.",
    ),
]

qdrant= QdrantVectorStore(
    client=client,
    collection_name="Qdrant_collection_2",
    embedding=embedding,
    retrieval_mode=RetrievalMode.HYBRID,
    sparse_embedding=sparse_embeddings,
    vector_name=dense_vector_name,
    sparse_vector_name=sparse_vector_name
)

qdrant.add_documents(documents=docs)
prompt = PromptTemplate(
    input_variables=["query", "context"],
    template="""You are an expert teacher. Help me with the following question:

**Question**: {query}

**Context from Documents**:
{context}

**Answer**:
"""
)

query= "AI ethics dilemmas in machine consciousness"


retriever=qdrant.as_retriever(search_kwargs={"k":3})
retrieved_docs = retriever.invoke(query)

print(f"📚 Retrieved {len(retrieved_docs)} documents:")
for doc in retrieved_docs:
    print(f"   - {doc.metadata.get('title', 'No title')}")

context = "\n\n".join([
    f"**{doc.metadata.get('title', 'Document')}** (by {doc.metadata.get('author', 'Unknown')})\n{doc.page_content}"
    for doc in retrieved_docs
])
parser=StrOutputParser()

print(context)

chain=prompt|model|parser

result= chain.invoke({"query":query,"context":context})


print(result)
