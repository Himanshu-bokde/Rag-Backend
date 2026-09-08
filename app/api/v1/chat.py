from fastapi import APIRouter
from qdrant_client.models import Filter, FieldCondition, MatchValue
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from sentence_transformers import CrossEncoder

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.config import settings

from google import genai


route = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


# query_filter = Filter(
#     must=[
#         FieldCondition(
#             key="metadata.department",
#             match=MatchValue(value="IT")
#         ),
#         FieldCondition(
#             key="metadata.document_type",
#             match=MatchValue(value="nodejs")
#         ),
#         FieldCondition(
#             key="metadata.year",
#             match=MatchValue(value=2026)
#         )
#     ]
# )


# Gemini client
client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


# Embedding model
embeddings_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# Connect to existing Qdrant collection
vector_db = QdrantVectorStore.from_existing_collection(
    url="http://localhost:6333",
    collection_name="learning_rag-demo",
    embedding=embeddings_model,
)


@route.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):

    # --------------------------------
    # 1. Get question from user
    # --------------------------------

    question = request.question

    print("User Question:", question)


    # --------------------------------
    # 2. Search Qdrant
    # --------------------------------

    search_results = vector_db.similarity_search(
        query=question,
        k=20,
        # filter=query_filter
    )

    print("Retrieved Chunks:", len(search_results))


# --------------------------------
# 3. Rerank retrieved chunks
# --------------------------------
    
    pairs = [
        (question,doc.page_content)
        for doc in search_results
    ]

    scores = reranker.predict(pairs)


    ranked_result = sorted(
        zip(search_results,scores),
        key=lambda x:x[1],
        reverse=True
    )

    top_n = 5

    final_results = ranked_result[:top_n]

    print("Final Chunks After Reranking:", len(final_results))

    for doc, score in final_results:
       print("Score:", score)
       print("Page:", doc.metadata.get("page"))
       print()


    # --------------------------------
    # 4. Build context
    # --------------------------------

    # context = ""

    # for doc in search_results:
    #     page_content = doc.page_content
    #     context += f"""
    #       {page_content}
    #     """


    context = ""

    for doc, score in final_results:

       page = doc.metadata.get("page", "Unknown")

       context += f"""
        [PDF Page: {page}]
        {doc.page_content}
        """


    # --------------------------------
    # 4. Create prompt
    # --------------------------------

    prompt = f"""
You are a helpful AI assistant.

Answer the user's question ONLY using the context retrieved
from the PDF.

Rules:

1. Use ONLY the provided context.
2. Do NOT use your own knowledge.
3. Do NOT make up information.
4. If the answer is not available in the context, say:
   "I could not find the answer in the provided PDF."
5. Always mention the PDF page number where the answer was found.
6. Tell the user which PDF page they can open to learn more.
7. Keep the answer clear and concise.

---------------- CONTEXT ----------------

{context}

---------------- END CONTEXT ----------------

---------------- USER QUESTION ----------------

{question}
"""


    # --------------------------------
    # 5. Send to Gemini
    # --------------------------------

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )


    # --------------------------------
    # 6. Get Gemini answer
    # --------------------------------

    answer = response.text

    print("\n================ ANSWER ================\n")
    print(answer)


    # --------------------------------
    # 7. Return to frontend
    # --------------------------------

    return ChatResponse(
        answer=answer
    )