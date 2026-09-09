from fastapi import APIRouter

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Prefetch,
    FusionQuery,
    Fusion,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
)

from sentence_transformers import SentenceTransformer, CrossEncoder

from app.schemas.chat import ChatRequest, ChatResponse
from app.core.config import settings

from google import genai

# Sparse embedding model
from fastembed import SparseTextEmbedding


route = APIRouter(
    prefix="/chat",
    tags=["Chat"]
)


# ============================================================
# 1. Gemini
# ============================================================

client = genai.Client(
    api_key=settings.GEMINI_API_KEY
)


# ============================================================
# 2. Qdrant
# ============================================================

qdrant = QdrantClient(
    url="http://localhost:6333"
)


COLLECTION_NAME = "learning_rag-demo"


# ============================================================
# 3. Embedding Models
# ============================================================

# Dense model
dense_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


# Sparse model
sparse_model = SparseTextEmbedding(
    model_name="prithivida/Splade_PP_en_v1"
)


# ============================================================
# 4. Cross Encoder
# ============================================================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)


# ============================================================
# 5. Optional Metadata Filter
# ============================================================

# Example:
#
# query_filter = Filter(
#     must=[
#         FieldCondition(
#             key="department",
#             match=MatchValue(value="IT")
#         ),
#         FieldCondition(
#             key="document_type",
#             match=MatchValue(value="nodejs")
#         ),
#         FieldCondition(
#             key="year",
#             match=MatchValue(value="2026")
#         )
#     ]
# )


# ============================================================
# 6. Chat API
# ============================================================

@route.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest):

    # --------------------------------------------------------
    # 1. Get user question
    # --------------------------------------------------------

    question = request.question

    print("\n========================================")
    print("User Question:", question)
    print("========================================")


    # --------------------------------------------------------
    # 2. Create Dense Query Vector
    # --------------------------------------------------------

    dense_vector = dense_model.encode(
        question
    ).tolist()


    # --------------------------------------------------------
    # 3. Create Sparse Query Vector
    # --------------------------------------------------------

    sparse_embedding = list(
        sparse_model.embed([question])
    )[0]


    sparse_vector = SparseVector(
        indices=sparse_embedding.indices.tolist(),
        values=sparse_embedding.values.tolist()
    )


    # --------------------------------------------------------
    # 4. Hybrid Search
    # --------------------------------------------------------

    results = qdrant.query_points(

        collection_name=COLLECTION_NAME,

        # Dense + Sparse retrieval
        prefetch=[

            # -------------------------
            # Dense Search
            # -------------------------

            Prefetch(
                query=dense_vector,
                using="dense",
                limit=20
            ),

            # -------------------------
            # Sparse Search
            # -------------------------

            Prefetch(
                query=sparse_vector,
                using="sparse",
                limit=20
            ),
        ],

        # -------------------------
        # RRF Fusion
        # -------------------------

        query=FusionQuery(
            fusion=Fusion.RRF
        ),

        # query_filter=query_filter,

        # Final hybrid candidates
        limit=20,

        # Return payload
        with_payload=True
    )


    hybrid_results = results.points

    print(
        "Hybrid Retrieved Chunks:",
        len(hybrid_results)
    )


    # --------------------------------------------------------
    # 5. Convert Qdrant results to text
    # --------------------------------------------------------

    documents = []

    for result in hybrid_results:

        payload = result.payload or {}

        text = payload.get(
            "text",
            ""
        )

        documents.append(
            {
                "text": text,
                "page": payload.get(
                    "page",
                    "Unknown"
                ),
                "document_id": payload.get(
                    "document_id"
                ),
                "score": result.score
            }
        )


    # --------------------------------------------------------
    # 6. Cross Encoder Reranking
    # --------------------------------------------------------

    pairs = [
        (
            question,
            document["text"]
        )
        for document in documents
    ]


    scores = reranker.predict(
        pairs
    )


    # --------------------------------------------------------
    # 7. Combine documents + reranker scores
    # --------------------------------------------------------

    ranked_results = sorted(
        zip(documents, scores),
        key=lambda x: x[1],
        reverse=True
    )


    # --------------------------------------------------------
    # 8. Select Top 5
    # --------------------------------------------------------

    top_n = 5

    final_results = ranked_results[:top_n]


    print(
        "Final Chunks After Reranking:",
        len(final_results)
    )


    # --------------------------------------------------------
    # 9. Debug Results
    # --------------------------------------------------------

    for document, score in final_results:

        print(
            "Reranker Score:",
            score
        )

        print(
            "PDF Page:",
            document["page"]
        )

        print(
            "Text:",
            document["text"][:200]
        )

        print("----------------------------------------")


    # --------------------------------------------------------
    # 10. Build Context
    # --------------------------------------------------------

    context = ""


    for document, score in final_results:

        page = document["page"]

        text = document["text"]

        context += f"""

[PDF Page: {page}]

{text}

"""


    # --------------------------------------------------------
    # 11. Build Prompt
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # 12. Send to Gemini
    # --------------------------------------------------------

    response = client.models.generate_content(

        model="gemini-3.6-flash",

        contents=prompt
    )


    # --------------------------------------------------------
    # 13. Get Answer
    # --------------------------------------------------------

    answer = response.text


    print("\n================ ANSWER ================\n")

    print(answer)


    # --------------------------------------------------------
    # 14. Return Response
    # --------------------------------------------------------

    return ChatResponse(
        answer=answer
    )