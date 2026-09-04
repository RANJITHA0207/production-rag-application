from app.ingestion import load_documents, chunk_text
from app.retrieval import HybridRetriever
from app.reranker import CrossEncoderReranker
from app.llm import LLMClient


class RAGPipeline:

    def __init__(self, document_folder="data/documents"):

        # Load documents
        documents = load_documents(document_folder)

        # Create chunks
        self.chunks = []

        for document in documents:
            chunks = chunk_text(document["text"])

            for chunk in chunks:
                self.chunks.append({
                    "source": document["source"],
                    "text": chunk
                })

        # Extract chunk text for retrieval
        chunk_texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # Initialize retrieval components
        self.retriever = HybridRetriever(chunk_texts)
        self.reranker = CrossEncoderReranker()

        # Initialize LLM
        self.llm = LLMClient()

    def retrieve(self, question, top_k=5):

        # Hybrid retrieval
        results = self.retriever.search(question)

        # Cross-encoder reranking
        reranked_results = self.reranker.rerank(
            question,
            results,
            top_k=top_k
        )

        # Add source information
        final_results = []

        for result in reranked_results:

            for chunk in self.chunks:

                if chunk["text"] == result["text"]:

                    final_results.append({
                        "source": chunk["source"],
                        "text": result["text"],
                        "score": result["rerank_score"]
                    })

                    break

        return final_results

    def answer(self, question, top_k=5):

        # Retrieve relevant documents
        results = self.retrieve(
            question,
            top_k=top_k
        )

        # Build context with citation labels
        context_parts = []

        for index, result in enumerate(results, start=1):
            context_parts.append(
                f"[Source {index}] {result['source']}\n"
                f"{result['text']}"
            )

        context = "\n\n".join(context_parts)

        # Generate answer
        answer = self.llm.generate_answer(
            question,
            context
        )

        return {
            "answer": answer,
            "sources": results
        }


if __name__ == "__main__":

    pipeline = RAGPipeline()

    question = "What products does TechFlow Solutions provide?"

    response = pipeline.answer(question)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")

    for index, source in enumerate(
        response["sources"],
        start=1
    ):

        print(f"\n[{index}] {source['source']}")
        print(source["text"])
        print(
            f"Rerank score: "
            f"{source['score']:.4f}"
        )
