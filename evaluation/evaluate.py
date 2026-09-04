import json
import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parent.parent)
)

from app.rag_pipeline import RAGPipeline


def load_questions():
    with open(
        "evaluation/questions.json",
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def evaluate():

    questions = load_questions()

    pipeline = RAGPipeline()

    total = len(questions)
    passed = 0

    print("\nRAG RETRIEVAL EVALUATION")
    print("=" * 50)

    for index, item in enumerate(questions, start=1):

        question = item["question"]
        expected_source = item["expected_source"]
        expected_text = item["expected_text"]

        results = pipeline.retrieve(
            question,
            top_k=5
        )

        source_match = any(
            result["source"] == expected_source
            for result in results
        )

        text_match = any(
            expected_text.lower()
            in result["text"].lower()
            for result in results
        )

        passed_test = source_match and text_match

        if passed_test:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\n[{index}] {status}")
        print(f"Question: {question}")
        print(f"Expected information: {expected_text}")

    accuracy = (
        passed / total * 100
        if total > 0
        else 0
    )

    print("\n" + "=" * 50)
    print(f"Passed: {passed}/{total}")
    print(f"Retrieval accuracy: {accuracy:.1f}%")
    print("=" * 50)


if __name__ == "__main__":
    evaluate()