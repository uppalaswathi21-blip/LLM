"""Small beginner evaluation set; review answers manually for correctness."""
from chat import answer_question, build_retriever, prompt, CHAT_MODEL
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

TEST_CASES = [
    {
        "question": "How many casual leave days do employees receive?",
        "expected_source": "leave_policy.txt"
    },
    {
        "question": "Who approves a training request?",
        "expected_source": "training_policy.txt"
    },
    {
        "question": "What security rules apply when working remotely?",
        "expected_source": "remote_work_policy.txt"
    },
    {
        "question": "What is the company's travel reimbursement limit?",
        "expected_source": None
    }
]


def main():
    retriever = build_retriever()
    model = ChatOllama(model=CHAT_MODEL, temperature=0)
    chain = prompt | model | StrOutputParser()

    source_hits = 0
    for index, test in enumerate(TEST_CASES, start=1):
        answer, sources, latency_ms = answer_question(
            test["question"], retriever, chain
        )
        expected = test["expected_source"]
        source_ok = expected in sources if expected else (
            "could not find this information" in answer.lower()
        )
        source_hits += int(source_ok)

        print("=" * 70)
        print(f"Test {index}: {test['question']}")
        print(f"Answer: {answer}")
        print(f"Retrieved sources: {sources}")
        print(f"Expected source/behavior: {expected or 'abstain'}")
        print(f"Source/abstention check: {'PASS' if source_ok else 'REVIEW'}")
        print(f"Latency: {latency_ms} ms")

    print("=" * 70)
    print(f"Basic checks passed: {source_hits}/{len(TEST_CASES)}")
    print("These are simple checks, not a complete measure of answer quality.")
    print("Manually review correctness, relevance, and groundedness.")


if __name__ == "__main__":
    main()