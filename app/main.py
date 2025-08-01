from rag_engine.rag_engine import initialize_rag_pipeline, process_query

def main():
    question = "Latest on Trump"  # Replace with any input or integrate with a CLI
    print(f"🔍 Query: {question}")

    rag_pipeline = initialize_rag_pipeline()
    chat_history = []

    answer, _ = process_query(question, rag_pipeline, chat_history)

    print("\n📢 Final Answer:\n", answer)

if __name__ == "__main__":
    main()
