import argparse
import sys
from main import answer_questions_from_url, llm, embedding_model

# --- Command-Line Interface ---
def main():
    """
    A command-line tool to ask questions about a PDF document from a URL.
    """
    # This check ensures that the foundational models from main.py were loaded correctly.
    if not llm or not embedding_model:
        print("\n--- FATAL ERROR ---")
        print("The foundational models (LLM or embedding model) failed to load.")
        print("Please check the startup logs of the main application for errors.")
        sys.exit(1) # Exit with an error code

    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF document available at a URL.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "doc_url",
        type=str,
        help="The full URL of the PDF document to analyze."
    )
    parser.add_argument(
        "-q", "--question",
        action='append',
        required=True,
        help="A question to ask about the document. You can provide this argument multiple times for multiple questions."
    )

    args = parser.parse_args()

    try:
        # Call the core logic function with the provided arguments
        answers = answer_questions_from_url(args.doc_url, args.question)
        
        # Print the results in a clean, readable format
        print("--- ANSWERS ---")
        for i, (question, answer) in enumerate(zip(args.question, answers)):
            print(f"\nQ{i+1}: {question}")
            print(f"A{i+1}: {answer}")
        print("\n---------------")

    except Exception as e:
        print(f"\n--- An error occurred ---", file=sys.stderr)
        print(f"{e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
