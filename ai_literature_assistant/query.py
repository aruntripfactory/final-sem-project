#!/usr/bin/env python3
"""
AI Research Assistant - Query Interface
Usage:
  python -m ai_literature_assistant.query "Your question"
  python -m ai_literature_assistant.query --interactive
"""

import os
import sys
import json
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("ERROR: OpenAI API key not found.")
    print("Set it using:")
    print("export OPENAI_API_KEY='your-api-key-here'")
    sys.exit(1)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rag.pipeline import RAGPipeline


# Displays the response in a readable CLI format
# def display_response(response: dict, show_sources: bool = True):
#     print("\n" + "=" * 80)
#     print("AI RESEARCH ASSISTANT RESPONSE")
#     print("=" * 80)

#     if not response.get("success"):
#         print(f"Error: {response.get('error', 'Unknown error')}")
#         return

#     print(f"\nQUESTION: {response['query']}")
#     print(
#         f"Time: {datetime.fromisoformat(response['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}"
#     )
#     print(f"Model: {response.get('model', 'unknown')}")

#     print("\n" + "-" * 80)
#     print("ANSWER:")
#     print("-" * 80)
#     print(response["answer"])

#     if show_sources and response.get("sources"):
#         print("\n" + "-" * 80)
#         print(
#             f"SOURCES ({response['retrieval']['documents_retrieved']} documents):"
#         )
#         print("-" * 80)

#         for i, source in enumerate(response["sources"], 1):
#             print(f"\n[{i}] Document ID: {source['id']}")
#             print(f"    Relevance: {source['similarity']}")

#             if source.get("metadata"):
#                 meta = source["metadata"]
#                 print(f"    Type: {meta.get('chunk_type', 'unknown')}")
#                 print(f"    Paper: {meta.get('paper_id', 'unknown')}")
#                 if meta.get("section") and meta["section"] != "unknown":
#                     print(f"    Section: {meta.get('section')}")

#             print(f"    Preview: {source['content_preview']}")


def display_response(response: dict):
    if not response.get("success"):
        print("Error:", response.get("error", "Unknown error"))
        return

    print("\nANSWER:\n")
    print(response["answer"])


# Saves response to a JSON file
def save_response(response: dict, filename: str = None):
    if filename is None:
        safe_query = "".join(
            c if c.isalnum() else "_" for c in response["query"][:50]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"query_results/{safe_query}_{timestamp}.json"

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(response, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {filename}")


# Runs interactive CLI mode
def interactive_mode(pipeline: RAGPipeline):
    print("\n" + "=" * 80)
    print("INTERACTIVE MODE - AI RESEARCH ASSISTANT")
    print("=" * 80)
    print("Commands:")
    print("  /filter <key> <value>   Set metadata filter")
    print("  /clear                 Clear filters")
    print("  /stats                 Show database stats")
    print("  /save                  Save last response")
    print("  /quit or /exit         Exit")
    print("=" * 80)

    current_filter = None
    last_response = None

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            if not user_input:
                continue

            if user_input.startswith("/"):
                cmd = user_input[1:].split()

                if cmd[0] in ("quit", "exit"):
                    print("Exiting interactive mode.")
                    break

                elif cmd[0] == "filter" and len(cmd) >= 3:
                    current_filter = {cmd[1]: " ".join(cmd[2:])}
                    print(f"Filter set: {current_filter}")
                    continue

                elif cmd[0] == "clear":
                    current_filter = None
                    print("Filter cleared.")
                    continue

                elif cmd[0] == "stats":
                    stats = pipeline.retriever.get_collection_stats()
                    print("\nSystem statistics:")
                    print(f"Total documents: {stats['total_documents']}")
                    if stats.get("chunk_types"):
                        for k, v in stats["chunk_types"].items():
                            print(f"  {k}: {v}")
                    continue

                elif cmd[0] == "save" and last_response:
                    save_response(last_response)
                    continue

                else:
                    print("Unknown command.")
                    continue

            response = pipeline.query(
                question=user_input,
                filter_by=current_filter,
                include_sources=True,
                generate_response=True,
            )

            display_response(response)
            last_response = response

        except KeyboardInterrupt:
            print("\nExiting interactive mode.")
            break
        except Exception as e:
            print(f"Error: {e}")


# Entry point
def main():
    parser = argparse.ArgumentParser(
        description="AI Research Assistant - Query your research papers"
    )

    parser.add_argument("question", nargs="?", help="Your question")
    parser.add_argument("-i", "--interactive", action="store_true")
    parser.add_argument("-f", "--filter", type=str, help="key=value filter")
    parser.add_argument("-r", "--results", type=int, default=5)
    parser.add_argument("--no-chatgpt", action="store_true")
    parser.add_argument("-s", "--save", action="store_true")
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )

    args = parser.parse_args()

    # print("Starting AI Research Assistant...")
    pipeline = RAGPipeline(openai_model=args.model)

    if args.no_chatgpt:
        pipeline.use_chatgpt = False

    filter_by = None
    if args.filter:
        key, value = args.filter.split("=")
        filter_by = {key.strip(): value.strip()}

    if args.interactive:
        interactive_mode(pipeline)

    elif args.question:
        response = pipeline.query(
            question=args.question,
            n_retrieve=args.results,
            filter_by=filter_by,
            include_sources=True,
            generate_response=not args.no_chatgpt,
        )

        display_response(response)

        if args.save or args.output:
            save_response(response, args.output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
