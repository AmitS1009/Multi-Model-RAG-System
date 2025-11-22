import subprocess
import sys
import os

def run_command(command, description):
    print(f"\n{description}")
    print(f"Running: {command}")

    result = subprocess.run(command, shell=True)

    if result.returncode != 0:
        print("\nError running command.")
        return False

    print("Completed.")
    return True


def main():
    print()
    print("Multi-Modal RAG Pipeline")
    print()

    steps = [
        ("python config.py", "Step 0: Creating directories"),
        ("python process_document.py", "Step 1: Extracting document data"),
        ("python create_embeddings.py", "Step 2: Creating embeddings")
    ]

    for command, description in steps:
        if not run_command(command, description):
            print("\nPipeline failed.")
            sys.exit(1)

    print()
    print("PIPELINE COMPLETE")


if __name__ == "__main__":
    main()
