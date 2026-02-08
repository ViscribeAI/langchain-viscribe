"""Example of using ViscribeAI tools with a local image file.

This example shows how to pass a local image path to the tools. Each tool
reads the file, converts it to base64, and sends it to the Viscribe API.

Requirements:
    pip install langchain-viscribe

Environment variables:
    VISCRIBE_API_KEY: Your ViscribeAI API key (loaded from .env file)
"""

from pathlib import Path

from dotenv import load_dotenv

from langchain_viscribe.tools import (
    AskImageTool,
    ClassifyImageTool,
    DescribeImageTool,
)

# Path to cat.png next to this script (works when run from repo root or examples/)
IMAGE_PATH = str((Path(__file__).resolve().parent / "cat.png"))

# Load environment variables from .env file
load_dotenv()


def main():
    """Run tools using the local image file examples/cat.png."""

    # Resolve path and check file exists
    path = Path(IMAGE_PATH)
    if not path.exists():
        print(f"Image not found: {path}")
        print("Please ensure examples/cat.png exists.")
        return

    # Example 1: Describe the local image
    print("=" * 80)
    print("Example 1: Describe local image (image_path)")
    print("=" * 80)
    tool = DescribeImageTool()
    result = tool.invoke({
        "image_path": IMAGE_PATH,
        "generate_tags": True,
    })
    print(f"Description: {result['image_description']}")
    print(f"Tags: {', '.join(result['tags'])}")
    print(f"Credits used: {result['credits_used']}")

    # Example 2: Ask a question about the local image
    print("\n" + "=" * 80)
    print("Example 2: Ask about local image (image_path)")
    print("=" * 80)
    ask_tool = AskImageTool()
    result = ask_tool.invoke({
        "image_path": IMAGE_PATH,
        "question": "What is the main subject of this image and what colors are prominent?",
    })
    print(f"Answer: {result['answer']}")

    # Example 3: Classify the local image
    print("\n" + "=" * 80)
    print("Example 3: Classify local image (image_path)")
    print("=" * 80)
    import json

    classify_tool = ClassifyImageTool()
    result = classify_tool.invoke({
        "image_path": IMAGE_PATH,
        "classes": json.dumps(["cat", "dog", "bird", "wildlife", "pet", "outdoor", "indoor"]),
        "multi_label": True,
    })
    print(f"Classification: {result['classification']}")


if __name__ == "__main__":
    main()
