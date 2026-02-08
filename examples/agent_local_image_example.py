"""Example of using a LangChain agent with a local image file.

This example uses the same create_agent API as agent_example.py but passes
a local image path to the tools. The tools read the file, convert to base64,
and call the Viscribe API.

Requirements:
    pip install langchain langchain-openai langchain-viscribe

Environment variables:
    VISCRIBE_API_KEY: Your ViscribeAI API key (loaded from .env file)
    OPENAI_API_KEY: Your OpenAI API key (loaded from .env file)
"""

from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from langchain_viscribe.tools import (
    AskImageTool,
    ClassifyImageTool,
    CompareImagesTool,
    DescribeImageTool,
    ExtractImageTool,
    GetCreditsTool,
)

# Local image path (absolute)
IMAGE_PATH = "/Users/mperini/Projects/langchain-viscribe/examples/cat.png"

# Load environment variables from .env file
load_dotenv()


def main():
    """Run the image analysis agent with the local image file."""

    path = Path(IMAGE_PATH)
    if not path.exists():
        print(f"Image not found: {path}")
        print("Please ensure the file exists.")
        return

    # Initialize all available tools (they support image_path)
    tools = [
        DescribeImageTool(),
        AskImageTool(),
        ClassifyImageTool(),
        ExtractImageTool(),
        CompareImagesTool(),
        GetCreditsTool(),
    ]

    system_prompt = (
        "You are a helpful AI assistant with advanced image analysis capabilities. "
        "You can describe images, answer questions about them, classify them, "
        "extract structured data, and compare multiple images. "
        "When the user gives you a path to a local image file, use the image_path "
        "parameter in your tools (not image_url). Use your tools to provide accurate "
        "and detailed information. Always analyze the tool responses and provide a final answer."
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    # Example 1: Describe the local image
    print("\n" + "=" * 80)
    print("Example 1: Describe the local image")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content=f"Describe this image in detail. Use the image at this path: {IMAGE_PATH}")
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 2: Ask about the local image
    print("\n" + "=" * 80)
    print("Example 2: Visual Question Answering (local image)")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content=f"Look at the image at {IMAGE_PATH} and tell me what the main subject is and what colors you see.")
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 3: Classify the local image
    print("\n" + "=" * 80)
    print("Example 3: Classify the local image")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content=f"Classify the image at {IMAGE_PATH} into one or more of: cat, dog, bird, fish, wildlife, pet, indoor, outdoor.")
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 4: Multi-step analysis of the local image
    print("\n" + "=" * 80)
    print("Example 4: Multi-step analysis (local image)")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content=(
                f"Analyze the image at {IMAGE_PATH} and provide: "
                "1) A detailed description, 2) The main colors, and 3) Whether it looks like indoors or outdoors."
            ))
        ]
    })
    print(f"\nAgent Response: {result}")


if __name__ == "__main__":
    main()
