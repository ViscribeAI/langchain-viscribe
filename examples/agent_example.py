"""Example of using ViscribeAI tools with a LangChain agent.

This example uses the current LangChain agent API (create_agent) as documented
at https://docs.langchain.com/oss/python/langchain/agents.

Requirements:
    pip install langchain langchain-openai langchain-viscribe

Environment variables:
    VISCRIBE_API_KEY: Your ViscribeAI API key (loaded from .env file)
    OPENAI_API_KEY: Your OpenAI API key (loaded from .env file)
"""

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage

from langchain_viscribe.tools import (
    AskImageTool,
    ClassifyImageTool,
    CompareImagesTool,
    DescribeImageTool,
    ExtractImageTool,
    GetCreditsTool,
)

# Load environment variables from .env file
load_dotenv()

def main():
    """Run the image analysis agent."""

    # Initialize all available tools
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
        "Use your tools to provide accurate and detailed information about images."
        "always analyze the tool responses and provide a final answer"
    )

    # Create the LLM and agent using the current LangChain API
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    agent = create_agent(
        llm,
        tools=tools,
        system_prompt=system_prompt,
    )

    # Example 1: Describe an image
    print("\n" + "=" * 80)
    print("Example 1: Describe an image")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Describe this image in detail: https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg")
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 2: Ask specific questions about an image
    print("\n" + "=" * 80)
    print("Example 2: Visual Question Answering")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Look at this image and tell me what color the cat is: https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg")
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 3: Classify an image
    print("\n" + "=" * 80)
    print("Example 3: Image Classification")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content="Classify this image into one of these categories: cat, dog, bird, fish - https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg")
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 4: Complex multi-step task
    print("\n" + "=" * 80)
    print("Example 4: Multi-step Analysis")
    print("=" * 80)
    result = agent.invoke({
        "messages": [
            HumanMessage(content=(
                "Analyze this image and provide: 1) A detailed description, "
                "2) The main colors present, and 3) Whether it's indoors or outdoors. "
                "Image URL: https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg"
            ))
        ]
    })
    print(f"\nAgent Response: {result}")

    # Example 5: Check credits
    print("\n" + "=" * 80)
    print("Example 5: Check API Credits")
    print("=" * 80)
    result = agent.invoke({
        "messages": [HumanMessage(content="How many API credits do I have remaining?")]
    })
    print(f"\nAgent Response: {result}")


if __name__ == "__main__":
    main()
