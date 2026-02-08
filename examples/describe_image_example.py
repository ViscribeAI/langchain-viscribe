"""Example of using DescribeImageTool.

This example shows how to use the DescribeImageTool to generate
natural language descriptions and tags for images.

Requirements:
    pip install langchain-viscribe

Environment variables:
    VISCRIBE_API_KEY: Your ViscribeAI API key (loaded from .env file)
"""

from dotenv import load_dotenv

from langchain_viscribe.tools import DescribeImageTool

# Load environment variables from .env file
load_dotenv()


def main():
    """Demonstrate DescribeImageTool usage."""

    # Initialize the tool
    tool = DescribeImageTool()

    # Example 1: Describe an image from URL
    print("=" * 80)
    print("Example 1: Describe image from URL")
    print("=" * 80)
    result = tool.invoke({
        "image_url": "https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg",
        "generate_tags": True,
    })
    print(f"Description: {result['image_description']}")
    print(f"Tags: {', '.join(result['tags'])}")
    print(f"Credits used: {result['credits_used']}")

    # Example 2: Describe with custom instruction
    print("\n" + "=" * 80)
    print("Example 2: Describe with custom instruction")
    print("=" * 80)
    result = tool.invoke({
        "image_url": "https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg",
        "instruction": "Focus on the colors and lighting in this image",
        "generate_tags": False,
    })
    print(f"Description: {result['image_description']}")

    # Example 3: Describe without tags
    print("\n" + "=" * 80)
    print("Example 3: Description only (no tags)")
    print("=" * 80)
    result = tool.invoke({
        "image_url": "https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg",
        "generate_tags": False,
    })
    print(f"Description: {result['image_description']}")

    # Example 4: Async usage
    print("\n" + "=" * 80)
    print("Example 4: Async usage")
    print("=" * 80)
    import asyncio

    async def async_describe():
        result = await tool.ainvoke({
            "image_url": "https://images.squarespace-cdn.com/content/v1/607f89e638219e13eee71b1e/1684821560422-SD5V37BAG28BURTLIXUQ/michael-sum-LEpfefQf4rU-unsplash.jpg",
            "generate_tags": True,
        })
        return result

    result = asyncio.run(async_describe())
    print(f"Description: {result['image_description']}")
    print(f"Tags: {', '.join(result['tags'])}")


if __name__ == "__main__":
    main()
