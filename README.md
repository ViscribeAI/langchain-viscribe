# langchain-viscribe

AI-powered image analysis tools for LangChain. Seamlessly integrate ViscribeAI's image understanding capabilities into your LangChain agents.

## Installation

```bash
pip install langchain-viscribe
```

For the agent example below you also need:

```bash
pip install langchain langchain-openai
export OPENAI_API_KEY="your-openai-key"
```

## First example: agent comparing a URL image and a local image

Minimal script using LangChain’s `create_agent` with `CompareImagesTool` to compare an image from a URL and a local file:

```python
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_viscribe.tools import CompareImagesTool

load_dotenv()

tools = [CompareImagesTool()]
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
agent = create_agent(
    llm,
    tools=tools,
    system_prompt="You compare images. Use the CompareImages tool when the user asks to compare two images.",
)

result = agent.invoke({
    "messages": [
        HumanMessage(content=(
            "Compare the image at https://example.com/photo.jpg with the local image at /path/to/my/image.png "
            "and describe similarities and differences."
        ))
    ]
})

# Final answer is the last AI message
print(result["messages"][-1].content)
```

Replace the URL and `/path/to/my/image.png` with your image URL and local file path. The tool accepts `image1_url` / `image2_url`, `image1_base64` / `image2_base64`, or `image1_path` / `image2_path`.

## Quick Start

```python
from langchain_viscribe.tools import DescribeImageTool

tool = DescribeImageTool()

# From URL
result = tool.invoke({
    "image_url": "https://example.com/image.jpg",
    "generate_tags": True
})

# From local file (read and sent as base64)
result = tool.invoke({
    "image_path": "/path/to/your/image.png",
    "generate_tags": True
})

print(result["image_description"])
print(result["tags"])
```

## Configuration

### Option 1: Using .env file (Recommended)

Create a `.env` file in your project root:

```bash
cp .env.example .env
```

Then add your API key to `.env`:

```
VISCRIBE_API_KEY=vscrb-your-api-key-here
```

The tools will automatically load the environment variables from `.env`.

### Option 2: Export environment variable

```bash
export VISCRIBE_API_KEY="vscrb-your-api-key-here"
```

### Option 3: Pass directly to tools

```python
tool = DescribeImageTool(api_key="vscrb-your-api-key-here")
```

## Available Tools

### DescribeImageTool
Generate natural language descriptions and tags for images.

### ExtractImageTool
Extract structured data from images (receipts, forms, documents).

### ClassifyImageTool
Classify images into predefined categories (single or multi-label).

### AskImageTool
Visual Question Answering - ask questions about images.

### CompareImagesTool
Compare two images and describe similarities and differences.

### GetCreditsTool
Check your remaining API credits.

### SubmitFeedbackTool
Submit feedback on API responses to improve quality.

## Examples

Example scripts are in the [examples/](examples/) directory.

### Individual tools

| Script | Description |
|--------|-------------|
| [describe_image_example.py](examples/describe_image_example.py) | Describe images and get tags (URL input) |
| [local_image_example.py](examples/local_image_example.py) | Use tools with a **local image file** (`image_path`); uses `examples/cat.png` |

Run: `python examples/describe_image_example.py` or `python examples/local_image_example.py`

### Agent integration

For agent examples, install the LangChain stack and set `OPENAI_API_KEY`:

```bash
pip install langchain langchain-openai
export OPENAI_API_KEY="your-openai-key"
```

| Script | Description |
|--------|-------------|
| [agent_example.py](examples/agent_example.py) | Agent with `create_agent` and image tools (image URLs) |
| [agent_local_image_example.py](examples/agent_local_image_example.py) | Same agent using a **local image** (`examples/cat.png`) via `image_path` |

Run: `python examples/agent_example.py` or `python examples/agent_local_image_example.py`

Agent examples use the current [LangChain Agents API](https://docs.langchain.com/oss/python/langchain/agents) (`create_agent`, message-based state).

## Documentation

For comprehensive documentation, visit [https://docs.viscribe.ai](https://docs.viscribe.ai)

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Support

- Issues: [https://github.com/viscribeai/langchain-viscribe/issues](https://github.com/viscribeai/langchain-viscribe/issues)
- Documentation: [https://docs.viscribe.ai](https://docs.viscribe.ai)
- Website: [https://viscribe.ai](https://viscribe.ai)
