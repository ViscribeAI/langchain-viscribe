"""Tool for generating natural language descriptions of images."""

from typing import Dict, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, Field, model_validator
from viscribe import Client

from langchain_viscribe.tools._utils import load_image_path_to_base64


class DescribeImageInput(BaseModel):
    """Input schema for DescribeImageTool."""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to describe. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image string. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to a local image file. The file will be read and sent as base64. Provide exactly one of image_url, image_base64, or image_path.",
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Optional instruction to guide the description generation (e.g., 'Focus on colors and mood', 'Describe the people in detail').",
    )
    generate_tags: bool = Field(
        default=True,
        description="Whether to generate tags for the image in addition to the description.",
    )

    @model_validator(mode="after")
    def check_image_source(self):
        """Ensure exactly one image source is provided."""
        sources = sum([bool(self.image_url), bool(self.image_base64), bool(self.image_path)])
        if sources == 0:
            raise ValueError("Provide exactly one of image_url, image_base64, or image_path")
        if sources > 1:
            raise ValueError("Provide only one of image_url, image_base64, or image_path")
        return self


class DescribeImageTool(BaseTool):
    """Tool for generating natural language descriptions and tags for images.

    This tool uses ViscribeAI's image analysis API to generate comprehensive
    descriptions of images and optional tags. It's useful for accessibility,
    content understanding, image cataloging, and creating alt text.

    Setup:
        Install langchain-viscribe and set your API key:

        .. code-block:: bash

            pip install langchain-viscribe
            export VISCRIBE_API_KEY="vscrb-your-api-key"

    Key init args:
        api_key: str
            ViscribeAI API key. If not provided, will be read from VISCRIBE_API_KEY environment variable.
        client: Optional[Client]
            ViscribeAI client instance. Automatically initialized if not provided.

    Instantiate:
        .. code-block:: python

            from langchain_viscribe.tools import DescribeImageTool

            tool = DescribeImageTool()
            # Or with explicit API key
            tool = DescribeImageTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            # With image URL
            result = tool.invoke({
                "image_url": "https://example.com/image.jpg",
                "generate_tags": True
            })
            print(result["image_description"])
            print(result["tags"])

            # With base64 image
            result = tool.invoke({
                "image_base64": "base64_encoded_string_here",
                "instruction": "Focus on the main subject"
            })

            # With custom instruction
            result = tool.invoke({
                "image_url": "https://example.com/photo.jpg",
                "instruction": "Describe the colors and mood of this image",
                "generate_tags": False
            })

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({
                "image_url": "https://example.com/image.jpg"
            })

    Tool response format:
        {
            "request_id": "unique-request-id",
            "credits_used": 1,
            "image_description": "A detailed description of the image...",
            "tags": ["tag1", "tag2", "tag3"]  # if generate_tags=True
        }
    """

    name: str = "DescribeImage"
    description: str = (
        "Generate natural language descriptions and tags for an image. "
        "Useful for understanding image content, creating alt text, accessibility, "
        "and image cataloging. Takes an image URL, a base64-encoded image, or the path to a local image file."
    )
    args_schema: Type[BaseModel] = DescribeImageInput
    return_direct: bool = True
    client: Optional[Client] = None
    api_key: str

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that API key exists and initialize client."""
        values["api_key"] = get_from_dict_or_env(values, "api_key", "VISCRIBE_API_KEY")
        values["client"] = Client(api_key=values["api_key"])
        return values

    def _run(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
        instruction: Optional[str] = None,
        generate_tags: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Generate description for an image.

        Args:
            image_url: URL of the image to describe
            image_base64: Base64 encoded image string
            image_path: Path to a local image file (read and sent as base64)
            instruction: Optional instruction to guide description
            generate_tags: Whether to generate tags
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, image_description, and tags
        """
        if not self.client:
            raise ValueError("Client not initialized")

        if image_path:
            image_base64 = load_image_path_to_base64(image_path)
            image_url = None

        response = self.client.describe_image(
            image_url=image_url,
            image_base64=image_base64,
            instruction=instruction,
            generate_tags=generate_tags,
        )

        return {
            "request_id": response.request_id,
            "credits_used": response.credits_used,
            "image_description": response.image_description,
            "tags": response.tags,
        }

    async def _arun(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
        instruction: Optional[str] = None,
        generate_tags: bool = True,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Generate description asynchronously.

        Args:
            image_url: URL of the image to describe
            image_base64: Base64 encoded image string
            image_path: Path to a local image file (read and sent as base64)
            instruction: Optional instruction to guide description
            generate_tags: Whether to generate tags
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, image_description, and tags
        """
        return self._run(
            image_url=image_url,
            image_base64=image_base64,
            image_path=image_path,
            instruction=instruction,
            generate_tags=generate_tags,
            run_manager=run_manager.get_sync() if run_manager else None,
        )
