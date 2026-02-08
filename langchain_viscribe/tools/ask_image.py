"""Tool for Visual Question Answering on images."""

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


class AskImageInput(BaseModel):
    """Input schema for AskImageTool."""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to ask about. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image string. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to a local image file. The file will be read and sent as base64. Provide exactly one of image_url, image_base64, or image_path.",
    )
    question: str = Field(
        description="Question to ask about the image (e.g., 'What color is the car?', 'How many people are in this image?', 'Is this indoors or outdoors?').",
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


class AskImageTool(BaseTool):
    """Tool for Visual Question Answering (VQA) on images.

    This tool enables you to ask natural language questions about images
    and receive accurate answers. It's useful for interactive image analysis,
    extracting specific information, and building conversational agents that
    can understand visual content.

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

            from langchain_viscribe.tools import AskImageTool

            tool = AskImageTool()
            # Or with explicit API key
            tool = AskImageTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            # Ask about colors
            result = tool.invoke({
                "image_url": "https://example.com/car.jpg",
                "question": "What color is the car?"
            })
            print(result["answer"])

            # Count objects
            result = tool.invoke({
                "image_url": "https://example.com/group.jpg",
                "question": "How many people are in this image?"
            })

            # Identify location/setting
            result = tool.invoke({
                "image_base64": "base64_encoded_string",
                "question": "Is this indoors or outdoors?"
            })

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({
                "image_url": "https://example.com/image.jpg",
                "question": "What is the main subject?"
            })

    Tool response format:
        {
            "request_id": "unique-request-id",
            "credits_used": 1,
            "answer": "The answer to your question..."
        }
    """

    name: str = "AskImage"
    description: str = (
        "Ask questions about an image and get natural language answers (Visual Question Answering). "
        "Useful for extracting specific information from images, interactive analysis, "
        "and understanding visual content. Takes an image URL, a base64-encoded image, or the path to a local image file."
    )
    args_schema: Type[BaseModel] = AskImageInput
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
        question: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Ask a question about an image.

        Args:
            image_url: URL of the image
            image_base64: Base64 encoded image string
            image_path: Path to a local image file (read and sent as base64)
            question: Question to ask about the image
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and answer
        """
        if not self.client:
            raise ValueError("Client not initialized")

        if image_path:
            image_base64 = load_image_path_to_base64(image_path)
            image_url = None

        response = self.client.ask_image(
            image_url=image_url,
            image_base64=image_base64,
            question=question,
        )

        return {
            "request_id": response.request_id,
            "credits_used": response.credits_used,
            "answer": response.answer,
        }

    async def _arun(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
        question: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Ask a question about an image asynchronously.

        Args:
            image_url: URL of the image
            image_base64: Base64 encoded image string
            image_path: Path to a local image file (read and sent as base64)
            question: Question to ask about the image
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and answer
        """
        return self._run(
            image_url=image_url,
            image_base64=image_base64,
            image_path=image_path,
            question=question,
            run_manager=run_manager.get_sync() if run_manager else None,
        )
