"""Tool for classifying images into predefined categories."""

import json
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


class ClassifyImageInput(BaseModel):
    """Input schema for ClassifyImageTool."""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to classify. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image string. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to a local image file. The file will be read and sent as base64. Provide exactly one of image_url, image_base64, or image_path.",
    )
    classes: str = Field(
        description='JSON array of class names to classify into. Example: \'["cat", "dog", "bird"]\'',
    )
    class_descriptions: Optional[str] = Field(
        default=None,
        description='Optional JSON object mapping class names to descriptions. Example: \'{"cat": "A feline animal", "dog": "A canine animal"}\'',
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Optional instruction to guide classification (e.g., 'Focus on the main subject', 'Consider the overall scene').",
    )
    multi_label: bool = Field(
        default=False,
        description="If True, allows multiple classes to be returned. If False, returns only the single best match.",
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


class ClassifyImageTool(BaseTool):
    """Tool for classifying images into predefined categories.

    This tool enables single-label or multi-label image classification using
    custom categories. It's useful for content moderation, image organization,
    product categorization, and automated tagging systems.

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

            from langchain_viscribe.tools import ClassifyImageTool

            tool = ClassifyImageTool()
            # Or with explicit API key
            tool = ClassifyImageTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            import json

            # Single-label classification
            result = tool.invoke({
                "image_url": "https://example.com/animal.jpg",
                "classes": json.dumps(["cat", "dog", "bird", "fish"]),
                "multi_label": False
            })
            print(result["classification"])  # ["cat"]

            # Multi-label classification
            result = tool.invoke({
                "image_url": "https://example.com/scene.jpg",
                "classes": json.dumps(["outdoor", "sunny", "beach", "people"]),
                "multi_label": True
            })
            print(result["classification"])  # ["outdoor", "sunny", "beach"]

            # With class descriptions
            result = tool.invoke({
                "image_url": "https://example.com/product.jpg",
                "classes": json.dumps(["electronics", "furniture", "clothing"]),
                "class_descriptions": json.dumps({
                    "electronics": "Devices and electronic equipment",
                    "furniture": "Tables, chairs, and home furnishings",
                    "clothing": "Apparel and accessories"
                })
            })

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({
                "image_url": "https://example.com/image.jpg",
                "classes": json.dumps(["class1", "class2"])
            })

    Tool response format:
        {
            "request_id": "unique-request-id",
            "credits_used": 1,
            "classification": ["class1", "class2"]  # List of matched classes
        }
    """

    name: str = "ClassifyImage"
    description: str = (
        "Classify images into predefined categories. Supports single-label or multi-label classification. "
        "Useful for content moderation, image organization, automated tagging, and product categorization. "
        "Takes an image URL, a base64-encoded image, or the path to a local image file."
    )
    args_schema: Type[BaseModel] = ClassifyImageInput
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
        classes: Optional[str] = None,
        class_descriptions: Optional[str] = None,
        instruction: Optional[str] = None,
        multi_label: bool = False,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Classify an image.

        Args:
            image_url: URL of the image
            image_base64: Base64 encoded image string
            image_path: Path to a local image file (read and sent as base64)
            classes: JSON string of class names
            class_descriptions: JSON string of class descriptions (optional)
            instruction: Optional instruction to guide classification
            multi_label: Allow multiple classes if True
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and classification list
        """
        if not self.client:
            raise ValueError("Client not initialized")

        if image_path:
            image_base64 = load_image_path_to_base64(image_path)
            image_url = None

        if classes is None:
            raise ValueError("classes parameter is required")

        # Parse JSON strings
        parsed_classes = json.loads(classes)
        parsed_descriptions = (
            json.loads(class_descriptions) if class_descriptions else None
        )

        response = self.client.classify_image(
            image_url=image_url,
            image_base64=image_base64,
            classes=parsed_classes,
            class_descriptions=parsed_descriptions,
            instruction=instruction,
            multi_label=multi_label,
        )

        return {
            "request_id": response.request_id,
            "credits_used": response.credits_used,
            "classification": response.classification,
        }

    async def _arun(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
        classes: Optional[str] = None,
        class_descriptions: Optional[str] = None,
        instruction: Optional[str] = None,
        multi_label: bool = False,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Classify an image asynchronously.

        Args:
            image_url: URL of the image
            image_base64: Base64 encoded image string
            classes: JSON string of class names
            class_descriptions: JSON string of class descriptions (optional)
            instruction: Optional instruction to guide classification
            multi_label: Allow multiple classes if True
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and classification list
        """
        return self._run(
            image_url=image_url,
            image_base64=image_base64,
            image_path=image_path,
            classes=classes,
            class_descriptions=class_descriptions,
            instruction=instruction,
            multi_label=multi_label,
            run_manager=run_manager.get_sync() if run_manager else None,
        )
