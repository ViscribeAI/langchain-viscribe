"""Tool for comparing two images."""

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


class CompareImagesInput(BaseModel):
    """Input schema for CompareImagesTool."""

    image1_url: Optional[str] = Field(
        default=None,
        description="URL of the first image. Provide exactly one of image1_url, image1_base64, or image1_path for the first image.",
    )
    image1_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded string of the first image. Provide exactly one of image1_url, image1_base64, or image1_path.",
    )
    image1_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to the first image file. The file will be read and sent as base64. Provide exactly one of image1_url, image1_base64, or image1_path.",
    )
    image2_url: Optional[str] = Field(
        default=None,
        description="URL of the second image. Provide exactly one of image2_url, image2_base64, or image2_path for the second image.",
    )
    image2_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded string of the second image. Provide exactly one of image2_url, image2_base64, or image2_path.",
    )
    image2_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to the second image file. The file will be read and sent as base64. Provide exactly one of image2_url, image2_base64, or image2_path.",
    )
    instruction: Optional[str] = Field(
        default="Describe the similarities and differences between these two images.",
        description="Instruction to guide the comparison (e.g., 'Focus on color differences', 'Compare the layout and composition').",
    )

    @model_validator(mode="after")
    def check_images(self):
        """Validate that exactly one source is provided for each image."""
        s1 = sum([bool(self.image1_url), bool(self.image1_base64), bool(self.image1_path)])
        if s1 == 0:
            raise ValueError("For the first image, provide exactly one of image1_url, image1_base64, or image1_path")
        if s1 > 1:
            raise ValueError("For the first image, provide only one of image1_url, image1_base64, or image1_path")
        s2 = sum([bool(self.image2_url), bool(self.image2_base64), bool(self.image2_path)])
        if s2 == 0:
            raise ValueError("For the second image, provide exactly one of image2_url, image2_base64, or image2_path")
        if s2 > 1:
            raise ValueError("For the second image, provide only one of image2_url, image2_base64, or image2_path")
        return self


class CompareImagesTool(BaseTool):
    """Tool for comparing two images and describing their similarities and differences.

    This tool enables side-by-side comparison of images to identify similarities,
    differences, changes, or relationships. Useful for quality control, change
    detection, A/B testing analysis, and visual comparison tasks.

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

            from langchain_viscribe.tools import CompareImagesTool

            tool = CompareImagesTool()
            # Or with explicit API key
            tool = CompareImagesTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            # Compare two images from URLs
            result = tool.invoke({
                "image1_url": "https://example.com/before.jpg",
                "image2_url": "https://example.com/after.jpg"
            })
            print(result["comparison_result"])

            # With custom instruction
            result = tool.invoke({
                "image1_url": "https://example.com/design-v1.jpg",
                "image2_url": "https://example.com/design-v2.jpg",
                "instruction": "Focus on the layout and color scheme differences"
            })

            # Mix URLs and base64
            result = tool.invoke({
                "image1_url": "https://example.com/original.jpg",
                "image2_base64": "base64_encoded_string_here",
                "instruction": "Identify any changes or modifications"
            })

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({
                "image1_url": "https://example.com/image1.jpg",
                "image2_url": "https://example.com/image2.jpg"
            })

    Tool response format:
        {
            "request_id": "unique-request-id",
            "credits_used": 1,
            "comparison_result": "Detailed comparison of the two images..."
        }
    """

    name: str = "CompareImages"
    description: str = (
        "Compare two images and describe their similarities and differences. "
        "Useful for change detection, quality control, A/B testing, and visual comparison tasks. "
        "Each image can be provided as a URL, a base64-encoded string, or the path to a local image file."
    )
    args_schema: Type[BaseModel] = CompareImagesInput
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
        image1_url: Optional[str] = None,
        image1_base64: Optional[str] = None,
        image1_path: Optional[str] = None,
        image2_url: Optional[str] = None,
        image2_base64: Optional[str] = None,
        image2_path: Optional[str] = None,
        instruction: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Compare two images.

        Args:
            image1_url: URL of the first image
            image1_base64: Base64 encoded string of the first image
            image1_path: Path to the first image file (read and sent as base64)
            image2_url: URL of the second image
            image2_base64: Base64 encoded string of the second image
            image2_path: Path to the second image file (read and sent as base64)
            instruction: Optional instruction to guide comparison
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and comparison_result
        """
        if not self.client:
            raise ValueError("Client not initialized")

        if image1_path:
            image1_base64 = load_image_path_to_base64(image1_path)
            image1_url = None
        if image2_path:
            image2_base64 = load_image_path_to_base64(image2_path)
            image2_url = None

        response = self.client.compare_images(
            image1_url=image1_url,
            image1_base64=image1_base64,
            image2_url=image2_url,
            image2_base64=image2_base64,
            instruction=instruction,
        )

        return {
            "request_id": response.request_id,
            "credits_used": response.credits_used,
            "comparison_result": response.comparison_result,
        }

    async def _arun(
        self,
        image1_url: Optional[str] = None,
        image1_base64: Optional[str] = None,
        image1_path: Optional[str] = None,
        image2_url: Optional[str] = None,
        image2_base64: Optional[str] = None,
        image2_path: Optional[str] = None,
        instruction: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Compare two images asynchronously.

        Args:
            image1_url: URL of the first image
            image1_base64: Base64 encoded string of the first image
            image1_path: Path to the first image file (read and sent as base64)
            image2_url: URL of the second image
            image2_base64: Base64 encoded string of the second image
            image2_path: Path to the second image file (read and sent as base64)
            instruction: Optional instruction to guide comparison
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and comparison_result
        """
        return self._run(
            image1_url=image1_url,
            image1_base64=image1_base64,
            image1_path=image1_path,
            image2_url=image2_url,
            image2_base64=image2_base64,
            image2_path=image2_path,
            instruction=instruction,
            run_manager=run_manager.get_sync() if run_manager else None,
        )
