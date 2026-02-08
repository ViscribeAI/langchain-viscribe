"""Tool for extracting structured data from images."""

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


class ExtractImageInput(BaseModel):
    """Input schema for ExtractImageTool."""

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to extract data from. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64 encoded image string. Provide exactly one of image_url, image_base64, or image_path.",
    )
    image_path: Optional[str] = Field(
        default=None,
        description="Absolute or relative path to a local image file. The file will be read and sent as base64. Provide exactly one of image_url, image_base64, or image_path.",
    )
    fields: Optional[str] = Field(
        default=None,
        description='JSON string of simple fields to extract. Each field should have "name", "type" (text/number/array_text/array_number), and optional "description". '
        'Example: \'[{"name": "price", "type": "number", "description": "Product price"}, {"name": "title", "type": "text"}]\'',
    )
    advanced_schema: Optional[str] = Field(
        default=None,
        description='JSON string of advanced schema for complex extraction (use for nested structures). Must have "type": "object" and "properties". '
        'Example: \'{"type": "object", "properties": {"title": {"type": "string"}, "items": {"type": "array", "items": {"type": "object", "properties": {"name": {"type": "string"}}}}}}\'',
    )
    instruction: Optional[str] = Field(
        default=None,
        description="Optional instruction to guide the extraction process.",
    )

    @model_validator(mode="after")
    def check_inputs(self):
        """Validate inputs."""
        sources = sum([bool(self.image_url), bool(self.image_base64), bool(self.image_path)])
        if sources == 0:
            raise ValueError("Provide exactly one of image_url, image_base64, or image_path")
        if sources > 1:
            raise ValueError("Provide only one of image_url, image_base64, or image_path")
        if not self.fields and not self.advanced_schema:
            raise ValueError("Either fields or advanced_schema must be provided")
        if self.fields and self.advanced_schema:
            raise ValueError("Provide either fields or advanced_schema, not both")
        return self


class ExtractImageTool(BaseTool):
    """Tool for extracting structured data from images.

    This tool enables extraction of structured data from documents, receipts,
    forms, screenshots, and other images containing text or structured information.
    Supports both simple field extraction and complex nested schema extraction.

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

            from langchain_viscribe.tools import ExtractImageTool

            tool = ExtractImageTool()
            # Or with explicit API key
            tool = ExtractImageTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            import json

            # Simple field extraction
            result = tool.invoke({
                "image_url": "https://example.com/receipt.jpg",
                "fields": json.dumps([
                    {"name": "total", "type": "number", "description": "Total amount"},
                    {"name": "date", "type": "text", "description": "Receipt date"},
                    {"name": "items", "type": "array_text", "description": "Item names"}
                ])
            })
            print(result["extracted_data"])
            # {"total": 42.99, "date": "2024-01-15", "items": ["Coffee", "Sandwich"]}

            # Advanced schema extraction (nested structures)
            result = tool.invoke({
                "image_url": "https://example.com/invoice.jpg",
                "advanced_schema": json.dumps({
                    "type": "object",
                    "properties": {
                        "invoice_number": {"type": "string"},
                        "total": {"type": "number"},
                        "line_items": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "description": {"type": "string"},
                                    "quantity": {"type": "number"},
                                    "price": {"type": "number"}
                                }
                            }
                        }
                    }
                })
            })

            # With instruction
            result = tool.invoke({
                "image_url": "https://example.com/form.jpg",
                "fields": json.dumps([
                    {"name": "name", "type": "text"},
                    {"name": "email", "type": "text"}
                ]),
                "instruction": "Extract the contact information from the form"
            })

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({
                "image_url": "https://example.com/document.jpg",
                "fields": json.dumps([{"name": "title", "type": "text"}])
            })

    Tool response format:
        {
            "request_id": "unique-request-id",
            "credits_used": 1,
            "extracted_data": {
                "field1": "value1",
                "field2": 123,
                ...
            }
        }
    """

    name: str = "ExtractImage"
    description: str = (
        "Extract structured data from images (receipts, documents, forms, screenshots, etc.). "
        "Supports simple fields (name, type, description) or complex JSON schemas for nested structures. "
        "Useful for document processing, form filling, and data extraction. "
        "Takes an image URL, a base64-encoded image, or the path to a local image file."
    )
    args_schema: Type[BaseModel] = ExtractImageInput
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
        fields: Optional[str] = None,
        advanced_schema: Optional[str] = None,
        instruction: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Extract structured data from an image.

        Args:
            image_url: URL of the image
            image_base64: Base64 encoded image string
            image_path: Path to a local image file (read and sent as base64)
            fields: JSON string of simple fields to extract
            advanced_schema: JSON string of advanced schema for complex extraction
            instruction: Optional instruction to guide extraction
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and extracted_data
        """
        if not self.client:
            raise ValueError("Client not initialized")

        if image_path:
            image_base64 = load_image_path_to_base64(image_path)
            image_url = None

        # Parse JSON strings
        parsed_fields = None
        parsed_schema = None

        if fields:
            parsed_fields = json.loads(fields)
        if advanced_schema:
            parsed_schema = json.loads(advanced_schema)

        response = self.client.extract_image(
            image_url=image_url,
            image_base64=image_base64,
            fields=parsed_fields,
            advanced_schema=parsed_schema,
            instruction=instruction,
        )

        return {
            "request_id": response.request_id,
            "credits_used": response.credits_used,
            "extracted_data": response.extracted_data,
        }

    async def _arun(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        image_path: Optional[str] = None,
        fields: Optional[str] = None,
        advanced_schema: Optional[str] = None,
        instruction: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Extract structured data from an image asynchronously.

        Args:
            image_url: URL of the image
            image_base64: Base64 encoded image string
            fields: JSON string of simple fields to extract
            advanced_schema: JSON string of advanced schema for complex extraction
            instruction: Optional instruction to guide extraction
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing request_id, credits_used, and extracted_data
        """
        return self._run(
            image_url=image_url,
            image_base64=image_base64,
            image_path=image_path,
            fields=fields,
            advanced_schema=advanced_schema,
            instruction=instruction,
            run_manager=run_manager.get_sync() if run_manager else None,
        )
