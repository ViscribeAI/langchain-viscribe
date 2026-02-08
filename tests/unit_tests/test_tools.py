"""Unit tests for langchain-viscribe tools."""

import json
from unittest.mock import patch

import pytest
from dotenv import load_dotenv
from langchain_tests.unit_tests import ToolsUnitTests

# Load environment variables from .env file
load_dotenv()

from langchain_viscribe.tools import (
    AskImageTool,
    ClassifyImageTool,
    CompareImagesTool,
    DescribeImageTool,
    ExtractImageTool,
    GetCreditsTool,
    SubmitFeedbackTool,
)
from tests.unit_tests.mocks import MockClient


class TestDescribeImageToolUnit(ToolsUnitTests):
    """Unit tests for DescribeImageTool."""

    @property
    def tool_constructor(self):
        return DescribeImageTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.describe_image.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": "https://example.com/image.jpg",
            "generate_tags": True,
        }


class TestDescribeImageToolCustom:
    """Custom unit tests for DescribeImageTool."""

    def test_invoke_with_url(self):
        """Test invoking with image URL."""
        with patch("langchain_viscribe.tools.describe_image.Client", MockClient):
            tool = DescribeImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_url": "https://example.com/image.jpg",
                    "generate_tags": True,
                }
            )
            assert "image_description" in result
            assert "tags" in result
            assert "request_id" in result
            assert "credits_used" in result
            assert isinstance(result["tags"], list)

    def test_invoke_with_base64(self):
        """Test invoking with base64 image."""
        with patch("langchain_viscribe.tools.describe_image.Client", MockClient):
            tool = DescribeImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_base64": "base64_string_here",
                    "generate_tags": False,
                }
            )
            assert "image_description" in result

    def test_validation_error_no_image(self):
        """Test validation error when no image is provided."""
        with patch("langchain_viscribe.tools.describe_image.Client", MockClient):
            tool = DescribeImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            with pytest.raises(Exception):
                tool.invoke({"generate_tags": True})

    def test_validation_error_both_images(self):
        """Test validation error when both URL and base64 are provided."""
        with patch("langchain_viscribe.tools.describe_image.Client", MockClient):
            tool = DescribeImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            with pytest.raises(Exception):
                tool.invoke(
                    {
                        "image_url": "https://example.com/image.jpg",
                        "image_base64": "base64_string",
                    }
                )

    def test_invoke_with_image_path(self):
        """Test invoking with local image path (read and sent as base64)."""
        with patch("langchain_viscribe.tools.describe_image.Client", MockClient), patch(
            "langchain_viscribe.tools.describe_image.load_image_path_to_base64",
            return_value="base64_string_from_file",
        ):
            tool = DescribeImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_path": "/absolute/path/to/cat.png",
                    "generate_tags": True,
                }
            )
            assert "image_description" in result
            assert "tags" in result
            assert "request_id" in result
            assert "credits_used" in result


class TestAskImageToolUnit(ToolsUnitTests):
    """Unit tests for AskImageTool."""

    @property
    def tool_constructor(self):
        return AskImageTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.ask_image.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": "https://example.com/image.jpg",
            "question": "What color is the car?",
        }


class TestAskImageToolCustom:
    """Custom unit tests for AskImageTool."""

    def test_invoke_with_question(self):
        """Test asking a question about an image."""
        with patch("langchain_viscribe.tools.ask_image.Client", MockClient):
            tool = AskImageTool(api_key="vscrb-12345678-1234-1234-1234-123456789012")
            result = tool.invoke(
                {
                    "image_url": "https://example.com/car.jpg",
                    "question": "What color is the car?",
                }
            )
            assert "answer" in result
            assert "request_id" in result
            assert "credits_used" in result
            assert isinstance(result["answer"], str)


class TestClassifyImageToolUnit(ToolsUnitTests):
    """Unit tests for ClassifyImageTool."""

    @property
    def tool_constructor(self):
        return ClassifyImageTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.classify_image.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": "https://example.com/image.jpg",
            "classes": json.dumps(["cat", "dog", "bird"]),
        }


class TestClassifyImageToolCustom:
    """Custom unit tests for ClassifyImageTool."""

    def test_invoke_single_label(self):
        """Test single-label classification."""
        with patch("langchain_viscribe.tools.classify_image.Client", MockClient):
            tool = ClassifyImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_url": "https://example.com/image.jpg",
                    "classes": json.dumps(["cat", "dog", "bird"]),
                    "multi_label": False,
                }
            )
            assert "classification" in result
            assert isinstance(result["classification"], list)
            assert len(result["classification"]) == 1

    def test_invoke_multi_label(self):
        """Test multi-label classification."""
        with patch("langchain_viscribe.tools.classify_image.Client", MockClient):
            tool = ClassifyImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_url": "https://example.com/image.jpg",
                    "classes": json.dumps(["cat", "dog", "indoor", "outdoor"]),
                    "multi_label": True,
                }
            )
            assert "classification" in result
            assert isinstance(result["classification"], list)
            assert len(result["classification"]) >= 1

    def test_invoke_with_descriptions(self):
        """Test classification with class descriptions."""
        with patch("langchain_viscribe.tools.classify_image.Client", MockClient):
            tool = ClassifyImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_url": "https://example.com/image.jpg",
                    "classes": json.dumps(["cat", "dog"]),
                    "class_descriptions": json.dumps(
                        {
                            "cat": "A feline animal",
                            "dog": "A canine animal",
                        }
                    ),
                }
            )
            assert "classification" in result


class TestExtractImageToolUnit(ToolsUnitTests):
    """Unit tests for ExtractImageTool."""

    @property
    def tool_constructor(self):
        return ExtractImageTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.extract_image.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": "https://example.com/receipt.jpg",
            "fields": json.dumps(
                [
                    {"name": "total", "type": "number"},
                    {"name": "date", "type": "text"},
                ]
            ),
        }


class TestExtractImageToolCustom:
    """Custom unit tests for ExtractImageTool."""

    def test_invoke_with_simple_fields(self):
        """Test extraction with simple fields."""
        with patch("langchain_viscribe.tools.extract_image.Client", MockClient):
            tool = ExtractImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_url": "https://example.com/receipt.jpg",
                    "fields": json.dumps(
                        [
                            {
                                "name": "total",
                                "type": "number",
                                "description": "Total amount",
                            },
                            {"name": "date", "type": "text"},
                        ]
                    ),
                }
            )
            assert "extracted_data" in result
            assert isinstance(result["extracted_data"], dict)
            assert "request_id" in result

    def test_invoke_with_advanced_schema(self):
        """Test extraction with advanced schema."""
        with patch("langchain_viscribe.tools.extract_image.Client", MockClient):
            tool = ExtractImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image_url": "https://example.com/invoice.jpg",
                    "advanced_schema": json.dumps(
                        {
                            "type": "object",
                            "properties": {
                                "invoice_number": {"type": "string"},
                                "total": {"type": "number"},
                            },
                        }
                    ),
                }
            )
            assert "extracted_data" in result
            assert isinstance(result["extracted_data"], dict)

    def test_validation_error_no_fields_or_schema(self):
        """Test validation error when neither fields nor schema provided."""
        with patch("langchain_viscribe.tools.extract_image.Client", MockClient):
            tool = ExtractImageTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            with pytest.raises(Exception):
                tool.invoke({"image_url": "https://example.com/image.jpg"})


class TestCompareImagesToolUnit(ToolsUnitTests):
    """Unit tests for CompareImagesTool."""

    @property
    def tool_constructor(self):
        return CompareImagesTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.compare_images.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {
            "image1_url": "https://example.com/image1.jpg",
            "image2_url": "https://example.com/image2.jpg",
        }


class TestCompareImagesToolCustom:
    """Custom unit tests for CompareImagesTool."""

    def test_invoke_with_urls(self):
        """Test comparing two images from URLs."""
        with patch("langchain_viscribe.tools.compare_images.Client", MockClient):
            tool = CompareImagesTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image1_url": "https://example.com/image1.jpg",
                    "image2_url": "https://example.com/image2.jpg",
                }
            )
            assert "comparison_result" in result
            assert isinstance(result["comparison_result"], str)
            assert "request_id" in result

    def test_invoke_with_mixed_sources(self):
        """Test comparing with URL and base64."""
        with patch("langchain_viscribe.tools.compare_images.Client", MockClient):
            tool = CompareImagesTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "image1_url": "https://example.com/image1.jpg",
                    "image2_base64": "base64_string_here",
                }
            )
            assert "comparison_result" in result

    def test_validation_error_missing_image2(self):
        """Test validation error when second image is missing."""
        with patch("langchain_viscribe.tools.compare_images.Client", MockClient):
            tool = CompareImagesTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            with pytest.raises(Exception):
                tool.invoke({"image1_url": "https://example.com/image1.jpg"})


class TestGetCreditsToolUnit(ToolsUnitTests):
    """Unit tests for GetCreditsTool."""

    @property
    def tool_constructor(self):
        return GetCreditsTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.get_credits.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {}


class TestGetCreditsToolCustom:
    """Custom unit tests for GetCreditsTool."""

    def test_invoke_get_credits(self):
        """Test getting credits."""
        with patch("langchain_viscribe.tools.get_credits.Client", MockClient):
            tool = GetCreditsTool(api_key="vscrb-12345678-1234-1234-1234-123456789012")
            result = tool.invoke({})
            assert "remaining_credits" in result
            assert "total_credits_used" in result
            assert isinstance(result["remaining_credits"], int)
            assert isinstance(result["total_credits_used"], int)


class TestSubmitFeedbackToolUnit(ToolsUnitTests):
    """Unit tests for SubmitFeedbackTool."""

    @property
    def tool_constructor(self):
        return SubmitFeedbackTool

    @property
    def tool_constructor_params(self):
        with patch("langchain_viscribe.tools.submit_feedback.Client", MockClient):
            return {"api_key": "vscrb-12345678-1234-1234-1234-123456789012"}

    @property
    def tool_invoke_params_example(self):
        return {
            "request_id": "test-request-id",
            "rating": 5,
        }


class TestSubmitFeedbackToolCustom:
    """Custom unit tests for SubmitFeedbackTool."""

    def test_invoke_with_feedback(self):
        """Test submitting feedback."""
        with patch("langchain_viscribe.tools.submit_feedback.Client", MockClient):
            tool = SubmitFeedbackTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "request_id": "test-request-id",
                    "rating": 5,
                    "feedback_text": "Excellent result",
                }
            )
            assert "feedback_id" in result
            assert "request_id" in result
            assert "message" in result
            assert "feedback_timestamp" in result

    def test_invoke_without_text(self):
        """Test submitting feedback without text."""
        with patch("langchain_viscribe.tools.submit_feedback.Client", MockClient):
            tool = SubmitFeedbackTool(
                api_key="vscrb-12345678-1234-1234-1234-123456789012"
            )
            result = tool.invoke(
                {
                    "request_id": "test-request-id",
                    "rating": 4,
                }
            )
            assert "feedback_id" in result
            assert "message" in result
