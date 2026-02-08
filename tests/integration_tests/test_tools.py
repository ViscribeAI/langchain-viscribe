"""Integration tests for langchain-viscribe tools.

These tests use the real ViscribeAI API and require VISCRIBE_API_KEY environment variable.
Tests will be skipped if the API key is not set.
"""

import json
import os

import pytest
from dotenv import load_dotenv
from langchain_tests.integration_tests import ToolsIntegrationTests

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

# Test image URLs (public domain images for testing)
TEST_IMAGE_URL = "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/240px-Cat03.jpg"
TEST_IMAGE_URL_2 = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/240px-Cat_November_2010-1a.jpg"


class TestDescribeImageToolIntegration(ToolsIntegrationTests):
    """Integration tests for DescribeImageTool."""

    @property
    def tool_constructor(self):
        return DescribeImageTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": TEST_IMAGE_URL,
            "generate_tags": True,
        }


class TestAskImageToolIntegration(ToolsIntegrationTests):
    """Integration tests for AskImageTool."""

    @property
    def tool_constructor(self):
        return AskImageTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": TEST_IMAGE_URL,
            "question": "What animal is in this image?",
        }


class TestClassifyImageToolIntegration(ToolsIntegrationTests):
    """Integration tests for ClassifyImageTool."""

    @property
    def tool_constructor(self):
        return ClassifyImageTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": TEST_IMAGE_URL,
            "classes": json.dumps(["cat", "dog", "bird", "fish"]),
        }


class TestExtractImageToolIntegration(ToolsIntegrationTests):
    """Integration tests for ExtractImageTool."""

    @property
    def tool_constructor(self):
        return ExtractImageTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        return {
            "image_url": TEST_IMAGE_URL,
            "fields": json.dumps(
                [
                    {
                        "name": "animal_type",
                        "type": "text",
                        "description": "Type of animal",
                    },
                    {
                        "name": "colors",
                        "type": "array_text",
                        "description": "Main colors visible",
                    },
                ]
            ),
        }


class TestCompareImagesToolIntegration(ToolsIntegrationTests):
    """Integration tests for CompareImagesTool."""

    @property
    def tool_constructor(self):
        return CompareImagesTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        return {
            "image1_url": TEST_IMAGE_URL,
            "image2_url": TEST_IMAGE_URL_2,
        }


class TestGetCreditsToolIntegration(ToolsIntegrationTests):
    """Integration tests for GetCreditsTool."""

    @property
    def tool_constructor(self):
        return GetCreditsTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        return {}


class TestSubmitFeedbackToolIntegration(ToolsIntegrationTests):
    """Integration tests for SubmitFeedbackTool."""

    @property
    def tool_constructor(self):
        return SubmitFeedbackTool

    @property
    def tool_constructor_params(self):
        api_key = os.getenv("VISCRIBE_API_KEY")
        if not api_key:
            pytest.skip("VISCRIBE_API_KEY environment variable not set")
        return {"api_key": api_key}

    @property
    def tool_invoke_params_example(self):
        # Note: This uses a mock request_id for testing
        # In real usage, you would use a request_id from a previous API call
        return {
            "request_id": "test-request-id-for-integration",
            "rating": 5,
            "feedback_text": "Integration test feedback",
        }
