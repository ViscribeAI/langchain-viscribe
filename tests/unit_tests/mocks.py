"""Mock client and tools for unit testing."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock
from uuid import UUID, uuid4

from viscribe import Client


class MockResponse:
    """Base mock response class."""

    pass


class MockDescribeImageResponse(MockResponse):
    """Mock response for describe_image."""

    def __init__(self):
        self.request_id = "test-request-id"
        self.credits_used = 1
        self.image_description = "A test image showing a cat sitting on a windowsill"
        self.tags = ["cat", "animal", "windowsill", "indoor"]


class MockAskImageResponse(MockResponse):
    """Mock response for ask_image."""

    def __init__(self):
        self.request_id = "test-request-id"
        self.credits_used = 1
        self.answer = "The car is blue"


class MockClassifyImageResponse(MockResponse):
    """Mock response for classify_image."""

    def __init__(self, multi_label: bool = False):
        self.request_id = "test-request-id"
        self.credits_used = 1
        if multi_label:
            self.classification = ["cat", "indoor"]
        else:
            self.classification = ["cat"]


class MockExtractImageResponse(MockResponse):
    """Mock response for extract_image."""

    def __init__(self):
        self.request_id = "test-request-id"
        self.credits_used = 1
        self.extracted_data = {
            "total": 42.99,
            "date": "2024-01-15",
            "items": ["Coffee", "Sandwich"],
        }


class MockCompareImagesResponse(MockResponse):
    """Mock response for compare_images."""

    def __init__(self):
        self.request_id = "test-request-id"
        self.credits_used = 1
        self.comparison_result = (
            "Both images show cats, but the first image shows a black cat "
            "while the second shows a white cat. The first is outdoors and "
            "the second is indoors."
        )


class MockCreditsResponse(MockResponse):
    """Mock response for get_credits."""

    def __init__(self):
        self.remaining_credits = 1000
        self.total_credits_used = 50


class MockFeedbackResponse(MockResponse):
    """Mock response for submit_feedback."""

    def __init__(self):
        self.feedback_id = uuid4()
        self.request_id = UUID("12345678-1234-1234-1234-123456789012")
        self.message = "Feedback submitted successfully"
        self.feedback_timestamp = datetime(2024, 1, 15, 10, 30, 0)


class MockClient(Client):
    """Mock ViscribeAI Client for testing.

    Inherits from the real Client class to pass pydantic validation,
    but overrides all methods to return mock data instead of making API calls.
    """

    def __init__(self, api_key: Optional[str] = None, *args, **kwargs):
        """Initialize mock client.

        Args:
            api_key: API key (not used in mock, but required for interface)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        # Don't call super().__init__() to avoid real client initialization
        # Just set the minimum required attributes
        self._api_key = api_key
        self.session = MagicMock()  # Mock session to avoid AttributeError

    def describe_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        instruction: Optional[str] = None,
        generate_tags: bool = True,
    ) -> MockDescribeImageResponse:
        """Mock describe_image method.

        Returns:
            MockDescribeImageResponse with test data
        """
        return MockDescribeImageResponse()

    def ask_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        question: Optional[str] = None,
    ) -> MockAskImageResponse:
        """Mock ask_image method.

        Returns:
            MockAskImageResponse with test data
        """
        return MockAskImageResponse()

    def classify_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        classes: Optional[List[str]] = None,
        class_descriptions: Optional[Dict[str, str]] = None,
        instruction: Optional[str] = None,
        multi_label: bool = False,
    ) -> MockClassifyImageResponse:
        """Mock classify_image method.

        Args:
            multi_label: Whether to return multiple classes

        Returns:
            MockClassifyImageResponse with test data
        """
        return MockClassifyImageResponse(multi_label=multi_label)

    def extract_image(
        self,
        image_url: Optional[str] = None,
        image_base64: Optional[str] = None,
        fields: Optional[List[Dict[str, Any]]] = None,
        advanced_schema: Optional[Dict[str, Any]] = None,
        instruction: Optional[str] = None,
    ) -> MockExtractImageResponse:
        """Mock extract_image method.

        Returns:
            MockExtractImageResponse with test data
        """
        return MockExtractImageResponse()

    def compare_images(
        self,
        image1_url: Optional[str] = None,
        image1_base64: Optional[str] = None,
        image2_url: Optional[str] = None,
        image2_base64: Optional[str] = None,
        instruction: Optional[str] = None,
    ) -> MockCompareImagesResponse:
        """Mock compare_images method.

        Returns:
            MockCompareImagesResponse with test data
        """
        return MockCompareImagesResponse()

    def get_credits(self) -> MockCreditsResponse:
        """Mock get_credits method.

        Returns:
            MockCreditsResponse with test data
        """
        return MockCreditsResponse()

    def submit_feedback(
        self,
        request_id: str,
        rating: int,
        feedback_text: Optional[str] = None,
    ) -> MockFeedbackResponse:
        """Mock submit_feedback method.

        Returns:
            MockFeedbackResponse with test data
        """
        return MockFeedbackResponse()
