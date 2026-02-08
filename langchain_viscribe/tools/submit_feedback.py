"""Tool for submitting feedback on API responses."""

from typing import Dict, Optional, Type

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, Field, model_validator
from viscribe import Client


class SubmitFeedbackInput(BaseModel):
    """Input schema for SubmitFeedbackTool."""

    request_id: str = Field(
        description="The request ID from a previous API call (returned in all tool responses as 'request_id').",
    )
    rating: int = Field(
        description="Rating from 1 to 5, where 1 is poor and 5 is excellent.",
        ge=1,
        le=5,
    )
    feedback_text: Optional[str] = Field(
        default=None,
        description="Optional text feedback providing details about the rating or suggestions for improvement.",
    )


class SubmitFeedbackTool(BaseTool):
    """Tool for submitting feedback on ViscribeAI API responses.

    This tool enables you to provide feedback on the quality of API responses
    to help improve the service. Requires a request_id from a previous API call
    and a rating from 1-5.

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

            from langchain_viscribe.tools import SubmitFeedbackTool

            tool = SubmitFeedbackTool()
            # Or with explicit API key
            tool = SubmitFeedbackTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            # Submit feedback on a previous request
            result = tool.invoke({
                "request_id": "req-123-456",
                "rating": 5,
                "feedback_text": "Excellent accuracy and detail"
            })
            print(result["message"])
            print(result["feedback_id"])

            # Minimal feedback (just rating)
            result = tool.invoke({
                "request_id": "req-789-012",
                "rating": 4
            })

            # Detailed feedback
            result = tool.invoke({
                "request_id": "req-345-678",
                "rating": 3,
                "feedback_text": "Good but could be more detailed in the description"
            })

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({
                "request_id": "req-123-456",
                "rating": 5
            })

    Tool response format:
        {
            "feedback_id": "fb-uuid-here",
            "request_id": "req-123-456",
            "message": "Feedback submitted successfully",
            "feedback_timestamp": "2024-01-15T10:30:00"
        }
    """

    name: str = "SubmitFeedback"
    description: str = (
        "Submit feedback on a previous Viscribe AI API response. "
        "Requires a request_id from a previous call and a rating (1-5). "
        "Useful for improving service quality and providing input on results."
    )
    args_schema: Type[BaseModel] = SubmitFeedbackInput
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
        request_id: str,
        rating: int,
        feedback_text: Optional[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Submit feedback on a previous API response.

        Args:
            request_id: The request ID from a previous API call
            rating: Rating from 1 to 5
            feedback_text: Optional text feedback
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing feedback_id, request_id, message, and feedback_timestamp
        """
        if not self.client:
            raise ValueError("Client not initialized")

        response = self.client.submit_feedback(
            request_id=request_id,
            rating=rating,
            feedback_text=feedback_text,
        )

        return {
            "feedback_id": str(response.feedback_id),
            "request_id": str(response.request_id),
            "message": response.message,
            "feedback_timestamp": response.feedback_timestamp.isoformat(),
        }

    async def _arun(
        self,
        request_id: str,
        rating: int,
        feedback_text: Optional[str] = None,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Submit feedback asynchronously.

        Args:
            request_id: The request ID from a previous API call
            rating: Rating from 1 to 5
            feedback_text: Optional text feedback
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing feedback_id, request_id, message, and feedback_timestamp
        """
        return self._run(
            request_id=request_id,
            rating=rating,
            feedback_text=feedback_text,
            run_manager=run_manager.get_sync() if run_manager else None,
        )
