"""Tool for checking API credits."""

from typing import Dict, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_core.utils import get_from_dict_or_env
from pydantic import model_validator
from viscribe import Client


class GetCreditsTool(BaseTool):
    """Tool for checking current API credits.

    This tool retrieves the current credit balance and total credits used
    for your ViscribeAI account. Useful for monitoring usage and ensuring
    sufficient credits are available before making API calls.

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

            from langchain_viscribe.tools import GetCreditsTool

            tool = GetCreditsTool()
            # Or with explicit API key
            tool = GetCreditsTool(api_key="vscrb-your-key")

    Use the tool:
        .. code-block:: python

            # Check credits
            result = tool.invoke({})
            print(f"Remaining credits: {result['remaining_credits']}")
            print(f"Total used: {result['total_credits_used']}")

    Async usage:
        .. code-block:: python

            result = await tool.ainvoke({})

    Tool response format:
        {
            "remaining_credits": 1000,
            "total_credits_used": 50
        }
    """

    name: str = "GetCredits"
    description: str = (
        "Get the current credits available in your Viscribe AI account. "
        "Returns remaining credits and total credits used. "
        "No parameters required."
    )
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
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> dict:
        """Get the available credits.

        Args:
            run_manager: Callback manager for the tool run

        Returns:
            Dictionary containing remaining_credits and total_credits_used
        """
        if not self.client:
            raise ValueError("Client not initialized")

        response = self.client.get_credits()

        return {
            "remaining_credits": response.remaining_credits,
            "total_credits_used": response.total_credits_used,
        }

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> dict:
        """Get the available credits asynchronously.

        Args:
            run_manager: Async callback manager for the tool run

        Returns:
            Dictionary containing remaining_credits and total_credits_used
        """
        return self._run(run_manager=run_manager.get_sync() if run_manager else None)
