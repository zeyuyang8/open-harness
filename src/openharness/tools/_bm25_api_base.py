"""Shared base class for tools that hit the BM25 HTTP API.

Centralises httpx client management and error handling so each
BM25-family tool only needs to define its endpoint, payload, and
response formatting.
"""

from __future__ import annotations

from typing import Any

import httpx

from openharness.tools.base import BaseTool, ToolResult

DEFAULT_API_URL = "http://localhost:8000"


class BM25ApiTool(BaseTool):
    """Base for tools backed by the BM25 HTTP API.

    Subclasses get:
    - A shared ``httpx.AsyncClient`` (connection-pooled, lazily created).
    - ``_post(endpoint, payload)`` that returns ``(data, None)`` on
      success or ``(None, ToolResult)`` with a user-friendly error.
    """

    def __init__(self, *, api_url: str = DEFAULT_API_URL) -> None:
        self._api_url = api_url
        self._client: httpx.AsyncClient | None = None

    # -- helpers ----------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        """Return a lazily-created, reusable async HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self._api_url,
                timeout=30.0,
            )
        return self._client

    async def _post(
        self, endpoint: str, payload: dict[str, Any]
    ) -> tuple[Any | None, ToolResult | None]:
        """POST *payload* to *endpoint* and return decoded JSON.

        Returns ``(json_data, None)`` on success, or
        ``(None, ToolResult(is_error=True))`` on failure.
        """
        try:
            resp = await self._get_client().post(endpoint, json=payload)
            resp.raise_for_status()
            return resp.json(), None
        except httpx.HTTPError as exc:
            err = ToolResult(
                output=(
                    f"{self.name} API error: {exc}. "
                    f"Is the server running at {self._api_url}?"
                ),
                is_error=True,
            )
            return None, err

    # -- BaseTool defaults ------------------------------------------------

    def is_read_only(self, arguments) -> bool:  # type: ignore[override]
        del arguments
        return True
