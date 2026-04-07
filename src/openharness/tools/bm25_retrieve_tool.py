"""BM25 retrieval tool — queries a running BM25 HTTP API."""

from __future__ import annotations

import httpx
from pydantic import BaseModel, Field

from openharness.tools.base import BaseTool, ToolExecutionContext, ToolResult

DEFAULT_API_URL = "http://localhost:8000"


class BM25RetrieveInput(BaseModel):
    """Arguments for BM25 retrieval."""

    query: str = Field(description="Search query (supports AND, OR, \"phrase\", fuzzy~1, wildcard*)")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")
    api_url: str = Field(default=DEFAULT_API_URL, description="BM25 API base URL")


class BM25RetrieveTool(BaseTool):
    """Search a document corpus using BM25 keyword retrieval."""

    name = "bm25_retrieve"
    description = "Search a document corpus using BM25 keyword retrieval via a running API server."
    input_model = BM25RetrieveInput

    def is_read_only(self, arguments: BM25RetrieveInput) -> bool:
        del arguments
        return True

    async def execute(self, arguments: BM25RetrieveInput, context: ToolExecutionContext) -> ToolResult:
        del context
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{arguments.api_url}/retrieve",
                    json={"query": arguments.query, "k": arguments.k},
                )
                resp.raise_for_status()
        except httpx.HTTPError as exc:
            return ToolResult(
                output=f"BM25 API error: {exc}. Is the server running at {arguments.api_url}?",
                is_error=True,
            )

        hits = resp.json().get("hits", [])
        if not hits:
            return ToolResult(output="No results found.")

        lines: list[str] = []
        for h in hits:
            lines.append(f"[{h['rank']}] (score={h['score']:.2f}) {h['doc_id']}")
            if h.get("title"):
                lines.append(f"    Title: {h['title']}")
            text = h.get("text", "")
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"    {text}")
            lines.append("")
        return ToolResult(output="\n".join(lines))
