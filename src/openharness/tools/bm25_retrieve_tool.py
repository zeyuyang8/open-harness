"""BM25 retrieval tool -- queries a running BM25 HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from openharness.tools._bm25_api_base import BM25ApiTool
from openharness.tools.base import ToolExecutionContext, ToolResult


class BM25RetrieveInput(BaseModel):
    """Arguments for BM25 retrieval."""

    query: str = Field(description="Search query (supports AND, OR, \"phrase\", fuzzy~1, wildcard*)")
    k: int = Field(default=10, ge=1, le=1000, description="Number of results to return")


class BM25RetrieveTool(BM25ApiTool):
    """Search a document corpus using BM25 keyword retrieval."""

    name = "bm25_retrieve"
    description = "Search a document corpus using BM25 keyword retrieval via a running API server."
    input_model = BM25RetrieveInput

    async def execute(self, arguments: BM25RetrieveInput, context: ToolExecutionContext) -> ToolResult:
        del context
        data, err = await self._post("/retrieve", {"query": arguments.query, "k": arguments.k})
        if err:
            return err

        hits = data.get("hits", [])
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
