"""Corpus grep tool -- regex search over document corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field

from openharness.tools._bm25_api_base import BM25ApiTool
from openharness.tools.base import ToolExecutionContext, ToolResult


class GrepCorpusInput(BaseModel):
    """Arguments for corpus grep."""

    pattern: str = Field(description="Regex pattern to search for in document title and text")
    limit: int = Field(default=20, ge=1, le=100, description="Max documents to return")


class GrepCorpusTool(BM25ApiTool):
    """Search corpus documents by regex pattern matching on title and text."""

    name = "grep_corpus"
    description = (
        "Search the document corpus with a regex pattern (exact/substring match). "
        "Use when you need precise text matching rather than BM25 keyword relevance scoring."
    )
    input_model = GrepCorpusInput

    async def execute(self, arguments: GrepCorpusInput, context: ToolExecutionContext) -> ToolResult:
        del context
        data, err = await self._post(
            "/grep",
            {"pattern": arguments.pattern, "limit": arguments.limit},
        )
        if err:
            return err

        if not data:
            return ToolResult(output="No matches found.")

        lines: list[str] = []
        for h in data:
            lines.append(f"[{h['rank']}] {h['doc_id']}")
            if h.get("title"):
                lines.append(f"    Title: {h['title']}")
            text = h.get("text", "")
            if len(text) > 300:
                text = text[:300] + "..."
            lines.append(f"    {text}")
            lines.append("")
        return ToolResult(output="\n".join(lines))
