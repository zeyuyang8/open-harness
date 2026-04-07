"""TF-IDF keyword extraction tool -- queries a running BM25 HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field

from openharness.tools._bm25_api_base import BM25ApiTool
from openharness.tools.base import ToolExecutionContext, ToolResult


class TfidfExtractInput(BaseModel):
    """Arguments for TF-IDF keyword extraction."""

    doc_ids: list[str] = Field(description="Document IDs from bm25_retrieve results")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of keywords per document")


class TfidfExtractTool(BM25ApiTool):
    """Extract discriminative TF-IDF keywords from retrieved documents to refine search queries."""

    name = "tfidf_extract"
    description = (
        "Extract top TF-IDF keywords from documents by their IDs (from bm25_retrieve results). "
        "Use when BM25 retrieval returned some relevant results and you want to "
        "find discriminative terms to refine your search query."
    )
    input_model = TfidfExtractInput

    async def execute(self, arguments: TfidfExtractInput, context: ToolExecutionContext) -> ToolResult:
        del context
        data, err = await self._post(
            "/tfidf_keywords",
            {"doc_ids": arguments.doc_ids, "top_k": arguments.top_k},
        )
        if err:
            return err

        lines: list[str] = []
        for i, doc_result in enumerate(data):
            keywords = doc_result.get("keywords", [])
            terms = [f"{kw['term']}({kw['score']:.2f})" for kw in keywords]
            lines.append(f"[{arguments.doc_ids[i]}] {', '.join(terms)}")
        return ToolResult(output="\n".join(lines) if lines else "No keywords extracted.")
