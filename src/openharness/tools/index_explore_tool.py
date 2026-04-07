"""Index exploration tool -- queries inverted index term statistics."""

from __future__ import annotations

from pydantic import BaseModel, Field

from openharness.tools._bm25_api_base import BM25ApiTool
from openharness.tools.base import ToolExecutionContext, ToolResult


class IndexExploreInput(BaseModel):
    """Arguments for index exploration."""

    terms: list[str] = Field(default_factory=list, description="Terms to look up IDF/doc frequency for")
    prefix: str = Field(default="", description="Prefix to search vocabulary (e.g. 'edit' finds 'editing', 'editor')")
    prefix_limit: int = Field(default=20, ge=1, le=100, description="Max prefix matches to return")
    cooccur_term: str = Field(default="", description="Find terms that co-occur with this term in the same documents")
    cooccur_limit: int = Field(default=20, ge=1, le=100, description="Max co-occurring terms to return")


class IndexExploreTool(BM25ApiTool):
    """Explore the BM25 inverted index for term statistics, vocabulary discovery, and co-occurrence."""

    name = "index_explore"
    description = (
        "Explore the BM25 inverted index. Three capabilities: "
        "(1) Look up term IDF/doc frequency. "
        "(2) Discover vocabulary terms by prefix. "
        "(3) Find terms that co-occur with a given term -- use this to discover related concepts."
    )
    input_model = IndexExploreInput

    async def execute(self, arguments: IndexExploreInput, context: ToolExecutionContext) -> ToolResult:
        del context
        data, err = await self._post(
            "/index_explore",
            {
                "terms": arguments.terms,
                "prefix": arguments.prefix,
                "prefix_limit": arguments.prefix_limit,
                "cooccur_term": arguments.cooccur_term,
                "cooccur_limit": arguments.cooccur_limit,
            },
        )
        if err:
            return err

        lines: list[str] = []

        term_stats = data.get("term_stats", [])
        if term_stats:
            lines.append("Term statistics:")
            for ts in term_stats:
                lines.append(f"  {ts['term']}: df={ts['df']}, idf={ts['idf']:.4f}")

        prefix_matches = data.get("prefix_matches", [])
        if prefix_matches:
            lines.append(f"Prefix matches for '{arguments.prefix}':")
            for pm in prefix_matches:
                lines.append(f"  {pm['term']}: df={pm['df']}, idf={pm['idf']:.4f}")

        cooccurring = data.get("cooccurring", [])
        if cooccurring:
            lines.append(f"Terms co-occurring with '{arguments.cooccur_term}':")
            for co in cooccurring:
                lines.append(f"  {co['term']}: cooccur={co['cooccur']}, df={co['df']}, idf={co['idf']:.4f}")

        return ToolResult(output="\n".join(lines) if lines else "No results.")
