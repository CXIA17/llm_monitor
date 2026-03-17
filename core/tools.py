#!/usr/bin/env python3
"""
Agent Tools
===========

External tools that agents can invoke during generation:
- RAGTool: Retrieval-augmented generation over case law / evidence files
- LegalSearchTool: Web-based legal fact-checking (fallback when RAG is insufficient)

Usage:
    rag = RAGTool(corpus_dir="./evidence")
    rag.load()
    results = rag.search("negligence standard of care", top_k=3)

    search = LegalSearchTool(api_key="...")
    results = search.search("Smith v. Jones 2019 ruling")
"""

import os
import json
import re
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path


# ---------------------------------------------------------------------------
# Base Tool
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    """Standardised result returned by every tool."""
    tool_name: str
    query: str
    results: List[Dict[str, Any]]
    source: str  # "rag", "web_search", etc.
    success: bool = True
    error: Optional[str] = None

    def format_for_prompt(self, max_results: int = 3) -> str:
        """Format results into a string suitable for injection into a prompt."""
        if not self.success:
            return f"[{self.tool_name} ERROR]: {self.error}"
        if not self.results:
            return f"[{self.tool_name}]: No results found for '{self.query}'."

        lines = [f"[{self.tool_name} — {len(self.results)} result(s) for '{self.query}']:"]
        for i, r in enumerate(self.results[:max_results], 1):
            title = r.get("title", r.get("filename", f"Result {i}"))
            snippet = r.get("snippet", r.get("text", ""))
            cite = r.get("citation", r.get("url", ""))
            lines.append(f"  {i}. [{title}] {snippet}")
            if cite:
                lines.append(f"     Source: {cite}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "query": self.query,
            "results": self.results,
            "source": self.source,
            "success": self.success,
            "error": self.error,
        }


class AgentTool(ABC):
    """Base class for all tools an agent can invoke."""
    name: str = "base_tool"
    description: str = ""
    category: str = "general"  # "rag", "fact_check", "analysis", etc.

    @abstractmethod
    def search(self, query: str, **kwargs) -> ToolResult:
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# RAG Tool — searches a local corpus of case law / evidence files
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A chunk from an indexed document."""
    doc_id: str
    filename: str
    title: str
    text: str
    citation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[Any] = None  # numpy array when embeddings are loaded


class RAGTool(AgentTool):
    """
    Retrieval-Augmented Generation tool backed by a local corpus.

    Supports two modes:
      1. **TF-IDF** (default, zero-dependency): keyword-based search
      2. **Embedding** (optional): dense vector search via sentence-transformers

    Corpus directory can contain:
      - .txt files (plain-text documents, one per file)
      - .json / .jsonl files with structure:
            {"title": "...", "text": "...", "citation": "...", ...}
      - .md files (treated as plain text)

    Parameters:
        corpus_dir:  path to the directory of documents
        chunk_size:  approximate number of characters per chunk
        chunk_overlap: characters of overlap between adjacent chunks
        use_embeddings: if True, use sentence-transformers for dense search
        embedding_model: HuggingFace model id for embeddings
    """

    name = "case_law_rag"
    description = "Search case law database and evidence files"
    category = "rag"

    def __init__(
        self,
        corpus_dir: str = "./evidence",
        chunk_size: int = 800,
        chunk_overlap: int = 200,
        use_embeddings: bool = False,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        self.corpus_dir = Path(corpus_dir)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_embeddings = use_embeddings
        self.embedding_model_name = embedding_model

        # Internal state
        self._chunks: List[DocumentChunk] = []
        self._idf: Dict[str, float] = {}
        self._tf_index: Dict[str, List[Tuple[int, float]]] = {}  # term -> [(chunk_idx, tf)]
        self._embedder = None
        self._chunk_embeddings = None  # numpy matrix
        self._loaded = False

    # ---- Loading & Indexing ------------------------------------------------

    def load(self) -> int:
        """Load and index all documents. Returns number of chunks."""
        self._chunks = []
        if not self.corpus_dir.exists():
            os.makedirs(self.corpus_dir, exist_ok=True)
            return 0

        for fpath in sorted(self.corpus_dir.rglob("*")):
            if fpath.is_dir():
                continue
            ext = fpath.suffix.lower()
            if ext in (".txt", ".md"):
                self._load_text_file(fpath)
            elif ext == ".json":
                self._load_json_file(fpath)
            elif ext == ".jsonl":
                self._load_jsonl_file(fpath)

        self._build_tfidf_index()

        if self.use_embeddings:
            self._build_embedding_index()

        self._loaded = True
        return len(self._chunks)

    def add_document(self, title: str, text: str, citation: str = "", metadata: Optional[Dict] = None):
        """Programmatically add a document (e.g. evidence file at runtime)."""
        doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
        for chunk_text in self._split_text(text):
            self._chunks.append(DocumentChunk(
                doc_id=doc_id,
                filename="<runtime>",
                title=title,
                text=chunk_text,
                citation=citation,
                metadata=metadata or {},
            ))
        # Rebuild indexes
        self._build_tfidf_index()
        if self.use_embeddings and self._embedder is not None:
            self._build_embedding_index()

    # ---- Search ------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, **kwargs) -> ToolResult:
        """Search the corpus for relevant chunks."""
        if not self._loaded:
            self.load()

        if not self._chunks:
            return ToolResult(
                tool_name=self.name,
                query=query,
                results=[],
                source="rag",
                success=True,
            )

        if self.use_embeddings and self._chunk_embeddings is not None:
            ranked = self._search_embedding(query, top_k)
        else:
            ranked = self._search_tfidf(query, top_k)

        results = []
        for idx, score in ranked:
            chunk = self._chunks[idx]
            results.append({
                "title": chunk.title,
                "text": chunk.text,
                "snippet": chunk.text[:300],
                "citation": chunk.citation,
                "filename": chunk.filename,
                "score": round(score, 4),
                "doc_id": chunk.doc_id,
                "metadata": chunk.metadata,
            })

        return ToolResult(
            tool_name=self.name,
            query=query,
            results=results,
            source="rag",
        )

    # ---- Internal: file loaders -------------------------------------------

    def _load_text_file(self, fpath: Path):
        text = fpath.read_text(encoding="utf-8", errors="replace")
        title = fpath.stem.replace("_", " ").title()
        doc_id = hashlib.md5(fpath.name.encode()).hexdigest()[:12]
        for chunk_text in self._split_text(text):
            self._chunks.append(DocumentChunk(
                doc_id=doc_id,
                filename=fpath.name,
                title=title,
                text=chunk_text,
                citation=f"File: {fpath.name}",
            ))

    def _load_json_file(self, fpath: Path):
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        docs = data if isinstance(data, list) else [data]
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            text = doc.get("text", doc.get("content", ""))
            if not text:
                continue
            title = doc.get("title", doc.get("name", fpath.stem))
            citation = doc.get("citation", doc.get("source", ""))
            doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
            meta = {k: v for k, v in doc.items() if k not in ("text", "content", "title", "citation")}
            for chunk_text in self._split_text(text):
                self._chunks.append(DocumentChunk(
                    doc_id=doc_id,
                    filename=fpath.name,
                    title=title,
                    text=chunk_text,
                    citation=citation,
                    metadata=meta,
                ))

    def _load_jsonl_file(self, fpath: Path):
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                text = doc.get("text", doc.get("content", ""))
                if not text:
                    continue
                title = doc.get("title", doc.get("name", fpath.stem))
                citation = doc.get("citation", doc.get("source", ""))
                doc_id = hashlib.md5(text[:200].encode()).hexdigest()[:12]
                for chunk_text in self._split_text(text):
                    self._chunks.append(DocumentChunk(
                        doc_id=doc_id,
                        filename=fpath.name,
                        title=title,
                        text=chunk_text,
                        citation=citation,
                    ))

    # ---- Internal: chunking -----------------------------------------------

    def _split_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # Try to break at sentence boundary
            if end < len(text):
                for boundary in [". ", ".\n", "\n\n", "\n", " "]:
                    pos = text.rfind(boundary, start + self.chunk_size // 2, end + 100)
                    if pos != -1:
                        end = pos + len(boundary)
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - self.chunk_overlap
        return chunks

    # ---- Internal: TF-IDF -------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _build_tfidf_index(self):
        import math
        self._tf_index = {}
        doc_freq: Dict[str, int] = {}
        n = len(self._chunks)
        if n == 0:
            return

        for idx, chunk in enumerate(self._chunks):
            tokens = self._tokenize(chunk.text)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            total = len(tokens) or 1
            seen = set()
            for t, count in tf.items():
                self._tf_index.setdefault(t, []).append((idx, count / total))
                if t not in seen:
                    doc_freq[t] = doc_freq.get(t, 0) + 1
                    seen.add(t)

        self._idf = {t: math.log((n + 1) / (df + 1)) + 1 for t, df in doc_freq.items()}

    def _search_tfidf(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        tokens = self._tokenize(query)
        scores: Dict[int, float] = {}
        for t in tokens:
            idf = self._idf.get(t, 0)
            for idx, tf in self._tf_index.get(t, []):
                scores[idx] = scores.get(idx, 0) + tf * idf
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ---- Internal: Embedding search ----------------------------------------

    def _build_embedding_index(self):
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            print("WARNING: sentence-transformers not installed — falling back to TF-IDF")
            self.use_embeddings = False
            return

        self._embedder = SentenceTransformer(self.embedding_model_name)
        texts = [c.text for c in self._chunks]
        self._chunk_embeddings = self._embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    def _search_embedding(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        import numpy as np
        q_emb = self._embedder.encode([query], normalize_embeddings=True)
        scores = (self._chunk_embeddings @ q_emb.T).squeeze()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in top_indices]

    # ---- Serialization -----------------------------------------------------

    def get_stats(self) -> Dict[str, Any]:
        unique_docs = set(c.doc_id for c in self._chunks)
        return {
            "total_chunks": len(self._chunks),
            "unique_documents": len(unique_docs),
            "corpus_dir": str(self.corpus_dir),
            "use_embeddings": self.use_embeddings,
            "vocab_size": len(self._idf),
            "loaded": self._loaded,
        }


# ---------------------------------------------------------------------------
# Legal Web Search Tool — fact-checking via external search API
# ---------------------------------------------------------------------------

class LegalSearchTool(AgentTool):
    """
    Fact-checking tool that searches legal databases via web APIs.

    Acts as a fallback when the local RAG corpus lacks sufficient information.
    Labelled as a **fact-checking** tool so agents invoke it only when the
    internal case-law library cannot resolve a question.

    Supported backends:
      - "courtlistener" : CourtListener (free, US case law)
      - "google_scholar": Google Scholar (legal articles/cases)
      - "serpapi"        : SerpAPI with Google Scholar lens
      - "custom"         : user-provided search function

    Parameters:
        backend:    which search backend to use
        api_key:    API key for the chosen backend (if required)
        base_url:   override URL for custom backends
        max_results: max results per query
    """

    name = "legal_fact_check"
    description = (
        "Fact-checking tool: search legal databases and the web when the "
        "internal case-law library does not contain sufficient information "
        "to support or refute a claim."
    )
    category = "fact_check"

    def __init__(
        self,
        backend: str = "courtlistener",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_results: int = 5,
        custom_search_fn: Optional[Any] = None,
    ):
        self.backend = backend
        self.api_key = api_key or os.environ.get("LEGAL_SEARCH_API_KEY", "")
        self.base_url = base_url
        self.max_results = max_results
        self._custom_fn = custom_search_fn

    def search(self, query: str, top_k: Optional[int] = None, **kwargs) -> ToolResult:
        """Search for legal information to fact-check a claim."""
        k = top_k or self.max_results
        try:
            if self.backend == "courtlistener":
                return self._search_courtlistener(query, k)
            elif self.backend == "google_scholar":
                return self._search_google_scholar(query, k)
            elif self.backend == "serpapi":
                return self._search_serpapi(query, k)
            elif self.backend == "custom" and self._custom_fn:
                raw = self._custom_fn(query, k)
                return ToolResult(
                    tool_name=self.name, query=query,
                    results=raw if isinstance(raw, list) else [raw],
                    source="custom_web_search",
                )
            else:
                return ToolResult(
                    tool_name=self.name, query=query, results=[],
                    source=self.backend, success=False,
                    error=f"Unknown backend: {self.backend}",
                )
        except Exception as e:
            return ToolResult(
                tool_name=self.name, query=query, results=[],
                source=self.backend, success=False, error=str(e),
            )

    # ---- CourtListener (free US case-law API) ------------------------------

    def _search_courtlistener(self, query: str, top_k: int) -> ToolResult:
        import urllib.request
        import urllib.parse

        url = self.base_url or "https://www.courtlistener.com/api/rest/v4/search/"
        params = urllib.parse.urlencode({"q": query, "type": "o", "format": "json"})
        req = urllib.request.Request(
            f"{url}?{params}",
            headers={
                "Authorization": f"Token {self.api_key}" if self.api_key else "",
                "User-Agent": "LLM-Monitor-Legal-FactCheck/1.0",
            },
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for item in data.get("results", [])[:top_k]:
            results.append({
                "title": item.get("caseName", item.get("case_name", "Unknown")),
                "snippet": (item.get("snippet", "") or "")[:500],
                "citation": item.get("citation", [item.get("absoluteUrl", "")])[0]
                    if isinstance(item.get("citation"), list)
                    else item.get("citation", ""),
                "url": item.get("absoluteUrl", item.get("absolute_url", "")),
                "court": item.get("court", ""),
                "date_filed": item.get("dateFiled", item.get("date_filed", "")),
            })

        return ToolResult(
            tool_name=self.name, query=query, results=results,
            source="courtlistener",
        )

    # ---- Google Scholar (scraping fallback) --------------------------------

    def _search_google_scholar(self, query: str, top_k: int) -> ToolResult:
        import urllib.request
        import urllib.parse

        url = "https://scholar.google.com/scholar"
        params = urllib.parse.urlencode({"q": query, "hl": "en", "num": top_k})
        req = urllib.request.Request(
            f"{url}?{params}",
            headers={"User-Agent": "Mozilla/5.0 (compatible; LLM-Monitor/1.0)"},
        )

        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode()

        # Minimal parsing — extract titles and snippets
        results = []
        title_pattern = re.compile(r'<h3[^>]*class="gs_rt"[^>]*>(.*?)</h3>', re.DOTALL)
        snippet_pattern = re.compile(r'<div[^>]*class="gs_rs"[^>]*>(.*?)</div>', re.DOTALL)
        titles = title_pattern.findall(html)
        snippets = snippet_pattern.findall(html)

        for i in range(min(len(titles), top_k)):
            clean_title = re.sub(r"<[^>]+>", "", titles[i]).strip()
            clean_snippet = re.sub(r"<[^>]+>", "", snippets[i]).strip() if i < len(snippets) else ""
            results.append({
                "title": clean_title,
                "snippet": clean_snippet[:500],
                "citation": clean_title,
                "url": "",
            })

        return ToolResult(
            tool_name=self.name, query=query, results=results,
            source="google_scholar",
        )

    # ---- SerpAPI -----------------------------------------------------------

    def _search_serpapi(self, query: str, top_k: int) -> ToolResult:
        import urllib.request
        import urllib.parse

        url = self.base_url or "https://serpapi.com/search.json"
        params = urllib.parse.urlencode({
            "q": query,
            "engine": "google_scholar",
            "api_key": self.api_key,
            "num": top_k,
        })
        req = urllib.request.Request(f"{url}?{params}")

        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())

        results = []
        for item in data.get("organic_results", [])[:top_k]:
            results.append({
                "title": item.get("title", ""),
                "snippet": item.get("snippet", "")[:500],
                "citation": item.get("publication_info", {}).get("summary", ""),
                "url": item.get("link", ""),
            })

        return ToolResult(
            tool_name=self.name, query=query, results=results,
            source="serpapi",
        )

    def get_stats(self) -> Dict[str, Any]:
        return {
            "backend": self.backend,
            "has_api_key": bool(self.api_key),
            "max_results": self.max_results,
        }


# ---------------------------------------------------------------------------
# Tool Registry — manages available tools per agent
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registry of tools available to agents.

    Usage:
        tools = ToolRegistry()
        tools.register(RAGTool(corpus_dir="./evidence"))
        tools.register(LegalSearchTool(backend="courtlistener"))

        # During agent turn, the orchestrator can call:
        result = tools.invoke("case_law_rag", "negligence standard")
        result = tools.invoke("legal_fact_check", "duty of care precedent")
    """

    def __init__(self):
        self._tools: Dict[str, AgentTool] = {}

    def register(self, tool: AgentTool):
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[AgentTool]:
        return self._tools.get(name)

    def invoke(self, tool_name: str, query: str, **kwargs) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name, query=query, results=[],
                source="unknown", success=False,
                error=f"Tool '{tool_name}' not registered.",
            )
        return tool.search(query, **kwargs)

    def invoke_all(self, query: str, categories: Optional[List[str]] = None, **kwargs) -> List[ToolResult]:
        """Invoke all tools (optionally filtered by category)."""
        results = []
        for tool in self._tools.values():
            if categories and tool.category not in categories:
                continue
            results.append(tool.search(query, **kwargs))
        return results

    def list_tools(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description, "category": t.category}
            for t in self._tools.values()
        ]

    def has_category(self, category: str) -> bool:
        return any(t.category == category for t in self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools.keys())})"


# ---------------------------------------------------------------------------
# Convenience: create a standard courtroom tool-set
# ---------------------------------------------------------------------------

def create_court_tools(
    evidence_dir: str = "./evidence",
    case_law_dir: str = "./case_law",
    search_backend: str = "courtlistener",
    search_api_key: Optional[str] = None,
    use_embeddings: bool = False,
) -> ToolRegistry:
    """
    Build the default tool registry for a courtroom simulation.

    Returns a ToolRegistry with:
      - case_law_rag       (RAG over case-law corpus)
      - evidence_rag       (RAG over case-specific evidence)
      - legal_fact_check   (web search fallback for fact-checking)
    """
    registry = ToolRegistry()

    # Case-law corpus
    case_rag = RAGTool(
        corpus_dir=case_law_dir,
        chunk_size=800,
        chunk_overlap=200,
        use_embeddings=use_embeddings,
    )
    case_rag.name = "case_law_rag"
    case_rag.description = "Search the case law database for precedents, statutes, and legal standards"
    case_rag.load()
    registry.register(case_rag)

    # Evidence files
    evidence_rag = RAGTool(
        corpus_dir=evidence_dir,
        chunk_size=600,
        chunk_overlap=150,
        use_embeddings=use_embeddings,
    )
    evidence_rag.name = "evidence_rag"
    evidence_rag.description = "Search the specific evidence files submitted for this case"
    evidence_rag.load()
    registry.register(evidence_rag)

    # Fact-check web search (labelled as fact-checking so it is used as fallback)
    fact_check = LegalSearchTool(
        backend=search_backend,
        api_key=search_api_key,
        max_results=5,
    )
    fact_check.name = "legal_fact_check"
    fact_check.description = (
        "FACT-CHECKING TOOL: Search external legal databases and the web. "
        "Use ONLY when the internal case-law library and evidence files do not "
        "contain sufficient information to support or refute a legal claim."
    )
    registry.register(fact_check)

    return registry
