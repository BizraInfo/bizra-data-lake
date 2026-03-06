# Phase 67.05 — AKIS v2 Knowledge Extraction Pipeline
# ════════════════════════════════════════════════════

## Standing on Giants
- Shannon (1948): Information theory — extract signal from noise
- Berners-Lee (1989): URL as universal resource identifier
- Page & Brin (1998): Structured web extraction
- Al-Khwarizmi (780-850): Systematic procedure for knowledge processing

## Source

- `last update/BIZRA_AKIS_v2.py` (390 lines) — Multi-source extraction engine
- `last update/BIZRA_Knowledge_Extractor.py` (~400 lines) — Knowledge extraction v1
- `last update/BIZRA_KIS_v2.jsx` — React Knowledge Intelligence System dashboard

## Purpose

AKIS (Adaptive Knowledge Intelligence System) extracts structured knowledge from
any public URL, scores it for BIZRA relevance, and outputs clean text ready for
the Knowledge Intelligence Dashboard or the data pipeline.

This is the **sensory layer** of the Sovereignty OS — how the system perceives
and ingests external knowledge.

## Target

```
core/akis/
├── __init__.py
├── extractor.py       # Multi-source extraction engine
├── sources/
│   ├── __init__.py
│   ├── youtube.py     # YouTube video + channel extraction
│   ├── github.py      # GitHub repo extraction
│   ├── arxiv.py       # ArXiv paper extraction
│   └── web.py         # Generic web page extraction
├── relevance.py       # BIZRA relevance scoring
├── formatter.py       # Output formatting (text, JSON, clipboard)
└── cli.py             # CLI interface
```

## Architecture

```
URL → detect_source() → Source-Specific Extractor → Unified Result
                                                          │
                                                    ┌─────┴──────┐
                                                    │ relevance  │
                                                    │ scoring    │
                                                    └─────┬──────┘
                                                          │
                                              ┌───────────┼───────────┐
                                              │           │           │
                                         text output  JSON output  pipeline
                                         (terminal)  (structured) (04_GOLD/)
```

## Pseudocode

### Source Detection

```
MODULE extractor

FUNCTION detect_source(url: str) -> str:
    """Classify URL into source type for routing to correct extractor."""
    host = urlparse(url).hostname.lower()
    path = urlparse(url).path

    IF "youtube.com" IN host OR "youtu.be" IN host:
        IF "/@" IN path OR "/channel/" IN path OR "/c/" IN path:
            RETURN "youtube_channel"
        IF "/playlist" IN path:
            RETURN "youtube_playlist"
        RETURN "youtube_video"

    IF "github.com" IN host:
        RETURN "github"

    IF "arxiv.org" IN host:
        RETURN "arxiv"

    RETURN "web"
```

### YouTube Extraction

```
MODULE sources.youtube

FUNCTION extract_youtube_video(url: str) -> ExtractionResult:
    """Extract video metadata + transcript via yt-dlp.

    Pathways (fallback chain):
    1. yt-dlp --dump-json → metadata (title, channel, description, tags, chapters)
    2. yt-dlp --write-auto-sub (VTT format) → transcript
    3. yt-dlp --write-auto-sub (json3 format) → transcript (fallback)
    4. Description text → content (last resort)
    """
    result = ExtractionResult(type="youtube_video", url=url)

    # Metadata via yt-dlp JSON dump
    metadata = run(["yt-dlp", "--no-check-certificates", "--dump-json",
                     "--no-download", "--no-warnings", "--quiet", url])
    IF metadata.ok:
        m = json.loads(metadata.stdout)
        result.title = m.get("title", "")
        result.channel = m.get("channel", m.get("uploader", ""))
        result.description = m.get("description", "")[:3000]
        result.duration = m.get("duration", 0)
        result.views = m.get("view_count", 0)
        result.tags = m.get("tags", [])[:15]
        result.chapters = extract_chapters(m)
        vid_id = m.get("id", "")

    # Transcript via subtitles
    transcript = extract_subtitles(url, vid_id)
    IF transcript:
        result.transcript = transcript
    ELSE:
        result.transcript = result.description  # Fallback

    # Reference URLs from description
    result.references = extract_urls(result.description)

    RETURN result

FUNCTION extract_youtube_channel(url: str, max_videos: int = 20) -> ExtractionResult:
    """Extract channel video listing with BIZRA relevance ranking."""
    result = ExtractionResult(type="youtube_channel", url=url)

    # Flat playlist dump via yt-dlp
    listing = run(["yt-dlp", "--no-check-certificates", "--flat-playlist",
                    "--dump-json", "--playlist-end", str(max_videos),
                    "--no-warnings", f"{url}/videos"])

    IF listing.ok:
        FOR line IN listing.stdout.strip().split("\n"):
            IF NOT line.strip(): CONTINUE
            v = json.loads(line)
            result.videos.append(VideoEntry(
                id=v.get("id", ""),
                title=v.get("title", ""),
                url=f"https://www.youtube.com/watch?v={v.get('id', '')}",
                duration=v.get("duration", 0),
                views=v.get("view_count", 0),
                description=v.get("description", "")[:300]
            ))

    # BIZRA relevance scoring
    score_videos_for_relevance(result.videos)
    result.videos.sort(key=lambda v: v.bizra_score, reverse=True)

    RETURN result

FUNCTION parse_vtt(path: Path) -> str:
    """Parse VTT subtitle file into clean text.
    Deduplicates consecutive identical lines (auto-sub artifact).
    """
    content = path.read_text(errors="replace")
    lines = []
    FOR line IN content.split("\n"):
        line = line.strip()
        # Skip headers and timestamps
        IF NOT line OR line.startswith("WEBVTT") OR "-->" IN line:
            CONTINUE
        IF re.match(r'^\d+$', line):
            CONTINUE
        # Strip HTML tags
        line = re.sub(r'<[^>]+>', '', line)
        IF line AND (NOT lines OR line != lines[-1]):
            lines.append(line)
    RETURN " ".join(lines)
```

### GitHub Extraction

```
MODULE sources.github

FUNCTION extract_github(url: str) -> ExtractionResult:
    """Extract repo metadata, README, and top issues via GitHub API."""
    result = ExtractionResult(type="github", url=url)
    parts = parse_github_url(url)
    api = f"https://api.github.com/repos/{parts.owner}/{parts.repo}"

    # Metadata
    metadata = http_get(api)
    IF metadata.ok:
        m = metadata.json()
        result.title = m.get("full_name", "")
        result.description = m.get("description", "")
        result.stars = m.get("stargazers_count", 0)
        result.language = m.get("language", "")
        result.topics = m.get("topics", [])

    # README (raw)
    readme = http_get(f"{api}/readme",
                       headers={"Accept": "application/vnd.github.v3.raw"})
    IF readme.ok AND len(readme.text) > 50:
        result.readme = readme.text[:10000]

    # Top issues (by comment count — real discussions)
    issues = http_get(f"{api}/issues?state=all&per_page=5&sort=comments&direction=desc")
    IF issues.ok:
        result.issues = [
            f"{i['title']} — {(i.get('body') or '')[:200]}"
            FOR i IN issues.json()[:5]
            IF isinstance(issues.json(), list)
        ]

    # Combine
    result.content = "\n\n".join(filter(None, [
        f"# {result.title}", result.description, result.readme,
        "\n".join(result.issues or [])
    ]))
    result.references = extract_urls(result.readme or "")

    RETURN result
```

### BIZRA Relevance Scoring

```
MODULE relevance

BIZRA_KEYWORDS = [
    "agent", "multi-agent", "sovereign", "token", "governance", "scoring",
    "evaluation", "safety", "formal", "proof", "verification", "consensus",
    "offline", "edge", "local", "constitutional", "ethics", "redistribution",
    "soulbound", "dual", "protocol", "adversarial", "mcp", "orchestrat",
    "agentic", "reasoning", "graph", "topology", "scaling",
]

FUNCTION relevance_score(text: str) -> float:
    """Compute BIZRA relevance score for extracted text.

    Score = count of matching keywords / total keywords.
    Range: [0.0, 1.0]. Higher = more relevant to BIZRA research.
    """
    text_lower = text.lower()
    matches = sum(1 FOR kw IN BIZRA_KEYWORDS IF kw IN text_lower)
    RETURN matches / len(BIZRA_KEYWORDS)

FUNCTION score_videos_for_relevance(videos: List[VideoEntry]) -> None:
    """Score each video by BIZRA keyword density in title + description."""
    FOR video IN videos:
        combined = f"{video.title} {video.description}".lower()
        video.bizra_score = relevance_score(combined)
```

### Unified Extraction

```
FUNCTION extract(url: str) -> ExtractionResult:
    """Master extraction: detect source, route to extractor, unify output."""
    source = detect_source(url)

    IF source == "youtube_video":
        data = extract_youtube_video(url)
    ELIF source IN ("youtube_channel", "youtube_playlist"):
        data = extract_youtube_channel(url)
    ELIF source == "github":
        data = extract_github(url)
    ELIF source == "arxiv":
        data = extract_arxiv(url)
    ELSE:
        data = extract_web(url)

    # Unified post-processing
    text_parts = [data.get(k) FOR k IN ["transcript", "content", "readme", "description"]
                  IF data.get(k)]
    data.extracted_text = "\n\n".join(text_parts)
    data.word_count = len(data.extracted_text.split())
    data.extracted_at = datetime.now(timezone.utc).isoformat()
    data.relevance = relevance_score(data.extracted_text)

    RETURN data
```

## Integration with Data Pipeline

AKIS output feeds into the existing data pipeline:

```
AKIS extract(url)
    → 00_INTAKE/akis_{timestamp}.json      # Raw extraction
    → corpus_manager.py                     # Layer 1: documents.parquet
    → vector_engine.py                      # Layer 2: embeddings
    → langextract_engine.py                 # Layer 4: LLM assertions
```

## CLI

```
python -m core.akis <url>               # Extract and print
python -m core.akis <url> --json        # Structured JSON
python -m core.akis <channel> --scan    # Channel relevance ranking
python -m core.akis --batch urls.txt    # Batch processing
```

## TDD Anchors

```python
# tests/akis/test_extractor.py

def test_detect_source_youtube():
    assert detect_source("https://youtube.com/watch?v=abc") == "youtube_video"
    assert detect_source("https://youtube.com/@channel") == "youtube_channel"
    assert detect_source("https://youtu.be/abc") == "youtube_video"

def test_detect_source_github():
    assert detect_source("https://github.com/owner/repo") == "github"

def test_detect_source_web():
    assert detect_source("https://example.com/page") == "web"

def test_parse_vtt_deduplicates():
    vtt = "WEBVTT\n\n00:00.000 --> 00:01.000\nhello\nhello\nworld\n"
    result = parse_vtt_text(vtt)
    assert result == "hello world"  # Deduplicated

def test_relevance_scoring():
    high = relevance_score("multi-agent sovereign consensus protocol")
    low = relevance_score("cooking recipe for pasta carbonara")
    assert high > low
    assert 0.0 <= high <= 1.0

def test_extract_result_has_metadata():
    result = ExtractionResult(type="web", url="https://example.com")
    result.extracted_text = "test content"
    assert result.word_count == 2

# tests/akis/test_github_extraction.py (requires_network)
@pytest.mark.requires_network
def test_github_extraction_structure():
    result = extract_github("https://github.com/python/cpython")
    assert result.title
    assert result.stars > 0
    assert result.readme
```
