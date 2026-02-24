"""Orchestrates the full ingestion pipeline: detect -> parse -> dedup -> enrich -> output.

The pipeline reads conversation exports from multiple platforms, normalizes
them into a unified schema, deduplicates across platforms, enriches with
metadata, and writes the results as Parquet + JSON indices.

Ref: specs/user-zero-bootstrap/phase_01_multi_platform_ingestion.md S7

Standing on Giants: Shannon (channel capacity) - Lamport (ordered events)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.genesis.ingestion.dedup import deduplicate
from core.genesis.ingestion.enrichment import enrich
from core.genesis.ingestion.parsers import PARSER_MAP, detect_platform
from core.genesis.ingestion.schema import ConversationTurn, Platform

log = logging.getLogger(__name__)


class IngestionReport:
    """Summary statistics for an ingestion run."""

    def __init__(self) -> None:
        self.files_processed: int = 0
        self.files_skipped: int = 0
        self.total_turns_raw: int = 0
        self.total_turns_deduped: int = 0
        self.platforms: dict[str, int] = {}
        self.date_range: tuple[str, str] = ("", "")
        self.languages: dict[str, int] = {}
        self.elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "total_turns_raw": self.total_turns_raw,
            "total_turns_deduped": self.total_turns_deduped,
            "platforms": self.platforms,
            "date_range": list(self.date_range),
            "languages": self.languages,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }


class IngestPipeline:
    """Multi-platform conversation ingestion pipeline.

    Usage::

        pipeline = IngestPipeline(output_dir=Path("04_GOLD"))
        report = pipeline.run(input_dirs=[Path("00_INTAKE/conversations")])
    """

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        input_dirs: list[Path] | None = None,
        input_files: list[Path] | None = None,
        platform_hint: Platform | None = None,
    ) -> IngestionReport:
        """Execute the full ingestion pipeline.

        Args:
            input_dirs: Directories to scan for conversation files
                (``*.json``, ``*.jsonl``, ``*.md``).
            input_files: Explicit list of files to process.
            platform_hint: Force a specific platform parser instead of
                auto-detection.

        Returns:
            IngestionReport with summary statistics.
        """
        start = time.monotonic()
        report = IngestionReport()

        # Collect input files
        all_files: list[Path] = list(input_files or [])
        for d in input_dirs or []:
            if d.is_dir():
                all_files.extend(sorted(d.rglob("*.json")))
                all_files.extend(sorted(d.rglob("*.jsonl")))
                all_files.extend(sorted(d.rglob("*.md")))

        log.info("Found %d input files", len(all_files))

        # Parse all files
        all_turns: list[ConversationTurn] = []
        for filepath in all_files:
            try:
                turns = self._parse_file(filepath, platform_hint)
                if turns:
                    all_turns.extend(turns)
                    platform_name = turns[0].platform.value
                    report.platforms[platform_name] = report.platforms.get(
                        platform_name, 0
                    ) + len(turns)
                    report.files_processed += 1
                else:
                    report.files_skipped += 1
            except Exception as exc:
                log.warning("Failed to parse %s: %s", filepath.name, exc)
                report.files_skipped += 1

        report.total_turns_raw = len(all_turns)
        log.info(
            "Parsed %d total turns from %d files",
            len(all_turns),
            report.files_processed,
        )

        # Deduplicate
        deduped = deduplicate(all_turns)
        report.total_turns_deduped = len(deduped)

        # Enrich
        enriched = enrich(deduped)

        # Compute date range
        timestamps = [t.timestamp for t in enriched if t.timestamp]
        if timestamps:
            report.date_range = (
                min(timestamps).isoformat(),
                max(timestamps).isoformat(),
            )

        # Language stats
        for turn in enriched:
            if turn.language:
                report.languages[turn.language] = (
                    report.languages.get(turn.language, 0) + 1
                )

        # Write output
        self._write_parquet(enriched)
        self._write_indices(enriched)

        report.elapsed_seconds = time.monotonic() - start
        self._write_report(report)

        log.info(
            "Ingestion complete: %d -> %d turns in %.1fs",
            report.total_turns_raw,
            report.total_turns_deduped,
            report.elapsed_seconds,
        )
        return report

    def _parse_file(
        self, filepath: Path, platform_hint: Platform | None
    ) -> list[ConversationTurn]:
        """Parse a single file, auto-detecting platform if needed."""
        raw = filepath.read_bytes()

        if platform_hint:
            parser = PARSER_MAP.get(platform_hint)
            if parser:
                data = self._load_data(filepath, raw)
                return parser.parse(data, source_path=filepath)

        platform = detect_platform(raw)
        if platform is None:
            log.debug("Could not detect platform for %s", filepath.name)
            return []

        parser = PARSER_MAP[platform]
        data = self._load_data(filepath, raw)
        return parser.parse(data, source_path=filepath)

    def _load_data(self, filepath: Path, raw: bytes) -> Any:
        """Load file data as JSON, JSONL, or raw text."""
        text = raw.decode("utf-8", errors="replace")

        if filepath.suffix == ".jsonl":
            return text  # Let parser handle JSONL line-by-line

        if filepath.suffix == ".json":
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

        if filepath.suffix == ".md":
            return text

        # Unknown extension: try JSON, fall back to text
        try:
            return json.loads(text)
        except (json.JSONDecodeError, ValueError):
            return text

    def _write_parquet(self, turns: list[ConversationTurn]) -> None:
        """Write unified Parquet file with zstd compression."""
        if not turns:
            return

        try:
            import pandas as pd
            import pyarrow.parquet  # noqa: F401 — validates pyarrow is available

            records = []
            for t in turns:
                records.append(
                    {
                        "id": t.id,
                        "platform": t.platform.value,
                        "conversation_id": t.conversation_id,
                        "turn_index": t.turn_index,
                        "role": t.role.value,
                        "content": t.content,
                        "model": t.model,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                        "metadata_json": json.dumps(t.metadata),
                        "token_count": t.token_count,
                        "content_hash": t.content_hash,
                        "language": t.language,
                        "language_conf": t.language_conf,
                        "topics_json": json.dumps(t.topics),
                    }
                )

            df = pd.DataFrame(records)
            output_path = self.output_dir / "conversations_unified.parquet"
            df.to_parquet(
                output_path, engine="pyarrow", compression="zstd", index=False
            )
            log.info("Wrote %d rows to %s", len(df), output_path)
        except ImportError as e:
            log.warning("Parquet write skipped (missing dependency: %s)", e)

    def _write_indices(self, turns: list[ConversationTurn]) -> None:
        """Write platform index, timeline index, and dedup manifest."""
        platform_index: dict[str, list[str]] = {}
        timeline_index: dict[str, list[str]] = {}
        dedup_manifest: dict[str, dict[str, Any]] = {}

        for t in turns:
            pname = t.platform.value
            if pname not in platform_index:
                platform_index[pname] = []
            if t.conversation_id not in platform_index[pname]:
                platform_index[pname].append(t.conversation_id)

            if t.timestamp:
                month = t.timestamp.strftime("%Y-%m")
                if month not in timeline_index:
                    timeline_index[month] = []
                timeline_index[month].append(t.id)

            if t.content_hash:
                if t.content_hash not in dedup_manifest:
                    dedup_manifest[t.content_hash] = {
                        "canonical": t.id,
                        "duplicates": t.metadata.get("duplicate_ids", []),
                    }

        for name, data in [
            ("platform_index.json", platform_index),
            ("timeline_index.json", timeline_index),
            ("dedup_manifest.json", dedup_manifest),
        ]:
            path = self.output_dir / name
            path.write_text(json.dumps(data, indent=2))
            log.info("Wrote %s", path)

    def _write_report(self, report: IngestionReport) -> None:
        """Write ingestion report as JSON."""
        path = self.output_dir / "ingestion_report.json"
        path.write_text(json.dumps(report.to_dict(), indent=2))
        log.info("Wrote report: %s", path)
