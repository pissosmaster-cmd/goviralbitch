"""
Main pipeline orchestration for Content Skeleton Ripper.
Ported from ReelRecon — uses InstaClient instead of cookie-based session.

Split into three data-only phases:
  1. scrape_and_transcribe  — download videos, transcribe with local Whisper,
                              write transcripts + extraction prompt to disk
  2. aggregate_and_finish   — read extraction results (written by Claude),
                              aggregate patterns, write synthesis prompt
  3. finalize               — read synthesis results (written by Claude),
                              merge into data/recon/
"""

import os
import json
import uuid
import time
import traceback
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Callable
from enum import Enum

from .cache import TranscriptCache, is_valid_transcript
from .extractor import BatchedExtractor
from .aggregator import SkeletonAggregator, AggregatedData
from .synthesizer import PatternSynthesizer, SynthesisResult, generate_report
from recon.utils.logger import get_logger

# Import recon scrapers (replaces ReelRecon's cookie-based scraper)
from recon.scraper.instagram import InstaClient
from recon.scraper.downloader import (
    transcribe_video,
    load_whisper_model,
    download_direct,
)
from recon.config import load_config

logger = get_logger()

RECON_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "recon"


class JobStatus(Enum):
    PENDING = "pending"
    SCRAPING = "scraping"
    TRANSCRIBING = "transcribing"
    EXTRACTING = "extracting"
    AGGREGATING = "aggregating"
    SYNTHESIZING = "synthesizing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class JobProgress:
    status: JobStatus = JobStatus.PENDING
    phase: str = ""
    message: str = ""
    videos_scraped: int = 0
    videos_downloaded: int = 0
    videos_transcribed: int = 0
    transcripts_from_cache: int = 0
    valid_transcripts: int = 0
    skeletons_extracted: int = 0
    total_target: int = 0
    current_creator: str = ""
    current_creator_index: int = 0
    total_creators: int = 0
    reels_fetched: int = 0
    current_video_index: int = 0
    extraction_batch: int = 0
    extraction_total_batches: int = 0
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    errors: list[str] = field(default_factory=list)


@dataclass
class JobConfig:
    usernames: list[str]
    videos_per_creator: int = 3
    platform: str = "instagram"
    whisper_model: str = "small.en"


@dataclass
class JobResult:
    job_id: str
    success: bool
    config: JobConfig
    progress: JobProgress
    output_dir: Optional[str] = None
    skeletons: list[dict] = field(default_factory=list)
    aggregated: Optional[AggregatedData] = None
    synthesis: Optional[SynthesisResult] = None
    report_path: Optional[str] = None
    skeletons_path: Optional[str] = None
    synthesis_path: Optional[str] = None


class ReconPipeline:
    """
    Main pipeline — three data-only phases, no LLM calls.

    Phase 1: scrape_and_transcribe  — produces transcripts.json + extraction-prompt.md
    Phase 2: aggregate_and_finish   — reads extraction-results.json, produces synthesis-prompt.md
    Phase 3: finalize               — reads synthesis-results.json, saves final report
    """

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            base_dir = str(RECON_DATA_DIR)
        self.base_dir = Path(base_dir)
        self.output_dir = RECON_DATA_DIR / 'reports'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache = TranscriptCache()
        logger.info("PIPELINE", "ReconPipeline initialized")

    # ------------------------------------------------------------------
    # Phase 1 — Scrape + Transcribe
    # ------------------------------------------------------------------

    def scrape_and_transcribe(
        self,
        config: JobConfig,
        on_progress: Optional[Callable[[JobProgress], None]] = None,
    ) -> str:
        """Phase 1: Download videos, transcribe with local Whisper, save to output_dir.

        Returns output_dir path with transcripts.json and extraction-prompt.md ready.
        """
        job_id = f"sr_{uuid.uuid4().hex[:8]}"
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        output_dir = str(self.output_dir / f"{timestamp}_{job_id}")
        os.makedirs(output_dir, exist_ok=True)

        progress = JobProgress(
            status=JobStatus.SCRAPING,
            phase="Scraping videos...",
            started_at=datetime.utcnow().isoformat(),
            total_target=len(config.usernames) * config.videos_per_creator,
            total_creators=len(config.usernames),
        )
        self._notify(on_progress, progress)

        try:
            transcripts = self._scrape_and_transcribe(
                config=config, progress=progress, on_progress=on_progress,
            )

            valid_count = sum(1 for t in transcripts if is_valid_transcript(t.get('transcript', '')))
            progress.valid_transcripts = valid_count

            if valid_count == 0:
                raise ValueError("No valid transcripts to process")

            valid_transcripts = [t for t in transcripts if is_valid_transcript(t.get('transcript', ''))]

            # Write transcripts + extraction prompt via extractor
            progress.status = JobStatus.EXTRACTING
            progress.phase = "Preparing extraction prompt..."
            self._notify(on_progress, progress)

            extractor = BatchedExtractor()
            extractor.prepare_extraction(valid_transcripts, output_dir)

            progress.phase = f"Phase 1 complete — {len(valid_transcripts)} transcripts ready"
            progress.message = f"Extraction prompt written to {output_dir}"
            self._notify(on_progress, progress)

            logger.info("PIPELINE", f"Phase 1 complete: {output_dir}")
            return output_dir

        except Exception as e:
            logger.error("PIPELINE", f"Phase 1 failed: {e}")
            progress.status = JobStatus.FAILED
            progress.phase = "Failed"
            progress.errors.append(str(e))
            self._notify(on_progress, progress)
            raise

    # ------------------------------------------------------------------
    # Phase 2 — Load Extraction Results + Aggregate + Prepare Synthesis
    # ------------------------------------------------------------------

    def aggregate_and_finish(
        self,
        output_dir: str,
        on_progress: Optional[Callable[[JobProgress], None]] = None,
    ) -> str:
        """Phase 2: Read extraction results (written by Claude), aggregate, prepare synthesis.

        Called AFTER Claude has written extraction-results.json to output_dir.
        Returns output_dir (synthesis-prompt.md ready for Claude).
        """
        progress = JobProgress(
            status=JobStatus.AGGREGATING,
            phase="Loading extraction results...",
        )
        self._notify(on_progress, progress)

        try:
            # Load transcripts for metadata enrichment
            transcripts_path = os.path.join(output_dir, "transcripts.json")
            transcripts = None
            if os.path.exists(transcripts_path):
                with open(transcripts_path) as f:
                    transcripts = json.load(f)

            # Load extraction results
            extractor = BatchedExtractor()
            skeletons = extractor.load_extraction_results(output_dir, transcripts=transcripts)

            if not skeletons:
                raise ValueError("No valid skeletons in extraction results")

            # Aggregate patterns
            progress.phase = "Aggregating patterns..."
            progress.skeletons_extracted = len(skeletons)
            self._notify(on_progress, progress)

            aggregator = SkeletonAggregator()
            aggregated = aggregator.aggregate(skeletons)

            # Save skeletons for later use in finalize
            skeletons_out = os.path.join(output_dir, "skeletons.json")
            with open(skeletons_out, 'w', encoding='utf-8') as f:
                json.dump(skeletons, f, indent=2, default=str)

            # Prepare synthesis prompt
            progress.status = JobStatus.SYNTHESIZING
            progress.phase = "Preparing synthesis prompt..."
            self._notify(on_progress, progress)

            synthesizer = PatternSynthesizer()
            synthesizer.prepare_synthesis(skeletons, output_dir)

            progress.phase = f"Phase 2 complete — synthesis prompt ready"
            progress.message = f"Synthesis prompt written to {output_dir}"
            self._notify(on_progress, progress)

            logger.info("PIPELINE", f"Phase 2 complete: {output_dir}")
            return output_dir

        except Exception as e:
            logger.error("PIPELINE", f"Phase 2 failed: {e}")
            progress.status = JobStatus.FAILED
            progress.phase = "Failed"
            progress.errors.append(str(e))
            self._notify(on_progress, progress)
            raise

    # ------------------------------------------------------------------
    # Phase 3 — Load Synthesis Results + Save Final Report
    # ------------------------------------------------------------------

    def finalize(
        self,
        output_dir: str,
        job_config: Optional[JobConfig] = None,
        on_progress: Optional[Callable[[JobProgress], None]] = None,
    ) -> dict:
        """Phase 3: Read synthesis results, merge into data/recon/.

        Called AFTER Claude has written synthesis-results.json.
        Returns a dict with paths to the final report, skeletons, and synthesis files.
        """
        progress = JobProgress(
            status=JobStatus.COMPLETE,
            phase="Loading synthesis results...",
        )
        self._notify(on_progress, progress)

        try:
            # Load synthesis results
            synthesizer = PatternSynthesizer()
            synthesis = synthesizer.load_synthesis_results(output_dir)

            # Load skeletons for the report
            skeletons_path = os.path.join(output_dir, "skeletons.json")
            skeletons = []
            if os.path.exists(skeletons_path):
                with open(skeletons_path) as f:
                    skeletons = json.load(f)

            # Aggregate for the report summary
            aggregator = SkeletonAggregator()
            aggregated = aggregator.aggregate(skeletons)

            # Save synthesis results
            synthesis_out = os.path.join(output_dir, "synthesis.json")
            with open(synthesis_out, 'w', encoding='utf-8') as f:
                json.dump({
                    'success': synthesis.success,
                    'analysis': synthesis.analysis,
                    'templates': synthesis.templates,
                    'quick_wins': synthesis.quick_wins,
                    'warnings': synthesis.warnings,
                    'model_used': synthesis.model_used,
                    'synthesized_at': synthesis.synthesized_at,
                }, f, indent=2)

            # Generate markdown report
            config_dict = asdict(job_config) if job_config else {}
            report_content = generate_report(
                data=aggregated, synthesis=synthesis, job_config=config_dict,
            )
            report_path = os.path.join(output_dir, "report.md")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report_content)

            progress.status = JobStatus.COMPLETE
            progress.phase = "Analysis Complete"
            progress.message = f"Done: {len(skeletons)} skeletons analyzed"
            progress.completed_at = datetime.utcnow().isoformat()
            self._notify(on_progress, progress)

            logger.info("PIPELINE", f"Phase 3 complete: {output_dir}")

            return {
                'report': report_path,
                'skeletons': skeletons_path,
                'synthesis': synthesis_out,
                'skeleton_count': len(skeletons),
                'output_dir': output_dir,
            }

        except Exception as e:
            logger.error("PIPELINE", f"Phase 3 failed: {e}")
            progress.status = JobStatus.FAILED
            progress.phase = "Failed"
            progress.errors.append(str(e))
            self._notify(on_progress, progress)
            raise

    # ------------------------------------------------------------------
    # Internal — Scrape + Transcribe helpers
    # ------------------------------------------------------------------

    def _scrape_and_transcribe(
        self,
        config: JobConfig,
        progress: JobProgress,
        on_progress: Optional[Callable],
    ) -> list[dict]:
        """Scrape videos and get transcripts using Instaloader for IG."""
        transcripts = []

        # Load local Whisper model
        progress.status = JobStatus.TRANSCRIBING
        progress.message = f"Loading Whisper model ({config.whisper_model})..."
        self._notify(on_progress, progress)
        whisper_model = load_whisper_model(config.whisper_model)
        if not whisper_model:
            raise RuntimeError(f"Failed to load Whisper model '{config.whisper_model}'")

        # Setup InstaClient for IG competitors
        insta_client = None
        if config.platform == 'instagram':
            recon_config = load_config()
            if recon_config.ig_username and recon_config.ig_password:
                insta_client = InstaClient()
                if not insta_client.login(recon_config.ig_username, recon_config.ig_password):
                    raise RuntimeError("Instagram login failed. Check IG_USERNAME/IG_PASSWORD.")
            else:
                raise RuntimeError("IG credentials not configured. Set IG_USERNAME and IG_PASSWORD env vars or run settings.")

        temp_dir = RECON_DATA_DIR / 'temp'
        temp_dir.mkdir(parents=True, exist_ok=True)

        for idx, username in enumerate(config.usernames):
            progress.current_creator = username
            progress.current_creator_index = idx + 1
            progress.current_video_index = 0
            progress.phase = f"Processing @{username} ({idx + 1}/{len(config.usernames)})"
            progress.message = "Checking cache..."
            self._notify(on_progress, progress)

            # Check cache first
            cached = self._get_cached_transcripts(config.platform, username, config.videos_per_creator)
            if cached and len(cached) >= config.videos_per_creator:
                progress.transcripts_from_cache += len(cached[:config.videos_per_creator])
                transcripts.extend(cached[:config.videos_per_creator])
                continue

            # Fetch reel metadata
            progress.message = f"Fetching reels from @{username}..."
            self._notify(on_progress, progress)

            if config.platform == 'instagram' and insta_client:
                reels = insta_client.get_competitor_reels(username, max_reels=100)
                if not reels:
                    progress.errors.append(f"@{username}: No reels found")
                    continue
                progress.videos_scraped += len(reels)
                progress.reels_fetched = len(reels)
            else:
                progress.errors.append(f"@{username}: Platform {config.platform} not yet supported in pipeline")
                continue

            # Iterate through reels until we have enough valid transcripts
            valid_count = 0
            for reel in reels:
                if valid_count >= config.videos_per_creator:
                    break

                video_id = reel.get('shortcode', 'unknown')
                views_display = f"{reel.get('views', 0):,}"

                # Check cache
                cached_text = self.cache.get(config.platform, username, video_id)
                if cached_text and is_valid_transcript(cached_text):
                    valid_count += 1
                    transcripts.append({
                        'video_id': video_id, 'username': username,
                        'platform': config.platform, 'views': reel.get('views', 0),
                        'likes': reel.get('likes', 0), 'url': reel.get('url', ''),
                        'video_url': reel.get('video_url', ''),
                        'transcript': cached_text, 'from_cache': True
                    })
                    progress.transcripts_from_cache += 1
                    progress.videos_transcribed += 1
                    continue

                # Download and transcribe
                progress.message = f"Video {valid_count + 1}/{config.videos_per_creator}: Downloading ({views_display} views)"
                self._notify(on_progress, progress)

                video_path = temp_dir / f"{username}_{video_id}.mp4"
                video_url = reel.get('video_url', '')

                downloaded = False
                if video_url:
                    downloaded = download_direct(video_url, video_path)
                if not downloaded and insta_client:
                    downloaded = insta_client.download_reel(video_id, video_path)

                if not downloaded or not video_path.exists():
                    continue

                progress.videos_downloaded += 1
                progress.message = f"Video {valid_count + 1}/{config.videos_per_creator}: Transcribing ({views_display} views)"
                self._notify(on_progress, progress)

                # Transcribe with local Whisper
                transcript_text = transcribe_video(str(video_path), whisper_model)

                # Cleanup video
                try:
                    if video_path.exists():
                        video_path.unlink()
                except OSError:
                    pass

                if transcript_text and is_valid_transcript(transcript_text):
                    self.cache.set(config.platform, username, video_id, transcript_text)
                    transcripts.append({
                        'video_id': video_id, 'username': username,
                        'platform': config.platform, 'views': reel.get('views', 0),
                        'likes': reel.get('likes', 0), 'url': reel.get('url', ''),
                        'video_url': reel.get('video_url', ''),
                        'transcript': transcript_text, 'from_cache': False
                    })
                    valid_count += 1
                    progress.videos_transcribed += 1

            if valid_count < config.videos_per_creator:
                progress.errors.append(f"@{username}: Only {valid_count}/{config.videos_per_creator} valid transcripts")

        return transcripts

    def _get_cached_transcripts(self, platform: str, username: str, count: int) -> list[dict]:
        cached = []
        cache_pattern = f"{platform.lower()}_{username.lower()}_*.txt"
        cache_files = list(self.cache.cache_dir.glob(cache_pattern))
        for cache_file in cache_files[:count]:
            try:
                transcript_text = cache_file.read_text(encoding='utf-8')
                if is_valid_transcript(transcript_text):
                    parts = cache_file.stem.split('_')
                    video_id = parts[-1] if len(parts) >= 3 else cache_file.stem
                    cached.append({
                        'video_id': video_id, 'username': username,
                        'platform': platform, 'views': 0, 'likes': 0,
                        'url': '', 'transcript': transcript_text, 'from_cache': True
                    })
            except Exception:
                pass
        return cached

    def _notify(self, callback, progress):
        if callback:
            try:
                callback(progress)
            except Exception:
                pass


# ======================================================================
# Backward-compat aliases
# ======================================================================

# Old name kept as alias so existing imports still work
SkeletonRipperPipeline = ReconPipeline


def create_job_config(
    usernames: list[str],
    videos_per_creator: int = 3,
    platform: str = "instagram",
    whisper_model: str = "small.en",
    # Deprecated kwargs accepted but ignored for backward compat
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    transcribe_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> JobConfig:
    return JobConfig(
        usernames=usernames,
        videos_per_creator=videos_per_creator,
        platform=platform,
        whisper_model=whisper_model,
    )


def run_skeleton_ripper(
    usernames: list[str],
    videos_per_creator: int = 3,
    platform: str = "instagram",
    whisper_model: str = "small.en",
    on_progress: Optional[Callable] = None,
    # Deprecated kwargs accepted but ignored
    llm_provider: Optional[str] = None,
    llm_model: Optional[str] = None,
    transcribe_provider: Optional[str] = None,
    openai_api_key: Optional[str] = None,
) -> str:
    """Convenience wrapper — runs Phase 1 and returns output_dir."""
    config = create_job_config(
        usernames=usernames,
        videos_per_creator=videos_per_creator,
        platform=platform,
        whisper_model=whisper_model,
    )
    pipeline = ReconPipeline()
    return pipeline.scrape_and_transcribe(config, on_progress=on_progress)


# ======================================================================
# Provider discovery — moved from llm_client.py (UI still needs it)
# ======================================================================

def get_available_providers() -> list[dict]:
    """Return available LLM providers for the settings UI.

    This is informational only — the pipeline no longer calls LLMs directly.
    Kept so the web UI can still show which API keys are configured.
    """
    import requests as _requests

    _PROVIDER_DEFS = [
        {'id': 'openai', 'name': 'OpenAI', 'api_key_env': 'OPENAI_API_KEY', 'models': [
            {'id': 'gpt-4o-mini', 'name': 'GPT-4o Mini', 'cost_tier': 'low'},
            {'id': 'gpt-4o', 'name': 'GPT-4o', 'cost_tier': 'medium'},
        ]},
        {'id': 'anthropic', 'name': 'Anthropic', 'api_key_env': 'ANTHROPIC_API_KEY', 'models': [
            {'id': 'claude-3-haiku-20240307', 'name': 'Claude 3 Haiku', 'cost_tier': 'low'},
            {'id': 'claude-3-sonnet-20240229', 'name': 'Claude 3 Sonnet', 'cost_tier': 'medium'},
        ]},
        {'id': 'google', 'name': 'Google', 'api_key_env': 'GOOGLE_API_KEY', 'models': [
            {'id': 'gemini-1.5-flash', 'name': 'Gemini 1.5 Flash', 'cost_tier': 'low'},
            {'id': 'gemini-1.5-pro', 'name': 'Gemini 1.5 Pro', 'cost_tier': 'medium'},
        ]},
        {'id': 'local', 'name': 'Local (Ollama)', 'api_key_env': '', 'models': [
            {'id': 'qwen3', 'name': 'Qwen 3 (Recommended)', 'cost_tier': 'free'},
            {'id': 'llama3', 'name': 'Llama 3', 'cost_tier': 'free'},
            {'id': 'mistral', 'name': 'Mistral', 'cost_tier': 'free'},
        ]},
    ]

    result = []
    for pdef in _PROVIDER_DEFS:
        available = False
        models = []
        if pdef['id'] == 'local':
            try:
                resp = _requests.get('http://localhost:11434/api/tags', timeout=2)
                if resp.status_code == 200:
                    available = True
                    data = resp.json()
                    installed = {m['name'].split(':')[0] for m in data.get('models', [])}
                    models = [m for m in pdef['models'] if m['id'] in installed]
            except _requests.exceptions.RequestException:
                pass
        else:
            api_key = os.getenv(pdef['api_key_env'])
            if api_key:
                available = True
                models = pdef['models']

        result.append({
            'id': pdef['id'],
            'name': pdef['name'],
            'available': available,
            'models': models,
        })

    return result
