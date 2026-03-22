"""
Content Skeleton Ripper — Multi-creator content pattern analysis.
Ported from ReelRecon with imports adjusted for content-pipeline.

Usage:
    from recon.skeleton_ripper import ReconPipeline, create_job_config

    config = create_job_config(
        usernames=['creator1', 'creator2'],
        videos_per_creator=3,
    )

    pipeline = ReconPipeline()
    output_dir = pipeline.scrape_and_transcribe(config)
"""

from .pipeline import (
    ReconPipeline,
    SkeletonRipperPipeline,  # backward-compat alias
    JobConfig,
    JobProgress,
    JobResult,
    JobStatus,
    create_job_config,
    run_skeleton_ripper,
    get_available_providers,
)
from .extractor import BatchedExtractor
from .synthesizer import PatternSynthesizer
from .aggregator import SkeletonAggregator
from .cache import TranscriptCache

__all__ = [
    'ReconPipeline', 'SkeletonRipperPipeline',
    'JobConfig', 'JobProgress', 'JobResult', 'JobStatus',
    'create_job_config', 'run_skeleton_ripper',
    'get_available_providers',
    'BatchedExtractor', 'PatternSynthesizer', 'SkeletonAggregator',
    'TranscriptCache',
]
