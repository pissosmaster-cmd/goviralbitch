"""
Batched content skeleton extractor — file-based I/O.

Writes transcripts + prompt to disk for Claude Code to process,
then reads back the extraction results. No direct LLM dependency.
"""

import json
import os
from datetime import datetime
from typing import Optional

from .prompts import build_extraction_prompt, validate_skeleton
from recon.utils.logger import get_logger

logger = get_logger()


class BatchedExtractor:
    """Prepare extraction prompts and load results via the filesystem."""

    def prepare_extraction(self, transcripts: list[dict], output_dir: str) -> str:
        """Write transcripts + prompt to files for Claude to process.

        Creates:
          - <output_dir>/extraction-prompt.md  — the full prompt text
          - <output_dir>/transcripts.json      — raw transcript data

        Returns the path to the prompt file.
        """
        os.makedirs(output_dir, exist_ok=True)

        prompt = build_extraction_prompt(transcripts)
        prompt_path = os.path.join(output_dir, "extraction-prompt.md")
        data_path = os.path.join(output_dir, "transcripts.json")

        with open(prompt_path, "w") as f:
            f.write(prompt)
        with open(data_path, "w") as f:
            json.dump(transcripts, f, indent=2)

        logger.info("EXTRACT", f"Wrote prompt to {prompt_path} ({len(transcripts)} transcripts)")
        return prompt_path

    def load_extraction_results(self, output_dir: str, transcripts: Optional[list[dict]] = None) -> list[dict]:
        """Read Claude's extraction output from file and enrich/validate.

        Expects <output_dir>/extraction-results.json to contain a JSON array
        of skeleton objects produced by the LLM.

        If *transcripts* is provided the skeletons are enriched with metadata
        from the original transcript dicts (username, platform, views, etc.).
        Invalid skeletons are logged and skipped.
        """
        results_path = os.path.join(output_dir, "extraction-results.json")
        with open(results_path) as f:
            raw_skeletons = json.load(f)

        if isinstance(raw_skeletons, dict):
            raw_skeletons = [raw_skeletons]

        valid: list[dict] = []
        for skeleton in raw_skeletons:
            is_valid, error = validate_skeleton(skeleton)
            if not is_valid:
                vid = skeleton.get('video_id', 'unknown')
                logger.warning("EXTRACT", f"Skeleton {vid} failed validation: {error}")
                continue

            # Enrich with original transcript metadata when available
            if transcripts:
                video_id = skeleton.get('video_id')
                original = next((t for t in transcripts if t.get('video_id') == video_id), None)
                if original:
                    skeleton['creator_username'] = original.get('username', 'unknown')
                    skeleton['platform'] = original.get('platform', 'unknown')
                    skeleton['views'] = original.get('views', 0)
                    skeleton['likes'] = original.get('likes', 0)
                    skeleton['url'] = original.get('url', '')
                    skeleton['video_url'] = original.get('video_url', '')
                    skeleton['transcript'] = original.get('transcript', '')

            skeleton['extracted_at'] = datetime.utcnow().isoformat()
            valid.append(skeleton)

        logger.info("EXTRACT", f"Loaded {len(valid)}/{len(raw_skeletons)} valid skeletons from {results_path}")
        return valid
