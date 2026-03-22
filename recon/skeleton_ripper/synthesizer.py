"""
Pattern synthesizer for Content Skeleton Ripper — file-based I/O.

Writes skeletons + prompt to disk for Claude Code to process,
then reads back the synthesis results. No direct LLM dependency.
"""

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from .prompts import build_synthesis_prompt
from .aggregator import AggregatedData
from recon.utils.logger import get_logger

logger = get_logger()


@dataclass
class SynthesisResult:
    success: bool
    analysis: str = ""
    templates: list[dict] = field(default_factory=list)
    quick_wins: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    model_used: str = ""
    tokens_used: int = 0
    synthesized_at: str = ""


class PatternSynthesizer:
    """Prepare synthesis prompts and load results via the filesystem."""

    def prepare_synthesis(self, skeletons: list[dict], output_dir: str) -> str:
        """Write skeletons + prompt for Claude to process.

        Creates:
          - <output_dir>/synthesis-prompt.md — system + user prompt
          - <output_dir>/skeletons.json      — raw skeleton data

        Returns the path to the prompt file.
        """
        os.makedirs(output_dir, exist_ok=True)

        system, user = build_synthesis_prompt(skeletons)
        prompt_path = os.path.join(output_dir, "synthesis-prompt.md")
        data_path = os.path.join(output_dir, "skeletons.json")

        with open(prompt_path, "w") as f:
            f.write(f"## System\n{system}\n\n## Task\n{user}")
        with open(data_path, "w") as f:
            json.dump(skeletons, f, indent=2)

        logger.info("SYNTH", f"Wrote prompt to {prompt_path} ({len(skeletons)} skeletons)")
        return prompt_path

    def load_synthesis_results(self, output_dir: str) -> SynthesisResult:
        """Read Claude's synthesis output from file.

        Expects <output_dir>/synthesis-results.json with a JSON object
        containing at minimum an "analysis" key with the full text output.
        Optionally may contain pre-parsed "templates", "quick_wins", and
        "warnings" arrays. If those are absent the raw analysis text is
        parsed to extract them.
        """
        results_path = os.path.join(output_dir, "synthesis-results.json")
        with open(results_path) as f:
            data = json.load(f)

        # Support both structured and raw-text result formats
        if isinstance(data, str):
            analysis_text = data
            pre_parsed = {}
        elif isinstance(data, dict):
            analysis_text = data.get("analysis", "")
            pre_parsed = data
        else:
            return SynthesisResult(success=False, error=f"Unexpected result format in {results_path}")

        result = SynthesisResult(
            success=True,
            analysis=analysis_text.strip(),
            synthesized_at=datetime.utcnow().isoformat(),
        )

        # Use pre-parsed fields if the results file has them,
        # otherwise fall back to text extraction.
        result.templates = pre_parsed.get("templates") or self._extract_templates(analysis_text)
        result.quick_wins = pre_parsed.get("quick_wins") or self._extract_section_items(analysis_text, "Quick Wins")
        result.warnings = pre_parsed.get("warnings") or self._extract_section_items(analysis_text, "Warnings")

        logger.info("SYNTH", f"Loaded synthesis results from {results_path}")
        return result

    # ------------------------------------------------------------------
    # Text-parsing helpers (kept from original for raw-text results)
    # ------------------------------------------------------------------

    def _extract_templates(self, text: str) -> list[dict]:
        templates = []
        current_template = None
        for line in text.split('\n'):
            line = line.strip()
            if line.startswith('## Template') or line.startswith('### Template'):
                if current_template:
                    templates.append(current_template)
                name = line.split(':', 1)[-1].strip() if ':' in line else line
                current_template = {'name': name, 'components': {}}
            elif current_template and line.startswith('**') and ':**' in line:
                parts = line.split(':**', 1)
                key = parts[0].replace('**', '').strip().lower()
                value = parts[1].strip() if len(parts) > 1 else ''
                current_template['components'][key] = value
        if current_template:
            templates.append(current_template)
        return templates

    def _extract_section_items(self, text: str, section_name: str) -> list[str]:
        items = []
        in_section = False
        for line in text.split('\n'):
            stripped = line.strip()
            if section_name.lower() in stripped.lower() and stripped.startswith('#'):
                in_section = True
                continue
            if in_section and stripped.startswith('#'):
                break
            if in_section and (stripped.startswith('-') or stripped.startswith('*')):
                item = stripped[1:].strip()
                if item:
                    items.append(item)
        return items


def generate_report(data: AggregatedData, synthesis: SynthesisResult, job_config: Optional[dict] = None) -> str:
    lines = [
        "# Content Skeleton Analysis Report", "",
        f"*Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*", "",
    ]
    if job_config:
        lines.extend([
            "## Analysis Configuration",
            f"- **Creators analyzed:** {', '.join(job_config.get('usernames', []))}",
            f"- **Platform:** {job_config.get('platform', 'N/A')}",
            f"- **Videos per creator:** {job_config.get('videos_per_creator', 'N/A')}",
            f"- **LLM:** {synthesis.model_used}", "",
        ])
    lines.extend([
        "## Summary",
        f"- **Total videos analyzed:** {data.total_videos}",
        f"- **Total views:** {data.total_views:,}",
        f"- **Average hook length:** {data.avg_hook_word_count:.1f} words",
        f"- **Average video length:** {data.avg_duration_seconds:.0f} seconds", "",
        "---", "", synthesis.analysis, "",
        "---", "", "## Raw Skeletons Data", "",
        "See `skeletons.json` for full extracted skeleton data.", "",
    ])
    return "\n".join(lines)
