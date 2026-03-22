# Remove OpenAI Dependency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all OpenAI API dependencies so the system runs entirely through Claude Code (interactive LLM) + local Whisper (transcription) + YouTube Data API (search/analytics).

**Architecture:** The recon pipeline currently makes programmatic LLM calls via `llm_client.py` for skeleton extraction and pattern synthesis. We split the pipeline into data-processing phases (Python) and analysis phases (Claude Code interactive). Commands feed transcripts/data to Claude inline, Claude reasons and outputs structured JSON, commands persist results. No API keys needed except `YOUTUBE_DATA_API_KEY`.

**Tech Stack:** Python 3.10+, local Whisper (openai-whisper), yt-dlp, instaloader, Claude Code commands (.md)

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `recon/skeleton_ripper/pipeline.py` | Split into data phases (scrape/transcribe/aggregate) — remove LLM orchestration |
| Modify | `recon/scraper/downloader.py` | Default to local Whisper, remove OpenAI Whisper API path |
| Modify | `recon/config.py` | Remove `llm_provider`/`llm_model` config, remove OpenAI credential loading |
| Delete | `recon/skeleton_ripper/llm_client.py` | Entire multi-provider LLM abstraction — no longer needed |
| Modify | `recon/skeleton_ripper/extractor.py` | Convert to file-based I/O: read transcripts → write extraction prompt → read Claude's output |
| Modify | `recon/skeleton_ripper/synthesizer.py` | Convert to file-based I/O: read skeletons → write synthesis prompt → read Claude's output |
| Modify | `.claude/commands/viral-discover.md` | Add inline Claude analysis for skeleton extraction + synthesis after scraping |
| Modify | `.claude/commands/viral-setup.md` | Remove OpenAI API key setup, add local Whisper install step |
| Modify | `skills/last30days/scripts/lib/openai_reddit.py` | Replace OpenAI Responses API with WebSearch tool calls |
| Modify | `requirements.txt` | Remove openai-whisper comment, make it a real dependency |
| Modify | `.env.example` | Remove `OPENAI_API_KEY` (required), keep only `YOUTUBE_DATA_API_KEY` |
| Create | `recon/skeleton_ripper/prompts.py` | Extraction + synthesis prompt templates (moved from extractor/synthesizer) |

---

## Task 1: Switch Transcription to Local Whisper

**Files:**
- Modify: `recon/scraper/downloader.py`
- Modify: `recon/config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: Read current downloader.py transcription logic**

Read `recon/scraper/downloader.py` — understand `transcribe_video_openai()` vs `transcribe_video_local()` and the `WHISPER_AVAILABLE` flag.

- [ ] **Step 2: Make local Whisper the default and only path**

In `downloader.py`:
- Remove `transcribe_video_openai()` function entirely
- Remove the OpenAI API key import/usage
- Rename `transcribe_video_local()` to `transcribe_video()`
- Remove the `WHISPER_AVAILABLE` conditional — whisper is now required
- Keep the `load_whisper_model()` function

- [ ] **Step 3: Update config.py**

In `recon/config.py`:
- Remove `transcribe_provider` field (was "openai" or "local")
- Remove `openai_api_key` from credential loading
- Remove OpenAI from `env_map`

- [ ] **Step 4: Uncomment whisper in requirements.txt**

Change `# openai-whisper>=20231117` to `openai-whisper>=20231117` (uncomment it).

- [ ] **Step 5: Verify whisper is installed locally**

```bash
cd /Users/cartersmith/AntiGravity/goviralbitch
pip3 install openai-whisper
python3 -c "import whisper; print(whisper.available_models())"
```

- [ ] **Step 6: Commit**

```bash
git add recon/scraper/downloader.py recon/config.py requirements.txt
git commit -m "feat: switch to local Whisper, remove OpenAI transcription API"
```

---

## Task 2: Extract Prompt Templates

**Files:**
- Create: `recon/skeleton_ripper/prompts.py`
- Modify: `recon/skeleton_ripper/extractor.py`
- Modify: `recon/skeleton_ripper/synthesizer.py`

- [ ] **Step 1: Read extractor.py and synthesizer.py prompt generation**

Read `recon/skeleton_ripper/extractor.py` — find `get_extraction_prompt()`. Read `recon/skeleton_ripper/synthesizer.py` — find `get_synthesis_prompts()`. These contain the prompt templates that tell the LLM what to extract/synthesize.

- [ ] **Step 2: Create prompts.py with extracted templates**

Move the prompt templates into `recon/skeleton_ripper/prompts.py` as standalone functions:

```python
def build_extraction_prompt(transcripts: list[dict]) -> str:
    """Build the prompt for skeleton extraction from transcripts.
    Returns a prompt string that can be given to Claude inline."""
    # Move get_extraction_prompt() logic here
    ...

def build_synthesis_prompt(skeletons: list[dict]) -> tuple[str, str]:
    """Build system + user prompts for pattern synthesis.
    Returns (system_prompt, user_prompt) tuple."""
    # Move get_synthesis_prompts() logic here
    ...
```

- [ ] **Step 3: Refactor extractor.py to file-based I/O**

Remove LLMClient dependency. Instead:

```python
import json
from .prompts import build_extraction_prompt

class BatchedExtractor:
    def prepare_extraction(self, transcripts: list[dict], output_dir: str):
        """Write transcripts + prompt to files for Claude to process."""
        prompt = build_extraction_prompt(transcripts)
        with open(f"{output_dir}/extraction-prompt.md", "w") as f:
            f.write(prompt)
        with open(f"{output_dir}/transcripts.json", "w") as f:
            json.dump(transcripts, f, indent=2)
        return f"{output_dir}/extraction-prompt.md"

    def load_extraction_results(self, output_dir: str) -> list[dict]:
        """Read Claude's extraction output from file."""
        with open(f"{output_dir}/extraction-results.json") as f:
            return json.load(f)
```

- [ ] **Step 4: Refactor synthesizer.py to file-based I/O**

Same pattern — write prompt to file, read results from file:

```python
import json
from .prompts import build_synthesis_prompt

class PatternSynthesizer:
    def prepare_synthesis(self, skeletons: list[dict], output_dir: str):
        """Write skeletons + prompt for Claude to process."""
        system, user = build_synthesis_prompt(skeletons)
        with open(f"{output_dir}/synthesis-prompt.md", "w") as f:
            f.write(f"## System\n{system}\n\n## Task\n{user}")
        with open(f"{output_dir}/skeletons.json", "w") as f:
            json.dump(skeletons, f, indent=2)

    def load_synthesis_results(self, output_dir: str) -> dict:
        """Read Claude's synthesis output from file."""
        with open(f"{output_dir}/synthesis-results.json") as f:
            return json.load(f)
```

- [ ] **Step 5: Commit**

```bash
git add recon/skeleton_ripper/prompts.py recon/skeleton_ripper/extractor.py recon/skeleton_ripper/synthesizer.py
git commit -m "feat: extract prompt templates, convert extractor/synthesizer to file-based I/O"
```

---

## Task 3: Refactor Pipeline to Data-Only Phases

**Files:**
- Modify: `recon/skeleton_ripper/pipeline.py`
- Delete: `recon/skeleton_ripper/llm_client.py`

- [ ] **Step 1: Read current pipeline.py orchestration**

Understand the job lifecycle: PENDING → SCRAPING → TRANSCRIBING → EXTRACTING → AGGREGATING → SYNTHESIZING → COMPLETE.

- [ ] **Step 2: Split pipeline into two entry points**

```python
class ReconPipeline:
    def scrape_and_transcribe(self, job_config) -> str:
        """Phase 1: Download videos, transcribe, save to output_dir.
        Returns output_dir path."""
        # SCRAPING phase
        # TRANSCRIBING phase (local Whisper)
        # Write transcripts to output_dir/transcripts.json
        # Write extraction prompt to output_dir/extraction-prompt.md

    def aggregate_and_finish(self, output_dir: str) -> dict:
        """Phase 2: Read extraction results, aggregate, prepare synthesis.
        Called AFTER Claude has written extraction-results.json."""
        # AGGREGATING phase
        # Write synthesis prompt to output_dir/synthesis-prompt.md
        # (Claude does synthesis)
        # After Claude writes synthesis-results.json:
        # COMPLETE phase — merge into data/recon/
```

- [ ] **Step 3: Delete llm_client.py**

```bash
rm recon/skeleton_ripper/llm_client.py
```

Remove all imports of `LLMClient` from other files.

- [ ] **Step 4: Commit**

```bash
git add recon/skeleton_ripper/pipeline.py
git rm recon/skeleton_ripper/llm_client.py
git commit -m "feat: split pipeline into data phases, delete LLM client"
```

---

## Task 4: Update Discover Command for Inline Claude Analysis

**Files:**
- Modify: `.claude/commands/viral-discover.md`

- [ ] **Step 1: Read current viral-discover.md**

Understand where it currently invokes the recon pipeline and how it handles results.

- [ ] **Step 2: Add Claude-inline extraction step**

After the scrape_and_transcribe phase, add a section in the command that tells Claude:

```markdown
## Phase: Skeleton Extraction

After scraping and transcribing, read the extraction prompt at `{output_dir}/extraction-prompt.md` and the transcripts at `{output_dir}/transcripts.json`.

Analyze each transcript and extract:
- Hook (opening line/pattern)
- Structure (section breakdown)
- Key points
- Proof method (data, story, authority, social)
- CTA approach
- Pacing (fast/medium/slow)

Write your analysis as JSON to `{output_dir}/extraction-results.json` matching the schema in `schemas/competitor-reel.schema.json`.
```

- [ ] **Step 3: Add Claude-inline synthesis step**

After aggregation, add:

```markdown
## Phase: Pattern Synthesis

Read the synthesis prompt at `{output_dir}/synthesis-prompt.md` and the aggregated skeletons at `{output_dir}/skeletons.json`.

Synthesize patterns into:
- Content templates (reusable structures)
- Quick wins (actionable insights)
- Warnings (what to avoid)

Write results to `{output_dir}/synthesis-results.json`.
```

- [ ] **Step 4: Commit**

```bash
git add .claude/commands/viral-discover.md
git commit -m "feat: inline Claude analysis in discover command, no API calls"
```

---

## Task 5: Replace Reddit Search

**Files:**
- Modify: `skills/last30days/scripts/lib/openai_reddit.py`

- [ ] **Step 1: Read current openai_reddit.py**

Understand the OpenAI Responses API web_search integration for Reddit.

- [ ] **Step 2: Replace with direct Reddit search**

Replace the OpenAI Responses API call with a direct approach — use the existing `WebSearch` tool or `requests` to search Reddit directly:

```python
def search_reddit(query: str, depth: str = "default") -> list[dict]:
    """Search Reddit using direct web scraping instead of OpenAI API."""
    import subprocess
    import json

    # Use yt-dlp or requests to search Reddit directly
    # OR call the WebSearch tool from Claude Code context
    search_url = f"https://www.reddit.com/search.json?q={query}&sort=new&t=month&limit=25"
    headers = {"User-Agent": "goviralbitch/1.0"}
    response = requests.get(search_url, headers=headers, timeout=15)
    response.raise_for_status()
    data = response.json()

    results = []
    for post in data["data"]["children"]:
        p = post["data"]
        results.append({
            "title": p["title"],
            "subreddit": p["subreddit"],
            "score": p["score"],
            "url": f"https://reddit.com{p['permalink']}",
            "num_comments": p["num_comments"],
            "created_utc": p["created_utc"],
            "selftext": p.get("selftext", "")[:500],
        })
    return results
```

- [ ] **Step 3: Remove OpenAI model fallback chain**

In `skills/last30days/scripts/lib/models.py`, remove the GPT model references if they exist only for Reddit search.

- [ ] **Step 4: Commit**

```bash
git add skills/last30days/scripts/lib/openai_reddit.py skills/last30days/scripts/lib/models.py
git commit -m "feat: replace OpenAI Reddit search with direct Reddit API"
```

---

## Task 6: Update Setup Command + Environment

**Files:**
- Modify: `.claude/commands/viral-setup.md`
- Modify: `.env.example`

- [ ] **Step 1: Update viral-setup.md**

- Remove the OpenAI API key setup phase entirely
- Add a local Whisper install/verify step:
  ```
  pip3 install openai-whisper
  python3 -c "import whisper; print('Whisper OK')"
  ```
- Add ffmpeg check (required for Whisper):
  ```
  which ffmpeg || brew install ffmpeg
  ```
- Keep YouTube Data API key setup
- Keep Instagram/YouTube Analytics OAuth setup (optional)

- [ ] **Step 2: Update .env.example**

Remove:
```
OPENAI_API_KEY=sk-...
```

Keep:
```
YOUTUBE_DATA_API_KEY=AIza...
```

Keep optional:
```
ANTHROPIC_API_KEY=sk-ant-...  # Optional: only for recon pipeline API mode
INSTAGRAM_ACCESS_TOKEN=...
```

- [ ] **Step 3: Commit**

```bash
git add .claude/commands/viral-setup.md .env.example
git commit -m "feat: remove OpenAI from setup, add local Whisper + ffmpeg checks"
```

---

## Task 7: Update LLM Model References in Config

**Files:**
- Modify: `recon/config.py`

- [ ] **Step 1: Remove all LLM provider/model config**

In `recon/config.py`:
- Remove `llm_provider` field
- Remove `llm_model` field
- Remove any model selection logic
- Keep only scraping/transcription/data config

- [ ] **Step 2: Commit**

```bash
git add recon/config.py
git commit -m "chore: remove LLM provider config from recon"
```

---

## Task 8: Final Cleanup + Verification

**Files:**
- All modified files

- [ ] **Step 1: Grep for remaining OpenAI references**

```bash
cd /Users/cartersmith/AntiGravity/goviralbitch
grep -r "openai\|OPENAI\|gpt-4\|gpt-3" --include="*.py" --include="*.md" --include="*.txt" -l
```

Fix any remaining references.

- [ ] **Step 2: Grep for remaining API call patterns**

```bash
grep -r "api.openai.com\|chat/completions\|v1/responses" --include="*.py" -l
```

Should return empty.

- [ ] **Step 3: Verify Python imports**

```bash
cd /Users/cartersmith/AntiGravity/goviralbitch
python3 -c "from recon.skeleton_ripper.pipeline import ReconPipeline; print('Pipeline OK')"
python3 -c "from recon.scraper.downloader import transcribe_video; print('Transcription OK')"
python3 -c "from scoring.engine import load_brain_context; print('Scoring OK')"
```

- [ ] **Step 4: Run init script to verify setup**

```bash
bash scripts/init-viral-command.sh
```

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -m "chore: final cleanup — remove all OpenAI dependencies

Only required API key is YOUTUBE_DATA_API_KEY. LLM analysis runs
through Claude Code interactively. Transcription uses local Whisper."
```

---

## Dependency Summary (After Refactor)

| Dependency | Required? | Purpose |
|-----------|-----------|---------|
| `YOUTUBE_DATA_API_KEY` | Yes | Discovery search, analytics |
| `openai-whisper` + `ffmpeg` | Yes | Local video transcription |
| `yt-dlp` | Yes | Video downloading |
| `instaloader` | Optional | Instagram scraping |
| `INSTAGRAM_ACCESS_TOKEN` | Optional | Instagram analytics |
| YouTube OAuth token | Optional | YouTube Analytics deep metrics |
| Claude Code | Yes | All LLM reasoning (interactive) |
