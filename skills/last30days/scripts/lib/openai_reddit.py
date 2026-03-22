"""Reddit search via direct Reddit JSON API."""

import json
import subprocess
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional

from . import dates as dates_mod
from . import http

# Kept for backwards compatibility (referenced by tests)
MODEL_FALLBACK_ORDER = ["gpt-4o"]

# Depth configurations: how many results to request from Reddit
DEPTH_LIMIT = {
    "quick": 10,
    "default": 25,
    "deep": 50,
}


def _log_error(msg: str):
    """Log error to stderr."""
    sys.stderr.write(f"[REDDIT ERROR] {msg}\n")
    sys.stderr.flush()


def _log_info(msg: str):
    """Log info to stderr."""
    sys.stderr.write(f"[REDDIT] {msg}\n")
    sys.stderr.flush()


def _is_model_access_error(error: http.HTTPError) -> bool:
    """Check if error is due to model access/verification issues.

    Kept for backwards compatibility (referenced by tests).
    No longer used internally since we no longer call OpenAI.
    """
    if error.status_code not in (400, 403):
        return False
    if not error.body:
        return False
    body_lower = error.body.lower()
    return any(phrase in body_lower for phrase in [
        "verified",
        "organization must be",
        "does not have access",
        "not available",
        "not found",
    ])


def _extract_core_subject(topic: str) -> str:
    """Extract core subject from verbose query for retry."""
    noise = ['best', 'top', 'how to', 'tips for', 'practices', 'features',
             'killer', 'guide', 'tutorial', 'recommendations', 'advice',
             'prompting', 'using', 'for', 'with', 'the', 'of', 'in', 'on']
    words = topic.lower().split()
    result = [w for w in words if w not in noise]
    return ' '.join(result[:3]) or topic  # Keep max 3 words


def _build_subreddit_query(topic: str) -> str:
    """Build a subreddit-targeted search query for fallback.

    When standard search returns few results, try searching for the
    subreddit itself: 'r/kanye', 'r/howie', etc.
    """
    core = _extract_core_subject(topic)
    # Remove dots and special chars for subreddit name guess
    sub_name = core.replace('.', '').replace(' ', '').lower()
    return f"r/{sub_name} site:reddit.com"


def _url_encode(text: str) -> str:
    """Simple URL encoding for query parameters."""
    return urllib.parse.quote_plus(text)


def _reddit_get(url: str, timeout: int = 15) -> Dict[str, Any]:
    """Fetch JSON from Reddit, falling back to curl if urllib is blocked.

    Reddit performs TLS fingerprinting and blocks Python's urllib with 403.
    When that happens, we fall back to curl which Reddit allows through.
    """
    headers = {
        "User-Agent": http.USER_AGENT,
        "Accept": "application/json",
    }

    # Try urllib first (faster, no subprocess overhead)
    try:
        return http.get(url, headers=headers, timeout=timeout, retries=1)
    except http.HTTPError as e:
        if e.status_code != 403:
            raise
        # 403 likely means TLS fingerprint block — fall back to curl
        _log_info("urllib blocked (403), falling back to curl")

    # Curl fallback
    try:
        result = subprocess.run(
            [
                "curl", "-s", "-f",
                "-H", f"User-Agent: {http.USER_AGENT}",
                "-H", "Accept: application/json",
                "--max-time", str(timeout),
                url,
            ],
            capture_output=True,
            text=True,
            timeout=timeout + 5,
        )
        if result.returncode != 0:
            stderr = result.stderr.strip()
            raise http.HTTPError(f"curl failed (exit {result.returncode}): {stderr}")
        return json.loads(result.stdout)
    except subprocess.TimeoutExpired:
        raise http.HTTPError(f"curl timed out after {timeout}s")
    except json.JSONDecodeError as e:
        raise http.HTTPError(f"Invalid JSON from Reddit: {e}")


def _parse_reddit_children(children: list, id_prefix: str = "R") -> List[Dict[str, Any]]:
    """Parse Reddit API children into standardized item dicts.

    Args:
        children: List of Reddit API child objects
        id_prefix: Prefix for item IDs (R for search, RS for subreddit)

    Returns:
        List of parsed item dicts
    """
    items = []
    for child in children:
        if child.get("kind") != "t3":  # t3 = link/submission
            continue
        p = child.get("data", {})
        permalink = p.get("permalink", "")
        if not permalink:
            continue

        # Parse date from created_utc
        created_utc = p.get("created_utc")
        date_str = dates_mod.timestamp_to_date(created_utc) if created_utc else None

        item = {
            "id": f"{id_prefix}{len(items) + 1}",
            "title": str(p.get("title", "")).strip(),
            "url": f"https://www.reddit.com{permalink}",
            "subreddit": str(p.get("subreddit", "")).strip(),
            "date": date_str,
            "why_relevant": f"Reddit search result (score: {p.get('score', 0)}, "
                            f"comments: {p.get('num_comments', 0)})",
            "relevance": 0.75,  # Default relevance for direct search results
        }
        items.append(item)

    return items


def search_reddit(
    topic: str,
    from_date: str = "",
    to_date: str = "",
    depth: str = "default",
    # Legacy params kept for backwards compatibility — unused
    api_key: str = None,
    model: str = None,
    mock_response: Optional[Dict] = None,
    _retry: bool = False,
) -> List[Dict[str, Any]]:
    """Search Reddit for relevant threads using Reddit's public JSON API.

    No API key needed. Uses reddit.com/search.json endpoint.

    Args:
        topic: Search topic
        from_date: Start date (YYYY-MM-DD) — unused by Reddit API but kept for compat
        to_date: End date (YYYY-MM-DD) — unused by Reddit API but kept for compat
        depth: Research depth - "quick", "default", or "deep"
        api_key: (deprecated) No longer needed — kept for backwards compatibility
        model: (deprecated) No longer needed — kept for backwards compatibility
        mock_response: Mock response for testing (should be a Reddit-format dict)
        _retry: (deprecated) No longer needed

    Returns:
        List of parsed item dicts with keys: id, title, url, subreddit, date,
        why_relevant, relevance
    """
    if mock_response is not None:
        # Support mock: if it looks like a Reddit API response, parse it;
        # if it's already a list, return it directly
        if isinstance(mock_response, list):
            return mock_response
        children = mock_response.get("data", {}).get("children", [])
        return _parse_reddit_children(children)

    limit = DEPTH_LIMIT.get(depth, DEPTH_LIMIT["default"])

    params = urllib.parse.urlencode({
        "q": topic,
        "sort": "relevance",
        "t": "month",  # last 30 days
        "limit": limit,
        "type": "link",
        "raw_json": 1,
    })

    url = f"https://www.reddit.com/search.json?{params}"

    try:
        data = _reddit_get(url, timeout=15)
    except http.HTTPError as e:
        _log_error(f"Reddit search failed: {e}")
        if e.status_code == 429:
            _log_info("Reddit rate-limited (429)")
        return []
    except Exception as e:
        _log_error(f"Reddit search error: {e}")
        return []

    children = data.get("data", {}).get("children", [])
    items = _parse_reddit_children(children)

    # Rate limit: Reddit asks for 1 request per 2 seconds
    time.sleep(2)

    return items


def search_subreddits(
    subreddits: List[str],
    topic: str,
    from_date: str,
    to_date: str,
    count_per: int = 5,
) -> List[Dict[str, Any]]:
    """Search specific subreddits via Reddit's free JSON endpoint.

    No API key needed. Uses reddit.com/r/{sub}/search/.json endpoint.
    Used in Phase 2 supplemental search after entity extraction.

    Args:
        subreddits: List of subreddit names (without r/)
        topic: Search topic
        from_date: Start date (YYYY-MM-DD)
        to_date: End date (YYYY-MM-DD)
        count_per: Results to request per subreddit

    Returns:
        List of raw item dicts (same format as search_reddit output).
    """
    all_items = []
    core = _extract_core_subject(topic)

    for sub in subreddits:
        sub = sub.lstrip("r/")
        try:
            url = f"https://www.reddit.com/r/{sub}/search/.json"
            params = f"q={_url_encode(core)}&restrict_sr=on&sort=new&limit={count_per}&raw_json=1"
            full_url = f"{url}?{params}"

            data = _reddit_get(full_url, timeout=15)

            # Reddit search returns {"data": {"children": [...]}}
            children = data.get("data", {}).get("children", [])
            for child in children:
                if child.get("kind") != "t3":  # t3 = link/submission
                    continue
                post = child.get("data", {})
                permalink = post.get("permalink", "")
                if not permalink:
                    continue

                item = {
                    "id": f"RS{len(all_items)+1}",
                    "title": str(post.get("title", "")).strip(),
                    "url": f"https://www.reddit.com{permalink}",
                    "subreddit": str(post.get("subreddit", sub)).strip(),
                    "date": None,
                    "why_relevant": f"Found in r/{sub} supplemental search",
                    "relevance": 0.65,  # Slightly lower default for supplemental
                }

                # Parse date from created_utc
                created_utc = post.get("created_utc")
                if created_utc:
                    item["date"] = dates_mod.timestamp_to_date(created_utc)

                all_items.append(item)

        except http.HTTPError as e:
            _log_info(f"Subreddit search failed for r/{sub}: {e}")
            if e.status_code == 429:
                _log_info("Reddit rate-limited (429) -- skipping remaining subreddits")
                break
        except Exception as e:
            _log_info(f"Subreddit search error for r/{sub}: {e}")

    return all_items


def parse_reddit_response(response) -> List[Dict[str, Any]]:
    """Parse Reddit response to extract items.

    This function is kept for backwards compatibility. Since search_reddit()
    now returns parsed items directly, this is a pass-through when given a list,
    or parses a Reddit API dict response.

    Args:
        response: Either a list of items (from new search_reddit) or a
                  Reddit API dict response

    Returns:
        List of item dicts
    """
    # New format: search_reddit already returns a list
    if isinstance(response, list):
        return response

    # Legacy/mock format: Reddit API dict
    if isinstance(response, dict):
        # Check for error
        if "error" in response and response["error"]:
            err = response["error"]
            err_msg = err.get("message", str(err)) if isinstance(err, dict) else str(err)
            _log_error(f"Reddit API error: {err_msg}")
            return []
        children = response.get("data", {}).get("children", [])
        return _parse_reddit_children(children)

    return []
