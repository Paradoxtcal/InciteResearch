"""
Baseline repository discovery tools.
Supports GitHub Search API and Papers with Code API.
"""

from __future__ import annotations
import json
import urllib.request
import urllib.parse


def search_github_repos(query: str, limit: int = 10, language: str = "") -> list[dict]:
    """
    Search GitHub repositories via the public Search API.

    Notes:
    - Unauthenticated requests are rate-limited.
    - Set GITHUB_TOKEN to increase the rate limit.
    """
    params = {
        "q": f"{query} stars:>50",
        "sort": "stars",
        "order": "desc",
        "per_page": limit,
    }
    if language:
        params["q"] += f" language:{language}"

    url = f"https://api.github.com/search/repositories?{urllib.parse.urlencode(params)}"

    import os
    headers = {
        "User-Agent": "InciteResearch/1.0",
        "Accept": "application/vnd.github.v3+json",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        return [
            {
                "name": r.get("full_name", ""),
                "description": r.get("description", "") or "",
                "stars": r.get("stargazers_count", 0),
                "url": r.get("html_url", ""),
                "language": r.get("language", ""),
                "updated": r.get("updated_at", "")[:10],
                "source": "github",
            }
            for r in data.get("items", [])
        ]
    except Exception as e:
        print(f"  [WARNING] GitHub search failed: {e}")
        return []


def search_papers_with_code(query: str, limit: int = 5) -> list[dict]:
    """
    Search papers that have official code on Papers with Code.
    """
    params = urllib.parse.urlencode({"q": query, "items_per_page": limit})
    url = f"https://paperswithcode.com/api/v1/papers/?{params}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "InciteResearch/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        repos = []
        for paper in data.get("results", []):
            if paper.get("repository"):
                repos.append({
                    "name": paper.get("title", ""),
                    "description": paper.get("abstract", "")[:200],
                    "url": paper.get("repository", {}).get("url", ""),
                    "stars": paper.get("repository", {}).get("stars", 0),
                    "source": "papers_with_code",
                    "paper_url": f"https://paperswithcode.com/paper/{paper.get('id','')}",
                })
        return repos
    except Exception as e:
        print(f"  [WARNING] Papers with Code search failed: {e}")
        return []
