import re
from dataclasses import dataclass
from urllib.parse import urlsplit

class IngestionError(Exception):
    """Base exception for all ingestion tasks."""
    pass

class InvalidUrlError(IngestionError):
    """Raised when a provided URL violates structural or security rules."""
    pass

@dataclass(frozen=True)
class GitHubRepo:
    owner: str
    repo: str

# Owner: 1-39 chars, alphanumeric or hyphens, no leading/trailing hyphen
OWNER_REGEX = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?$")
# Repo: 1-100 chars, alphanumeric, dots, underscores, hyphens
REPO_REGEX = re.compile(r"^[A-Za-z0-9._-]{1,100}$")

def validate_github_url(url: str) -> GitHubRepo:
    """Strictly validates a GitHub repository root URL and extracts owner and repo."""
    if not isinstance(url, str):
        raise InvalidUrlError("URL must be a string.")
        
    if len(url) > 2048:
        raise InvalidUrlError("URL exceeds maximum length of 2048 characters.")

    # 1. Validate raw string before any parsing to catch whitespace/control character tricks
    if any(c.isspace() or ord(c) < 32 or ord(c) == 127 for c in url):
        raise InvalidUrlError("URL contains whitespace or control characters.")
        
    if "\\" in url:
        raise InvalidUrlError("URL contains invalid backslash characters.")
        
    if not url.isascii():
        raise InvalidUrlError("URL contains non-ASCII characters.")

    # 2. Parse URL and strictly validate network location and scheme
    try:
        parsed = urlsplit(url)
    except Exception as e:
        raise InvalidUrlError(f"URL parsing failed: {str(e)}")

    if parsed.scheme != "https":
        raise InvalidUrlError("URL scheme must be exactly 'https'.")

    if parsed.netloc != "github.com":
        raise InvalidUrlError("URL host must be exactly 'github.com'.")

    if parsed.query or parsed.fragment:
        raise InvalidUrlError("URL must not contain query strings or fragments.")

    # 3. Path sanitization and structural checking
    path = parsed.path
    if "%" in path:
        raise InvalidUrlError("URL path contains forbidden URL-encoded characters.")

    # Remove exactly one trailing slash if present
    if path.endswith("/"):
        path = path[:-1]

    # Split and clean elements
    segments = [s for s in path.split("/") if s]

    # Must contain exactly two elements: [owner, repo]
    if len(segments) != 2:
        if "tree" in segments or "blob" in segments:
            raise InvalidUrlError("Provide the repository root URL, not a branch or file path.")
        raise InvalidUrlError("URL path must contain exactly the owner and repository segments.")

    owner, repo = segments[0], segments[1]

    # Remove optional .git suffix from repository name
    if repo.lower().endswith(".git"):
        repo = repo[:-4]

    # 4. Strict Regex Validation & Explicit Traversal Protection
    if not OWNER_REGEX.match(owner):
        raise InvalidUrlError("Invalid GitHub owner format.")

    if not REPO_REGEX.match(repo) or repo in (".", ".."):
        raise InvalidUrlError("Invalid GitHub repository name format.")

    return GitHubRepo(owner=owner, repo=repo)
