"""Tests for app.ingestion.url_validator."""

import pytest

from app.core.exceptions import InvalidUrlError
from app.ingestion.url_validator import GitHubRepo, validate_github_url


@pytest.mark.parametrize(
    ("url", "owner", "repo"),
    [
        ("https://github.com/openai/openai-python", "openai", "openai-python"),
        ("https://github.com/openai/openai-python/", "openai", "openai-python"),
        ("https://github.com/openai/openai-python.git", "openai", "openai-python"),
        ("https://github.com/a/b", "a", "b"),
        ("https://github.com/my-org/my_repo.v2", "my-org", "my_repo.v2"),
    ],
)
def test_valid_urls(url: str, owner: str, repo: str) -> None:
    assert validate_github_url(url) == GitHubRepo(owner=owner, repo=repo)


@pytest.mark.parametrize(
    "url",
    [
        "http://github.com/user/repo",  # wrong scheme
        "https://gitlab.com/user/repo",  # wrong host
        "https://www.github.com/user/repo",  # subdomain not allowed
        "https://github.com/user",  # missing repo
        "https://github.com/",  # missing owner and repo
        "https://github.com",  # no path
        "https://github.com/user/repo/extra",  # extra segment
        "https://github.com/user/repo/tree/main",  # branch URL
        "https://github.com//repo",  # empty owner
        "https://github.com/user/repo//",  # empty trailing segment
        "https://github.com/user/repo?tab=readme",  # query
        "https://github.com/user/repo#section",  # fragment
        "https://github.com/user/repo?",  # empty query
        "https://user:password@github.com/user/repo",  # credentials
        "https://github.com:8080/user/repo",  # port
        "https://github.com:443/user/repo",  # explicit port
        "",  # empty
        "github.com/user/repo",  # no scheme
    ],
)
def test_invalid_urls(url: str) -> None:
    with pytest.raises(InvalidUrlError):
        validate_github_url(url)


@pytest.mark.parametrize(
    "value", [None, 123, ["https://github.com/a/b"], b"https://github.com/a/b"]
)
def test_non_string_input(value: object) -> None:
    with pytest.raises(InvalidUrlError):
        validate_github_url(value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "url",
    [
        " https://github.com/user/repo",  # leading space
        "https://github.com/user/repo ",  # trailing space
        "https://github.com/user/repo\n",  # trailing newline
        "https://github.com/us\ner/repo",  # embedded newline
        "https://github.com/user/re\tpo",  # embedded tab
        "https://github.com/user/repo\x00",  # NUL byte
        "https://g\u0456thub.com/user/repo",  # Cyrillic 'i' in host
        "https://github.com/\u043ewner/repo",  # Cyrillic 'o' in owner
        "https://github.com/user/r\u00e9po",  # accented char in repo
        "https://github.com/user%2Frepo",  # encoded slash
        "https://github.com/user/%2e%2e",  # encoded dots
        "https://github.com/user\\repo",  # backslash
        "https://github.com\\@evil.com/user/repo",  # backslash host trick
        "https://github.com.attacker.com/user/repo",  # host spoofing
        "https://github.com@attacker.com/user/repo",  # userinfo spoofing
        "https://attacker.com/github.com/user/repo",  # host in path
        "https://github.com/user/..",  # repo is ..
        "https://github.com/user/.",  # repo is .
        "https://github.com/user/.git",  # empty repo after suffix strip
        "https://github.com/-user/repo",  # leading hyphen owner
        "https://github.com/user-/repo",  # trailing hyphen owner
        "https://github.com/" + "a" * 40 + "/repo",  # owner too long
        "https://github.com/user/" + "r" * 101,  # repo too long
        "https://github.com/user/repo/" + "a" * 3000,  # URL too long
    ],
)
def test_security_rejections(url: str) -> None:
    with pytest.raises(InvalidUrlError):
        validate_github_url(url)


def test_result_is_immutable() -> None:
    result = validate_github_url("https://github.com/openai/openai-python")
    with pytest.raises(AttributeError):
        result.owner = "other"  # type: ignore[misc]
