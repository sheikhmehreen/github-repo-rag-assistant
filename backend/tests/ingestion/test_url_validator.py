import pytest
from app.ingestion.url_validator import validate_github_url, GitHubRepo, InvalidUrlError

@pytest.mark.parametrize(
    "valid_url,expected_owner,expected_repo",
    [
        ("https://github.com", "openai", "openai-python"),
        ("https://github.com", "openai", "openai-python"),
        ("https://github.com.git", "openai", "openai-python"),
        ("https://github.com", "some-owner", "some_repo.dots"),
        ("https://github.com", "a", "b"),
    ],
)
def test_validate_github_url_success(valid_url, expected_owner, expected_repo):
    result = validate_github_url(valid_url)
    assert isinstance(result, GitHubRepo)
    assert result.owner == expected_owner
    assert result.repo == expected_repo


@pytest.mark.parametrize(
    "invalid_url,error_msg_snippet",
    [
        ("http://github.com", "scheme must be exactly 'https'"),
        ("https://github.com", "host must be exactly 'github.com'"),
        ("https://evil.com", "host must be exactly 'github.com'"),
        ("https://github.com", "host must be exactly 'github.com'"),
        ("https://github.com", "host must be exactly 'github.com'"),
        ("https://github.com", "must contain exactly the owner and repository"),
        ("https://github.comrepo/tree/main", "Provide the repository root URL"),
        ("https://github.comrepo/blob/main/README.md", "Provide the repository root URL"),
        ("https://github.comrepo?query=1", "must not contain query strings"),
        ("https://github.comrepo#fragment", "must not contain query strings"),
        ("https://github.comrepo%2Fextra", "contains forbidden URL-encoded"),
        ("https://github.com..", "Invalid GitHub repository name"),
        ("https://github.com.", "Invalid GitHub repository name"),
        ("https://github.com\nowner/repo", "contains whitespace or control characters"),
        ("https://github.com\t/repo", "contains whitespace or control characters"),
        ("https://github.com\\repo", "contains invalid backslash characters"),
        ("https://github.comоwner/repo", "contains non-ASCII characters"),
        ("", "URL path must contain exactly"),
        (123, "URL must be a string"),
        ("A" * 2050, "exceeds maximum length"),
        ("https://github.com-owner/repo", "Invalid GitHub owner format"),
        ("https://github.com-/repo", "Invalid GitHub owner format"),
    ],
)
def test_validate_github_url_failures(invalid_url, error_msg_snippet):
    with pytest.raises(InvalidUrlError) as exc_info:
        validate_github_url(invalid_url)
    assert error_msg_snippet in str(exc_info.value)
