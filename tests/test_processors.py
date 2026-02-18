"""Tests for HTML processors."""

from pathlib import Path

from bs4 import BeautifulSoup

from chatty.infra.processor_utils import HtmlHeadTitleMeta


def test_real_site_html_processing():
    """Test processing real site HTML data."""
    # Load real test data
    test_data_path = Path(__file__).parent / "test_data" / "site.html"
    html_content = test_data_path.read_text()

    processor = HtmlHeadTitleMeta()
    result = processor.process(html_content)
    soup = BeautifulSoup(result, "html.parser")

    # Verify structure
    assert soup.find("html") is not None
    assert soup.find("head") is not None

    # Check title is preserved
    title = soup.find("title")
    assert title is not None
    assert title.get_text() == "Xinyu's Digital Space"

    # Check wanted meta tags are preserved
    meta_tags = soup.find_all("meta")
    assert meta_tags is not None
    assert len(meta_tags) >= 5  # Should have at least description, keywords,

    # Should have description, keywords, author, and og tags
    description = soup.find("meta", {"name": "description"})
    assert description is not None
    assert "Digital space of Xinyu Huang" in description.get("content")

    keywords = soup.find("meta", {"name": "keywords"})
    assert keywords is not None
    assert "Xinyu Huang" in keywords.get("content")

    author = soup.find("meta", {"name": "author"})
    assert author is not None
    assert author.get("content") == "Xinyu Huang"

    og_title = soup.find("meta", {"property": "og:title"})
    assert og_title is not None
    assert og_title.get("content") == "Xinyu's Digital Space"

    og_url = soup.find("meta", {"property": "og:url"})
    assert og_url is not None
    assert og_url.get("content") == "https://x3huang.dev"

    # Verify unwanted tags are filtered out
    assert soup.find("script") is None
    assert soup.find("link") is None
    assert soup.find("meta", {"name": "viewport"}) is None
    assert soup.find("meta", {"charset": True}) is None
