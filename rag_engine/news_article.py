from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
from datetime import datetime
import json
from langchain_core.documents import Document
import tldextract

def extract_metadata(soup, url):
    def get_meta(name, prop=None):
        tag = soup.find('meta', attrs={name: prop or name})
        return tag['content'] if tag and tag.has_attr('content') else None

    # Attempt to parse structured data
    ld_json = soup.find('script', type='application/ld+json')
    structured_data = {}
    if ld_json:
        try:
            structured_data = json.loads(ld_json.string)
        except Exception:
            pass

    # domain = urlparse(url).netloc
    domain = tldextract.extract(url).domain
    def extract_author(author_data):
        if isinstance(author_data, list):
            return ", ".join([a.get("name", "") for a in author_data if isinstance(a, dict)])
        elif isinstance(author_data, dict):
            return author_data.get("name")
        return None

    return {
        "source": url,
        "domain": domain,
        "title": soup.title.string if soup.title else None,
        "description": get_meta("name", "description") or get_meta("property", "og:description"),
        "publish_date": (
            datetime.fromisoformat(get_meta("property", "article:published_time") or
            structured_data.get("datePublished"))
        ),
        "author": get_meta("name", "author") or extract_author(structured_data.get("author")),
        "publisher": structured_data.get("publisher", {}).get("name", domain),
        "language": soup.html.get("lang") if soup.html else None,
        "scraped_at": datetime.utcnow().isoformat(),
        "quality_score": None
    }

def load_web_content(url):
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(strip=True)
    metadata = extract_metadata(soup, url)
    return Document(page_content=text, metadata=metadata)