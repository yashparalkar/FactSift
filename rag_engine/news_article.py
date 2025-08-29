import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
from datetime import datetime
import json
from langchain_core.documents import Document
import tldextract
from dateutil.parser import parse as date_parse
import extruct
from newspaper import Article
import trafilatura

# Option 1: Using extruct for comprehensive metadata extraction
def extract_metadata_extruct(soup, url, html_content=None):
    """Enhanced metadata extraction using extruct library"""
    domain = tldextract.extract(url).domain
    
    # Extract structured data using extruct
    structured_data = {}
    if html_content:
        try:
            data = extruct.extract(html_content, base_url=url)
            structured_data = data
        except Exception as e:
            print(f"Extruct extraction failed: {e}")
    
    # Helper function for meta tags
    def get_meta(name_attr, content_attr):
        selectors = [
            f'meta[{name_attr}="{content_attr}"]',
            f'meta[{name_attr}="{content_attr.lower()}"]',
            f'meta[{name_attr}="{content_attr.replace(":", "_")}"]'
        ]
        for selector in selectors:
            tag = soup.select_one(selector)
            if tag and tag.get('content'):
                return tag.get('content')
        return None
    
    # Extract publication date from multiple sources
    pub_date = None
    date_candidates = [
        get_meta('property', 'article:published_time'),
        get_meta('name', 'date'),
        get_meta('name', 'publishdate'),
        get_meta('property', 'og:updated_time'),
        soup.select_one('time[datetime]')
    ]
    
    # Try structured data
    if structured_data.get('json-ld'):
        for item in structured_data['json-ld']:
            if isinstance(item, dict):
                date_fields = ['datePublished', 'dateCreated', 'dateModified']
                for field in date_fields:
                    if item.get(field):
                        date_candidates.append(item[field])
    
    # Parse the first valid date
    for candidate in date_candidates:
        if candidate:
            try:
                if hasattr(candidate, 'get') and candidate.get('datetime'):
                    pub_date = date_parse(candidate.get('datetime'))
                elif isinstance(candidate, str):
                    pub_date = date_parse(candidate)
                break
            except:
                continue
    
    # Extract author information
    author = None
    author_candidates = [
        get_meta('name', 'author'),
        get_meta('property', 'article:author'),
        get_meta('name', 'twitter:creator'),
    ]
    
    # Check structured data for author
    if structured_data.get('json-ld'):
        for item in structured_data['json-ld']:
            if isinstance(item, dict) and item.get('author'):
                author_data = item['author']
                if isinstance(author_data, list):
                    author = ", ".join([a.get("name", "") for a in author_data if isinstance(a, dict)])
                elif isinstance(author_data, dict):
                    author = author_data.get("name")
                elif isinstance(author_data, str):
                    author = author_data
                break
    
    if not author:
        for candidate in author_candidates:
            if candidate:
                author = candidate
                break
    
    return {
        "source": url,
        "domain": domain,
        "title": soup.title.string.strip() if soup.title else get_meta('property', 'og:title'),
        "description": (get_meta('name', 'description') or 
                       get_meta('property', 'og:description') or
                       get_meta('name', 'twitter:description')),
        "publish_date": pub_date.isoformat() if pub_date else None,
        "author": author,
        "publisher": get_meta('property', 'og:site_name') or domain,
        "language": (soup.html.get('lang') if soup.html else None) or get_meta('property', 'og:locale'),
        "scraped_at": datetime.utcnow().isoformat(),
        "quality_score": None,
        "structured_data": structured_data  # Include raw structured data
    }

# Option 2: Using newspaper3k for article extraction
def load_web_content_newspaper(url):
    """Using newspaper3k library for better article extraction"""
    try:
        article = Article(url)
        article.download()
        article.parse()
        
        # newspaper3k automatically extracts many metadata fields
        metadata = {
            "source": url,
            "domain": tldextract.extract(url).domain,
            "title": article.title,
            "description": article.meta_description,
            "publish_date": article.publish_date.isoformat() if article.publish_date else None,
            "author": ", ".join(article.authors) if article.authors else None,
            "publisher": article.meta_data.get('og', {}).get('site_name', tldextract.extract(url).domain),
            "language": article.meta_lang,
            "scraped_at": datetime.utcnow().isoformat(),
            "top_image": article.top_image,
            "movies": article.movies,
            "keywords": article.keywords,
            "summary": article.summary,
            "quality_score": None
        }
        
        return Document(page_content=article.text, metadata=metadata)
    
    except Exception as e:
        print(f"Newspaper extraction failed: {e}")
        # Fallback to original method
        return load_web_content_original(url)

# Option 3: Using trafilatura for robust content extraction
def load_web_content_trafilatura(url):
    """Using trafilatura for robust content and metadata extraction"""
    try:
        # Download the page
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            raise Exception("Failed to download page")
        
        # Extract text content
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=True)
        
        # Extract metadata
        metadata_dict = trafilatura.extract_metadata(downloaded)
        
        # Parse with BeautifulSoup for additional metadata
        soup = BeautifulSoup(downloaded, 'html.parser')
        enhanced_metadata = extract_metadata_extruct(soup, url, downloaded)
        
        # Combine trafilatura metadata with enhanced metadata
        if metadata_dict:
            enhanced_metadata.update({
                "title": metadata_dict.title or enhanced_metadata.get("title"),
                "author": metadata_dict.author or enhanced_metadata.get("author"),
                "publish_date": metadata_dict.date.isoformat() if metadata_dict.date else enhanced_metadata.get("publish_date"),
                "description": metadata_dict.description or enhanced_metadata.get("description"),
                "categories": metadata_dict.categories,
                "tags": metadata_dict.tags,
                "sitename": metadata_dict.sitename or enhanced_metadata.get("publisher")
            })
        
        return Document(page_content=text or "", metadata=enhanced_metadata)
    
    except Exception as e:
        print(f"Trafilatura extraction failed: {e}")
        return load_web_content_original(url)

# Original method as fallback
def load_web_content_original(url):
    """Original method as fallback"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    response = requests.get(url, headers=headers, timeout=10)
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text(strip=True)
    metadata = extract_metadata_extruct(soup, url, response.text)
    return Document(page_content=text, metadata=metadata)

# Hybrid approach - try multiple methods
def load_web_content_hybrid(url):
    """Hybrid approach that tries multiple extraction methods"""
    methods = [
        ("trafilatura", load_web_content_trafilatura),
        ("newspaper", load_web_content_newspaper),
        ("original", load_web_content_original)
    ]
    
    for method_name, method_func in methods:
        try:
            print(f"Trying {method_name} method...")
            result = method_func(url)
            
            # Check if we got reasonable results
            if (result.page_content and len(result.page_content) > 100 and 
                result.metadata.get('title')):
                print(f"Success with {method_name}")
                return result
        except Exception as e:
            print(f"{method_name} failed: {e}")
            continue
    
    raise Exception("All extraction methods failed")