from googlesearch import search

def simple_google_search(query: str, num_results: int=10):
    """
    Simple Google search using googlesearch-python
    """
    try:
        results = []
        for url in search(query, num_results=num_results, sleep_interval=1):
            results.append(url)
        return results
    except Exception as e:
        print(f"Error: {e}")
        return []