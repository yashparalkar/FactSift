from duckduckgo_search import DDGS

def simple_google_search(query: str, num_results: int=10):
    with DDGS() as ddgs:
        results = ddgs.text("Latest on Donald Trump", max_results = num_results)
        res = []
        for r in results:
            res.append(r['href'])
        return res