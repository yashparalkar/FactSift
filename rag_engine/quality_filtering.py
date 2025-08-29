from datetime import datetime

def credibility_scores(results):
    """Filter for reputable news sources and assign quality scores."""
    
    reputable_domains = {
        'reuters', 'ap', 'bbc', 'cnn', 'hindustantimes', 'hindu', 'livemint',
        'nytimes', 'wsj', 'bloomberg', 'techcrunch', 'arstechnica', 'timesofindia', 'forbes',
        'theverge', 'wired', 'theguardian', 'economist', 'financialtimes', 'npr',
        'cnbc', 'axios', 'politico', 'nbcnews', 'thehill', 'vox', 'buzzfeednews',
        'ndtv'
    }

    scores = []
    for result in results:
        domain_score = 0.5  # Default score
        content_score = 0.0

        # --- Domain score ---
        domain = result.metadata.get('domain')
        if domain:
            if any(rep in domain.lower() for rep in reputable_domains):
                domain_score = 1.0
            else:
                domain_score = 0.5
        else:
            print("⚠️ Domain missing for one of the documents")

        # --- Content + recency score ---
        try:
            if result.metadata.get('publish_date'):
                days_old = (datetime.now() - result.metadata['publish_date'].replace(tzinfo=None)).days
                content_score += max(0, 10 - days_old)  # Recent articles = higher score
        except Exception:
            print("⚠️ Missing or invalid publish date")

        length = len(result.page_content)
        if 200 < length < 5000:
            content_score += 3

        # Store raw content score
        result.metadata['content_score'] = content_score
        scores.append(content_score)

        # Temporarily store domain credibility
        result.metadata['domain_score'] = domain_score

    # --- Normalize content scores ---
    if scores:
        max_score = max(scores)
        norm_scores = [s / max_score if max_score > 0 else 0 for s in scores]
    else:
        print("⚠️ No valid content scores. Assigning default quality_score = 0.5")
        norm_scores = [0.5 for _ in results]

    # --- Final quality score ---
    for i, result in enumerate(results):
        final_score = 0.5 * result.metadata.get('domain_score', 0.5) + 0.5 * norm_scores[i]
        result.metadata['quality_score'] = final_score

    return sorted(results, key=lambda x: x.metadata['quality_score'], reverse=True)