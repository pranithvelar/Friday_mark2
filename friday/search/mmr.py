import re
from typing import List, Set, Dict, Any, TypeVar, Generic
from friday.search.hybrid_search import HybridSearchResult

CJK_RE = re.compile(r'[\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\u1100-\u11ff]')

class MMRConfig:
    def __init__(self, enabled: bool = False, lambda_param: float = 0.5):
        self.enabled = enabled
        self.lambda_param = lambda_param

def tokenize(text: str) -> Set[str]:
    lower = text.lower()
    ascii_tokens = re.findall(r'[a-z0-9_]+', lower)
    
    chars = list(lower)
    cjk_data = []
    for i, char in enumerate(chars):
        if CJK_RE.match(char):
            cjk_data.append({"char": char, "index": i})
            
    bigrams = []
    for i in range(len(cjk_data) - 1):
        if cjk_data[i+1]["index"] == cjk_data[i]["index"] + 1:
            bigrams.append(cjk_data[i]["char"] + cjk_data[i+1]["char"])
            
    unigrams = [d["char"] for d in cjk_data]
    
    return set(ascii_tokens + bigrams + unigrams)

def jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
        
    smaller = set_a if len(set_a) <= len(set_b) else set_b
    larger = set_b if len(set_a) <= len(set_b) else set_a
    
    intersection_size = sum(1 for token in smaller if token in larger)
    union_size = len(set_a) + len(set_b) - intersection_size
    
    return intersection_size / union_size if union_size > 0 else 0.0

def max_similarity_to_selected(
    candidate_tokens: Set[str], 
    selected_items: List[Any], 
    token_cache: Dict[str, Set[str]],
    id_getter
) -> float:
    if not selected_items:
        return 0.0
        
    max_sim = 0.0
    for selected in selected_items:
        selected_tokens = token_cache[id_getter(selected)]
        sim = jaccard_similarity(candidate_tokens, selected_tokens)
        if sim > max_sim:
            max_sim = sim
            
    return max_sim

def compute_mmr_score(relevance: float, max_similarity: float, lambda_param: float) -> float:
    return lambda_param * relevance - (1.0 - lambda_param) * max_similarity

T = TypeVar('T')

def mmr_rerank(
    items: List[T], 
    config: MMRConfig,
    id_getter,
    score_getter,
    content_getter
) -> List[T]:
    if not config.enabled or len(items) <= 1:
        return list(items)
        
    lam = max(0.0, min(1.0, config.lambda_param))
    if lam == 1.0:
        return sorted(items, key=score_getter, reverse=True)
        
    token_cache = {id_getter(item): tokenize(content_getter(item)) for item in items}
    
    scores = [score_getter(item) for item in items]
    max_score = max(scores)
    min_score = min(scores)
    score_range = max_score - min_score
    
    def normalize_score(score: float) -> float:
        if score_range == 0:
            return 1.0
        return (score - min_score) / score_range
        
    selected = []
    remaining = list(items)
    
    while remaining:
        best_item = None
        best_mmr_score = float('-inf')
        
        for candidate in remaining:
            norm_rel = normalize_score(score_getter(candidate))
            candidate_tokens = token_cache[id_getter(candidate)]
            max_sim = max_similarity_to_selected(candidate_tokens, selected, token_cache, id_getter)
            
            mmr_score = compute_mmr_score(norm_rel, max_sim, lam)
            
            if mmr_score > best_mmr_score or (mmr_score == best_mmr_score and score_getter(candidate) > (score_getter(best_item) if best_item else float('-inf'))):
                best_mmr_score = mmr_score
                best_item = candidate
                
        if best_item:
            selected.append(best_item)
            remaining.remove(best_item)
        else:
            break
            
    return selected

def apply_mmr_to_hybrid_results(results: List[HybridSearchResult], config: MMRConfig) -> List[HybridSearchResult]:
    return mmr_rerank(
        items=results,
        config=config,
        id_getter=lambda x: f"{x.path}:{x.start_line}:{x.chunk_id}",
        score_getter=lambda x: x.score,
        content_getter=lambda x: x.snippet
    )
