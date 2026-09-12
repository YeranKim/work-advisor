"""search_cases: 한국어 질의로 유사 업무 사례 Top-K 하이브리드 검색.

하이브리드 = 의미 기반 벡터 검색(multilingual-e5 + FAISS)
            + 키워드 기반 BM25 검색을 정규화해 가중 결합.
- 벡터: 동의어·의미가 달라도 잡음 (예: "로그인 안 됨" ↔ "인증 실패")
- BM25: 정확한 키워드·약어·시스템명("TLS", "SSH", "VPN")에 강함
- case_id 기준 원본 사례 전체 필드 반환 (절차/결재선/문서/도구 등)
"""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Optional

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

MODEL_NAME = "intfloat/multilingual-e5-base"
CASES = Path("data/processed/insurance_it_cases.jsonl")
IDX_FILE = Path("indexes/workcases.faiss")
META_FILE = Path("indexes/metadata.jsonl")

# 하이브리드 가중치 (벡터가 이미 강하므로 벡터에 더 무게)
VECTOR_WEIGHT = 0.6
BM25_WEIGHT = 0.4

# ── 약한 매칭 판정 ─────────────────────────────────────────────
# 최고 사례의 원본 벡터 유사도(vector_score)가 이 값 미만이면 '약한 매칭'.
# 정상 질의 vector_score ≈ 0.87+, 무관 질의 ≈ 0.80 이하.
WEAK_MATCH_THRESHOLD = 0.83
# 보조 신호: 벡터는 경계값을 넘겼지만 1위 사례의 bm25_norm 이 바닥(키워드 미매칭)이면 오탐.
WEAK_MATCH_BM25_FLOOR = 0.05
# 보조 신호 2: 질의 토큰 중 '변별력 있는' 토큰(전체 사례의 30% 이하에만 등장)이 1위 사례에
# 하나도 없으면 키워드 미매칭으로 본다. '절차'·'신청'처럼 거의 모든 사례에 있는 토큰은
# BM25 를 0 이 아니게 만들지만 근거가 되지 못한다 (예: "사원증 재발급 절차를 알려줘").
DISTINCTIVE_DF_RATIO = 0.3

# ── 담당자 라우팅 ──────────────────────────────────────────────
# 질의 ↔ 담당자 디렉터리 항목(팀·담당 업무 설명) 임베딩 유사도가 이 값 이상이고,
# 2위 팀보다 CONTACT_SIM_MARGIN 이상 높을 때만 그 팀으로 안내한다. 아니면 기본 창구(서비스데스크).
# 무관한 질의(회의실 예약, 사원증 재발급 등)는 모든 팀이 0.77~0.80 에 몰려 1·2위 차이가 거의 없다.
CONTACT_SIM_FLOOR = 0.82
CONTACT_SIM_MARGIN = 0.02


# 한글 토큰 끝에 자주 붙는 조사. '배치가'·'계정을'이 '배치'·'계정'과도 맞도록 어간을 함께 낸다.
_PARTICLES = ("에서는", "으로는", "에서", "으로", "이랑", "까지", "부터", "에게", "처럼",
              "이", "가", "은", "는", "을", "를", "의", "에", "로", "과", "와", "도", "만", "랑")


def _tokenize(text: str):
    """간단 한국어/영숫자 토큰화. 한글·영문·숫자 연속을 토큰으로.

    한글 토큰은 원형에 더해 조사를 뗀 어간도 함께 낸다(문서·질의 양쪽 동일 적용).
    """
    out = []
    for tok in re.findall(r"[가-힣]+|[a-zA-Z0-9]+", text.lower()):
        out.append(tok)
        if "가" <= tok[0] <= "힣":
            for p in _PARTICLES:
                if len(tok) - len(p) >= 2 and tok.endswith(p):
                    out.append(tok[: -len(p)])
                    break
    return out


def _minmax(vals):
    """점수 리스트를 0~1 로 min-max 정규화."""
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.0 for _ in vals]
    return [(v - lo) / (hi - lo) for v in vals]


def _is_case(rec: dict) -> bool:
    """사례 레코드 여부. contacts 등 비사례 레코드를 제외한다."""
    return rec.get("record_type") != "contacts" and "search_text" in rec


@lru_cache(maxsize=1)
def load_contacts() -> dict:
    """담당자 디렉터리(record_type=contacts) 반환. 없으면 빈 dict.

    '관련 사례를 찾지 못했을 때' 채팅 에이전트가 담당 팀·연락처를 안내하는 데 쓴다.
    """
    for line in CASES.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record_type") == "contacts":
            return rec.get("contacts", {})
    return {}


def directory_entries():
    """디렉터리를 (key, entry) 목록으로. 'default' 가 항상 맨 앞."""
    c = load_contacts()
    out = []
    if c.get("default"):
        out.append(("default", c["default"]))
    out.extend(c.get("by_category", {}).items())
    return out


def get_contact(category: Optional[str] = None):
    """카테고리별 담당자 안내를 반환. 없으면 기본 서비스데스크."""
    c = load_contacts()
    by_cat = c.get("by_category", {})
    if category and category in by_cat:
        return by_cat[category]
    return c.get("default", {})


def _norm_query(text: str) -> str:
    """키워드 규칙 비교용: 소문자화 + 공백 제거."""
    return re.sub(r"\s+", "", text.lower())


def _keyword_hits(query: str, keywords) -> list:
    """디렉터리 항목의 keywords 중 질의에 일치하는 것을 반환.

    '+' 로 이은 키워드(예: '재인증+주기')는 모든 조각이 질의에 포함될 때만 일치.
    """
    q = _norm_query(query)
    hits = []
    for kw in keywords or []:
        parts = [p for p in _norm_query(kw).split("+") if p]
        if parts and all(p in q for p in parts):
            hits.append(kw)
    return hits


@lru_cache(maxsize=1)
def _contact_embeddings():
    """디렉터리 항목(팀·담당 업무)을 passage 로 임베딩. (key 목록, 행렬)"""
    _, _, _, model, _ = _load()
    keys, texts = [], []
    for key, e in directory_entries():
        keys.append(key)
        texts.append("passage: " + f"담당 영역: {key}\n팀: {e.get('team','')}\n"
                     f"업무: {e.get('note','')}")
    emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return keys, emb


def _case_categories() -> set:
    cases, *_ = _load()
    return {c["category"] for c in cases}


def is_weak_match(results) -> tuple:
    """검색 결과만으로 본 약한 매칭 여부와 사유.

    (1) 결과 없음 (2) top vector_score < 임계값
    (3) top bm25_norm 이 바닥이거나, 변별력 있는 질의 키워드가 top 사례에 하나도 없음 (키워드 미매칭)
    """
    if not results:
        return True, "no_results"
    top = results[0]
    if top.get("vector_score", 0.0) < WEAK_MATCH_THRESHOLD:
        return True, "low_vector_similarity"
    if top.get("bm25_norm", 0.0) < WEAK_MATCH_BM25_FLOOR or not top.get("keyword_overlap"):
        return True, "no_keyword_overlap"
    return False, ""


def route_contact(query: str, results) -> dict:
    """질의를 담당 팀에 연결한다. 반환:

    {
      "weak_match": bool,      # 최종 판정 (사례로 답하지 말고 담당자 안내)
      "weak_reason": str,      # no_results / low_vector_similarity / no_keyword_overlap /
                               # directory_only_topic / ""
      "contact": {...}|None,   # weak_match 일 때 안내할 담당 팀 (key, routed_by, matched_keywords 포함)
      "related_contact": {...}|None,  # weak 이 아닐 때 1위 사례 카테고리의 담당 팀 (추가 문의처)
      "contact_similarity": {key: score}  # 디버그용
    }

    판정 순서:
      A. 디렉터리에만 있는 영역(사례 DB 에 카테고리가 없는 팀, 예: 성능·튜닝, 인증·세션 정책)의
         키워드 규칙이 일치하면 → 검색 결과가 그럴듯해 보여도 담당자 안내 (weak_match=True).
         (예: 'MFA 재인증 주기 늘리기'는 세션 만료 장애 사례와 벡터 유사도가 높지만 정책 변경 요청이다)
      B. 검색 신호로 약한 매칭이면 → 키워드 일치 팀 > 임베딩 유사도 ≥ CONTACT_SIM_FLOOR (2위와 CONTACT_SIM_MARGIN 이상 차이) 팀 > 기본 창구.
      C. 정상 매칭이면 → contact 없음, related_contact 에 1위 사례 카테고리 담당 팀.
    """
    entries = directory_entries()
    keys, emb = _contact_embeddings()
    _, _, _, model, _ = _load()
    q = model.encode(["query: " + query], convert_to_numpy=True,
                     normalize_embeddings=True)
    sims = {k: round(float(x), 4) for k, x in zip(keys, (q @ emb.T)[0])}
    covered = _case_categories()

    def _pack(key, entry, routed_by, hits=None):
        d = dict(entry)
        d.pop("keywords", None)
        d.update({"key": key, "routed_by": routed_by,
                  "matched_keywords": hits or [],
                  "similarity": sims.get(key)})
        return d

    weak, reason = is_weak_match(results)
    hits_by_key = {k: _keyword_hits(query, e.get("keywords")) for k, e in entries}

    # A. 디렉터리 전용 영역 키워드 일치 → 사례보다 담당자 우선
    for key, entry in entries:
        if key == "default" or key in covered:
            continue
        if hits_by_key[key]:
            return {"weak_match": True, "weak_reason": "directory_only_topic",
                    "contact": _pack(key, entry, "keyword", hits_by_key[key]),
                    "related_contact": None, "contact_similarity": sims}

    # B. 약한 매칭 → 키워드 > 유사도 > 기본
    if weak:
        best_kw = max((k for k, _ in entries if k != "default"),
                      key=lambda k: len(hits_by_key[k]), default=None)
        if best_kw and hits_by_key[best_kw]:
            entry = dict(entries)[best_kw]
            return {"weak_match": True, "weak_reason": reason,
                    "contact": _pack(best_kw, entry, "keyword", hits_by_key[best_kw]),
                    "related_contact": None, "contact_similarity": sims}
        ranked = sorted((k for k, _ in entries if k != "default"), key=lambda k: -sims[k])
        best_sim, second = ranked[0], (ranked[1] if len(ranked) > 1 else None)
        margin = sims[best_sim] - (sims[second] if second else 0.0)
        if sims[best_sim] >= CONTACT_SIM_FLOOR and margin >= CONTACT_SIM_MARGIN:
            return {"weak_match": True, "weak_reason": reason,
                    "contact": _pack(best_sim, dict(entries)[best_sim], "similarity"),
                    "related_contact": None, "contact_similarity": sims}
        return {"weak_match": True, "weak_reason": reason,
                "contact": _pack("default", get_contact(None), "default"),
                "related_contact": None, "contact_similarity": sims}

    # C. 정상 매칭 → 1위 사례 카테고리의 담당 팀을 추가 문의처로
    top_cat = results[0]["category"]
    related = None
    if top_cat in dict(entries):
        related = _pack(top_cat, dict(entries)[top_cat], "case_category")
    return {"weak_match": False, "weak_reason": "", "contact": None,
            "related_contact": related, "contact_similarity": sims}


@lru_cache(maxsize=1)
def _load():
    # 사례 레코드만 사용 (contacts 등 비사례 레코드는 벡터 인덱스와 순서를 맞추기 위해 제외)
    cases = [c for c in (json.loads(l) for l in
                         CASES.read_text(encoding="utf-8").splitlines())
             if _is_case(c)]
    meta = [json.loads(l) for l in META_FILE.read_text(encoding="utf-8").splitlines()]
    index = faiss.read_index(str(IDX_FILE))
    model = SentenceTransformer(MODEL_NAME)
    # BM25 는 사례 search_text 를 토큰화해 구축 (벡터 인덱스와 동일한 순서)
    bm25 = BM25Okapi([_tokenize(c["search_text"]) for c in cases])
    return cases, meta, index, model, bm25


@lru_cache(maxsize=1)
def _token_stats():
    """사례별 토큰 집합과 토큰별 문서 빈도(df). 변별력 있는 키워드 겹침 계산용."""
    cases, *_ = _load()
    sets = [set(_tokenize(c["search_text"])) for c in cases]
    df = {}
    for st in sets:
        for t in st:
            df[t] = df.get(t, 0) + 1
    return sets, df


def _keyword_overlap(query: str, row: int) -> list:
    """질의의 변별력 있는 토큰(df 비율 ≤ DISTINCTIVE_DF_RATIO) 중 해당 사례에 있는 것."""
    sets, df = _token_stats()
    n = len(sets)
    return sorted({t for t in _tokenize(query)
                   if len(t) >= 2 and df.get(t, 0) <= DISTINCTIVE_DF_RATIO * n and t in sets[row]})


def search_cases(query: str, top_k: int = 5,
                 category: Optional[str] = None,
                 status: Optional[str] = None):
    """한국어 질의로 유사 사례 하이브리드 검색. category/status 필터 옵션.

    벡터 cosine 점수와 BM25 점수를 각각 0~1 정규화 후 가중 합산해
    최종 순위를 결정한다. 모든 사례에 대해 점수를 계산해 결합한다.
    """
    cases, meta, index, model, bm25 = _load()
    n = len(cases)

    # 1) 벡터 점수: 전체 사례에 대해 코사인 유사도 (row 순서로 정렬)
    q = model.encode(["query: " + query], convert_to_numpy=True,
                     normalize_embeddings=True).astype("float32")
    v_scores, v_idxs = index.search(q, n)
    vec_by_row = [0.0] * n
    for s, i in zip(v_scores[0], v_idxs[0]):
        if i >= 0:
            vec_by_row[i] = float(s)

    # 2) BM25 점수: 전체 사례에 대해
    bm_by_row = list(bm25.get_scores(_tokenize(query)))

    # 3) 각각 0~1 정규화 후 가중 합산
    vec_n = _minmax(vec_by_row)
    bm_n = _minmax(bm_by_row)
    combined = [
        (VECTOR_WEIGHT * vec_n[i] + BM25_WEIGHT * bm_n[i], i)
        for i in range(n)
    ]
    combined.sort(reverse=True)

    results = []
    for hybrid_score, i in combined:
        c = cases[i]
        if category and c["category"] != category:
            continue
        if status and c["status"] != status:
            continue
        results.append({
            "case_id": c["case_id"],
            "score": round(float(hybrid_score), 4),
            "vector_score": round(vec_by_row[i], 4),
            "bm25_norm": round(bm_n[i], 4),
            "keyword_overlap": _keyword_overlap(query, i),
            "category": c["category"],
            "title": c["title"],
            "status": c["status"],
            "situation": c["situation"],
            "required_documents": c["required_documents"],
            "missing_document": c.get("missing_document"),
            "approval_route": c["approval_route"],
            "required_tools": c["required_tools"],
            "pre_checks": c["pre_checks"],
            "process_steps": c["process_steps"],
            "cautions": c["cautions"],
            "outcome": c["outcome"],
        })
        if len(results) >= top_k:
            break
    return results


if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "운영 DB 조회 계정이 필요한데 뭘 해야 해?"
    res = search_cases(q, top_k=5)
    print(f"질의: {q}\n")
    for r in res:
        print(f"[{r['score']}] {r['case_id']} ({r['category']}/{r['status']}) {r['title']}")
        print(f"    결재선: {' → '.join(r['approval_route'])}")
        print(f"    필요문서: {', '.join(r['required_documents'])}"
              + (f" (누락: {r['missing_document']})" if r['missing_document'] else ""))
        print(f"    도구: {', '.join(r['required_tools'])}")
        print()
