"""workcase-agent 검색 CLI (채팅 환경 연결용).

사용:
  python scripts/ask.py "운영 DB 조회 계정이 필요한데 뭘 해야 해?"
  python scripts/ask.py "VPN 신청" --top-k 3 --json
  python scripts/ask.py "방화벽 오픈" --category "방화벽·네트워크" --status 완료

--json  : 기계가 파싱하기 좋은 JSON을 stdout으로 출력 (채팅 Agent가 근거로 사용)
기본    : 사람이 읽기 좋은 요약 출력

JSON 필드:
  results          유사 사례 Top-K
  weak_match       True 면 사례로 답하지 말고 contact 로 담당자 안내
  weak_reason      no_results / low_vector_similarity / no_keyword_overlap / directory_only_topic
  contact          weak_match 일 때 안내할 담당 팀 (team/contact/channel/note + key/routed_by/matched_keywords)
  related_contact  weak_match 가 아닐 때 1위 사례 카테고리의 담당 팀 (추가 문의처로 답변 말미에 사용 가능)
"""
import argparse
import json
import sys
from pathlib import Path

# src 를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from workcase_agent.search import search_cases, route_contact  # noqa: E402


def _print_contact(label: str, c: dict):
    print(f"{label} → {c.get('team','')}")
    print(f"  연락처: {c.get('contact','')}")
    print(f"  접수  : {c.get('channel','')}")
    if c.get("note"):
        print(f"  담당  : {c.get('note','')}")
    why = c.get("routed_by", "")
    if why == "keyword":
        print(f"  근거  : 디렉터리 키워드 일치 {c.get('matched_keywords')}")
    elif why == "similarity":
        print(f"  근거  : 담당 업무 설명과의 유사도 {c.get('similarity')}")
    elif why == "default":
        print("  근거  : 특정 팀을 정하기 어려워 1차 접수 창구로 안내")


def main():
    ap = argparse.ArgumentParser(description="보험사 IT 업무 유사 사례 검색")
    ap.add_argument("query", help="한국어 업무 질의")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--category", default=None, help="카테고리 필터")
    ap.add_argument("--status", default=None, help="상태 필터 (완료/반려/보완 후 완료/진행 중)")
    ap.add_argument("--json", action="store_true", help="JSON 출력")
    args = ap.parse_args()

    results = search_cases(
        args.query, top_k=args.top_k,
        category=args.category, status=args.status,
    )
    # 약한 매칭 판정 + 담당자 라우팅 (규칙은 search.route_contact 참조)
    routing = route_contact(args.query, results)
    weak = routing["weak_match"]
    contact = routing["contact"]

    if args.json:
        print(json.dumps({
            "query": args.query,
            "results": results,
            "weak_match": weak,
            "weak_reason": routing["weak_reason"],
            "contact": contact,
            "related_contact": routing["related_contact"],
        }, ensure_ascii=False))
        return

    if weak:
        if routing["weak_reason"] == "directory_only_topic":
            print("이 요청은 과거 처리 사례로 답할 업무가 아니라 담당 팀에 직접 요청할 사항입니다.")
        else:
            print("질문에 정확히 맞는 사례를 찾지 못했습니다.")
        print()
        _print_contact("담당자 안내", contact)
        if results:
            print("\n참고 (직접 맞는 사례는 아님):")
            for r in results[:2]:
                print(f"  - {r['case_id']} ({r['status']}) {r['title']}")
        return

    print(f"질의: {args.query}")
    print(f"검색된 유사 사례: {len(results)}건\n")
    for r in results:
        print(f"[{r['score']}] {r['case_id']}  ({r['category']} / {r['status']})")
        print(f"  제목    : {r['title']}")
        print(f"  상황    : {r['situation']}")
        print(f"  필요문서: {', '.join(r['required_documents'])}"
              + (f"  (누락: {r['missing_document']})" if r['missing_document'] else ""))
        print(f"  결재선  : {' → '.join(r['approval_route'])}")
        print(f"  설치도구: {', '.join(r['required_tools'])}")
        print(f"  사전점검: {', '.join(r['pre_checks'])}")
        print("  처리절차:")
        for i, s in enumerate(r["process_steps"], 1):
            print(f"     {i}. {s}")
        print(f"  주의    : {' / '.join(r['cautions'])}")
        print(f"  결과    : {r['outcome']}")
        print()

    if routing["related_contact"]:
        _print_contact("추가 문의처", routing["related_contact"])


if __name__ == "__main__":
    main()
