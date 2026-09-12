"""웹 데모(docs/index.html) 빌드.

web/template.html 의 `__WORKADVISOR_DATA__` 자리에 사례 303건(요약 필드) + 담당자 디렉터리를
JSON 으로 심어 단일 HTML 을 만든다. 브라우저 안에서 BM25 검색 + 담당자 라우팅을 수행하고,
답변 문장은 페이지를 연 뷰어의 Claude(artifact `sample` 기능)가 생성한다.

  python scripts/build_web_demo.py        # -> docs/index.html (GitHub Pages 가 main:/docs 를 서빙)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = ROOT / "data/processed/insurance_it_cases.jsonl"
TEMPLATE = ROOT / "web/template.html"
OUT = ROOT / "docs/index.html"

FIELDS = ["case_id", "category", "request_type", "title", "situation", "target_system",
          "environment", "urgency", "requester_role", "required_documents", "missing_document",
          "approval_route", "required_tools", "pre_checks", "process_steps", "status",
          "outcome", "cautions"]


def main():
    rows = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    contacts = next(r["contacts"] for r in rows if r.get("record_type") == "contacts")
    cases = [{k: r.get(k) for k in FIELDS} for r in rows if r.get("record_type") != "contacts"]
    payload = {"cases": cases, "contacts": contacts,
               "categories": sorted({c["category"] for c in cases})}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # </script> 로 스크립트 블록이 끊기지 않도록 이스케이프
    data = data.replace("</", "<\\/")
    html = TEMPLATE.read_text(encoding="utf-8")
    assert "__WORKADVISOR_DATA__" in html
    from datetime import datetime
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    OUT.write_text(html.replace("__WORKADVISOR_DATA__", data).replace("__BUILD_STAMP__", stamp), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.0f} KB, cases={len(cases)}, "
          f"teams={len(contacts.get('by_category', {}))})")


if __name__ == "__main__":
    main()
