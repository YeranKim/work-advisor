"""답변 구성 — 검색 결과(사례 필드)를 근거로 한 답변 프롬프트와, LLM 이 없을 때의 결정적 보고서.

규칙은 skill/SKILL.md · skill/references/output_format.md 와 같다.
"""
import json
from typing import Optional

SYSTEM = (
    "당신은 보험사 사내 IT 업무를 과거 처리 사례를 근거로 안내하는 도우미 'Work Advisor' 입니다. "
    "주어진 사례의 실제 필드만 근거로 한국어(존댓말)로 답하고, 사례에 없는 내용은 만들지 않습니다."
)

CASE_FIELDS = ["case_id", "category", "status", "title", "situation", "required_documents",
               "missing_document", "approval_route", "required_tools", "pre_checks",
               "process_steps", "cautions", "outcome"]


def _slim(results):
    return [{k: r.get(k) for k in CASE_FIELDS} for r in results]


def history_text(turns, limit: int = 6) -> str:
    """앞선 대화를 짧은 텍스트로. turns = [{"role": "user"|"assistant", "text": ...}]"""
    prev = [t for t in turns if t.get("text")][-limit:]
    if not prev:
        return "(없음)"
    return "\n".join(("사용자: " if t["role"] == "user" else "안내: ") + str(t["text"])[:500] for t in prev)


def build_answer_prompt(query: str, results, related: Optional[dict], history: str = "(없음)") -> str:
    rel = f"{related['team']} · {related['contact']} · {related['channel']}" if related else "(없음)"
    return f"""[사용자 질문]
{query}

[앞선 대화]
{history}

[근거 사례] (JSON)
{json.dumps(_slim(results), ensure_ascii=False)}

[추가 문의처]
{rel}

작성 규칙:
- 권장 뼈대는 "## 제목" 아래 "### ① 상황 요약 / ### ② 권장 처리 절차 / ### ③ 준비할 정보·문서 / ### ④ 주의 사항 / ### ⑤ 참고 사례" 이지만 고정 틀은 아닙니다. 절차 전체를 묻는 질문은 5개를 대체로 다 쓰고, 단답형 질문(누구 결재?, 무슨 문서?)은 해당 부분만 간결히, 장애·문의는 원인 확인과 담당 연락을 앞세우세요.
- ② 는 process_steps 를 종합한 번호 목록. 사례마다 approval_route 가 다르면 "개인정보 포함 시 A→B→정보보호, 아니면 A→B" 처럼 조건별 분기를 명시.
- ③ 은 required_documents 의 합집합 + 공통 확인 정보(대상 시스템, 환경, 기간, 개인정보 여부).
- ④ 는 cautions 의 공통 항목과, 반려/보완 후 완료 사례의 missing_document 를 근거로 "처음부터 함께 제출" 같은 실질 조언.
- ⑤ 는 각 사례를 "case_id (상태) — 한 줄 요약" 으로.
- 맨 끝에 "추가 문의: 팀 · 연락처 · 접수 채널" 한 줄(추가 문의처가 있을 때만).
- 금지: 고정폭 구분선(━ ─ =), "model:" 같은 화자 접두사, 데이터 출처 고지("합성/가상 데이터", "사내 규정으로 최종 확인"), 내부 점수·파라미터 언급, 영역 한정 프레이밍.
- Markdown 표준 문법만 사용. 본문 텍스트로 바로 시작."""


def deterministic_report(results, related: Optional[dict]) -> str:
    """LLM 없이 상위 사례 필드를 그대로 정리한 보고서(Markdown)."""
    top = results[0]
    docs = list(dict.fromkeys(d for r in results for d in r.get("required_documents", [])))
    routes = list(dict.fromkeys(" → ".join(r.get("approval_route", [])) for r in results))
    missing = [f"{r['case_id']}: {r['missing_document']} 누락으로 {r['status']}"
               for r in results if r.get("missing_document") and r.get("status") != "완료"]
    cautions = list(dict.fromkeys(c for r in results for c in r.get("cautions", [])))[:4]
    out = [f"## {top['title']} 처리 안내", "", "### ① 상황 요약", top["situation"], "",
           "### ② 권장 처리 절차"]
    out += [f"{i}. {s}" for i, s in enumerate(top.get("process_steps", []), 1)]
    out += ["", f"결재선: {' / '.join(routes)}", "", "### ③ 준비할 정보·문서"]
    out += [f"- {d}" for d in docs]
    out += ["- 대상 시스템·환경(운영/개발/검증), 사용 기간, 개인정보 포함 여부", "", "### ④ 주의 사항"]
    out += [f"- {c}" for c in cautions]
    out += [f"- {m} → 처음부터 함께 제출" for m in missing]
    out += ["", "### ⑤ 참고 사례"]
    out += [f"- {r['case_id']} ({r['status']}) — {r['title']}" for r in results]
    if related:
        out += ["", f"추가 문의: {related['team']} · {related['contact']} · {related['channel']}"]
    return "\n".join(out)


def handoff_text(routing: dict) -> str:
    """담당자 안내(weak_match) 답변 텍스트. 대화 이력·CLI 겸용."""
    c = routing["contact"]
    lead = ("이 요청은 과거 처리 사례로 답할 업무가 아니라 담당 팀에 직접 요청할 사항입니다."
            if routing.get("weak_reason") == "directory_only_topic"
            else "이 질문에 정확히 맞는 사례가 없습니다.")
    return (f"{lead}\n\n- 담당 팀: {c['team']}\n- 연락처: {c['contact']}\n- 접수 채널: {c['channel']}\n"
            f"- 이 팀이 맡는 업무: {c.get('note', '')}")


ASK_FOR = {
    "인증·세션 정책": ["대상 시스템·앱(예: 보안접속 에이전트, 그룹웨어)", "원하는 재인증 주기·세션 시간 값과 업무상 사유"],
    "성능·튜닝": ["대상 DB·시스템과 환경(운영/개발)", "느린 쿼리(SQL)·화면과 발생 시간대"],
    "default": ["어떤 시스템·업무와 관련된 요청인지", "원하는 처리 결과(신청/변경/문의 중 무엇인지)"],
}


def ask_for(key: str):
    return ASK_FOR.get(key, ["대상 시스템과 환경(운영/개발/검증)", "요청 사유와 필요한 기간·범위"])
