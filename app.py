"""Work Advisor — Streamlit 웹 앱 (로컬 실행용).

저장소의 파이썬 검색·판정·라우팅(search.py)을 그대로 실행하고, 맞는 사례가 있을 때만
LLM(llm.py 어댑터: Claude / GPT / Gemini / Groq 중 키가 있는 것)으로 답변 문장을 만든다.
LLM 키가 없으면 상위 사례 필드를 정리한 보고서를 보여준다.

  streamlit run app.py
"""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)                      # search.py 가 data/·indexes/ 상대경로를 쓴다
sys.path.insert(0, str(ROOT / "src"))

from workcase_agent import llm  # noqa: E402
from workcase_agent.answer import (SYSTEM, ask_for, build_answer_prompt,  # noqa: E402
                                   deterministic_report, handoff_text, history_text)
from workcase_agent.search import directory_entries, route_contact, search_cases  # noqa: E402

TOP_K = 4
SCENARIO = [
    ("1턴 · 계정 발급", "계약계 DB 신규 조회 계정을 발급받으려고 하는데 어떻게 해야 해?"),
    ("2턴 · 접속 장애", "보안접속 에이전트 연결이 계속 끊기는데 어떡해야 해"),
    ("3턴 · 정책 변경 요청처", "MFA 세션의 재인증 주기를 늘리려면 어떻게 해야 해?"),
]
EXAMPLES = ["퇴직자 DB 계정 회수해야 하는데 절차 알려줘", "재택근무 VPN 신청 어떻게 해?",
            "운영 배포 롤백은 어떻게 해?", "쿼리가 너무 느려서 실행계획 튜닝 받고 싶어", "회의실 예약은 어떻게 해?"]
STATUS_ICON = {"완료": "✅", "보완 후 완료": "🟡", "반려": "⛔", "진행 중": "🔵"}

st.set_page_config(page_title="Work Advisor", page_icon="🧭", layout="wide")
st.markdown("""
<style>
.block-container{max-width:1080px;padding-top:1.6rem}
.wa-handoff{border:1px solid #d9a066;background:#fff7ec;border-radius:10px;padding:14px 18px;margin:8px 0 6px}
.wa-handoff .eyebrow{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:#8a4b0b;font-weight:600}
.wa-handoff .team{font-size:20px;font-weight:700;margin:2px 0 8px;color:#1c2b3a}
.wa-handoff table{font-size:14px;border-collapse:collapse}
.wa-handoff td{padding:2px 14px 2px 0;vertical-align:top;color:#1c2b3a}
.wa-handoff td:first-child{color:#8a4b0b;white-space:nowrap}
.wa-strip{font-size:12.5px;color:#5b6b7c;margin:0 0 6px}
.wa-strip b{color:#1b4f8a}
@media (prefers-color-scheme: dark){
 .wa-handoff{background:#33230f;border-color:#b26a1f}
 .wa-handoff .team,.wa-handoff td{color:#f1e6d6}
 .wa-handoff .eyebrow,.wa-handoff td:first-child{color:#f0b46a}
 .wa-strip{color:#9fb0c1}.wa-strip b{color:#8fbbee}
}
</style>""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="검색 모델을 불러오는 중입니다 (처음 한 번, 1~2분)…")
def warm_up():
    search_cases("계정 발급", top_k=1)   # 모델·인덱스 로딩
    return True


def run_pipeline(query: str):
    results = search_cases(query, top_k=TOP_K)
    routing = route_contact(query, results)
    return results, routing


def strip_html(results, routing):
    n = len(results)
    if routing["weak_match"]:
        why = "사례 DB 밖 업무" if routing["weak_reason"] == "directory_only_topic" else "맞는 사례 없음"
        return (f'<div class="wa-strip">① 사례 검색 <b>후보 {n}건</b> → ② 판정 <b>{why}</b> → '
                f'③ <b>담당자 연결 · {routing["contact"]["team"]}</b></div>')
    return (f'<div class="wa-strip">① 사례 검색 <b>후보 {n}건</b> → ② 판정 <b>맞는 사례 있음</b> → '
            f'③ <b>답변 생성</b></div>')


def handoff_html(c: dict) -> str:
    return (f'<div class="wa-handoff"><div class="eyebrow">담당 팀 연결</div><div class="team">{c["team"]}</div>'
            f'<table><tr><td>연락처</td><td><code>{c["contact"]}</code></td></tr>'
            f'<tr><td>접수 채널</td><td>{c["channel"]}</td></tr>'
            f'<tr><td>이 팀이 맡는 업무</td><td>{c.get("note", "")}</td></tr></table></div>')


def reason_text(routing) -> str:
    c = routing.get("contact") or routing.get("related_contact")
    if not c:
        return ""
    by = c.get("routed_by")
    if by == "keyword":
        kws = ", ".join(f"`{k}`" for k in c.get("matched_keywords", []))
        pre = "사례 DB 밖 업무로 보고 " if routing.get("weak_reason") == "directory_only_topic" else ""
        return f"질문에 {kws} 키워드가 있어 {pre}**{c['team']}** 로 연결했습니다."
    if by == "similarity":
        return f"맞는 사례가 없어, 담당 업무 설명과 가장 가까운 **{c['team']}** (유사도 {c.get('similarity')}) 로 연결했습니다."
    if by == "default":
        return "맞는 사례도, 특정할 담당 팀도 없어 1차 접수 창구로 안내했습니다."
    if by == "case_category":
        return f"1위 사례 카테고리({c['key']})의 담당 팀을 추가 문의처로 붙였습니다."
    return ""


def render_cases(results, picked=True):
    for r in results:
        icon = STATUS_ICON.get(r["status"], "•")
        with st.expander(f"{icon} {r['case_id']} · {r['title']}  ({r['category']} / {r['status']})"):
            st.markdown(f"**상황** {r['situation']}")
            docs = ", ".join(r["required_documents"]) + (f"  *(누락: {r['missing_document']})*" if r.get("missing_document") else "")
            st.markdown(f"**필요 문서** {docs}  \n**결재선** {' → '.join(r['approval_route'])}  \n"
                        f"**설치 도구** {', '.join(r['required_tools'])}  \n**사전 점검** {', '.join(r['pre_checks'])}")
            st.markdown("**처리 절차**\n" + "\n".join(f"{i}. {s}" for i, s in enumerate(r["process_steps"], 1)))
            st.markdown(f"**주의** {' / '.join(r['cautions'])}  \n**결과** {r['outcome']}")


def render_assistant(msg: dict):
    st.markdown(msg["strip"], unsafe_allow_html=True)
    if msg["kind"] == "handoff":
        st.markdown(msg["lead"])
        st.markdown(handoff_html(msg["contact"]), unsafe_allow_html=True)
        st.markdown("**접수 전에 알려주시면 좋은 정보**\n" + "\n".join(f"- {q}" for q in msg["clarify"]))
        with st.expander(f"판정 근거 · 검색 후보 {len(msg['results'])}건 (직접 맞는 사례는 아님)"):
            st.markdown(msg["reason"])
            render_cases(msg["results"])
    else:
        st.markdown(msg["text"])
        if msg.get("note"):
            st.info(msg["note"])
        with st.expander(f"근거 사례 {len(msg['results'])}건"):
            if msg.get("reason"):
                st.markdown(msg["reason"])
            render_cases(msg["results"])


def answer(query: str, history: list) -> dict:
    results, routing = run_pipeline(query)
    strip = strip_html(results, routing)
    if routing["weak_match"]:
        c = routing["contact"]
        text = handoff_text(routing)
        return {"role": "assistant", "kind": "handoff", "strip": strip, "results": results,
                "contact": c, "lead": text.split("\n")[0], "clarify": ask_for(c["key"]),
                "reason": reason_text(routing), "text": text}
    related = routing.get("related_contact")
    info = llm.available()
    note = None
    if info:
        try:
            text = llm.generate(build_answer_prompt(query, results, related, history_text(history)), system=SYSTEM)
        except llm.LLMError as e:
            text = deterministic_report(results, related)
            note = f"LLM 호출에 실패해 상위 사례 필드를 그대로 정리했습니다. ({str(e)[:120]})"
    else:
        text = deterministic_report(results, related)
        note = "LLM 키가 설정되지 않아 상위 사례 필드를 그대로 정리했습니다. 환경변수에 GEMINI_API_KEY 등을 넣으면 질문에 맞춘 문장을 생성합니다."
    return {"role": "assistant", "kind": "answer", "strip": strip, "results": results,
            "text": text, "note": note, "reason": reason_text(routing)}


# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.title("Work Advisor")
    st.caption("보험사 사내 IT 업무, 과거 사례로 절차를 안내하고 맞는 사례가 없으면 담당 팀을 연결합니다.")
    info = llm.available()
    st.markdown(f"사례 **303건** · 담당 팀 **13**  \n답변 생성: **{info['provider']} / {info['model']}**" if info
                else "사례 **303건** · 담당 팀 **13**  \n답변 생성: **LLM 없음 (사례 필드 정리 모드)**")
    st.subheader("시연 시나리오")
    for label, q in SCENARIO:
        if st.button(f"{label}\n{q}", key=f"sc-{label}", use_container_width=True):
            st.session_state["pending"] = q
    st.subheader("다른 질문 예시")
    for q in EXAMPLES:
        if st.button(q, key=f"ex-{q}", use_container_width=True):
            st.session_state["pending"] = q
    st.subheader("처리 흐름")
    st.markdown("1. **사례 검색** — 벡터(multilingual-e5)+BM25 하이브리드로 후보를 찾습니다.\n"
                "2. **판정** — 사례 DB 밖 영역 키워드 규칙 → 벡터 유사도·키워드 신호로 약한 매칭을 가립니다.\n"
                "3. **답변 또는 연결** — 맞는 사례가 있으면 절차·문서·결재선을 정리하고, 없으면 담당 팀을 안내합니다.")
    with st.expander("담당 팀 디렉터리"):
        for key, e in directory_entries():
            st.markdown(f"**{e['team']}**  \n`{e['contact']}`  \n{e['note']}")
    if st.button("대화 지우기", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── 본문 ──────────────────────────────────────────────────────
warm_up()
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if not st.session_state["messages"]:
    st.markdown("### 무엇을 처리해야 하나요?")
    st.markdown("계정 발급, VPN 신청, 방화벽 오픈, 배포, 장애 문의 같은 사내 IT 업무를 평소 말하듯 물어보세요. "
                "왼쪽 시연 시나리오는 계정 발급 → 접속 장애 → 정책 변경 요청처 안내로 이어지는 3턴 흐름입니다.")

for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        if m["role"] == "user":
            st.markdown(m["text"])
        else:
            render_assistant(m)

typed = st.chat_input("예: 퇴직자 DB 계정 회수해야 하는데 절차 알려줘")
query = typed or st.session_state.pop("pending", None)
if query:
    st.session_state["messages"].append({"role": "user", "text": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("사례를 찾고 판정하는 중…"):
            history = [{"role": m["role"], "text": m.get("text", "")} for m in st.session_state["messages"][:-1]]
            msg = answer(query, history)
        render_assistant(msg)
    st.session_state["messages"].append(msg)
