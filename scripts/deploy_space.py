"""Hugging Face Spaces(Streamlit) 배포 스크립트.

Space 를 만들고(없으면), 앱 실행에 필요한 파일만 올리고, 환경변수에 있는 LLM 키를 Space Secrets 에 넣는다.
키는 이 스크립트를 실행하는 셸의 환경변수로만 전달되며 파일이나 저장소에 남지 않는다.

사용 (본인 터미널에서):
  export HF_TOKEN=hf_...            # huggingface.co/settings/tokens 에서 Write 권한 토큰
  export GEMINI_API_KEY=...         # 또는 ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY
  venv/bin/python scripts/deploy_space.py <HF계정>/work-advisor

배포 후 주소: https://huggingface.co/spaces/<HF계정>/work-advisor
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SECRET_KEYS = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GROQ_API_KEY"]
VARIABLE_KEYS = ["LLM_PROVIDER", "LLM_MODEL"]

SPACE_README = """---
title: Work Advisor
emoji: 🧭
colorFrom: blue
colorTo: gray
sdk: streamlit
sdk_version: "{sdk_version}"
python_version: "3.11"
app_file: app.py
pinned: false
---

# Work Advisor — 보험사 IT 업무 사례 검색 Agent

과거 보험사 IT 업무 사례 303건을 하이브리드 검색(multilingual-e5 벡터 + BM25)해 처리 절차·필요 문서·결재선을
안내하고, 맞는 사례가 없거나 사례로 답할 업무가 아니면(정책 변경·성능 튜닝 의뢰 등) 담당 팀·연락처·접수 채널을 연결한다.

- 검색·판정·담당자 라우팅: 파이썬(`src/workcase_agent/search.py`), LLM 무관
- 답변 문장 생성: `src/workcase_agent/llm.py` 어댑터 — Claude / GPT / Gemini / Groq 중 Secrets 에 키가 있는 것을 사용
- 처음 접속 시 임베딩 모델 로딩으로 1~2분 걸릴 수 있음

소스 저장소: https://github.com/YeranKim/work-advisor
"""

SPACE_REQUIREMENTS = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.8.0
sentence-transformers==5.1.2
transformers==4.57.6
faiss-cpu==1.13.0
numpy==2.0.2
rank-bm25==0.2.2
requests>=2.31
streamlit>={sdk_version}
"""


def stage(tmp: Path, sdk_version: str):
    (tmp / "README.md").write_text(SPACE_README.format(sdk_version=sdk_version), encoding="utf-8")
    (tmp / "requirements.txt").write_text(SPACE_REQUIREMENTS.format(sdk_version=sdk_version), encoding="utf-8")
    shutil.copy(ROOT / "app.py", tmp / "app.py")
    shutil.copytree(ROOT / "src", tmp / "src", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (tmp / "data/processed").mkdir(parents=True)
    shutil.copy(ROOT / "data/processed/insurance_it_cases.jsonl", tmp / "data/processed/insurance_it_cases.jsonl")
    shutil.copytree(ROOT / "indexes", tmp / "indexes")
    (tmp / "scripts").mkdir()
    for f in ("ask.py", "show_case.py"):
        shutil.copy(ROOT / "scripts" / f, tmp / "scripts" / f)


def main():
    if len(sys.argv) != 2 or "/" not in sys.argv[1]:
        sys.exit("사용: python scripts/deploy_space.py <HF계정>/<space이름>")
    repo_id = sys.argv[1]
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        sys.exit("HF_TOKEN 환경변수가 필요합니다 (Write 권한).")
    try:
        import streamlit
        sdk_version = streamlit.__version__
    except ImportError:
        sdk_version = "1.40.0"

    api = HfApi(token=token)
    url = api.create_repo(repo_id, repo_type="space", space_sdk="streamlit", exist_ok=True, private=False)
    print("Space:", url)

    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        stage(tmp, sdk_version)
        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="space",
                          commit_message="deploy: work advisor streamlit app")
    print("파일 업로드 완료")

    for k in SECRET_KEYS:
        v = os.environ.get(k, "").strip()
        if v:
            api.add_space_secret(repo_id, k, v)
            print(f"Secret 설정: {k}")
    for k in VARIABLE_KEYS:
        v = os.environ.get(k, "").strip()
        if v:
            api.add_space_variable(repo_id, k, v)
            print(f"Variable 설정: {k}={v}")
    print(f"\n완료 → https://huggingface.co/spaces/{repo_id}  (빌드 5~10분, 이후 첫 접속 1~2분)")


if __name__ == "__main__":
    main()
