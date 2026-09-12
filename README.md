# workcase-agent — 보험사 IT 업무 사례 검색 Agent

과거 보험사 사내 IT 업무 사례(합성 303건)를 하이브리드 검색(벡터 + BM25)해, 신규 요청에 대한
**처리 절차 · 필요 문서 · 결재선 · 설치 프로그램 · 확인 사항**을 근거 기반으로 안내한다.

한국어 질의를 그대로 임베딩(multilingual-e5)해 FAISS 로 검색한다. 외부 LLM API 없이
로컬에서 검색이 동작하며, 답변 문장 생성은 이 저장소를 연 채팅 에이전트가 담당한다.

맞는 사례가 없거나(약한 매칭) 사례로 답할 업무가 아닌 요청(정책 변경·성능 튜닝 의뢰 등)은
**담당 팀·연락처·접수 채널**을 안내한다. 담당자 디렉터리는 13개 팀이다(사례 카테고리 11개 + 사례 DB 에 없는 영역 2개).

> **웹 데모** — https://yerankim.github.io/work-advisor-demo/ (브라우저 전용 정적 버전, 9장)
> **기술 정리 문서** — https://yerankim.github.io/work-advisor-demo/tech.html

---

## 1. 요구 사항

- Python 3.9 이상 (개발/검증 환경: macOS, Python 3.9.6)
- 인터넷 연결 (최초 1회, 임베딩 모델 `intfloat/multilingual-e5-base` 다운로드 약 1GB)
- 디스크 여유 약 2~3GB (torch + 모델 캐시 포함)

## 2. 설치

```bash
# 저장소 폴더로 이동
cd work-advisor

# 가상환경 생성 및 활성화 (Windows: .venv\Scripts\activate)
python3 -m venv .venv
source .venv/bin/activate

# 의존성 설치 (버전 고정)
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. 인덱스 준비

저장소에 이미 만들어진 인덱스(`indexes/`)가 포함되어 있으면 이 단계는 건너뛴다.
데이터만 있고 인덱스가 없거나, 데이터를 바꿨다면 재생성한다:

```bash
# 입력: data/processed/insurance_it_cases.jsonl (사례 303건 + 담당자 디렉터리 1줄)
# 출력: indexes/workcases.faiss, metadata.jsonl, manifest.json
python scripts/create_index.py
```

> 최초 실행 시 임베딩 모델을 내려받으며 시간이 걸린다(이후에는 캐시 사용).

## 3-1. (선택) DeepWork 스킬로 등록

DeepWork 앱에서 이 저장소를 "스킬"로 등록하면, 채팅에서 보험사 IT 업무를
물을 때 자동으로 이 검색 도구를 사용해 정해진 형식으로 답한다.

저장소 안의 `skill/` 폴더를 앱의 스킬 경로로 복사한다:

```bash
# macOS / Linux
mkdir -p ~/.deepwork/skills/work-advisor
cp -R skill/. ~/.deepwork/skills/work-advisor/
```

복사 후 `SKILL.md` 안의 `<프로젝트 루트>` 안내대로, 본인이 clone 한
저장소 경로를 사용하도록 맞춘다. (스킬은 CLI 를 호출할 때 저장소의
`scripts/ask.py` 를 그 경로로 실행한다.)

## 4. 사용법

### 4-1. 질문하기 (사람이 읽는 출력)
```bash
python scripts/ask.py "퇴직자 DB 계정 회수해야 하는데 절차 알려줘"
python scripts/ask.py "재택근무 VPN 신청 어떻게 해?" --top-k 3
```

### 4-2. 필터
```bash
python scripts/ask.py "방화벽 포트 오픈" --category "방화벽·네트워크"
python scripts/ask.py "포트 오픈" --status "반려"   # 반려 사례만
```
상태 값: `완료` / `반려` / `보완 후 완료` / `진행 중`

### 4-3. 사례 상세 보기
```bash
python scripts/show_case.py SYN-IT-DBA-0232
python scripts/show_case.py SYN-IT-DBA-0232 --json
```

### 4-4. 에이전트 연동용 JSON
```bash
python scripts/ask.py "질문" --top-k 4 --json
```

## 4-5. 3턴 시연 시나리오

계정 발급 → 접속 장애 진단 → 정책 변경 요청처 안내로 이어지는 후속 흐름 예시.
각 턴의 질의를 채팅에 입력하면 해당 보고서가 생성된다.

| 턴 | 입력 질의 | 검색 결과(핵심) | 답변 형태 |
|---|---|---|---|
| 1턴 | 계약계 DB 신규 조회 계정을 발급받으려고 하는데 어떻게 해야 해? | SYN-IT-DBA-0166·0001 (DB 계정·권한, 성공률 80%) | 계정 발급 절차 보고서 (결재선: 신청자 팀장 → 시스템 책임자) |
| 2턴 | 보안접속 에이전트 연결이 계속 끊기는데 어떡해야 해 | SYN-IT-INC-0301·0303·0302 (IT 장애·문의) | 장애 진단 보고서 (MFA 세션 만료 / 유휴 타임아웃 / 단말 정책 3원인) |
| 3턴 | MFA 세션의 재인증 주기를 늘리려면 어떻게 해야 해? | 담당자 디렉터리 '인증·세션 정책' | 담당자 안내 (인증인프라팀 · 내선 3320 · iam-auth@example.local) |

> 3턴은 개별 장애 처리 사례가 아니라 정책 변경 요청이다. 벡터 유사도만 보면 '세션 만료 끊김' 장애
> 사례가 상위에 오지만, `route_contact()` 의 디렉터리 키워드 규칙(`재인증+주기` 등)이 먼저 걸러
> `weak_match: true` + 인증인프라팀으로 안내한다.

## 4-6. 담당자 라우팅 (약한 매칭 처리)

`scripts/ask.py --json` 은 `weak_match` 와 함께 `contact` / `related_contact` 를 돌려준다.
판정은 `src/workcase_agent/search.py` 의 `route_contact()` 가 담당하며 순서는 다음과 같다.

| 단계 | 조건 | 결과 |
|---|---|---|
| A. 디렉터리 전용 영역 | 사례 DB 에 없는 팀(성능·튜닝, 인증·세션 정책)의 키워드 규칙이 질의에 일치 | `weak_match: true`, `weak_reason: directory_only_topic`, 해당 팀 |
| B. 약한 매칭 | 결과 없음 OR 1위 벡터 유사도 < 0.83 OR 1위 bm25_norm < 0.05 | 키워드 일치 팀 → 담당 업무 설명과의 임베딩 유사도 ≥ 0.80 인 팀 → 기본 창구 순 |
| C. 정상 매칭 | 그 외 | `contact: null`, `related_contact` 에 1위 사례 카테고리 담당 팀 |

키워드 규칙은 데이터 파일의 contacts 레코드 안 `keywords` 에 있다. `+` 로 이은 항목(예: `세션+유효시간`)은
모든 조각이 질의(소문자·공백 제거)에 포함될 때만 일치한다. 팀·연락처·키워드를 바꾸려면 그 레코드만 고치면 된다
(인덱스 재생성 불필요).

```bash
python scripts/ask.py "MFA 세션의 재인증 주기를 늘리려면 어떻게 해야 해?"
# → 담당자 안내 → 인증인프라팀 (IAM·MFA 운영) / 내선 3320 / iam-auth@example.local
python scripts/ask.py "쿼리가 너무 느려서 실행계획 튜닝 받고 싶어" --json
# → "weak_match": true, "weak_reason": "directory_only_topic", "contact": {"team": "DB 성능관리팀 …"}
```

## 5. 다룰 수 있는 업무 영역 (11개)

DB 계정·권한 / 사내 계정·그룹웨어 / VPN·원격접속 / 프로그램 설치 /
서버 접근·작업 / 방화벽·네트워크 / 배포·변경관리 / 배치·인터페이스 장애 /
개인정보·데이터 추출 / 인증서·암호화키 / IT 장애·문의

## 6. 폴더 구조

```
work-advisor/
├── requirements.txt
├── data/processed/insurance_it_cases.jsonl   # 합성 사례 303건 + 담당자 디렉터리(13개 팀) 1줄 (약 1MB)
├── indexes/                                  # FAISS 인덱스 + 메타데이터
│   ├── workcases.faiss
│   ├── metadata.jsonl
│   └── manifest.json
├── src/workcase_agent/search.py              # 검색 핵심 로직 + 약한 매칭 판정 + 담당자 라우팅
├── scripts/
│   ├── ask.py            # 질의 → 유사 사례 검색 (메인 진입점)
│   ├── show_case.py      # 사례 ID → 상세
│   ├── create_index.py   # 인덱스 (재)생성
│   ├── prepare_cases.py  # 데이터 → 사례 변환
│   ├── make_report.py    # HTML 보고서 (선택)
│   ├── build_web_demo.py # 정적 웹 데모(docs/index.html) 빌드
│   └── deploy_pages.sh   # 정적 웹 데모를 GitHub Pages 저장소로 푸시
├── web/template.html     # 정적 웹 데모 템플릿 (검색·라우팅 로직 JS + UI)
├── docs/index.html       # 정적 웹 데모 빌드 결과 (GitHub Pages)
└── skill/                                    # DeepWork 스킬 (선택 등록)
    ├── SKILL.md
    └── references/output_format.md
```

## 7. 다른 컴퓨터로 옮길 때 체크리스트

- [ ] `data/processed/insurance_it_cases.jsonl` 포함했는가 (필수 입력)
- [ ] `indexes/` 포함했는가 — 없으면 3장으로 재생성 (약 15초)
- [ ] `requirements.txt`로 동일 버전 설치했는가
- [ ] 최초 실행 시 모델 다운로드용 인터넷 연결이 되는가
- [ ] `.venv/`는 옮기지 말 것 (OS/경로 의존적, 새로 생성)

## 8. 한계 (중요)

- 데이터는 **완전 합성(가상)** 이다. 특정 보험사의 실제 규정·시스템·결재선이 아니다.
  실제 신청서명·결재선·설치 프로그램은 사내 규정으로 최종 확인해야 한다.
- 답변 문장 생성은 채팅 에이전트가 수행한다. CLI 단독으로는 "유사 사례 검색 결과"까지 제공한다.

## 9. 정적 웹 데모 (브라우저 안 JS 복제본)

`docs/index.html` 은 사례 303건과 담당자 디렉터리를 내장한 단일 HTML 이다(소스: `web/template.html`). 두 곳에 배포한다.

| 배포 | 링크 | 동작 |
|---|---|---|
| GitHub Pages (공개, 로그인 불필요) | https://yerankim.github.io/work-advisor-demo/ | 사례 검색(BM25) + 담당자 연결 + 상위 사례 필드 정리. 파이썬은 실행되지 않고 같은 규칙을 JS 로 옮긴 복제본 |
| claude.ai Artifact (claude.ai 로그인 필요) | https://claude.ai/code/artifact/49af41f2-d655-4e48-b4f5-a56ea2fe04fa | 위 기능 + 뷰어의 Claude 로 판정·답변 문장 생성 (뷰어 사용량 소모, API 키 불필요) |

- 검색: 브라우저 안에서 BM25(한글 조사 제거 + 2음절 조각)로 후보 사례를 찾는다. 임베딩 모델은 쓰지 않는다.
- 판정: 사례 DB 밖 영역 키워드 규칙(A 단계) → 페이지를 연 뷰어의 Claude 가 "후보가 같은 업무 유형인가"를 판정.
- 출력: 맞는 사례가 있으면 Claude 가 5개 섹션 답변을 생성하고, 없으면 담당 팀 카드(팀·연락처·접수 채널·담당 업무)를
  디렉터리 값 그대로 보여준다. Claude 를 쓸 수 없는 환경에서는 검색 전용 모드로 상위 사례 필드를 정리해 보여준다.

```bash
python scripts/build_web_demo.py     # 데이터/템플릿을 바꿨을 때 재빌드 → docs/index.html
bash scripts/deploy_pages.sh https://github.com/<계정>/work-advisor-demo.git   # GitHub Pages 저장소로 푸시
```

GitHub Pages 는 github.com/new 에서 공개 저장소를 만든 뒤 위 스크립트로 푸시하고, 저장소 Settings > Pages 에서
Branch 를 `main` / `(root)` 로 지정하면 1~2분 뒤 열린다.

