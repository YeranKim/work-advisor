#!/usr/bin/env bash
# docs/index.html(웹 데모)을 GitHub Pages 용 저장소의 main 브랜치로 푸시한다.
#
# 사전 준비 (1회): github.com/new 에서 공개 저장소를 만든다 (예: work-advisor-demo, README 없이 빈 저장소).
# 사용:
#   bash scripts/deploy_pages.sh https://github.com/<계정>/work-advisor-demo.git
# 푸시 후 저장소 Settings > Pages > Source 를 "Deploy from a branch", Branch 를 main / (root) 로 지정하면
# 1~2분 뒤 https://<계정>.github.io/work-advisor-demo/ 에서 열린다.
set -euo pipefail
REMOTE="${1:?원격 저장소 URL 을 인자로 주세요}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/docs/index.html" ] || { echo "docs/index.html 이 없습니다. 먼저 python scripts/build_web_demo.py"; exit 1; }
TMP="$(mktemp -d)"
cp "$ROOT/docs/index.html" "$TMP/index.html"
cp "$ROOT/docs/work-advisor-presentation.html" "$TMP/presentation.html" 2>/dev/null || true
touch "$TMP/.nojekyll"
cat > "$TMP/README.md" <<'MD'
# Work Advisor 웹 데모

보험사 IT 업무 질문에 과거 사례로 절차를 안내하고, 맞는 사례가 없으면 담당 팀을 연결하는 정적 데모.
사례 검색(BM25)과 담당자 연결 판정은 브라우저 안에서 동작하며 외부 API 를 쓰지 않는다.
소스 저장소: https://github.com/YeranKim/work-advisor
MD
cd "$TMP"
git init -q -b main
git add -A
git -c user.name="work-advisor" -c user.email="work-advisor@users.noreply.github.com" commit -q -m "deploy: work advisor web demo"
git push "$REMOTE" main
echo "푸시 완료 → 저장소 Settings > Pages 에서 Branch: main / (root) 지정"
rm -rf "$TMP"
