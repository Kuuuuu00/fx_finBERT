import re

from utils.stopwords import stopword_density

_CLEANUP_PATTERNS = [
    # 저작권 태그
    re.compile(r"<저작권자[^>]*>"),
    # 연합인포맥스 저작권 문구
    re.compile(r"\(c\)\s*연합인포맥스[^\n]*", re.IGNORECASE),
    # 무단 전재-재배포 금지
    re.compile(r"무단\s*전재[^\n]*재배포\s*금지[^\n]*"),
    # 기자 서명: (서울=연합뉴스) 홍길동 기자
    re.compile(r"\([가-힣a-zA-Z\s]+\s*=\s*[가-힣a-zA-Z\s]+\)\s*[가-힣]{2,4}\s*기자\s*=?"),
    # ※ 제보 안내 줄
    re.compile(r"※[^\n]*"),
    # ▶ 구독 안내 줄
    re.compile(r"▶[^\n]*구독[^\n]*"),
    # (끝) 기사 종료 표시
    re.compile(r"\(끝\)\s*$", re.MULTILINE),
    # ▶ 로 시작하는 줄 전체
    re.compile(r"^\s*▶[^\n]*", re.MULTILINE),
    # [출처] 태그
    re.compile(r"\[[^\]]{1,20}\]\s*$", re.MULTILINE),
]

_MARKET_CLOSE_FX: frozenset = frozenset(["환율", "원/달러", "원달러", "달러/원", "외환시장"])
_MARKET_CLOSE_IND: frozenset = frozenset(["마감", "종가"])

_AD_INDICATORS: list[str] = ["바로가기", "구독신청", "이벤트", "회원가입"]


def clean_article_body(text: str) -> str:
    """저작권 문구, 기자명 라인, 광고 꼬리말 등 정형 노이즈 제거."""
    for pattern in _CLEANUP_PATTERNS:
        text = pattern.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_market_summary(title: str, body: str) -> bool:
    """단순 환율 마감 시황 보도 판별 (제목에 환율 키워드 + 마감/종가 동시 포함)."""
    has_fx = any(kw in title for kw in _MARKET_CLOSE_FX)
    has_close = any(kw in title for kw in _MARKET_CLOSE_IND)
    return has_fx and has_close


def is_irrelevant_article(title: str, body: str) -> tuple[bool, str]:
    """광고/안내 기사 판별 (절대 기준, 불용어 밀도와 별개로 적용).

    fx 키워드 화이트리스트는 사용하지 않음 — 외생 충격 기사(FOMC, 지정학 등)를
    환율 직접 언급 없이도 포함해야 하기 때문. (plan.md Phase 1 Step 3 참조)
    """
    text = title + " " + body
    if sum(text.count(w) for w in _AD_INDICATORS) >= 3:
        return True, "advertisement"
    return False, "keep"
