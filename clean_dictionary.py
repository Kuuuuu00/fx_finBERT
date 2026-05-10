"""
clean_dictionary.py
─────────────────────────────────────────────────────────────────────────────
이미 생성된 tone_dictionary.csv 에서 불용어가 포함된 N-gram 행을 제거합니다.

사용법:
    uv run python clean_dictionary.py
    uv run python clean_dictionary.py --input tone_dictionary.csv --output tone_dictionary_cleaned.csv
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# 불용어 설정
# ══════════════════════════════════════════════════════════════════════════════

STOPWORD_KEYWORDS: frozenset = frozenset({
    # 1. 언론사 / 뉴스 메타 단어
    "연합뉴스", "이데일리", "블룸버그", "로이터", "뉴시스", "뉴스1",
    "속보", "긴급", "제보", "특파원", "기자", "마감",

    # 2. 매체 · 꼬리말 · 저작권 · 서비스 명칭
    "연합인포맥스", "인포맥스", "투데이", "한경", "매경", "조선비즈", "아시아경제",
    "신문", "일보", "게티", "모바일", "구독", "신청", "채널", "추가",
    "무단", "전재", "재배포", "배포", "금지", "저작권", "저작권자", "캡처",
    "ⓒ", "©", "사진", "제공", "보여요",
    "신문협회", "디지털", "온라인", "유료뉴스", "플러스", "리얼타임", "이용", "규칙", "게재",
    "미디어", "터미널", "단말기", "최상", "안내", "판매", "기사",

    # 3. 광고 · 홍보 · 웹/앱 UI · 호객 텍스트
    "트래블러", "상담센터", "종목", "진단", "안방", "클릭", "팩트", "체크", "영상",
    "재테크", "습관", "멀티미디어", "포토", "매거진", "닷컴", "바로가기", "글방",
    "실시간", "바로보기", "구매", "신규", "오픈", "트렌드", "총집결", "많이",
    "만나", "가장", "빠른", "종합", "스마트", "방송", "마켓", "전문가", "성공", "부르",

    # 4. 현장 스케치 노이즈 · 부사 · 직함
    "딜링룸", "하나은행", "본점", "을지로", "열린", "오전", "오후", "이날", "중구",
    "자세", "기록", "직원", "업무", "변조", "대응", "센터", "현지", "시간",
    "진공", "동취", "재단",

    # 5. 노이즈 인명 (기자 · 공인)
    "우정민", "최진석", "이종혁", "이장원", "정선영", "이대희", "종국", "민경원", "승지",
    "이창용", "제롬", "파월", "도널드", "트럼프", "한국은행총재", "연준의장", "대통령", 

    # 주요인물 등장보다, 해당 인물이 한 발언이 중요하기 때문에 인명은 필터링

    # 6. 뉴스 플랫폼 서비스 홍보 문구
    "뉴스",

    # 7. 환율 고시표 노이즈
    # 7-1. 소수 통화 단위
    "크로네", "페세타", "디르함", "디나르", "실링", "루피", "링깃",
    "바트", "코루나", "즐로티", "포린트", "레알", "랜드", "페소", "루블",
    # 7-2. 고시표 전용 국가명
    "덴마크", "노르웨이", "벨기에", "오스트리아",
    "쿠웨이트", "아랍에미리트", "바레인", "카타르",
    # 7-3. 고시 업무 전용 용어
    "기준율", "기준환율", "고시",
    "서울외국환중개", "한국자금중개",

    # 8. 언론/통신사/채널 (추가)
    "경향신문", "경향닷컴", "대한민국", "희망", "언론",
    "파이낸셜타임스", "보도", "뉴욕", "타임스", "블룸버그통신", "월스트리트저널",
    "집계", "트위터", "페이스북", "유튜브", "네이버", "카톡", "라인", "방송채널",

    # 9. 기자 · 직함 · 기관명 (추가)
    "이경민", "대신증권", "신호경", "이민혁", "국민은행", "연구원",
    "박정우", "노무라", "증권", "이코노미스트", "백석현", "신한은행",
    "홍승모", "손은정", "우리선물", "주원", "현대경제연구원", "경제연구실장",
    "최현석", "문정희", "오정석", "kb선물", "촬영", "이세원", "제작",
    "이태호", "최자윤", "일러스트", "남궁선", "권소현", "이진철",

    # 10. 기타 안내 문구 / 노이즈 (추가)
    "놓치", "아쉬운", "핫뉴스", "브리프", "하루", "무료", "지금", "다운",
    "화면", "번호", "사전", "동의", "없이", "링크", "코리아", "재판매",
    "db", "종로구", "지로", "의사봉", "두드리", "장마", "감전",
    "포인트", "상황", "스타", "벅스", "커피", "드세요", "예하",
    "키움증권", "한지훈", "선희",

    # 11-A. 주식 리딩방 / 스팸 광고 / 앱 홍보 (1~5gram 확장 추가)
    "고수익", "비법", "매매내역", "수익비결", "정식버전", "버전", "솔루션", "체험",
    "초과달성", "내역",

    # 11-B. 지역 기사 / 날씨 / 사건사고 / 무관한 일상어 (1~5gram 확장 추가)
    "축제", "유아", "사용자", "가슴", "갤러리", "성폭행", "충전", "마당", "일반인",
    "연수", "관광", "내국", "시설", "주민", "가구", "복지", "익명", "채용", "인근",
    "올림픽", "방역",

    # 11-C. 특정 지역 지국 / 기자 이름 / 특정 언론사 (1~5gram 확장 추가)
    "춘천", "제주", "제주시", "제주도", "서귀포시", "인천", "세종", "경기도", "용인",
    "동해", "최승현", "강홍", "선재규", "조휘", "형민", "클로드", "마이클", "페이",
    "교도", "로이터통신", "데일리", "기사문", "그래픽",

    # 11-D. 무의미한 기사 서술어 / 상투어 (1~5gram 확장 추가)
    "기자회견", "간담회", "덧붙였", "밝혔", "설명", "나타냈", "따르", "보인다",
    "보였", "나오", "마쳤", "전장",

    # 11-E. 뉴스 제작 / 서비스 / 광고 마케팅
    "편집", "취재", "칼럼", "사설", "응답", "인물", "이미지", "뱅크", "홈페이지",
    "문자", "시세", "추천", "마케팅", "청약",

    # 11-F. 특정 인명 및 직함
    "재닛옐런", "트리셰", "츠토무", "야마모토", "후진타오", "박옥희",
    "부행장", "부사장", "교수", "박사", "수상",

    # 11-G. 시황 기사 상투어 및 출처 인용구
    "서울", "로컬", "중개사", "시중은행", "딜러", "소식통", "인용",
    "금융통화위원회", "본회의", "한국은행본관", "취해", "드러난", "진로", "구인",

    # 11-H. 경제와 무관한 일상어 / 감정 표현
    "사랑", "이혼", "결혼", "자녀", "학교", "공원", "비키니", "눈물", "퀴즈",
    "자랑", "호텔", "아찔",

    # 11. 스팸 / 광고 / 주식 리딩방 관련
    "로또", "당첨자", "상담", "추천주", "유료", "전화", "머니", "색기",
    "자동", "화제", "혜택",

    # 12. 국내외 인물, 직책, 부서명 (경제 지표 자체가 아닌 출처/화자 정보)
    "최종구", "기획재정부장관", "윤증현", "강만수", "이성태", "한은총재",
    "박상재", "이주열", "김성순", "홍남기", "부총리", "김중수", "박재완",
    "정미영", "삼성선물", "최정희", "변지영", "박상희", "정태선", "문정현",
    "김유정", "이재헌", "저우샤오촨", "스콧", "헨리", "폴슨", "재무장관",
    "부시", "찰스", "버락", "오바마", "버냉키", "스노", "가이트너",
    "앨런그린스펀", "아베", "사사키", "토루", "제임스", "다니가키", "후쿠이",
    "도시히코", "데이비드", "메르켈", "히로시", "국제금융국장", "기자간담회",

    # 13. 특정 기업 및 개별 브랜드 (거시 환율과 무관)
    "롯데쇼핑", "SK하이닉스", "현대자동차", "LG", "포스코", "기아차",
    "동아일보", "아이폰", "스타워즈", "자이언트",

    # 14. 일상생활, 사회 사건, 방송 등 비경제 키워드
    "도민", "학생", "개미", "아빠", "시민", "친구", "가족", "연예인",
    "아내", "가수", "남자", "사자", "병원", "쉼표", "상품권", "교통",
    "월세", "음식", "차량", "아파트", "스포츠", "월드컵", "영화", "동영상",
    "화보", "축구", "드레스", "로켓", "컴퓨터", "별장", "사망", "경찰",
    "수사", "태풍", "폭행", "생활", "라디오", "앵커", "언제", "청사",
    "보수", "박싱", "바꾸", "가늠", "프리즘", "맞춤",

    # 15. 광고/스팸 및 매체 홍보 (추가)
    "창간", "기념", "재야", "고수", "기법", "전수", "특급", "사이트", "웹사이트",
    "플랫폼", "게임", "즐겨", "즐기", "경품", "스페셜", "브로커", "증정",

    # 16. 일상어 / 감정표현 / 비경제 (추가)
    "남편", "부부", "여성", "사람", "얼굴", "몸매", "점심", "바다", "느낌",
    "손안", "발칵",

    # 17. 무의미한 서술어 및 부사 (추가)
    "마치", "한다", "합니다", "해볼", "어쩔", "새로운", "어떻", "제대로",
    "마냥", "조금",

    # 18. 사회/정치 및 무관한 사실 (추가)
    "정당", "비상계엄", "대학", "성적", "적발", "이집트", "발전소",

    # 19. 특정 기업 및 인명/장소 (추가)
    "삼성생명", "NH투자증권", "KEB하나은행", "한국전력", "인텔",
    "이광주", "시진핑", "고이즈미", "파리",
    # ⚠️ 주의: "점도표", "스텝" 은 통화정책 핵심 단어 — 절대 포함 금지
})

# 한국어 성씨+이름 패턴: 1-gram NNP 중 인명으로 의심되는 토큰 제거
_KOREAN_SURNAMES = (
    "김|이|박|최|정|강|조|윤|장|임|한|오|서|신|권|황|안|송|류|전|홍|"
    "고|문|양|손|배|백|허|유|남|심|노|하|곽|성|차|주|우|구|나|민|진|"
    "지|엄|채|원|방|천|공|변|라|석|추|왕|도|위|설|반|길|여|탁"
)
_PERSON_NAME_RE: re.Pattern = re.compile(
    rf"^(?:{_KOREAN_SURNAMES})[가-힣]{{1,2}}/NNP$"
)


def is_stopword_ngram(ngram: str) -> bool:
    """N-gram 하나가 불용어 조건에 해당하면 True 반환."""
    words = [tok.split("/")[0] for tok in ngram.split(";")]

    # 조건 1: 키워드 포함
    if any(w in STOPWORD_KEYWORDS for w in words):
        return True

    # 조건 2: 인명 패턴 (1-gram NNP만)
    if ";" not in ngram and _PERSON_NAME_RE.match(ngram):
        return True

    return False


def print_top_words(df: pd.DataFrame, top_n: int = 10):
    """Top-N Hawkish / Dovish 단어를 가독성 좋게 출력합니다."""

    hawkish = df[df["tone"] == "Hawkish"].nlargest(top_n, "polarity_score")
    dovish  = df[df["tone"] == "Dovish"].nsmallest(top_n, "polarity_score")

    n_hawk = (df["tone"] == "Hawkish").sum()
    n_dove = (df["tone"] == "Dovish").sum()
    n_neut = (df["tone"] == "Neutral").sum()

    W = 68

    print()
    print("=" * W)
    print("  환율 방향성 Tone Dictionary 구축 결과")
    print("=" * W)
    print(f"  전체 어휘    : {len(df):>6,}개")
    print(f"  Hawkish      : {n_hawk:>6,}개  (환율 상승 / 원화 약세 연관어)")
    print(f"  Dovish       : {n_dove:>6,}개  (환율 하락 / 원화 강세 연관어)")
    print(f"  Neutral      : {n_neut:>6,}개")
    print()
    print("  Tone 비율:")
    total = n_hawk + n_dove + n_neut
    bar_h = "█" * round(n_hawk / total * 40)
    bar_d = "█" * round(n_dove / total * 40)
    bar_n = "█" * round(n_neut / total * 40)
    print(f"  Hawkish  {n_hawk/total*100:5.1f}%  {bar_h}")
    print(f"  Dovish   {n_dove/total*100:5.1f}%  {bar_d}")
    print(f"  Neutral  {n_neut/total*100:5.1f}%  {bar_n}")

    # ── Top-N Hawkish ─────────────────────────────────────────────────────────
    print()
    print("─" * W)
    print(f"  🔺 Top-{top_n} Hawkish 단어  (환율 상승 / 원화 약세 연관)")
    print(f"  {'순위':<4} {'N-gram':<32} {'Score':>8} {'DF↑':>5} {'DF↓':>5} {'Size':>4}")
    print("  " + "─" * (W - 2))
    for rank, (_, row) in enumerate(hawkish.iterrows(), 1):
        word = row["ngram"]
        if len(word) > 30:
            word = word[:27] + "..."
        print(f"  {rank:<4} {word:<32} {row['polarity_score']:>8.4f} "
              f"{int(row['df_up']):>5} {int(row['df_down']):>5} {int(row['ngram_size']):>4}")

    # ── Top-N Dovish ──────────────────────────────────────────────────────────
    print()
    print("─" * W)
    print(f"  🔻 Top-{top_n} Dovish 단어   (환율 하락 / 원화 강세 연관)")
    print(f"  {'순위':<4} {'N-gram':<32} {'Score':>8} {'DF↑':>5} {'DF↓':>5} {'Size':>4}")
    print("  " + "─" * (W - 2))
    for rank, (_, row) in enumerate(dovish.iterrows(), 1):
        word = row["ngram"]
        if len(word) > 30:
            word = word[:27] + "..."
        print(f"  {rank:<4} {word:<32} {row['polarity_score']:>8.4f} "
              f"{int(row['df_up']):>5} {int(row['df_down']):>5} {int(row['ngram_size']):>4}")

    # ── N-gram 유형별 Tone 분포 ──────────────────────────────────────────────
    print()
    print("─" * W)
    print("  N-gram 유형별 Tone 분포:")
    print(f"  {'N-gram':<8} {'Hawkish':>8} {'Dovish':>8} {'Neutral':>8} {'합계':>8}")
    print("  " + "─" * (W - 2))
    for n in sorted(df["ngram_size"].unique()):
        sub = df[df["ngram_size"] == n]
        h = (sub["tone"] == "Hawkish").sum()
        d = (sub["tone"] == "Dovish").sum()
        nu = (sub["tone"] == "Neutral").sum()
        print(f"  {n}-gram  {h:>8} {d:>8} {nu:>8} {len(sub):>8}")

    # ── 대표 단어별 극성 점수 요약 ──────────────────────────────────────────
    print()
    print("─" * W)
    key_words = [
        "환율/NNG", "달러/NNG", "원화/NNG", "금리/NNG",
        "강세/NNG", "약세/NNG", "상승/NNG", "하락/NNG",
        "긴축/NNG", "완화/NNG", "불안/NNG", "안정/NNG",
    ]
    found = df[df["ngram"].isin(key_words)].set_index("ngram")
    if not found.empty:
        print("  주요 환율 관련 1-gram 극성 점수:")
        print(f"  {'단어':<16} {'Score':>8} {'LogPol':>8} {'Tone':<10} {'DF↑':>5} {'DF↓':>5}")
        print("  " + "─" * (W - 2))
        for w in key_words:
            if w in found.index:
                r = found.loc[w]
                print(f"  {w:<16} {r['polarity_score']:>8.4f} {r['log_polarity']:>8.4f} "
                      f"{r['tone']:<10} {int(r['df_up']):>5} {int(r['df_down']):>5}")

    print("=" * W)
    print()


def process_one(input_path: Path, output_path: Path, top_n: int) -> None:
    """단일 tone_dictionary CSV를 정제하고 저장합니다."""
    if not input_path.exists():
        log.error("입력 파일을 찾을 수 없습니다: %s", input_path)
        return

    # ── 로드 ───────────────────────────────────────────────────────────────
    df = pd.read_csv(input_path, dtype=str)
    log.info("로드: %s  →  %d행", input_path.name, len(df))

    # ngram 컬럼 확인
    ngram_col = next((c for c in df.columns if c.lower() == "ngram"), None)
    if ngram_col is None:
        log.error("ngram 컬럼을 찾을 수 없습니다. 실제 컬럼: %s", list(df.columns))
        return

    # ── 필터링 ─────────────────────────────────────────────────────────────
    mask_keep = ~df[ngram_col].apply(is_stopword_ngram)

    removed   = (~mask_keep).sum()
    kept      = mask_keep.sum()

    # 제거된 항목 중 키워드 vs 인명 분류 집계
    removed_kw     = df.loc[~mask_keep, ngram_col].apply(
        lambda g: any(tok.split("/")[0] in STOPWORD_KEYWORDS for tok in g.split(";"))
    ).sum()
    removed_person = removed - removed_kw

    log.info("필터링 결과:")
    log.info("  제거: %d행  (키워드 포함: %d개 / 인명 패턴: %d개)", removed, removed_kw, removed_person)
    log.info("  유지: %d행", kept)
    log.info("  제거율: %.1f%%", removed / len(df) * 100 if len(df) > 0 else 0)

    df_clean = df[mask_keep].reset_index(drop=True)

    # ── 저장 ───────────────────────────────────────────────────────────────
    df_clean.to_csv(output_path, index=False, encoding="utf-8-sig")
    log.info("저장 완료: %s  (%d행)", output_path, len(df_clean))

    # ── 수치 컬럼 타입 변환 (CSV 로드 시 str → 숫자) ───────────────────────
    for col in ("ngram_size", "df_up", "df_down", "df_total"):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce").fillna(0).astype(int)
    for col in ("p_up", "p_down", "polarity_score", "log_polarity"):
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    # ── Top-N 출력 (build_dictionary.py 동일 포맷) ─────────────────────────
    if "tone" in df_clean.columns and "polarity_score" in df_clean.columns:
        print_top_words(df_clean, top_n)


def main():
    parser = argparse.ArgumentParser(description="tone_dictionary 불용어 정제 (1w / 2w / 3w)")
    parser.add_argument("--prefix", default="tone_dictionary",
                        help="입력 CSV 접두사 (기본: tone_dictionary → _1w/_2w/_3w.csv)")
    parser.add_argument("--top-n",  type=int, default=20,
                        help="출력할 Top-N Hawkish / Dovish 단어 수")
    args = parser.parse_args()

    HORIZONS = [
        (f"{args.prefix}_1w.csv",  f"{args.prefix}_cleaned_1w.csv",  "1W"),
        (f"{args.prefix}_2w.csv",  f"{args.prefix}_cleaned_2w.csv",  "2W"),
        (f"{args.prefix}_3w.csv",  f"{args.prefix}_cleaned_3w.csv",  "3W"),
    ]

    for inp, out, label in HORIZONS:
        print(f"\n{'=' * 68}")
        print(f"  [{label}]  {inp}  →  {out}")
        print(f"{'=' * 68}")
        log.info("[%s] 처리 시작: %s", label, inp)
        process_one(Path(inp), Path(out), args.top_n)
        log.info("[%s] 처리 완료\n", label)


if __name__ == "__main__":
    main()
