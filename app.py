import streamlit as st
import pandas as pd
import numpy as np
import joblib
import platform
import html
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score,
    recall_score,
    roc_auc_score,
    precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# =========================================================
# 디자인 시스템 (Design tokens)
# =========================================================
# 하나의 팔레트 / 하나의 서체(Pretendard)로 사이드바 · 카드 · 차트를 통일합니다.

INK = "#181B2A"          # 제목/본문 진한 잉크색
TEXT = "#3F4356"          # 기본 본문
MUTED = "#868DA6"         # 캡션/보조 텍스트
BORDER = "#E4E7F1"        # 카드/구분선
BG = "#F6F7FB"            # 페이지 배경 (쿨 페이퍼톤)
SURFACE = "#FFFFFF"       # 카드 배경

PRIMARY = "#3D4FE0"       # 브랜드 메인 (인디고 블루)
PRIMARY_DARK = "#2B39B3"
PRIMARY_TINT = "#EEF0FD"  # 연한 브랜드 배경
NEUTRAL = "#D7DAE6"       # 미결제 등 보조 계열
ACCENT_AMBER = "#F2A93B"  # threshold / 강조선
ACCENT_TEAL = "#17B897"   # 긍정 지표(정밀도 등)
NEGATIVE = "#E2593B"      # 음수 delta / 경고

# 하위 호환용 별칭 (기존 로직에서 참조)
BLUE = PRIMARY
LIGHT_BLUE = PRIMARY_TINT
GRAY = NEUTRAL
TEXT_GRAY = TEXT

APP_DIR = Path(__file__).resolve().parent
FONT_DIR = APP_DIR / "fonts"


def register_fonts() -> str:
    """Pretendard 폰트를 matplotlib에 등록하고, 실패 시 OS별 한글 폰트로 대체한다."""
    family_name = None
    if FONT_DIR.exists():
        for font_file in sorted(FONT_DIR.glob("Pretendard-*.otf")):
            try:
                fm.fontManager.addfont(str(font_file))
                if family_name is None:
                    family_name = fm.FontProperties(fname=str(font_file)).get_name()
            except Exception:
                pass

    if family_name:
        plt.rcParams["font.family"] = family_name
        return family_name

    system = platform.system()
    if system == "Darwin":
        fallback = "AppleGothic"
    elif system == "Windows":
        fallback = "Malgun Gothic"
    else:
        fallback = "NanumGothic"
    plt.rcParams["font.family"] = fallback
    return fallback


FONT_FAMILY = register_fonts()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["text.color"] = INK
plt.rcParams["axes.edgecolor"] = BORDER
plt.rcParams["axes.labelcolor"] = TEXT
plt.rcParams["xtick.color"] = MUTED
plt.rcParams["ytick.color"] = MUTED
plt.rcParams["font.size"] = 10.5


def style_axes(ax, grid_axis="y"):
    """모든 차트에 공통으로 적용하는 미니멀 스타일."""
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(BORDER)
    ax.tick_params(length=0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=BORDER, linewidth=0.9, alpha=0.9, zorder=0)
        ax.set_axisbelow(True)
    return ax


def finalize_fig(fig):
    fig.patch.set_facecolor("white")
    fig.tight_layout()


st.set_page_config(
    page_title="공유오피스 3일 체험 · 결제 전환 예측",
    page_icon="💼",
    layout="wide",
)

# =========================================================
# 전역 CSS
# =========================================================
st.markdown(
    f"""
    <style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

    html, body, [class*="css"] {{
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}

    .stApp {{
        background-color: {BG};
    }}

    /* ---------- 본문 컨테이너 여백 ---------- */
    .block-container {{
        padding-top: 4rem;
        padding-bottom: 3rem;
        max-width: 1180px;
    }}

    /* ---------- 타이포그래피 ---------- */
    h1, h2, h3 {{
        color: {INK} !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
    }}
    h1 {{ font-size: 2.0rem !important; }}
    h2, .stSubheader {{ font-size: 1.28rem !important; }}
    p, li, span, label {{ color: {TEXT}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{
        color: {MUTED} !important;
    }}

    /* ---------- 히어로 헤더 ---------- */
    .hero {{
        background: linear-gradient(135deg, {INK} 0%, #2A2E4A 100%);
        border-radius: 20px;
        padding: 34px 38px;
        margin-bottom: 28px;
        box-shadow: 0 12px 28px rgba(24, 27, 42, 0.18);
    }}
    .hero-eyebrow {{
        display: inline-block;
        background: rgba(255,255,255,0.12);
        color: #C9CEFF;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        padding: 5px 12px;
        border-radius: 999px;
        margin-bottom: 14px;
    }}
    .hero-title {{
        color: #FFFFFF;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin: 0 0 8px 0;
    }}
    .hero-sub {{
        color: #B9BDD6;
        font-size: 0.98rem;
        line-height: 1.6;
        margin: 0;
        max-width: 760px;
    }}

    /* ---------- 섹션 라벨 ---------- */
    .section-label {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 6px 0 14px 0;
    }}
    .section-label .dot {{
        width: 8px; height: 8px; border-radius: 50%;
        background: {PRIMARY};
        flex-shrink: 0;
    }}
    .section-label span {{
        font-size: 1.05rem;
        font-weight: 800;
        color: {INK};
    }}
    .section-caption {{
        color: {MUTED};
        font-size: 0.88rem;
        margin: -8px 0 16px 18px;
    }}

    /* ---------- st.metric 카드화 ---------- */
    div[data-testid="stMetric"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 18px 20px 16px 20px;
        box-shadow: 0 2px 10px rgba(24, 27, 42, 0.04);
        position: relative;
        overflow: hidden;
    }}
    div[data-testid="stMetric"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 3px;
        background: linear-gradient(90deg, {PRIMARY}, {ACCENT_TEAL});
    }}
    div[data-testid="stMetricLabel"] {{
        color: {MUTED} !important;
        font-size: 0.82rem !important;
        font-weight: 600 !important;
    }}
    div[data-testid="stMetricValue"] {{
        color: {INK} !important;
        font-size: 1.55rem !important;
        font-weight: 800 !important;
    }}
    div[data-testid="stMetricDelta"] svg {{ display: inline; }}

    /* ---------- 카드/박스 ---------- */
    .card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px 22px;
        box-shadow: 0 2px 10px rgba(24, 27, 42, 0.04);
    }}
    .insight-box {{
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 18px 20px;
        background: {SURFACE};
        min-height: 150px;
        line-height: 1.7;
        color: {TEXT};
        box-shadow: 0 2px 10px rgba(24, 27, 42, 0.04);
    }}
    .insight-box b {{ color: {INK}; }}

    /* ---------- 결제 전환 요약 (가로 바) ---------- */
    .conv-summary-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 20px 24px 22px 24px;
        box-shadow: 0 2px 10px rgba(24, 27, 42, 0.04);
    }}
    .conv-summary-title {{
        font-size: 0.98rem;
        font-weight: 800;
        color: {INK};
        margin: 0 0 4px 0;
    }}
    .conv-summary-caption {{
        font-size: 0.85rem;
        color: {MUTED};
        margin: 0 0 16px 0;
    }}
    .conv-bar-track {{
        width: 100%;
        height: 14px;
        border-radius: 999px;
        background: {NEUTRAL};
        overflow: hidden;
        display: flex;
    }}
    .conv-bar-fill {{
        height: 100%;
        background: {PRIMARY};
        border-radius: 999px 0 0 999px;
    }}
    .conv-bar-labels {{
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        font-size: 0.88rem;
        color: {TEXT};
    }}
    .conv-bar-labels b {{ color: {INK}; }}
    .conv-dot {{
        display: inline-block;
        width: 9px; height: 9px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    .blue-box {{
        border-left: 4px solid {PRIMARY};
        background: {PRIMARY_TINT};
        padding: 16px 20px;
        border-radius: 12px;
        line-height: 1.75;
        color: {TEXT};
    }}
    .blue-box b {{ color: {INK}; }}

    /* ---------- 결측치 처리 규칙 카드 ---------- */
    .rule-card {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-left: 4px solid {PRIMARY};
        border-radius: 14px;
        padding: 20px 24px;
        box-shadow: 0 2px 10px rgba(24, 27, 42, 0.04);
    }}
    .rule-card-header {{
        color: {INK};
        font-size: 0.98rem;
        line-height: 1.6;
        margin-bottom: 14px;
    }}
    .rule-card-header b {{ color: {PRIMARY_DARK}; }}
    .rule-card-sub {{
        color: {MUTED};
        font-size: 0.84rem;
        font-weight: 700;
        margin-bottom: 10px;
    }}
    .rule-grid {{
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 14px;
    }}
    .rule-item {{
        background: {PRIMARY_TINT};
        border-radius: 10px;
        padding: 12px 14px;
    }}
    .rule-tag {{
        display: inline-block;
        color: {PRIMARY_DARK};
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 6px;
    }}
    .rule-detail {{
        color: {TEXT};
        font-size: 0.84rem;
        line-height: 1.55;
    }}
    .rule-footnote {{
        display: flex;
        gap: 8px;
        align-items: flex-start;
        background: #FFF7E8;
        border: 1px solid #F5DFAF;
        border-radius: 10px;
        padding: 10px 14px;
        color: #8A5A0A;
        font-size: 0.84rem;
        line-height: 1.55;
    }}
    .result-box {{
        border-radius: 14px;
        padding: 16px 20px;
        font-size: 1.02rem;
        line-height: 1.6;
    }}
    .result-positive {{
        background: #E9FBF4;
        border: 1px solid #B7EDD8;
        color: #0C7A56;
    }}
    .result-negative {{
        background: #F5F3FF;
        border: 1px solid #E1DCFA;
        color: #4C4A73;
    }}
    .small-note {{
        color: {MUTED};
        font-size: 0.86rem;
    }}
    .chip {{
        display: inline-block;
        background: {PRIMARY_TINT};
        color: {PRIMARY_DARK};
        font-size: 0.78rem;
        font-weight: 700;
        padding: 4px 11px;
        border-radius: 999px;
        margin-right: 6px;
    }}

    /* ---------- 최종 피처 / Threshold 강조 카드 ---------- */
    .feature-highlight-card {{
        background: linear-gradient(135deg, {PRIMARY} 0%, {PRIMARY_DARK} 100%);
        border-radius: 18px;
        padding: 26px 28px;
        margin: 4px 0 4px 0;
        box-shadow: 0 12px 26px rgba(61, 79, 224, 0.25);
    }}
    .fh-columns {{
        display: flex;
        align-items: stretch;
        gap: 28px;
    }}
    .fh-section {{
        display: flex;
        flex-direction: column;
    }}
    .fh-section-features {{ flex: 2 1 340px; }}
    .fh-section-threshold {{ flex: 1 1 160px; }}
    .fh-divider {{
        width: 1px;
        align-self: stretch;
        background: rgba(255,255,255,0.28);
    }}
    .fh-section-title {{
        display: flex;
        align-items: center;
        gap: 7px;
        color: rgba(255,255,255,0.8);
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: 10px;
    }}
    .fh-value {{
        color: #FFFFFF;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.01em;
        margin-bottom: 14px;
    }}
    .fh-section-threshold .fh-value {{ margin-bottom: 6px; }}
    .fh-threshold-desc {{
        color: rgba(255,255,255,0.78);
        font-size: 0.82rem;
        line-height: 1.55;
    }}
    .feature-badge-row {{
        display: flex;
        gap: 10px;
        flex-wrap: wrap;
    }}
    .feature-badge {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(255,255,255,0.16);
        color: #FFFFFF;
        font-weight: 700;
        font-size: 0.88rem;
        padding: 7px 15px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,0.32);
    }}
    .feature-badge .fb-num {{
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 18px; height: 18px;
        border-radius: 50%;
        background: #FFFFFF;
        color: {PRIMARY_DARK};
        font-size: 0.72rem;
        font-weight: 800;
    }}

    /* ---------- 사이드바 ---------- */
    section[data-testid="stSidebar"] {{
        background: {INK};
    }}
    section[data-testid="stSidebar"] * {{
        color: #E7E8F3 !important;
    }}
    .sb-brand {{
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 6px 2px 18px 2px;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        margin-bottom: 18px;
    }}
    .sb-brand-icon {{
        width: 40px; height: 40px;
        border-radius: 11px;
        background: linear-gradient(135deg, {PRIMARY}, #6B78F5);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.25rem;
        flex-shrink: 0;
    }}
    .sb-brand-title {{
        font-size: 0.98rem;
        font-weight: 800;
        color: #FFFFFF !important;
        line-height: 1.3;
    }}
    .sb-brand-sub {{
        font-size: 0.74rem;
        color: #9CA1C4 !important;
    }}
    .sb-foot {{
        margin-top: 18px;
        padding: 14px 14px;
        border-radius: 12px;
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.10);
        font-size: 0.78rem;
        line-height: 1.7;
        color: #C4C8E2 !important;
    }}
    section[data-testid="stSidebar"] div[data-baseweb="radio"] label {{
        padding: 9px 12px;
        border-radius: 10px;
        margin-bottom: 2px;
        transition: background 0.15s ease;
    }}
    section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {{
        background: rgba(255,255,255,0.07);
    }}

    /* ---------- 버튼 ---------- */
    .stButton > button[kind="primary"] {{
        background-color: {PRIMARY};
        border-color: {PRIMARY};
        color: white;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1rem;
    }}
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span,
    .stButton > button[kind="primary"] div {{
        color: white !important;
    }}
    .stButton > button[kind="primary"]:hover {{
        background-color: {PRIMARY_DARK};
        border-color: {PRIMARY_DARK};
        color: white;
    }}
    .stButton > button[kind="primary"]:hover p,
    .stButton > button[kind="primary"]:hover span,
    .stButton > button[kind="primary"]:hover div {{
        color: white !important;
    }}
    .stDownloadButton > button {{
        border-radius: 10px;
        border-color: {BORDER};
        color: {INK};
        font-weight: 600;
    }}
    .stDownloadButton > button:hover {{
        border-color: {PRIMARY};
        color: {PRIMARY};
    }}

    /* ---------- 탭 ---------- */
    button[data-baseweb="tab"] {{
        font-weight: 600;
        color: {MUTED};
    }}
    button[data-baseweb="tab"][aria-selected="true"] {{
        color: {PRIMARY} !important;
        border-bottom-color: {PRIMARY} !important;
        font-weight: 800;
    }}
    button[data-baseweb="tab"]:hover {{
        color: {PRIMARY_DARK} !important;
    }}

    /* ---------- 슬라이더 ---------- */
    /* 스트림릿 기본 빨간색(rgb 255,75,75 계열) 인라인 스타일을 브랜드 블루로 치환 */
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(255, 75, 75"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgb(255,75,75"] {{
        background-color: {PRIMARY} !important;
        border-color: {PRIMARY} !important;
    }}
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgba(255, 75, 75"],
    div[data-testid="stSlider"] [data-baseweb="slider"] [style*="rgba(255,75,75"] {{
        box-shadow: 0 0 0 0.2rem rgba(61, 79, 224, 0.18) !important;
    }}
    /* 트랙 레일(연회색) */
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div {{
        background-color: {BORDER} !important;
    }}
    /* 채워진 구간(브랜드 블루) */
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div > div {{
        background-color: {PRIMARY} !important;
    }}
    /* 손잡이(thumb) */
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {{
        background-color: {PRIMARY} !important;
        border: 3px solid #FFFFFF !important;
        box-shadow: 0 0 0 1.5px {PRIMARY}, 0 2px 6px rgba(61, 79, 224, 0.35) !important;
    }}
    /* 현재 값 라벨(손잡이 위 숫자) */
    div[data-testid="stThumbValue"] {{
        color: {INK} !important;
        font-weight: 700 !important;
    }}
    /* 좌우 최소/최대 눈금 텍스트 */
    div[data-testid="stTickBarMin"], div[data-testid="stTickBarMax"] {{
        color: {MUTED} !important;
    }}

    /* ---------- 데이터프레임 / 구분선 ---------- */
    div[data-testid="stDataFrame"] {{
        border: 1px solid {BORDER};
        border-radius: 12px;
        overflow: hidden;
    }}
    hr {{ border-color: {BORDER} !important; }}



    /* ---------- Bright mode fixes for Streamlit Cloud ---------- */
    /* st.code() block */
    div[data-testid="stCodeBlock"],
    div[data-testid="stCodeBlock"] pre,
    div[data-testid="stCodeBlock"] code {{
        background: #FFFFFF !important;
        color: #181B2A !important;
        border-color: #E4E7F1 !important;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    div[data-testid="stCodeBlock"] {{
        border: 1px solid #E4E7F1 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }}
    div[data-testid="stCodeBlock"] pre {{
        padding: 16px 18px !important;
    }}

    /* Download button */
    div[data-testid="stDownloadButton"] button {{
        background: #FFFFFF !important;
        color: #181B2A !important;
        border: 1px solid #D9DDE8 !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
    }}
    div[data-testid="stDownloadButton"] button * {{
        color: #181B2A !important;
    }}
    div[data-testid="stDownloadButton"] button:hover {{
        border-color: #3D4FE0 !important;
        color: #3D4FE0 !important;
    }}
    div[data-testid="stDownloadButton"] button:hover * {{
        color: #3D4FE0 !important;
    }}

    /* File uploader */
    [data-testid="stFileUploader"] section {{
        background: #FFFFFF !important;
        border: 2px dashed #D9DDE8 !important;
        border-radius: 14px !important;
    }}
    [data-testid="stFileUploader"] section * {{
        color: #3F4356 !important;
    }}
    [data-testid="stFileUploader"] button {{
        background: #FFFFFF !important;
        color: #374151 !important;
        border: 1px solid #D9DDE8 !important;
        border-radius: 10px !important;
    }}
    [data-testid="stFileUploader"] button * {{
        color: #374151 !important;
    }}

    /* Data editor / dataframe bright override */
    [data-testid="stDataEditor"],
    [data-testid="stDataEditor"] div,
    [data-testid="stDataEditor"] table,
    [data-testid="stDataEditor"] thead,
    [data-testid="stDataEditor"] tbody,
    [data-testid="stDataEditor"] tr,
    [data-testid="stDataEditor"] th,
    [data-testid="stDataEditor"] td {{
        background-color: #FFFFFF !important;
        color: #181B2A !important;
    }}
    [data-testid="stDataEditor"] th,
    [data-testid="stDataEditor"] [role="columnheader"] {{
        background-color: #F6F7FB !important;
        color: #181B2A !important;
        font-weight: 800 !important;
    }}
    [data-testid="stDataEditor"] input,
    [data-testid="stDataEditor"] textarea {{
        background-color: #FFFFFF !important;
        color: #181B2A !important;
    }}
    [data-testid="stDataEditor"] [role="gridcell"],
    [data-testid="stDataEditor"] [role="columnheader"],
    [data-testid="stDataEditor"] [role="row"] {{
        background-color: #FFFFFF !important;
        color: #181B2A !important;
        border-color: #E4E7F1 !important;
    }}
    [data-testid="stDataEditor"] [role="columnheader"] {{
        background-color: #F6F7FB !important;
    }}

    /* ---------- 알림 박스 ---------- */
    div[data-testid="stAlert"] {{
        border-radius: 12px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# Streamlit Cloud의 dataframe/data_editor는 canvas 기반이라 CSS가 잘 먹지 않아,
# 표시용 표와 수기 입력 영역은 아래 custom light UI로 대체합니다.
st.markdown(
    """
    <style>
    .light-table-wrap {
        background: #FFFFFF;
        border: 1px solid #E4E7F1;
        border-radius: 14px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(24, 27, 42, 0.04);
        margin: 8px 0 18px 0;
    }
    table.light-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #181B2A;
        background: #FFFFFF;
        font-size: 0.92rem;
    }
    table.light-table thead th {
        background: #F6F7FB;
        color: #3F4356;
        font-weight: 800;
        text-align: left;
        padding: 12px 14px;
        border-bottom: 1px solid #E4E7F1;
        white-space: nowrap;
    }
    table.light-table tbody td {
        background: #FFFFFF;
        color: #181B2A;
        padding: 12px 14px;
        border-bottom: 1px solid #EEF0F6;
        white-space: nowrap;
    }
    table.light-table tbody tr:last-child td { border-bottom: none; }
    table.light-table tbody tr:hover td { background: #F9FAFE; }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="input"] input {
        background-color: #FFFFFF !important;
        color: #181B2A !important;
        border-color: #D9DDE8 !important;
    }
    div[data-testid="stTextInput"] > div,
    div[data-testid="stNumberInput"] > div,
    div[data-baseweb="input"] {
        background-color: #FFFFFF !important;
        color: #181B2A !important;
    }
    div[data-testid="stTextInput"] input::placeholder {
        color: #9CA3AF !important;
    }
    .manual-table-head {
        display: grid;
        grid-template-columns: 1.2fr 1fr 1fr 1.5fr;
        gap: 12px;
        margin-top: 14px;
        margin-bottom: 6px;
        color: #3F4356;
        font-weight: 800;
        font-size: 0.92rem;
    }
    .manual-row-sep {
        height: 1px;
        background: #EEF0F6;
        margin: 4px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def artifact_path(filename: str) -> Path:
    """앱 파일과 같은 폴더를 먼저 찾고, 없으면 한 단계 상위 폴더를 찾는다."""
    same_dir = APP_DIR / filename
    parent_dir = APP_DIR.parent / filename

    if same_dir.exists():
        return same_dir
    if parent_dir.exists():
        return parent_dir

    raise FileNotFoundError(
        f"{filename} 파일을 찾지 못했습니다. app.py와 같은 폴더 또는 상위 폴더에 배치하세요."
    )


def section_label(icon: str, title: str, caption: str = ""):
    st.markdown(
        f"""<div class="section-label"><span class="dot"></span><span>{icon} {title}</span></div>""",
        unsafe_allow_html=True,
    )
    if caption:
        st.markdown(f"""<div class="section-caption">{caption}</div>""", unsafe_allow_html=True)


def render_light_table(table_df: pd.DataFrame, max_rows: int | None = None, index: bool = False):
    """Streamlit 기본 dataframe이 다크모드로 보이는 문제를 피하기 위한 HTML 표 렌더러."""
    display_df = table_df.copy()
    if max_rows is not None:
        display_df = display_df.head(max_rows)
    html_table = display_df.to_html(
        index=index,
        escape=True,
        classes="light-table",
        border=0,
    )
    st.markdown(f'<div class="light-table-wrap">{html_table}</div>', unsafe_allow_html=True)


def format_percent_table(table_df: pd.DataFrame, percent_cols: list[str]) -> pd.DataFrame:
    """지정 컬럼을 보기 좋은 퍼센트 문자열로 변환."""
    formatted = table_df.copy()
    for col in percent_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(
                lambda x: f"{float(x):.1f}%" if pd.notna(x) and float(x) > 1 else f"{float(x) * 100:.1f}%" if pd.notna(x) else ""
            )
    return formatted


def parse_int_input(value, default: int, min_value: int, max_value: int) -> int:
    """text_input으로 받은 숫자를 안전하게 정수로 변환하고 범위를 제한."""
    try:
        parsed = int(float(str(value).strip()))
    except Exception:
        parsed = default
    return int(max(min_value, min(max_value, parsed)))


# =========================
# 데이터 / 모델 로드
# =========================

@st.cache_data
def load_data():
    return joblib.load(artifact_path("feature_data.pkl"))


@st.cache_resource
def load_model():
    model = joblib.load(artifact_path("xgb_payment_model.pkl"))
    threshold = joblib.load(artifact_path("best_threshold.pkl"))
    return model, float(threshold)


df = load_data()
final_model, best_threshold = load_model()

BEST_FEATURES = ["방문일수", "총출입횟수", "신청후첫방문일수"]
VISIT_DAY_VALUES = [0, 1, 2, 3]
DAYS_TO_FIRST_VALUES = [-1, 0, 1, 2]
TOTAL_ACCESS_MAX = 50

# 모델이 실제로 요구하는 피처와 화면상 BEST_FEATURES가 다르면 모델 기준으로 보정
if hasattr(final_model, "feature_names_in_"):
    MODEL_FEATURES = list(final_model.feature_names_in_)
else:
    MODEL_FEATURES = BEST_FEATURES

if MODEL_FEATURES != BEST_FEATURES:
    BEST_FEATURES = MODEL_FEATURES

ID_CANDIDATES = ["user_uuid", "고객ID", "customer_id", "id", "ID"]


# =========================
# 공통 함수
# =========================

def prepare_model_input(input_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """예측용 입력 데이터를 모델 피처 순서에 맞추고 결측/논리 규칙을 적용한다."""
    working_df = input_df.copy()

    missing_columns = [col for col in BEST_FEATURES if col not in working_df.columns]
    if missing_columns:
        raise ValueError(f"필수 컬럼이 없습니다: {missing_columns}")

    raw_feature_df = working_df[BEST_FEATURES].copy()
    model_input = raw_feature_df.apply(pd.to_numeric, errors="coerce")

    # 문자값 등이 숫자로 변환되지 않아 생긴 결측까지 포함해서 기록
    missing_mask = model_input.isna()
    missing_detail = missing_mask.apply(
        lambda row: ", ".join(row.index[row].tolist()) if row.any() else "없음",
        axis=1,
    )
    missing_count = missing_mask.sum(axis=1)

    # -------------------------
    # 결측치 처리 규칙
    # -------------------------
    # 1) 방문일수 결측: 방문일수=0, 총출입횟수=0, 신청후첫방문일수=-1
    visit_missing = model_input["방문일수"].isna()
    model_input.loc[visit_missing, ["방문일수", "총출입횟수", "신청후첫방문일수"]] = [0, 0, -1]

    # 2) 총출입횟수 결측: 총출입횟수만 2로 대체
    access_missing = model_input["총출입횟수"].isna() & ~visit_missing
    model_input.loc[access_missing, "총출입횟수"] = 2

    # 3) 신청후첫방문일수 결측: 신청후첫방문일수만 0으로 대체
    days_missing = model_input["신청후첫방문일수"].isna() & ~visit_missing
    model_input.loc[days_missing, "신청후첫방문일수"] = 0

    # -------------------------
    # 입력값 범위 제한
    # -------------------------
    model_input["방문일수"] = model_input["방문일수"].round().clip(0, 3)
    model_input["총출입횟수"] = model_input["총출입횟수"].round().clip(0, TOTAL_ACCESS_MAX)
    model_input["신청후첫방문일수"] = model_input["신청후첫방문일수"].round().clip(-1, 2)

    # -------------------------
    # 논리 규칙 보정
    # -------------------------
    # 신청후첫방문일수가 -1이면 미방문자로 보고 방문일수/총출입횟수를 0으로 보정
    # 방문일수가 0이면 미방문자로 보고 총출입횟수=0, 신청후첫방문일수=-1로 보정
    no_visit_mask = (model_input["신청후첫방문일수"] == -1) | (model_input["방문일수"] == 0)
    model_input.loc[no_visit_mask, ["방문일수", "총출입횟수", "신청후첫방문일수"]] = [0, 0, -1]

    model_input = model_input[BEST_FEATURES].astype(int)

    preprocess_notes = []
    for idx in model_input.index:
        notes = []
        if missing_count.loc[idx] > 0:
            notes.append("결측치 대체")
        if no_visit_mask.loc[idx]:
            notes.append("미방문 규칙 적용")
        preprocess_notes.append(", ".join(notes) if notes else "없음")

    missing_info = pd.DataFrame({
        "결측_피처수": missing_count,
        "결측_피처목록": missing_detail,
        "전처리_메모": preprocess_notes,
    })

    return model_input, missing_info

def predict_customers(input_df: pd.DataFrame) -> pd.DataFrame:
    """여러 고객의 결제 전환 확률을 예측한다."""
    model_input, missing_info = prepare_model_input(input_df)

    probabilities = final_model.predict_proba(model_input)[:, 1]
    predictions = (probabilities >= best_threshold).astype(int)

    result_df = input_df.copy()

    id_cols = [col for col in ID_CANDIDATES if col in result_df.columns]
    base_cols = id_cols + BEST_FEATURES
    base_cols = [col for col in base_cols if col in result_df.columns]

    result_df = result_df[base_cols].copy()

    # 결과표에는 실제 예측에 사용된 보정 후 값을 표시
    for feature in BEST_FEATURES:
        result_df[feature] = model_input[feature].values

    result_df["결제전환확률"] = probabilities
    result_df["결제전환확률(%)"] = probabilities * 100
    result_df["예측결과"] = np.where(predictions == 1, "결제 예상", "미결제 예상")
    result_df["적용_threshold"] = best_threshold
    result_df["결측_피처수"] = missing_info["결측_피처수"].values
    result_df["결측_피처목록"] = missing_info["결측_피처목록"].values
    result_df["전처리_메모"] = missing_info["전처리_메모"].values
    result_df["입력데이터_상태"] = np.where(
        result_df["전처리_메모"] == "없음",
        "정상",
        "결측/논리 규칙 보정 적용",
    )

    return result_df.sort_values("결제전환확률", ascending=False).reset_index(drop=True)


def read_uploaded_csv(uploaded_file) -> pd.DataFrame:
    """UTF-8 / CP949 CSV를 최대한 안전하게 읽는다."""
    try:
        return pd.read_csv(uploaded_file)
    except UnicodeDecodeError:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="cp949")


def display_prediction_summary(result_df: pd.DataFrame):
    """예측 결과 요약 카드."""
    total_count = len(result_df)
    positive_count = int((result_df["예측결과"] == "결제 예상").sum())
    avg_proba = result_df["결제전환확률"].mean()
    missing_rows = int((result_df["결측_피처수"] > 0).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("예측 고객 수", f"{total_count:,}명")
    c2.metric("결제 예상 고객", f"{positive_count:,}명")
    c3.metric("평균 결제 확률", f"{avg_proba * 100:.1f}%")
    c4.metric("결측 포함 행", f"{missing_rows:,}건")


# =========================
# 사이드바
# =========================

st.sidebar.markdown(
    """
    <div class="sb-brand">
        <div class="sb-brand-icon">💼</div>
        <div>
            <div class="sb-brand-title">체험 전환 예측</div>
            <div class="sb-brand-sub">공유오피스 3일 체험 분석</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "발표 섹션",
    ["📋 데이터 소개", "📊 결과", "🔮 예측 데모"],
    label_visibility="collapsed",
)

st.sidebar.markdown(
    f"""
    <div class="sb-foot">
        <b style="color:#FFFFFF;">최종 피처</b><br>
        {', '.join(BEST_FEATURES)}<br><br>
        <b style="color:#FFFFFF;">Threshold</b><br>
        {best_threshold:.2f}
    </div>
    """,
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════
# PAGE 1 — 데이터 소개
# ══════════════════════════════════════════
if page == "📋 데이터 소개":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">DATA OVERVIEW</div>
            <p class="hero-title">📋 3일 체험, 그다음은 결제로 이어질까요?</p>
            <p class="hero-sub">
                공유오피스 3일 무료 체험을 신청한 고객의 방문 · 출입 · 결제 데이터를 분석해
                <b style="color:#FFFFFF;">유료 전환을 예측하는 모델</b>을 구축했습니다.
                아래는 전체 데이터의 기본 현황과 결제자·미결제자의 행동 차이입니다.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("전체 신청자", f"{len(df):,}명")
    col2.metric("결제자", f"{int(df['is_payment'].sum()):,}명")
    col3.metric("미결제자", f"{int((df['is_payment'] == 0).sum()):,}명")
    col4.metric("결제율", f"{df['is_payment'].mean() * 100:.1f}%")

    st.write("")

    total_n = len(df)
    paid_n = int(df["is_payment"].sum())
    unpaid_n = total_n - paid_n
    paid_pct = paid_n / total_n * 100
    unpaid_pct = 100 - paid_pct

    st.markdown(
        f"""
        <div class="conv-summary-card">
            <p class="conv-summary-title">결제 전환 요약</p>
            <p class="conv-summary-caption">전체 신청자 중 결제자와 미결제자 비중입니다.</p>
            <div class="conv-bar-track">
                <div class="conv-bar-fill" style="width:{paid_pct:.1f}%;"></div>
            </div>
            <div class="conv-bar-labels">
                <span><span class="conv-dot" style="background:{PRIMARY};"></span>결제 {paid_pct:.1f}% · {paid_n:,}명</span>
                <span><span class="conv-dot" style="background:{NEUTRAL};"></span>미결제 {unpaid_pct:.1f}% · {unpaid_n:,}명</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")
    section_label("🔍", "결제자 vs 미결제자 행동 비교", "Delta 값은 결제자 평균 − 미결제자 평균 기준입니다.")

    pay_df = df[df["is_payment"] == 1]
    non_df = df[df["is_payment"] == 0]

    c1, c2, c3 = st.columns(3)
    c1.metric(
        "평균 방문일수",
        f"{pay_df['방문일수'].mean():.1f}일",
        f"{pay_df['방문일수'].mean() - non_df['방문일수'].mean():+.1f}일",
    )
    c2.metric(
        "평균 총출입횟수",
        f"{pay_df['총출입횟수'].mean():.1f}회",
        f"{pay_df['총출입횟수'].mean() - non_df['총출입횟수'].mean():+.1f}회",
    )
    c3.metric(
        "첫 방문까지 소요",
        f"{pay_df['신청후첫방문일수'].mean():.1f}일",
        f"{pay_df['신청후첫방문일수'].mean() - non_df['신청후첫방문일수'].mean():+.1f}일",
    )

    st.write("")
    st.write("")

    section_label("📶", "방문일수별 결제율")
    visit_pay = df.groupby("방문일수")["is_payment"].agg(["sum", "count"])
    visit_pay["결제율"] = visit_pay["sum"] / visit_pay["count"] * 100

    fig2, ax2 = plt.subplots(figsize=(9.4, 4.0))
    colors = [PRIMARY for _ in visit_pay["결제율"]]
    bars = ax2.bar(visit_pay.index.astype(str), visit_pay["결제율"], color=colors, width=0.5,
                    zorder=3)
    for bar, val in zip(bars, visit_pay["결제율"]):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            val + 2.5,
            f"{val:.1f}%",
            ha="center",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
    ax2.set_xlabel("방문일수", fontsize=10.5, color=MUTED)
    ax2.set_ylabel("결제율 (%)", fontsize=10.5, color=MUTED)
    ax2.set_ylim(0, 100)
    style_axes(ax2)
    finalize_fig(fig2)
    st.pyplot(fig2)
    plt.close(fig2)


# ══════════════════════════════════════════
# PAGE 2 — 결과
# ══════════════════════════════════════════
elif page == "📊 결과":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">MODEL RESULT</div>
            <p class="hero-title">📊 최종 모델 결과</p>
            <p class="hero-sub">XGBoost 기반 결제 전환 예측 모델의 성능과 피처 선택 과정을 정리했습니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="blue-box">
        <b>🧭 최종 피처 선택 과정</b><br>
        총 8개의 피처와 1개의 파생피처를 추가했고, 인코딩이 필요한 범주형 데이터는 인코딩까지 진행했습니다.<br>
        이후 전체 피처를 기준으로 XGBoost 모델을 학습해 F1, Recall, Precision 수치를 확인했고,
        Ablation Test를 통해 F1 점수를 낮추는 피처를 제외했습니다.<br>
        마지막으로 남은 피처들의 전체 조합을 확인해 약 511개의 피처 조합 중 최적의 F1 점수를 낸 피처 조합을 최종 피처로 선택했습니다.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    feature_badges_html = "".join(
        f'<span class="feature-badge"><span class="fb-num">{i+1}</span>{feat}</span>'
        for i, feat in enumerate(BEST_FEATURES)
    )
    st.markdown(
        f"""
        <div class="feature-highlight-card">
            <div class="fh-columns">
                <div class="fh-section fh-section-features">
                    <div class="fh-section-title">🧩 최종 피처</div>
                    <div class="fh-value">{len(BEST_FEATURES)}개</div>
                    <div class="feature-badge-row">
                        {feature_badges_html}
                    </div>
                </div>
                <div class="fh-divider"></div>
                <div class="fh-section fh-section-threshold">
                    <div class="fh-section-title">🎚️ Threshold</div>
                    <div class="fh-value">{best_threshold:.2f}</div>
                    <div class="fh-threshold-desc">결제 확률이 이 값 이상이면<br>'결제 예상'으로 분류합니다.</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    X = df[BEST_FEATURES]
    y = df["is_payment"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=0,
        stratify=y,
    )
    proba = final_model.predict_proba(X_test)[:, 1]
    pred = (proba >= best_threshold).astype(int)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.4f}")
    m2.metric("F1", f"{f1_score(y_test, pred, zero_division=0):.4f}")
    m3.metric("재현율", f"{recall_score(y_test, pred):.4f}")
    m4.metric("정밀도", f"{precision_score(y_test, pred, zero_division=0):.4f}")

    st.write("")
    st.write("")

    graph_col_a, graph_col_b = st.columns(2)

    with graph_col_a:
        section_label("🧩", "혼동행렬")
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        from matplotlib.colors import LinearSegmentedColormap
        custom_cmap = LinearSegmentedColormap.from_list("primary_cmap", ["#FFFFFF", PRIMARY_TINT, PRIMARY])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["미결제", "결제"])
        disp.plot(ax=ax, cmap=custom_cmap, colorbar=False, values_format="d")
        ax.set_aspect("auto")
        for text in ax.texts:
            text.set_fontsize(13)
            text.set_fontweight("bold")
        ax.set_xlabel("예측값", fontsize=10.5, color=MUTED)
        ax.set_ylabel("실제값", fontsize=10.5, color=MUTED)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with graph_col_b:
        section_label("🎯", "피처 중요도")
        imp_df = pd.DataFrame({
            "피처": BEST_FEATURES,
            "중요도": final_model.feature_importances_,
        }).sort_values("중요도", ascending=True)

        fig2, ax2 = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
        bar_colors = [PRIMARY if i == len(imp_df) - 1 else "#9CA6F2" for i in range(len(imp_df))]
        ax2.barh(imp_df["피처"], imp_df["중요도"], color=bar_colors, height=0.5, zorder=3)
        for i, (feat, val) in enumerate(zip(imp_df["피처"], imp_df["중요도"])):
            ax2.text(val + imp_df["중요도"].max() * 0.02, i, f"{val:.3f}",
                      va="center", fontsize=9.5, fontweight="bold", color=INK)
        ax2.set_xlabel("중요도", fontsize=10.5, color=MUTED)
        ax2.set_ylabel("")
        style_axes(ax2, grid_axis="x")
        st.pyplot(fig2, use_container_width=True)
        plt.close(fig2)

    st.write("")
    explain_col_a, explain_col_b = st.columns(2)

    with explain_col_a:
        tn, fp, fn, tp = cm.ravel()
        st.markdown(
            f"""
            <div class="insight-box">
            <b>💬 혼동행렬 해석</b><br><br>
            · 실제 결제자 중 맞게 예측: <b>{tp}/{tp + fn}명 ({tp / (tp + fn) * 100:.1f}%)</b><br>
            · 결제자 놓친 수(FN): <b>{fn}명</b><br>
            · 미결제자 잘못 예측(FP): <b>{fp}명</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with explain_col_b:
        st.markdown(
            """
            <div class="insight-box">
            <b>💡 피처 중요도 인사이트</b><br><br>
            · <b>방문일수</b>가 가장 강력한 전환 예측 피처<br>
            · <b>총출입횟수</b>가 높을수록 시설 이용에 적극적<br>
            · <b>신청 후 첫방문 일수</b>가 짧을수록 전환 가능성 높음
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    st.write("")
    section_label("📈", "Threshold별 성능 변화")
    ths = np.arange(0.2, 0.71, 0.05)
    rows = []
    for t in ths:
        p = (proba >= t).astype(int)
        rows.append({
            "Threshold": round(t, 2),
            "재현율": recall_score(y_test, p),
            "정밀도": precision_score(y_test, p, zero_division=0),
            "F1": f1_score(y_test, p, zero_division=0),
        })
    th_df = pd.DataFrame(rows)

    fig3, ax3 = plt.subplots(figsize=(9, 3.6))
    ax3.plot(th_df["Threshold"], th_df["재현율"], marker="o", markersize=5, label="재현율", color=PRIMARY, linewidth=2.2)
    ax3.plot(th_df["Threshold"], th_df["정밀도"], marker="o", markersize=5, label="정밀도", color=ACCENT_TEAL, linewidth=2.2)
    ax3.plot(th_df["Threshold"], th_df["F1"], marker="o", markersize=5, label="F1", color=NEGATIVE, linewidth=2.2, alpha=0.85)
    ax3.axvline(best_threshold, color=ACCENT_AMBER, linestyle="--", linewidth=1.6,
                label=f"선택 Threshold ({best_threshold:.2f})")
    ax3.set_xlabel("Threshold", fontsize=10.5, color=MUTED)
    ax3.set_ylabel("점수", fontsize=10.5, color=MUTED)
    ax3.legend(fontsize=9, frameon=False, loc="lower left")
    style_axes(ax3)
    finalize_fig(fig3)
    st.pyplot(fig3)
    plt.close(fig3)


# ══════════════════════════════════════════
# PAGE 3 — 예측 데모
# ══════════════════════════════════════════
elif page == "🔮 예측 데모":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">LIVE DEMO</div>
            <p class="hero-title">🔮 예측 데모</p>
            <p class="hero-sub">고객의 방문 정보를 입력하거나 CSV 파일을 업로드하면 결제 전환 확률을 예측합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="rule-card">
            <div class="rule-card-header">
                현재 모델은 <b>{', '.join(BEST_FEATURES)}</b> 3개 피처를 기준으로 예측합니다.
            </div>
            <div class="rule-card-sub">결측치가 있을 경우 다음 규칙으로 대체합니다</div>
            <div class="rule-grid">
                <div class="rule-item">
                    <span class="rule-tag">방문일수 결측</span>
                    <div class="rule-detail">방문일수 0<br>총출입횟수 0<br>신청후첫방문일수 -1</div>
                </div>
                <div class="rule-item">
                    <span class="rule-tag">총출입횟수 결측</span>
                    <div class="rule-detail">총출입횟수 2</div>
                </div>
                <div class="rule-item">
                    <span class="rule-tag">신청후첫방문일수 결측</span>
                    <div class="rule-detail">신청후첫방문일수 0</div>
                </div>
            </div>
            <div class="rule-footnote">
                ⚠️ 신청후첫방문일수가 -1이거나 방문일수가 0이면 미방문자로 보고
                방문일수 0 · 총출입횟수 0 · 신청후첫방문일수 -1로 보정합니다.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    demo_tab, csv_tab, manual_tab = st.tabs(["🙋 단일 예측", "📁 CSV 업로드 예측", "✍️ 수기 다중 입력"])

    # -------------------------
    # 1. 단일 예측
    # -------------------------
    with demo_tab:
        st.subheader("단일 고객 예측")

        col1, col2, col3 = st.columns(3)
        with col1:
            visit_days_raw = st.slider("방문일수", 0, 3, 2)
        with col2:
            days_to_first_raw = st.select_slider(
                "신청 후 첫방문까지 걸린 일수",
                options=DAYS_TO_FIRST_VALUES,
                value=0,
                help="-1은 미방문을 의미합니다.",
            )

        force_no_visit = (visit_days_raw == 0) or (days_to_first_raw == -1)

        with col3:
            total_access_raw_text = st.text_input(
                "총 출입횟수",
                value=str(0 if force_no_visit else 4),
                disabled=force_no_visit,
                help=f"0~{TOTAL_ACCESS_MAX} 사이의 정수를 입력하세요.",
            )
            total_access_raw = parse_int_input(total_access_raw_text, default=0 if force_no_visit else 4, min_value=0, max_value=TOTAL_ACCESS_MAX)

        if force_no_visit:
            visit_days = 0
            total_access = 0
            days_to_first = -1
            st.info("방문일수가 0이거나 신청 후 첫방문까지 걸린 일수가 -1이면 미방문자로 처리되어 총 출입횟수는 0으로 적용됩니다.")
        else:
            visit_days = visit_days_raw
            total_access = total_access_raw
            days_to_first = days_to_first_raw

        st.caption(f"예측에 적용되는 값: 방문일수 {visit_days}, 총출입횟수 {total_access}, 신청후첫방문일수 {days_to_first}")

        st.divider()

        if st.button("단일 고객 예측하기", type="primary", use_container_width=True):
            input_df = pd.DataFrame([{
                "방문일수": visit_days,
                "총출입횟수": total_access,
                "신청후첫방문일수": days_to_first,
            }])

            result_df = predict_customers(input_df)
            proba = float(result_df.loc[0, "결제전환확률"])
            pred_label = result_df.loc[0, "예측결과"]

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("결제 확률", f"{proba * 100:.1f}%")
            col_b.metric("예측 결과", pred_label)
            col_c.metric("Threshold", f"{best_threshold:.2f}")

            st.write("")

            fig, ax = plt.subplots(figsize=(8.5, 1.4))
            ax.barh([0], [1], color="#EFF1F7", height=0.42, zorder=2)
            fill_color = PRIMARY if proba >= best_threshold else "#9CA6F2"
            ax.barh([0], [proba], color=fill_color, height=0.42, zorder=3)
            ax.axvline(best_threshold, color=ACCENT_AMBER, linestyle="--", linewidth=1.8,
                       label=f"Threshold ({best_threshold:.2f})", zorder=4)
            ax.text(proba, 0.42, f"{proba*100:.1f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK)
            ax.set_xlim(0, 1)
            ax.set_yticks([])
            ax.set_xlabel("결제 확률", fontsize=10, color=MUTED)
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.legend(loc="upper left", bbox_to_anchor=(0, -0.45), fontsize=8.5, frameon=False, ncol=1)
            fig.patch.set_facecolor("white")
            fig.subplots_adjust(top=0.82, bottom=0.42, left=0.03, right=0.98)
            st.pyplot(fig)
            plt.close(fig)

            if proba >= best_threshold:
                st.markdown(
                    "<div class='result-box result-positive'>✅ <b>결제 가능성이 높은 사용자로 예측되었습니다.</b></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div class='result-box result-negative'>ℹ️ <b>결제 가능성이 낮은 사용자로 예측되었습니다.</b></div>",
                    unsafe_allow_html=True,
                )

            st.divider()
            section_label("📊", "방문일수별 결제 확률 비교", "다른 조건은 고정하고 방문일수만 바꿨을 때")

            sim_rows = []
            for d in range(0, 4):
                sim_total_access = 0 if d == 0 else total_access
                sim_days_to_first = -1 if d == 0 else days_to_first
                row = pd.DataFrame([{
                    "방문일수": d,
                    "총출입횟수": sim_total_access,
                    "신청후첫방문일수": sim_days_to_first,
                }])
                p = float(final_model.predict_proba(row[BEST_FEATURES])[0, 1])
                sim_rows.append({
                    "방문일수": d,
                    "결제확률": p,
                    "예측": "결제 예상" if p >= best_threshold else "미결제 예상",
                })

            sim_df = pd.DataFrame(sim_rows)

            fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))
            colors = [PRIMARY if p >= best_threshold else NEUTRAL for p in sim_df["결제확률"]]
            bars = ax2.bar(sim_df["방문일수"].astype(str), sim_df["결제확률"], color=colors, width=0.55, zorder=3)
            ax2.axhline(best_threshold, color=ACCENT_AMBER, linestyle="--", linewidth=1.4,
                        label=f"Threshold ({best_threshold:.2f})", zorder=2)
            ax2.set_ylim(0, 1)
            ax2.set_xlabel("방문일수", fontsize=10.5, color=MUTED)
            ax2.set_ylabel("결제 확률", fontsize=10.5, color=MUTED)
            ax2.legend(fontsize=8.5, frameon=False)
            for bar, val in zip(bars, sim_df["결제확률"]):
                ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val * 100:.0f}%",
                          ha="center", fontsize=9.5, fontweight="bold", color=INK)
            style_axes(ax2)
            finalize_fig(fig2)
            st.pyplot(fig2)
            plt.close(fig2)

            sim_display_df = sim_df.copy()
            sim_display_df["결제확률"] = sim_display_df["결제확률"].map(lambda x: f"{x:.1%}")
            render_light_table(sim_display_df)

    # -------------------------
    # 2. CSV 업로드 예측
    # -------------------------
    with csv_tab:
        st.subheader("CSV 업로드로 여러 고객 예측")
        st.markdown("CSV 파일에는 아래 3개 컬럼이 필요합니다.")
        required_cols_html = "".join(
            f'<span style="display:inline-flex;align-items:center;gap:6px;background:#EEF0FD;color:#2B39B3;border:1px solid #D7DBFA;border-radius:999px;padding:7px 13px;font-weight:700;font-size:0.9rem;margin-right:8px;margin-bottom:8px;">✓ {col}</span>'
            for col in BEST_FEATURES
        )
        st.markdown(
            f"""
            <div style="
                background:#FFFFFF;
                border:1px solid #E4E7F1;
                border-radius:14px;
                padding:18px 20px 12px 20px;
                margin:10px 0 16px 0;
                box-shadow:0 2px 10px rgba(24, 27, 42, 0.04);
            ">
                <div style="color:#181B2A;font-weight:800;font-size:0.95rem;margin-bottom:12px;">필수 컬럼</div>
                <div>{required_cols_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        sample_df = pd.DataFrame({
            "고객ID": ["고객_1", "고객_2", "고객_3"],
            "방문일수": [2, 1, np.nan],
            "총출입횟수": [4, 2, 0],
            "신청후첫방문일수": [0, 1, -1],
        })

        st.download_button(
            label="샘플 CSV 다운로드",
            data=sample_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="payment_prediction_sample.csv",
            mime="text/csv",
        )

        uploaded_file = st.file_uploader("예측할 고객 CSV 파일 업로드", type=["csv"])

        if uploaded_file is not None:
            try:
                uploaded_df = read_uploaded_csv(uploaded_file)
                st.write("업로드 데이터 미리보기")
                render_light_table(uploaded_df.head(20))

                missing_columns = [col for col in BEST_FEATURES if col not in uploaded_df.columns]
                if missing_columns:
                    st.error(f"필수 컬럼이 없습니다: {missing_columns}")
                else:
                    result_df = predict_customers(uploaded_df)
                    st.divider()
                    display_prediction_summary(result_df)

                    st.write("")
                    st.subheader("고객별 예측 결과")
                    display_cols = [col for col in ["user_uuid", "고객ID", "customer_id", "id", "ID"] if col in result_df.columns]
                    display_cols += BEST_FEATURES + [
                        "결제전환확률(%)",
                        "예측결과",
                        "결측_피처수",
                        "결측_피처목록",
                        "전처리_메모",
                        "입력데이터_상태",
                    ]
                    display_cols = [col for col in display_cols if col in result_df.columns]

                    result_display_df = result_df[display_cols].copy()
                    if "결제전환확률(%)" in result_display_df.columns:
                        result_display_df["결제전환확률(%)"] = result_display_df["결제전환확률(%)"].map(lambda x: f"{x:.1f}%")
                    render_light_table(result_display_df, max_rows=50)

                    st.download_button(
                        label="예측 결과 CSV 다운로드",
                        data=result_df.to_csv(index=False).encode("utf-8-sig"),
                        file_name="payment_prediction_result.csv",
                        mime="text/csv",
                    )
            except Exception as e:
                st.error("CSV 예측 중 오류가 발생했습니다.")
                st.exception(e)

    # -------------------------
    # 3. 수기 다중 입력
    # -------------------------
    with manual_tab:
        st.subheader("수기 입력으로 여러 고객 예측")
        st.caption("행을 추가해서 여러 고객을 한 번에 입력할 수 있습니다. 빈 값은 지정된 결측치 처리 규칙에 따라 대체됩니다.")

        default_manual_df = pd.DataFrame({
            "고객ID": ["고객_1", "고객_2", "고객_3"],
            "방문일수": [2, 1, 0],
            "총출입횟수": [4, 2, 0],
            "신청후첫방문일수": [0, 1, -1],
        })

        row_count = st.slider("입력할 고객 수", min_value=1, max_value=10, value=3, step=1)
        st.markdown(
            """
            <div class="manual-table-head">
                <div>고객ID</div>
                <div>방문일수</div>
                <div>총출입횟수</div>
                <div>신청 후 첫방문까지 걸린 일수</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        manual_rows = []
        for i in range(row_count):
            default_row = default_manual_df.iloc[i].to_dict() if i < len(default_manual_df) else {
                "고객ID": f"고객_{i + 1}",
                "방문일수": 0,
                "총출입횟수": 0,
                "신청후첫방문일수": -1,
            }
            c_id, c_visit, c_access, c_first = st.columns([1.2, 1, 1, 1.5])
            with c_id:
                customer_id = st.text_input(
                    "고객ID",
                    value=str(default_row["고객ID"]),
                    key=f"manual_customer_id_{i}",
                    label_visibility="collapsed",
                )
            with c_visit:
                visit_input = st.text_input(
                    "방문일수",
                    value=str(default_row["방문일수"]),
                    key=f"manual_visit_days_{i}",
                    label_visibility="collapsed",
                )
            with c_access:
                access_input = st.text_input(
                    "총출입횟수",
                    value=str(default_row["총출입횟수"]),
                    key=f"manual_total_access_{i}",
                    label_visibility="collapsed",
                )
            with c_first:
                first_input = st.text_input(
                    "신청 후 첫방문까지 걸린 일수",
                    value=str(default_row["신청후첫방문일수"]),
                    key=f"manual_days_to_first_{i}",
                    label_visibility="collapsed",
                )

            manual_rows.append({
                "고객ID": customer_id if str(customer_id).strip() else f"고객_{i + 1}",
                "방문일수": parse_int_input(visit_input, default=0, min_value=0, max_value=3),
                "총출입횟수": parse_int_input(access_input, default=0, min_value=0, max_value=TOTAL_ACCESS_MAX),
                "신청후첫방문일수": parse_int_input(first_input, default=-1, min_value=-1, max_value=2),
            })
            st.markdown('<div class="manual-row-sep"></div>', unsafe_allow_html=True)

        edited_df = pd.DataFrame(manual_rows)

        if st.button("수기 입력 고객 예측하기", type="primary", use_container_width=True):
            try:
                edited_df = edited_df.dropna(how="all").copy()
                result_df = predict_customers(edited_df)
                display_prediction_summary(result_df)

                st.write("")
                st.subheader("수기 입력 고객 예측 결과")
                display_cols = ["고객ID"] + BEST_FEATURES + [
                    "결제전환확률(%)",
                    "예측결과",
                    "결측_피처수",
                    "결측_피처목록",
                    "전처리_메모",
                    "입력데이터_상태",
                ]
                display_cols = [col for col in display_cols if col in result_df.columns]

                manual_result_display_df = result_df[display_cols].copy()
                if "결제전환확률(%)" in manual_result_display_df.columns:
                    manual_result_display_df["결제전환확률(%)"] = manual_result_display_df["결제전환확률(%)"].map(lambda x: f"{x:.1f}%")
                render_light_table(manual_result_display_df, max_rows=50)

                st.download_button(
                    label="수기 입력 예측 결과 CSV 다운로드",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name="manual_payment_prediction_result.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error("수기 입력 예측 중 오류가 발생했습니다.")
                st.exception(e)
