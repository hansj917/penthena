# =================================================================================
# PENTHENA AI Agent - v19.3 (초기 화면 UI 단순화)
# =================================================================================
# 작성자: Google Gemini/GPT along with Dave Han
# 업데이트 날짜: 2025-06-15
#
# 주요 변경 사항:
# 1. [UI/UX] 메인 화면의 예시 프롬프트 버튼 및 관련 컨테이너 완전 제거
# 2. 이전 버전의 모든 UI/UX 개선 및 버그 수정 사항 포함
# =================================================================================

import streamlit as st
import openai, os, json, time, re
from dotenv import load_dotenv
from datetime import datetime, timedelta
import pytz, requests
import xml.etree.ElementTree as ET
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from tavily import TavilyClient
from concurrent.futures import ThreadPoolExecutor

# --- 1. 기본 설정 및 환경변수 로드 ---
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily = TavilyClient(api_key=tavily_api_key)

def load_css(file_name):
    """지정된 CSS 파일을 읽어 Streamlit 앱에 적용하는 함수"""
    try:
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"경고: '{file_name}' 파일을 찾을 수 없습니다.")

# --- 2. 디자인 시스템 및 상수 정의 ---
CHART_COLOR_PALETTE = ['#6C5FF5', '#889AF5', '#B6BCFA', '#DDE0FD', '#78C4D3', '#A5D8E2']
CHART_BG_COLOR = "rgba(0,0,0,0)"
CHART_FONT_COLOR = "#EAEBF0"
GRID_COLOR = "rgba(255, 255, 255, 0.1)"

# --- 3. 사이드바 UI 컴포넌트 ---
def display_world_clocks():
    """세계 주요 도시의 현재 시간을 표시하는 사이드바 컴포넌트"""
    st.markdown("<h5><span class='icon-dot'></span> 세계 시간</h5>", unsafe_allow_html=True)
    now_korea = datetime.now(pytz.timezone("Asia/Seoul"))
    st.markdown(f"<p class='sidebar-date'>{now_korea.strftime('%Y년 %m월 %d일')}</p>", unsafe_allow_html=True)
    
    timezones = {
        "서울": "Asia/Seoul", 
        "뉴욕": "America/New_York", 
        "도쿄": "Asia/Tokyo", 
        "호치민": "Asia/Ho_Chi_Minh"
    }
    
    for city, tz in timezones.items():
        try:
            now = datetime.now(pytz.timezone(tz))
            st.markdown(f"""<div class="info-item"><span class="info-label">{city}</span><span class="info-value">{now.strftime('%H:%M')}</span></div>""", unsafe_allow_html=True)
        except Exception: 
            st.markdown(f"""<div class="info-item"><span class="info-label">{city}</span><span class="info-value error">로드 실패</span></div>""", unsafe_allow_html=True)

def display_exchange_rates():
    """주요 통화 환율 정보를 표시하는 사이드바 컴포넌트"""
    st.markdown("<h5><span class='icon-dot'></span> 환율 정보</h5>", unsafe_allow_html=True)
    url = "https://www.ecb.europa.eu/stats/eurofxref/eurofxref-daily.xml"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        ns = {'ecb': 'http://www.ecb.int/vocabulary/1.0'}
        
        rates = {c.get('currency'): float(c.get('rate')) for c in root.findall('.//ecb:Cube[@currency]', ns)}
        rates['EUR'] = 1.0
        
        usd_per_eur = rates.get('USD', 1)
        krw_per_eur = rates.get('KRW', 0)
        jpy_per_eur = rates.get('JPY', 1)

        usd_krw_rate = krw_per_eur / usd_per_eur
        jpy_krw_rate = (krw_per_eur / jpy_per_eur) * 100

        st.markdown(f"""<div class="info-item"><span class="info-label">USD (1달러)</span><span class="info-value">{usd_krw_rate:,.2f} 원</span></div>""", unsafe_allow_html=True)
        st.markdown(f"""<div class="info-item"><span class="info-label">JPY (100엔)</span><span class="info-value">{jpy_krw_rate * 100:,.2f} 원</span></div>""", unsafe_allow_html=True)
    except Exception: 
        st.markdown("""<div class="info-item"><span class="info-label">환율 정보</span><span class="info-value error">로드 실패</span></div>""", unsafe_allow_html=True)

# --- 4. 데이터 파싱 및 시각화 컴포넌트 ---

def parse_table_from_text(text: str) -> pd.DataFrame:
    """마크다운 형식의 텍스트에서 테이블을 추출하여 Pandas DataFrame으로 변환"""
    match = re.search(r'^\s*\|.*\|(?:\n|\r\n?)\|.*\|(?:\n|\r\n?)(?:\|.*\|\s*(?:\n|\r\n?))*', text, re.MULTILINE)
    if not match: return pd.DataFrame()
    
    table_str = match.group(0)
    lines = table_str.strip().split('\n')
    header = [h.strip() for h in lines[0].split('|') if h.strip()]
    data = []
    for line in lines[2:]:
        if '|' in line:
            values = [v.strip() for v in line.split('|') if v.strip()]
            if len(values) == len(header): data.append(values)
    return pd.DataFrame(data, columns=header)

def create_empty_chart(message: str) -> go.Figure:
    """데이터 분석 실패 시 표시할 빈 차트를 생성"""
    fig = go.Figure()
    fig.add_annotation(text=message, align='center', showarrow=False, font=dict(color="orange", size=14))
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False), paper_bgcolor=CHART_BG_COLOR, plot_bgcolor="rgba(255, 255, 255, 0.03)")
    return fig

def extract_numeric(series: pd.Series) -> pd.Series:
    """문자열이 포함된 Series에서 숫자만 추출하여 숫자형 Series로 변환"""
    return series.astype(str).str.extract(r'(\d+\.?\d*)').iloc[:, 0].astype(float)

def create_market_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.empty or df.shape[1] < 2: return create_empty_chart("AI가 유효한 시장 규모 데이터를<br>생성하지 못했습니다.")
    try:
        label_col, value_col = df.columns[0], df.columns[1]
        def convert_korean_to_numeric(value_str):
            try:
                value_str = str(value_str).replace(',', '').strip()
                num_part_match = re.search(r'[\d\.]+', value_str)
                if not num_part_match: return 0
                num = float(num_part_match.group(0))
                multipliers = {'조': 1e12, '억': 1e8, '만': 1e4}
                for unit, multiplier in multipliers.items():
                    if unit in value_str: num *= multiplier; break
                return num
            except (ValueError, IndexError): return 0
        df[value_col] = df[value_col].apply(convert_korean_to_numeric)
        if 'plan_data' not in st.session_state: st.session_state.plan_data = {}
        st.session_state.plan_data['market_analysis'] = df.to_dict('records')
        fig = go.Figure(data=[go.Pie(labels=df[label_col], values=df[value_col], hole=.6, marker_colors=CHART_COLOR_PALETTE, textinfo='label+percent', textfont_size=14, hoverinfo='label+value+percent')])
        fig.update_layout(showlegend=False, font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, margin=dict(t=10, b=10, l=10, r=10), annotations=[dict(text='시장', x=0.5, y=0.5, font_size=20, showarrow=False, font_color=CHART_FONT_COLOR)])
        return fig
    except Exception as e: return create_empty_chart(f"시장 차트 렌더링 오류: {e}")

def create_forrester_wave_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.shape[1] < 4:
        return create_empty_chart("AI가 유효한 경쟁사 데이터를<br>생성하지 못했습니다.")
    try:
        df.columns = ['Competitor', 'Strategy', 'Current Offering', 'Market Presence']
        df = df[~df['Competitor'].str.contains('Competitor [A-Z]', case=False, na=False)]
        df = df[~df['Competitor'].str.contains('경쟁사 [A-Z]', case=False, na=False)]
        if df.empty or '분석' in df['Competitor'].iloc[0]: return create_empty_chart("AI가 실제 경쟁사 이름 대신<br>Placeholder를 생성했습니다.")

        df['Strategy'] = extract_numeric(df['Strategy'])
        df['Current Offering'] = extract_numeric(df['Current Offering'])
        df['Market Presence'] = extract_numeric(df['Market Presence'])

        df.dropna(subset=['Strategy', 'Current Offering'], inplace=True)
        df['Market Presence'].fillna(5, inplace=True)

        if df.empty:
            return create_empty_chart("AI가 생성한 데이터에 유효한<br>숫자 점수가 없어 차트를 그릴 수 없습니다.")
            
        x_mid, y_mid = 5.0, 5.0
        fig = px.scatter(df, x="Strategy", y="Current Offering", size="Market Presence", color="Competitor", hover_name="Competitor", hover_data={'Strategy': ':.1f', 'Current Offering': ':.1f', 'Market Presence': ':.1f'}, size_max=50, text="Competitor", color_discrete_sequence=px.colors.qualitative.Vivid)
        fig.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='DarkSlateGrey'), opacity=0.7))
        fig.update_layout(xaxis_title="전략 (Strategy)", yaxis_title="현재 오퍼링 (Current Offering)", font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, showlegend=False, xaxis=dict(range=[0, 10.5], showgrid=True, gridcolor=GRID_COLOR, zeroline=False), yaxis=dict(range=[0, 10.5], showgrid=True, gridcolor=GRID_COLOR, zeroline=False), margin=dict(t=20, b=20, l=20, r=20))
        fig.add_shape(type="rect", x0=x_mid, y0=y_mid, x1=10.5, y1=10.5, fillcolor="rgba(108, 95, 245, 0.1)", layer="below", line_width=0)
        fig.add_vline(x=x_mid, line_width=1, line_dash="dash", line_color="grey")
        fig.add_hline(y=y_mid, line_width=1, line_dash="dash", line_color="grey")
        fig.add_annotation(x=9.8, y=9.8, text="Leaders", showarrow=False, font=dict(color="#6C5FF5", size=14), xanchor='right', yanchor='top')
        fig.add_annotation(x=0.2, y=9.8, text="Strong Performers", showarrow=False, font=dict(color="grey"), xanchor='left', yanchor='top')
        fig.add_annotation(x=0.2, y=0.2, text="Contenders", showarrow=False, font=dict(color="grey"), xanchor='left', yanchor='bottom')
        fig.add_annotation(x=9.8, y=0.2, text="Challengers", showarrow=False, font=dict(color="grey"), xanchor='right', yanchor='bottom')
        return fig
    except Exception as e: 
        return create_empty_chart(f"Forrester Wave 차트 렌더링 오류: {e}")

def display_persona_cards(text: str):
    df = parse_table_from_text(text)
    if df.empty or df.shape[1] < 5: 
        st.warning("페르소나 데이터를 분석할 수 없습니다."); st.text(text); return
    try:
        df.columns = ['Persona', 'Age', 'Occupation', 'Goal', 'Pain_Point']
        if 'plan_data' not in st.session_state: st.session_state.plan_data = {}
        st.session_state.plan_data['personas'] = df.to_dict('records')
        cols = st.columns(min(len(df), 3))
        for i, row in df.iterrows():
            with cols[i % min(len(df), 3)]:
                st.markdown(f"""<div class="persona-card"><div class="persona-icon"></div><h4>{row['Persona']}</h4><p class="persona-info">{row['Occupation']} • {row['Age']}세</p><hr class="persona-divider"><h5>핵심 목표</h5><p>{row['Goal']}</p><h5>가장 큰 고충</h5><p>{row['Pain_Point']}</p></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"페르소나 카드 생성 오류: {e}"); st.markdown(text)

def create_priority_chart(text: str, topic: str) -> go.Figure:
    df = parse_table_from_text(text)
    required_cols = ['Reach', 'Impact', 'Confidence', 'Effort']
    header_map = {'기능': 'Feature', 'feature': 'Feature', '도달': 'Reach', 'reach': 'Reach', '영향': 'Impact', 'impact': 'Impact', '확신': 'Confidence', 'confidence': 'Confidence', '노력': 'Effort', 'effort': 'Effort'}
    
    if not df.empty:
        df.rename(columns=lambda c: header_map.get(c.strip().lower(), c.strip()), inplace=True)

    if df.empty or not all(col in df.columns for col in required_cols):
        with st.spinner("AI 출력 형식 오류 감지. 수정을 위해 재요청합니다..."):
            correction_prompt = f"이전 답변의 마크다운 테이블에 RICE Score 필수 컬럼이 누락되었습니다. 주제 '{topic}'에 대해, 컬럼명이 반드시 **'기능', 'Reach', 'Impact', 'Confidence', 'Effort'**인 마크다운 테이블을 다시 생성해 주십시오."
            corrected_text = stream_and_display_step(correction_prompt)
            df = parse_table_from_text(corrected_text)
            if df.empty: return create_empty_chart("AI의 답변 형식을 수정하는데 실패했습니다.")
            df.rename(columns=lambda c: header_map.get(c.strip().lower(), c.strip()), inplace=True)
            if not all(col in df.columns for col in required_cols):
                return create_empty_chart("AI가 필수 컬럼을 생성하지 못했습니다.")
    try:
        feature_col = 'Feature'
        if feature_col not in df.columns:
            potential_feature_col = df.columns[0]
            if potential_feature_col not in required_cols:
                df.rename(columns={potential_feature_col: 'Feature'}, inplace=True); feature_col = 'Feature'
            else: return create_empty_chart("기능(Feature)에 해당하는 컬럼을<br>찾을 수 없습니다.")
        
        df[required_cols] = df[required_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
        df['Score'] = (df['Reach'] * df['Impact'] * df['Confidence']) / df['Effort'].replace(0, 1)
        if 'plan_data' not in st.session_state: st.session_state.plan_data = {}
        st.session_state.plan_data['key_features'] = df.sort_values(by='Score', ascending=False).to_dict('records')
        
        fig = px.scatter(df, x="Effort", y="Impact", size="Score", color=feature_col, hover_name=feature_col, size_max=60, color_discrete_sequence=CHART_COLOR_PALETTE)
        fig.update_layout(title_text='기능 우선순위 (RICE Score)', title_x=0.5, xaxis_title="Effort (노력)", yaxis_title="Impact (영향력)", font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, legend_title_text='핵심 기능', margin=dict(t=40, b=40, l=40, r=40))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor=GRID_COLOR)
        return fig
    except Exception as e: return create_empty_chart(f"우선순위 차트 렌더링 오류: {e}")

def create_roadmap_gantt_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.empty: return create_empty_chart("AI가 유효한 로드맵 데이터를<br>생성하지 못했습니다.")
    try:
        task_col_name, start_col_name, finish_col_name = df.columns[0], df.columns[1], df.columns[2]
        df[start_col_name] = pd.to_datetime(df[start_col_name], errors='coerce')
        df[finish_col_name] = pd.to_datetime(df[finish_col_name], errors='coerce')
        df.dropna(subset=[start_col_name, finish_col_name], inplace=True)
        if df.empty: return create_empty_chart("유효한 날짜 데이터가 없습니다.")
        resource_col_name = df.columns[3] if len(df.columns) > 3 else None
        fig = px.timeline(df, x_start=start_col_name, x_end=finish_col_name, y=task_col_name, color=resource_col_name, color_discrete_sequence=CHART_COLOR_PALETTE)
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(title_text='실행 로드맵', title_x=0.5, font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, legend_title_text='담당', margin=dict(t=40, b=40, l=40, r=40))
        return fig
    except Exception as e: return create_empty_chart(f"로드맵 차트 렌더링 오류: {e}")

def create_kpi_bar_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.empty or len(df.columns) < 2: return create_empty_chart("AI가 유효한 KPI 데이터를<br>생성하지 못했습니다.")
    try:
        kpi_col, target_col = df.columns[0], df.columns[1]
        df['numeric_target'] = pd.to_numeric(df[target_col].str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        fig = px.bar(df, x=kpi_col, y='numeric_target', text=target_col, color_discrete_sequence=CHART_COLOR_PALETTE)
        fig.update_traces(textposition='outside')
        fig.update_layout(title_text="핵심 목표(KPI)", title_x=0.5, font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, xaxis_title="", yaxis_title="", margin=dict(t=40, b=20, l=20, r=20))
        return fig
    except Exception as e: return create_empty_chart(f"KPI 차트 렌더링 오류: {e}")

def create_pie_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.empty or len(df.columns) < 2: return create_empty_chart("AI가 유효한 비율 데이터를<br>생성하지 못했습니다.")
    try:
        label_col, value_col = df.columns[0], df.columns[1]
        df[value_col] = pd.to_numeric(df[value_col].str.replace(r'[^0-9.]', '', regex=True), errors='coerce')
        df.dropna(inplace=True)
        fig = px.pie(df, names=label_col, values=value_col, hole=0.4, color_discrete_sequence=CHART_COLOR_PALETTE)
        fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.05]*len(df))
        fig.update_layout(showlegend=False, font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, margin=dict(t=10, b=10, l=10, r=10))
        return fig
    except Exception as e: return create_empty_chart(f"파이 차트 렌더링 오류: {e}")

def display_message_and_target_card(text: str):
    try:
        target_pattern = r"(?:타겟|타겟 고객|주 타겟)\s*[:]?\s*(.*?)(?=\n(?:-|\*|#)|메시지|핵심 메시지|슬로건|$)"
        message_pattern = r"(?:메시지|핵심 메시지|슬로건)\s*[:]?\s*\"?(.*?)\"?"
        target_match = re.search(target_pattern, text, re.IGNORECASE | re.DOTALL)
        message_match = re.search(message_pattern, text, re.IGNORECASE | re.DOTALL)
        target = target_match.group(1).strip() if target_match else "타겟 정보 분석 실패"
        message = message_match.group(1).strip() if message_match else "메시지 정보 분석 실패"
        card_html = f"""<div class="message-card-container"><div class="message-card-target"><h6>TARGET AUDIENCE</h6><p>{target}</p></div><div class="message-card-slogan"><h6>CORE MESSAGE</h6><p>"{message}"</p></div></div>"""
        st.markdown(card_html, unsafe_allow_html=True)
    except Exception:
        st.warning("메시지 및 타겟 정보 분석에 실패했습니다. AI 원본 답변을 확인해주세요.")
        st.text(text)

def display_content_strategy_cards(text: str):
    df = parse_table_from_text(text)
    if df.empty:
        st.warning("콘텐츠 전략 데이터를 분석할 수 없습니다."); st.text(text); return
    try:
        df.columns = ['Format', 'Content']
        cols = st.columns(len(df) if len(df) <= 3 else 3)
        for i, row in df.iterrows():
            with cols[i % 3]:
                st.markdown(f"""<div class="content-strategy-card"><div class="content-card-header">{row['Format']}</div><div class="content-card-body">{row['Content']}</div></div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"콘텐츠 전략 카드 생성 오류: {e}"); st.dataframe(df)

def create_sunburst_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.empty or len(df.columns) < 3: return create_empty_chart("AI가 유효한 채널 믹스 데이터를<br>생성하지 못했습니다.")
    try:
        df.columns = ['Category', 'Channel', 'Budget']
        df['Budget'] = pd.to_numeric(df['Budget'].str.replace('%','').str.strip(), errors='coerce')
        df.dropna(inplace=True)
        fig = px.sunburst(df, path=['Category', 'Channel'], values='Budget', color='Category', color_discrete_map={'(?)':'#262730', 'Owned':'#6C5FF5', 'Paid':'#889AF5', 'Earned':'#B6BCFA'})
        fig.update_traces(textinfo='label+percent entry', insidetextorientation='radial')
        fig.update_layout(title_text='채널 믹스 전략 (Sunburst)', title_x=0.5, font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, margin=dict(t=40, b=20, l=20, r=20))
        return fig
    except Exception as e: return create_empty_chart(f"썬버스트 차트 렌더링 오류: {e}")
        
def create_customer_segment_chart(text: str) -> go.Figure:
    df = parse_table_from_text(text)
    if df.empty or len(df.columns) < 3: return create_empty_chart("AI가 유효한 타겟 고객 데이터를<br>생성하지 못했습니다.")
    try:
        df.columns = ['Group', 'Characteristic', 'Reach_Rate']
        df['Reach_Rate'] = pd.to_numeric(df['Reach_Rate'].str.replace('%','').str.strip(), errors='coerce')
        df.dropna(inplace=True)
        df = df.sort_values(by='Reach_Rate', ascending=True)
        fig = px.bar(df, x='Reach_Rate', y='Group', orientation='h', text='Characteristic', color='Reach_Rate', color_continuous_scale='Viridis')
        fig.update_traces(textposition='inside', insidetextanchor='end', textfont_size=12)
        fig.update_layout(xaxis_title="예상 도달률 (%)", yaxis_title="", font_color=CHART_FONT_COLOR, paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, margin=dict(t=20, b=40, l=20, r=20))
        return fig
    except Exception as e: return create_empty_chart(f"타겟 고객 차트 렌더링 오류: {e}")
        
def create_text_display_fig(text: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=text.replace("\n", "<br>"), align='left', showarrow=False, x=0.05, y=0.95, xref="paper", yref="paper", font=dict(color="white", size=14), xanchor='left', yanchor='top')
    fig.update_layout(xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False), paper_bgcolor=CHART_BG_COLOR, plot_bgcolor=CHART_BG_COLOR, height=300, margin=dict(t=20, b=20, l=20, r=20))
    return fig

def stream_and_display_step(prompt: str) -> str:
    full_response = ""
    system_prompt = "You are a world-class business consultant. Respond in Korean. When asked for a table, generate it in a simple, clean Markdown format. You must not add any conversational text or explanations before or after the table. Only the markdown table is allowed."
    try:
        stream = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], stream=True)
        for chunk in stream:
            full_response += (chunk.choices[0].delta.content or "")
        return full_response
    except Exception as e:
        st.error(f"AI 응답 처리 중 오류 발생: {e}"); return ""

def get_deep_dive_analysis(topic: str, step_title: str, ai_response: str) -> str:
    deep_dive_prompt = f"""...""" # 전체 프롬프트 내용
    try:
        response = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": deep_dive_prompt}], temperature=0.7)
        return response.choices[0].message.content
    except Exception as e:
        return f"심층 분석 중 오류가 발생했습니다: {e}"

def perform_web_research_and_synthesis(topic: str) -> str:
    st.markdown("<h3>실시간 웹 리서치</h3>", unsafe_allow_html=True)

    # ─── 서울 타임존 기준 현재 연도를 계산 ─────────────────────────────
    current_year = datetime.now(pytz.timezone("Asia/Seoul")).year

    with st.spinner("리서치 전략 수립 및 검색 쿼리 생성 중..."):
        query_gen_prompt = (
            f'주제 "{topic}"에 대한 심층 분석을 위해, 시장 크기, 경쟁사, 최신 기술, '
            f'타겟 고객 관점의 효과적인 웹 검색 쿼리 4개를 JSON 리스트 형식으로 생성해줘. '
            f'각 쿼리에 반드시 "{current_year}년"을 포함해줘. 다른 설명은 제외해줘.'
        )
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": query_gen_prompt}],
                response_format={"type": "json_object"},
                temperature=0
            )
            search_queries = json.loads(response.choices[0].message.content).get("queries", [])
        except Exception as e:
            st.error(f"검색 쿼리 생성 실패: {e}")
            return ""

    if not search_queries:
        st.warning("분석에 필요한 검색 쿼리를 생성하지 못했습니다.")
        return ""

    with st.container(border=True):
        st.markdown("##### 생성된 검색 쿼리")
        st.json(search_queries)

    # ─── 웹 검색 실행 및 결과 수집 ─────────────────────────────────
    search_results_text = ""
    with st.spinner("4개의 웹 검색을 동시에 실행합니다... (성능 최적화)"):
        def search_task(query):
            try:
                return tavily.search(query=query, search_depth="advanced")
            except Exception:
                return None

        with ThreadPoolExecutor(max_workers=4) as executor:
            for result in executor.map(search_task, search_queries):
                if result:
                    for hit in result.get('results', []):
                        search_results_text += (
                            f"제목: {hit['title']}\n"
                            f"URL: {hit['url']}\n"
                            f"내용: {hit['content']}\n\n"
                        )

    if not search_results_text:
        st.error("웹 리서치 중 오류가 발생했거나, 검색 결과가 없습니다.")
        return ""

    # ─── AI 종합 브리핑 생성 ─────────────────────────────────────────
    synthesis_prompt = (
        f'"{topic}"에 대한 다음 웹 리서치 결과를 바탕으로, '
        f'다음 각 항목에 대해 분석하고, 반드시 "- *항목명*" 형식으로 시작하여 '
        f'구조화된 "{current_year}년 기준 초기 리서치 브리핑"을 작성해줘: '
        f'핵심 요약, 시장 동향, 주요 경쟁사, 주요 타겟 고객, 핵심 기술, 주요 통계'
    )

    try:
        st.success("웹 리서치 완료! AI가 결과를 종합하여 브리핑을 실시간으로 생성합니다...")
        response_placeholder = st.empty()

        def stream_generator():
            stream = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": synthesis_prompt}],
                stream=True
            )
            for chunk in stream:
                yield chunk.choices[0].delta.content or ""

        full_context = "".join(response_placeholder.write_stream(stream_generator()))
        return full_context

    except Exception as e:
        st.error(f"리서치 종합 실패: {e}")
        return ""


def display_research_briefing(context: str):
    st.markdown("<h3>AI 리서치 브리핑 (재구성)</h3>", unsafe_allow_html=True)
    sections = {"핵심 요약": "summary", "시장 동향": "trends", "핵심 플레이어": "competitors", "주요 경쟁사": "competitors", "주요 타겟 고객": "audience", "핵심 기술": "tech", "주요 통계": "stats"}
    parsed_sections = {}
    for key in sections.keys():
        pattern = re.compile(rf"-\s*\*+{key}\*+.*?\n(.*?)(?=\n\s*-\s*\*|\Z)", re.DOTALL | re.IGNORECASE)
        match = pattern.search(context)
        if match and key not in parsed_sections:
            parsed_sections[key] = match.group(1).strip()
    if not parsed_sections:
        st.warning("AI 브리핑을 섹션별로 분석하는 데 실패했습니다. 원본 텍스트를 표시합니다."); st.text(context); return

    briefing_html = '<div class="briefing-container">'
    for key, content in parsed_sections.items():
        icon_class = sections.get(key, "summary")
        briefing_html += f"""<div class="briefing-section"><div class="briefing-header"><div class="briefing-icon {icon_class}"></div><h4>{key}</h4></div><div class="briefing-content">{content.replace(chr(10), "<br>")}</div></div>"""
    briefing_html += '</div>'
    st.markdown(briefing_html, unsafe_allow_html=True)

def execute_pipeline(pipeline_name: str, steps: dict, topic: str, research_context: str, competitor_names: list = []):
    st.markdown(f"<div class='pipeline-header'>{pipeline_name} 생성을 시작합니다.</div>", unsafe_allow_html=True)
    step_titles = list(steps.keys())
    progress_placeholder = st.empty()
    if 'deep_dive_status' not in st.session_state: st.session_state.deep_dive_status = {}

    def update_progress_bar(current_step_index):
        progress_items = []
        for i, title in enumerate(step_titles):
            if i < current_step_index: progress_items.append(f"<span class='progress-item done'>{title}</span>")
            elif i == current_step_index: progress_items.append(f"<span class='progress-item working'>{title}</span>")
            else: progress_items.append(f"<span class='progress-item pending'>{title}</span>")
        progress_html = f"<div class='progress-bar'>{''.join(progress_items)}</div>"
        progress_placeholder.markdown(progress_html, unsafe_allow_html=True)

    step_items = list(steps.items())
    master_step_index = 0
    while master_step_index < len(step_items):
        title, details = step_items[master_step_index]
        group_name = details.get("layout_group")
        if group_name:
            group_steps_data = []
            temp_index = master_step_index
            while temp_index < len(step_items) and step_items[temp_index][1].get("layout_group") == group_name:
                group_steps_data.append(step_items[temp_index]); temp_index += 1
            update_progress_bar(master_step_index)
            st.markdown(f'<h3>{master_step_index + 1}. {group_name}</h3>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="step-content-card">', unsafe_allow_html=True)
                cols = st.columns(len(group_steps_data))
                full_group_text_result = ""
                for i, (step_title, step_details) in enumerate(group_steps_data):
                    with cols[i]:
                        if step_title == "경쟁사 구도" and not competitor_names:
                            st.markdown(f"<h5>{step_title}</h5>", unsafe_allow_html=True)
                            fig = create_empty_chart("웹 리서치에서 유효한<br>경쟁사를 찾을 수 없었습니다.")
                            st.plotly_chart(fig, use_container_width=True)
                            continue
                        with st.spinner(f"{step_title} 분석 중..."):
                            previous_steps_summary = ""
                            if st.session_state.get('plan_data'):
                                if 'personas' in st.session_state.plan_data: previous_steps_summary += f"- 타겟 페르소나: {', '.join([p['Persona'] for p in st.session_state.plan_data['personas']])}\n"
                                if 'key_features' in st.session_state.plan_data and st.session_state.plan_data['key_features']: previous_steps_summary += f"- 핵심 기능: {st.session_state.plan_data['key_features'][0].get('Feature', 'N/A')}\n"
                            
                            augmented_prompt = f'"{step_details["prompt_template"].format(p=topic)}"'
                            text_result = stream_and_display_step(augmented_prompt)
                            full_group_text_result += f"--- {step_title} ---\n{text_result}\n\n"
                        if text_result:
                            st.markdown(f"<h5>{step_title}</h5>", unsafe_allow_html=True)
                            display_type = step_details.get("display_type", "chart")
                            if display_type == "chart":
                                if step_details["func"] == create_priority_chart: fig = step_details["func"](text_result, topic)
                                else: fig = step_details["func"](text_result)
                                st.plotly_chart(fig, use_container_width=True, key=f"{pipeline_name}_{master_step_index}_{i}")
                            elif display_type == "custom":
                                step_details["func"](text_result)
                        else: st.error(f"{step_title} 데이터 분석 실패")
                st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("AI 원본 답변 보기 (그룹)"): st.text(full_group_text_result or "AI 답변 없음")
            deep_dive_key = f"deep_dive_{pipeline_name}_{master_step_index}"
            if st.button(f"'{group_name}' 분석 근거 더보기", key=f"show_{deep_dive_key}", use_container_width=True):
                with st.spinner("심층 분석 중..."):
                    deep_dive_content = get_deep_dive_analysis(topic, group_name, full_group_text_result)
                    st.info(deep_dive_content)
            if master_step_index < len(steps) - len(group_steps_data): st.markdown("<hr class='step-divider'>", unsafe_allow_html=True)
            master_step_index += len(group_steps_data)
        else:
            update_progress_bar(master_step_index)
            st.markdown(f'<h3>{master_step_index+1}. {title}</h3>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="step-content-card">', unsafe_allow_html=True)
                with st.spinner(f"{title} 분석 중..."):
                    previous_steps_summary = ""
                    if st.session_state.get('plan_data'):
                        if 'personas' in st.session_state.plan_data: previous_steps_summary += f"- 타겟 페르소나: {', '.join([p['Persona'] for p in st.session_state.plan_data['personas']])}\n"
                        if 'key_features' in st.session_state.plan_data and st.session_state.plan_data['key_features']:
                            previous_steps_summary += f"- 핵심 기능: {st.session_state.plan_data['key_features'][0].get('Feature', 'N/A')}\n"
                    augmented_prompt = f'"{details["prompt_template"].format(p=topic)}"'
                    text_result = stream_and_display_step(augmented_prompt)
                if text_result:
                    display_type = details.get("display_type", "chart")
                    if display_type == "chart":
                        if details["func"] == create_priority_chart:
                            fig = details["func"](text_result, topic)
                        else:
                            fig = details["func"](text_result)
                        st.plotly_chart(fig, use_container_width=True, key=f"{pipeline_name}_{master_step_index}")
                    elif display_type == "custom":
                        details["func"](text_result)
                else: st.error("데이터 분석 실패")
                st.markdown('</div>', unsafe_allow_html=True)
            with st.expander("AI 원본 답변 보기"): st.text(text_result or "AI 답변 없음")
            deep_dive_key = f"deep_dive_{pipeline_name}_{master_step_index}"
            if st.button(f"'{title}' 분석 근거 더보기", key=f"show_{deep_dive_key}", use_container_width=True):
                 with st.spinner("심층 분석 중..."):
                    deep_dive_content = get_deep_dive_analysis(topic, title, text_result)
                    st.info(deep_dive_content)
            if master_step_index < len(steps) - 1: st.markdown("<hr class='step-divider'>", unsafe_allow_html=True)
            master_step_index += 1
    update_progress_bar(len(steps))

def get_competitor_data(topic: str, research_context: str) -> list[str]:
    """리서치 내용에서 경쟁사 목록만 정확히 추출하는 함수"""
    with st.spinner("AI 리서치 브리핑에서 경쟁사 목록 추출 중..."):
        prompt = f"다음은 '{topic}'에 대한 리서치 내용입니다. 이 내용에서 언급된 **주요 경쟁사 또는 핵심 플레이어의 이름**만 추출하여 JSON 형식의 리스트로 응답해주십시오. 다른 설명은 절대 추가하지 마세요. 예: {{\"competitors\": [\"Google\", \"Cloudflare\", \"Akamai\"]}}\n\n---\n{research_context}"
        try:
            response = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], response_format={"type": "json_object"}, temperature=0)
            competitors = json.loads(response.choices[0].message.content).get("competitors", [])
            return competitors
        except Exception:
            return []

def product_planning_pipeline(topic: str, research_context: str, competitor_names: list):
    st.markdown("<h2>I. 제품 기획 분석</h2>", unsafe_allow_html=True)
    competitor_list_str = ", ".join(competitor_names) if competitor_names else "리서치에서 발견된 경쟁사 없음"
    forrester_prompt = f"다음 경쟁사 리스트({competitor_list_str})를 기반으로 Forrester Wave 분석을 수행해줘. **리스트에 없는 경쟁사를 절대 임의로 추가하지 마시오.** 각 경쟁사별 Strategy와 Current Offering은 **0.0에서 10.0 사이 숫자 값**으로만 표현해야 해. 만약 리스트가 비어있다면, 'Competitor' 컬럼에 '분석할 경쟁사 없음'이라고 답변하는 테이블을 생성해줘. 컬럼명은 'Competitor', 'Strategy', 'Current Offering', 'Market Presence'로 정확히 생성해야 해."
    steps = {
        "시장 스냅샷": {"prompt_template": "주제 '{p}'의 시장 규모(TAM, SAM, SOM)를 추정하고 성장 가능성을 분석하는 마크다운 테이블(컬럼: 구분, 규모(원), 근거)을 만들어줘.", "display_type": "chart", "func": create_market_chart, "layout_group": "초기 시장 분석"},
        "경쟁사 구도": {"prompt_template": forrester_prompt, "display_type": "chart", "func": create_forrester_wave_chart, "layout_group": "초기 시장 분석"},
        "타겟 페르소나": {"prompt_template": "주제 '{p}'의 핵심 타겟 페르소나 3명을 정의하고, 각 페르소나의 이름, 나이, 직업, 핵심 목표(Goal), 가장 큰 고충(Pain Point)을 마크다운 테이블로 만들어줘.", "display_type": "custom", "func": display_persona_cards},
        "기능 우선순위": {"prompt_template": "주제 '{p}'에 필요한 핵심 기능 5가지를 정의하고, RICE 점수를 10점 만점으로 평가하는 마크다운 테이블을 만들어줘. 컬럼명은 반드시 **'기능', 'Reach', 'Impact', 'Confidence', 'Effort'**로 생성해줘.", "display_type": "chart", "func": create_priority_chart},
        "제품 출시 로드맵": {"prompt_template": f"'{topic}'의 1년 로드맵을 5단계로 나누어 마크다운 테이블(Task, Start, Finish, Resource)로 만들어줘. 날짜는 {datetime.now().year+1}-MM-DD 형식으로.", "display_type": "chart", "func": create_roadmap_gantt_chart}
    }
    execute_pipeline("제품 기획", steps, topic, research_context, competitor_names)

def promotion_planning_pipeline(topic: str, research_context: str):
    st.markdown("<h2>II. 프로모션 기획 분석</h2>", unsafe_allow_html=True)
    steps = {
        "프로모션 목표(KPI)": {"prompt_template": "앞선 분석 내용을 바탕으로, '{p}' 프로모션을 위한 구체적인 핵심 목표 3가지를 'KPI', '목표치' 컬럼의 마크다운 테이블로 만들어줘.", "display_type": "chart", "func": create_kpi_bar_chart, "layout_group": "프로모션 목표 및 타겟"},
        "타겟 고객": {"prompt_template": "앞서 정의된 타겟 페르소나 기반으로 '{p}' 프로모션의 타겟 고객 그룹 3개를 '고객 그룹', '특징', '예상 도달률(%)' 컬럼의 마크다운 테이블로 만들어줘.", "display_type": "chart", "func": create_customer_segment_chart, "layout_group": "프로모션 목표 및 타겟"},
        "핵심 오퍼": {"prompt_template": "경쟁사와 트렌드, 우리 제품 기능을 고려하여 '{p}' 프로모션의 매력적인 혜택 3가지를 '오퍼 내용', '조건', '매력도(10점)' 컬럼의 마크다운 테이블로 만들어줘.", "display_type": "chart", "func": create_kpi_bar_chart},
        "홍보 채널": {"prompt_template": "타겟 고객이 주로 활동하는 채널 중심으로 '{p}' 프로모션 홍보 채널 4개와 예상 예산 비율(%)을 '채널', '예산 비율(%)' 컬럼의 마크다운 테이블로 만들어줘.", "display_type": "chart", "func": create_pie_chart},
        "기간 및 일정": {"prompt_template": f"'{topic}' 프로모션의 주요 일정을 4단계로 나누어 마크다운 테이블(Task, Start, Finish)로 만들어줘. 날짜는 {datetime.now().year+1}-MM-DD 형식으로.", "display_type": "chart", "func": create_roadmap_gantt_chart}
    }
    execute_pipeline("프로모션 기획", steps, topic, research_context)

def marketing_campaign_pipeline(topic: str, research_context: str):
    st.markdown("<h2>III. 마케팅 캠페인 분석</h2>", unsafe_allow_html=True)
    steps = {
        "캠페인 목표": {"prompt_template": "'{p}' 마케팅 캠페인의 구체적인 목표 3가지를 '목표', '핵심지표(KPI)' 컬럼을 가진 마크다운 테이블로 만들어줘. 답변은 테이블 외에 다른 설명을 포함하지 마.", "display_type": "chart", "func": create_kpi_bar_chart},
        "타겟 & 핵심 메시지": {"prompt_template": "캠페인의 **가장 중요한 타겟 고객** 한 그룹을 명확히 정의하고, 그들의 마음을 움직일 수 있는 **짧고 강력한 핵심 슬로건**을 제안해줘. 답변은 반드시 '타겟: [간결한 타겟 설명]'과 '핵심 메시지: \"[기억하기 쉬운 슬로건]\"' 형식으로, 다른 설명 없이 두 줄로만 작성해줘.", "display_type": "custom", "func": display_message_and_target_card},
        "콘텐츠 전략": {"prompt_template": "최신 트렌드를 반영하여 '{p}' 캠페인을 위한 핵심 콘텐츠 아이디어 3가지를 '콘텐츠 형식', '주요 내용' 컬럼의 마크다운 테이블로 만들어줘.", "display_type": "custom", "func": display_content_strategy_cards},
        "채널 믹스": {"prompt_template": "'{p}' 캠페인에 활용할 미디어 채널 믹스를 '채널 구분(Owned/Paid/Earned)', '채널명', '예산 비중(%)' 컬럼의 마크다운 테이블로 만들어줘.", "display_type": "chart", "func": create_sunburst_chart},
        "실행 타임라인": {"prompt_template": f"'{topic}' 캠페인의 3개월 타임라인을 주요 마일스톤별로 나누어 마크다운 테이블(Task, Start, Finish)로 만들어줘. 날짜는 {datetime.now().year+1}-MM-DD 형식으로.", "display_type": "chart", "func": create_roadmap_gantt_chart}
    }
    execute_pipeline("마케팅 캠페인 기획", steps, topic, research_context)

def classify_intent(user_prompt: str) -> str:
    system_prompt = "사용자 요청을 'PLANNING_REQUEST' 또는 'GENERAL_QUESTION'으로 분류해. 제품, 서비스, 프로모션, 마케팅, 사업 기획 및 분석은 PLANNING_REQUEST야. 그 외는 GENERAL_QUESTION이야."
    try:
        response = openai.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}], temperature=0)
        intent = response.choices[0].message.content.strip()
        st.toast(f"요청 의도 분석: {intent}")
        return intent
    except Exception as e: st.error(f"의도 분석 중 오류: {e}"); return "GENERAL_QUESTION"

def run_intelligent_agent(user_prompt):
    if 'plan_data' not in st.session_state:
        st.session_state.plan_data = {}
    
    intent = classify_intent(user_prompt)

    if intent == "PLANNING_REQUEST":
        research_context = perform_web_research_and_synthesis(user_prompt)
        if not research_context:
            st.error("초기 리서치에 실패하여 분석을 진행할 수 없습니다."); return
        
        display_research_briefing(research_context)
        
        competitor_names = get_competitor_data(user_prompt, research_context)
        
        product_planning_pipeline(user_prompt, research_context, competitor_names)
        promotion_planning_pipeline(user_prompt, research_context)
        marketing_campaign_pipeline(user_prompt, research_context)

        st.success("모든 비즈니스 플랜 분석이 완료되었습니다.")
    else:
        with st.spinner("답변 생성 중..."):
            response = openai.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": user_prompt}])
            st.markdown(response.choices[0].message.content)

def main():
    st.set_page_config(page_title="PENTHENA AI Agent", layout="wide", initial_sidebar_state="expanded")
    load_css("style.css")
    
    with st.sidebar:
        st.markdown('<div class="sidebar-logo">PENTHENA</div>', unsafe_allow_html=True)
        if st.button("새 분석 시작", use_container_width=True, type="primary", key="new_analysis_sidebar"):
            st.session_state.clear(); st.rerun()
        st.divider()
        display_world_clocks()
        st.divider()
        display_exchange_rates()
        st.divider()
        
        col1, col2 = st.columns([0.8, 0.2])
        with col1: st.markdown("<h5><span class='icon-dot'></span> 분석 기록</h5>", unsafe_allow_html=True)
        with col2:
            if st.button("삭제", use_container_width=True, key="clear_history"):
                st.session_state.prompt_history = []; st.rerun()
        
        if 'prompt_history' not in st.session_state: st.session_state.prompt_history = []
        for i, prompt_text in enumerate(st.session_state.prompt_history[:5]):
            if st.button(prompt_text, key=f"history_{i}", use_container_width=True):
                st.session_state.clear()
                st.session_state.messages = [{"role": "user", "content": prompt_text}]
                st.rerun()

    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    st.title("PENTHENA Intelligence")
    st.write("분석하고 싶은 기획 주제를 자유롭게 입력해주세요. AI가 실시간 웹 리서치를 통해 통합 비즈니스 플랜을 생성합니다.")

    if 'messages' not in st.session_state: st.session_state.messages = []
    
   

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"], unsafe_allow_html=True)

    if prompt := st.chat_input("새로운 기획 분석을 시작합니다..."):
        st.session_state.clear()
        st.session_state.messages = [{"role": "user", "content": prompt}]
        if 'prompt_history' not in st.session_state: st.session_state.prompt_history = []
        st.session_state.prompt_history.insert(0, prompt)
        st.rerun()

    if st.session_state.get('messages') and 'last_prompt' not in st.session_state:
        user_prompt = st.session_state.messages[-1]["content"]
        st.session_state.last_prompt = user_prompt
        with st.chat_message("assistant"):
            run_intelligent_agent(user_prompt)

    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
