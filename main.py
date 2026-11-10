import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 페이지 기본 설정
st.set_page_config(
    page_title="스쿨존 사고 위험 등급 예측 및 안전 관리 웹앱",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 초등학교 주변 어린이 보호구역(스쿨존) 사고 위험 등급 예측 및 안전 관리 모델")
st.markdown("""
이 웹앱은 전국 어린이 보호구역 표준데이터를 활용하여 **CCTV 설치 대수**와 **보호구역 도로 폭** 등의 환경 요인을 바탕으로 
스쿨존의 사고 위험 등급(고위험/중위험/저위험)을 예측하고, 머신러닝 분류 모델의 성능을 비교·분석하는 도구입니다.
""")

# 1. 데이터 로드 및 전처리 함수
@st.cache_data
def load_and_preprocess_data():
    # 파일 읽기 (CP949 인코딩 적용)
    df = pd.read_csv('전국어린이보호구역표준데이터.csv', encoding='cp949')
    
    # 결측치 처리 및 수치형 변환
    df['CCTV설치대수'] = pd.to_numeric(df['CCTV설치대수'], errors='coerce').fillna(0)
    df['보호구역도로폭_num'] = pd.to_numeric(df['보호구역도로폭'], errors='coerce')
    
    # 도로 폭 결측치는 중앙값으로 채우기
    median_width = df['보호구역도로폭_num'].median()
    df['보호구역도로폭_num'] = df['보호구역도로폭_num'].fillna(median_width)
    
    # 파생 변수 생성: 사고 위험 등급 (Risk_Level) 분류 규칙 정의
    def assign_risk_level(row):
        cctv = row['CCTV설치대수']
        width = row['보호구역도로폭_num']
        
        # 임계값 기준 분류 (저위험, 중위험, 고위험)
        if width >= 12 and cctv <= 1:
            return '고위험'
        elif width >= 8 or cctv <= 2:
            return '중위험'
        else:
            return '저위험'
            
    df['Risk_Level'] = df.apply(assign_risk_level, axis=1)
    
    # 모델 학습용 수치형 타겟 변수 생성 (0: 저위험, 1: 중위험, 2: 고위험)
    risk_mapping = {'저위험': 0, '중위험': 1, '고위험': 2}
    df['Risk_Target'] = df['Risk_Level'].map(risk_mapping)
    
    return df

try:
    df = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터 파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 사이드바 메뉴 구성
st.sidebar.header("🔍 탐색 및 예측 메뉴")
menu = st.sidebar.selectbox(
    "메뉴를 선택하세요", 
    ["1. 데이터 개요 및 탐색적 시각화", "2. 머신러닝 모델 학습 및 성능 평가", "3. 스쿨존 위험 등급 실시간 예측 시뮬레이터"]
)

# -------------------------------------------------------------
# [메뉴 1] 데이터 개요 및 탐색적 시각화
# -------------------------------------------------------------
if menu == "1. 데이터 개요 및 탐색적 시각화":
    st.header("📊 전국 어린이 보호구역 데이터 탐색")
    st.markdown("수집된 원본 데이터의 구조와 주요 환경 요인별 분포를 Plotly 차트로 시각화합니다.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("총 스쿨존 시설 수", f"{len(df):,}개")
    col2.metric("평균 CCTV 설치 대수", f"{df['CCTV설치대수'].mean():.2f}대")
    col3.metric("평균 도로 폭", f"{df['보호구역도로폭_num'].mean():.1f}m")
    
    with st.expander("원천 데이터 상위 10개 미리보기"):
        st.dataframe(df[['시설종류', '대상시설명', '소재지도로명주소', 'CCTV설치대수', '보호구역도로폭_num', 'Risk_Level']].head(10))
    
    st.subheader("📈 환경 요인별 시각화 분석")
    chart_type = st.radio("확인할 시각화 주제 선택", ["위험 등급별 시설 분포", "CCTV 설치 대수와 도로 폭의 관계"])
    
    if chart_type == "위험 등급별 시설 분포":
        risk_counts = df['Risk_Level'].value_counts().reset_index()
        risk_counts.columns = ['Risk_Level', 'Count']
        fig = px.bar(
            risk_counts, x='Risk_Level', y='Count', color='Risk_Level',
            title="스쿨존 사고 위험 등급별 시설 분포 현황",
            labels={'Risk_Level': '사고 위험 등급', 'Count': '시설 개수'},
            color_discrete_map={'저위험': '#2ecc71', '중위험': '#f1c40f', '고위험': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        fig = px.scatter(
            df.sample(min(1000, len(df))), x='보호구역도로폭_num', y='CCTV설치대수', color='Risk_Level',
            title="도로 폭과 CCTV 설치 대수에 따른 위험 등급 산점도",
            labels={'보호구역도로폭_num': '보호구역 도로 폭 (m)', 'CCTV설치대수': 'CCTV 설치 대수', 'Risk_Level': '위험 등급'},
            color_discrete_map={'저위험': '#2ecc71', '중위험': '#f1c40f', '고위험': '#e74c3c'}
        )
        st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# [메뉴 2] 머신러닝 모델 학습 및 성능 평가
# -------------------------------------------------------------
elif menu == "2. 머신러닝 모델 학습 및 성능 평가":
    st.header("🤖 분류 모델 학습 및 혼동행렬 비교 평가")
    st.markdown("독립 변수(`CCTV설치대수`, `보호구역도로폭_num`)를 이용해 사고 위험 등급을 예측하는 두 가지 분류 모델을 학습합니다.")
    
    # 데이터 준비
    X = df[['CCTV설치대수', '보호구역도로폭_num']]
    y = df['Risk_Target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 모델 정의
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    dt_model = DecisionTreeClassifier(random_state=42)
    
    lr_model.fit(X_train, y_train)
    dt_model.fit(X_train, y_train)
    
    y_pred_lr = lr_model.predict(X_test)
    y_pred_dt = dt_model.predict(X_test)
    
    # 지표 계산 함수 (매크로 평균 적용 및 0 division 방지)
    def get_metrics(y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
        return acc, prec, rec, f1
        
    lr_acc, lr_prec, lr_rec, lr_f1 = get_metrics(y_test, y_pred_lr)
    dt_acc, dt_prec, dt_rec, dt_f1 = get_metrics(y_test, y_pred_dt)
    
    st.subheader("📋 모델 성능 비교 표")
    metrics_df = pd.DataFrame({
        "평가 지표": ["정확도 (Accuracy)", "정밀도 (Precision)", "재현율 (Recall)", "F1 스코어 (F1-score)"],
        "로지스틱 회귀(확률로 답하는 모델)": [f"{lr_acc:.4f}", f"{lr_prec:.4f}", f"{lr_rec:.4f}", f"{lr_f1:.4f}"],
        "의사결정트리(질문으로 답하는 모델)": [f"{dt_acc:.4f}", f"{dt_prec:.4f}", f"{dt_rec:.4f}", f"{dt_f1:.4f}"]
    })
    st.table(metrics_df)
    st.info("※ 참고: 다중 분류 환경에서 정밀도, 재현율, F1 스코어는 매크로 평균(Macro average) 공식을 적용하여 산출하였습니다.")
    
    st.subheader("🔲 혼동행렬 (Confusion Matrix) 시각화")
    col1, col2 = st.columns(2)
    
    class_names = ['저위험', '중위험', '고위험']
    
    with col1:
        st.markdown("**로지스틱 회귀(확률로 답하는 모델)**")
        cm_lr = confusion_matrix(y_test, y_pred_lr)
        fig_lr = px.imshow(
            cm_lr, text_auto=True, x=class_names, y=class_names,
            labels=dict(x="예측값", y="실제값", color="빈도수"),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_lr, use_container_width=True)
        
    with col2:
        st.markdown("**의사결정트리(질문으로 답하는 모델)**")
        cm_dt = confusion_matrix(y_test, y_pred_dt)
        fig_dt = px.imshow(
            cm_dt, text_auto=True, x=class_names, y=class_names,
            labels=dict(x="예측값", y="실제값", color="빈도수"),
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_dt, use_container_width=True)

# -------------------------------------------------------------
# [메뉴 3] 스쿨존 위험 등급 실시간 예측 시뮬레이터
# -------------------------------------------------------------
else:
    st.header("🎯 스쿨존 사고 위험 등급 실시간 예측 및 안전 관리 시뮬레이터")
    st.markdown("사이드바에서 도로 환경 조건을 조절하여 새로운 스쿨존의 위험 등급을 예측하고, 초등 교사 관점의 맞춤형 안전 대책을 확인해보세요.")
    
    # 모델 학습 재실행
    X = df[['CCTV설치대수', '보호구역도로폭_num']]
    y = df['Risk_Target']
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X, y)
    
    st.sidebar.subheader("🎛️ 도로 환경 조건 입력")
    input_cctv = st.sidebar.slider("CCTV 설치 대수 (대)", min_value=0, max_value=20, value=2, step=1)
    input_width = st.sidebar.slider("보호구역 도로 폭 (m)", min_value=3.0, max_value=30.0, value=8.0, step=0.5)
    
    # 예측 수행
    input_data = pd.DataFrame([[input_cctv, input_width]], columns=['CCTV설치대수', '보호구역도로폭_num'])
    pred_code = model.predict(input_data)[0]
    
    code_to_label = {0: '저위험', 1: '중위험', 2: '고위험'}
    pred_label = code_to_label[pred_code]
    
    st.subheader("🔍 실시간 예측 결과")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if pred_label == '고위험':
            st.error(f"🚨 예측된 사고 위험 등급: **{pred_label}**")
        elif pred_label == '중위험':
            st.warning(f"⚠️ 예측된 사고 위험 등급: **{pred_label}**")
        else:
            st.success(f"✅ 예측된 사고 위험 등급: **{pred_label}**")
            
    with col2:
        st.markdown("#### 💡 권장 안전 관리 대책 및 시사점")
        if pred_label == '고위험':
            st.markdown("""
            - **긴급 개선 조치 필요:** 해당 구역은 도로 폭에 비해 CCTV 및 단속 시설이 부족하여 과속 및 사고 위험이 높습니다.
            - **정책 제안:** 즉시 과속단속 카메라 추가 설치, 방호울타리(펜스) 및 과속방지턱 확충이 시급합니다.
            """)
        elif pred_label == '중위험':
            st.markdown("""
            - **예방적 모니터링 필요:** 통학로 주변 차량 속도 제한 표지판을 정비하고, 등하교 시간대 불법 주정차 단속을 강화해야 합니다.
            """)
        else:
            st.markdown("""
            - **안전한 통학 환경 유지:** 현재 시설 조건이 우수합니다. 정기적인 시설 점검과 어린이 교통안전 지도 캠페인을 유지하는 것이 좋습니다.
            """)
            
    # 전체 데이터 중 입력값과 유사한 위치 시각화
    st.subheader("📍 주변 스쿨존 분포 속 나의 위치")
    fig = px.scatter(
        df.sample(min(500, len(df))), x='보호구역도로폭_num', y='CCTV설치대수', color='Risk_Level',
        title="전국 스쿨존 분포와 현재 입력 조건 비교",
        labels={'보호구역도로폭_num': '도로 폭 (m)', 'CCTV설치대수': 'CCTV 설치 대수'},
        color_discrete_map={'저위험': '#2ecc71', '중위험': '#f1c40f', '고위험': '#e74c3c'}
    )
    # 현재 입력값 강조 점 추가
    fig.add_trace(go.Scatter(
        x=[input_width], y=[input_cctv],
        mode='markers',
        marker=dict(size=18, color='black', symbol='star'),
        name='현재 입력 조건'
    ))
    st.plotly_chart(fig, use_container_width=True)
