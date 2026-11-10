import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

st.set_page_config(page_title="어린이보호구역 안전 등급 예측 및 개선 시뮬레이터", page_icon="🚨", layout="wide")

st.title("🚨 어린이보호구역 안전 등급 예측 및 개선 시뮬레이터")
st.markdown("""
전국 어린이 보호구역 표준데이터를 활용하여 **CCTV 설치 여부, CCTV 설치 대수, 보호구역 도로 폭** 등의 주요 변수를 바탕으로 
해당 스쿨존이 **고위험 구역**에 해당하는지 예측하는 머신러닝 분류 모델을 학습하고 성능을 비교합니다.
""")

# 1. 데이터 로드 및 전처리
@st.cache_data
def load_and_preprocess_data():
    df = pd.read_csv('전국어린이보호구역표준데이터.csv', encoding='cp949')
    
    # 전처리 및 파생 변수 생성
    df['CCTV설치대수'] = pd.to_numeric(df['CCTV설치대수'], errors='coerce').fillna(0)
    df['보호구역도로폭_num'] = pd.to_numeric(df['보호구역도로폭'], errors='coerce')
    median_width = df['보호구역도로폭_num'].median()
    df['보호구역도로폭_num'] = df['보호구역도로폭_num'].fillna(median_width)
    
    # CCTV 설치 여부 범주형 처리 (Y: 1, 그외: 0)
    df['CCTV설치여부_num'] = df['CCTV설치여부'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
    
    # 타겟 변수 설정: 고위험 여부 (도로 폭이 12m 이상이면서 CCTV 대수가 1대 이하이거나 설치율이 낮으면 고위험 1, 아니면 0)
    def define_high_risk(row):
        if row['보호구역도로폭_num'] >= 12 and row['CCTV설치대수'] <= 1:
            return 1
        elif row['CCTV설치여부_num'] == 0 and row['보호구역도로폭_num'] >= 10:
            return 1
        else:
            return 0
            
    df['High_Risk'] = df.apply(define_high_risk, axis=1)
    
    # 시도명 추출 (소재지도로명주소에서 공백 기준 분리)
    def extract_sido(addr):
        if pd.isna(addr):
            return "기타"
        parts = str(addr).split()
        return parts[0] if len(parts) > 0 else "기타"
        
    df['시도명'] = df['소재지도로명주소'].apply(extract_sido)
    
    return df

try:
    df = load_and_preprocess_data()
except Exception as e:
    st.error(f"데이터 파일을 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()

# 2. 모델 학습 및 평가 데이터 준비
X = df[['CCTV설치여부_num', 'CCTV설치대수', '보호구역도로폭_num']]
y = df['High_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 두 모델 학습
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression(max_iter=1000, random_state=42)

rf_model.fit(X_train, y_train)
lr_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)
y_pred_lr = lr_model.predict(X_test)

# 다수 클래스 예측 모델 (훈련용에서 많은 쪽으로만 답하는 모델) 생성
majority_class = y_train.mode()[0]
y_pred_maj = np.full_like(y_test, majority_class)

# 테스트용 구역 총 개수 및 고위험 구역 개수
total_test_count = len(y_test)
actual_high_risk_count = int(y_test.sum())

st.markdown("---")
st.subheader("📋 테스트 데이터 개요")
st.info(f"테스트용 구역 총 개수: **{total_test_count:,}개** | 그중 실제 고위험 구역 개수: **{actual_high_risk_count:,}개**")

# 혼동행렬 및 지표 계산 함수 (문법 오류 수정 완료)
def compute_metrics_and_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    
    prec_rec_sum = prec + rec
    if prec_rec_sum == 0:
        f1 = 0.0
        f1_str = "계산할 수 없음"
    else:
        f1 = 2 * prec * rec / prec_rec_sum
        f1_str = f"{f1:.4f}"
        
    prec_str = f"{prec:.4f}" if (tp + fp) > 0 else "계산할 수 없음"
    rec_str = f"{rec:.4f}" if (tp + fn) > 0 else "계산할 수 없음"
    if (tp + fp) == 0:
        prec_str = "계산할 수 없음"
        
    return {
        'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'acc': acc, 'prec': prec, 'rec': rec, 'f1': f1,
        'prec_str': prec_str, 'rec_str': rec_str, 'f1_str': f1_str
    }

rf_res = compute_metrics_and_matrix(y_test, y_pred_rf)
lr_res = compute_metrics_and_matrix(y_test, y_pred_lr)
maj_res = compute_metrics_and_matrix(y_test, y_pred_maj)

# 3. 혼동행렬 2x2 표 시각화
st.subheader("🔲 모델별 혼동행렬 (Confusion Matrix)")
col1, col2 = st.columns(2)

def display_confusion_matrix_table(res, title):
    st.markdown(f"**{title}**")
    cm_data = {
        "예측: 안전(0)": [f"TN: {res['tn']:,}", f"FN: {res['fn']:,}"],
        "예측: 고위험(1)": [f"FP: {res['fp']:,}", f"TP: {res['tp']:,}"],
        "행 합계": [f"{res['tn'] + res['fp']:,}", f"{res['fn'] + res['tp']:,}"]
    }
    cm_df = pd.DataFrame(cm_data, index=["실제: 안전(0)", "실제: 고위험(1)"])
    st.table(cm_df)

with col1:
    display_confusion_matrix_table(rf_res, "랜덤 포레스트(다양한 기준으로 고위험 구역을 예측하는 앙상블 모델)")

with col2:
    display_confusion_matrix_table(lr_res, "로지스틱 회귀(확률로 위험도를 산출하는 선형 모델)")

# 4. 성능 지표 비교 표
st.subheader("📊 두 모델 및 기준 모델 성능 비교 표")

metrics_summary_df = pd.DataFrame({
    "평가 지표": ["정확도 (Accuracy)", "정밀도 (Precision)", "재현율 (Recall)", "F1 스코어 (F1-score)"],
    "랜덤 포레스트": [f"{rf_res['acc']:.4f}", rf_res['prec_str'], rf_res['rec_str'], rf_res['f1_str']],
    "로지스틱 회귀": [f"{lr_res['acc']:.4f}", lr_res['prec_str'], lr_res['rec_str'], lr_res['f1_str']],
    "훈련용에서 많은 쪽으로만 답하는 모델": [f"{maj_res['acc']:.4f}", maj_res['prec_str'], maj_res['rec_str'], maj_res['f1_str']]
})
st.table(metrics_summary_df)

# 5. 놓친 위험 구역 및 헛짚은 구역 목록 비교 섹션
st.subheader("🔍 오류 분석 목록 (놓친 위험 구역 vs 헛짚은 구역)")

selected_model_name = st.selectbox(
    "어느 모델의 오류 목록을 확인하시겠습니까?", 
    ["랜덤 포레스트(다양한 기준으로 고위험 구역을 예측하는 앙상블 모델)", "로지스틱 회귀(확률로 위험도를 산출하는 선형 모델)"]
)

chosen_preds = y_pred_rf if "랜덤 포레스트" in selected_model_name else y_pred_lr

test_df = df.loc[X_test.index].copy()
test_df['Actual'] = y_test
test_df['Pred'] = chosen_preds

missed_risk_df = test_df[(test_df['Actual'] == 1) & (test_df['Pred'] == 0)].copy()
false_alarm_df = test_df[(test_df['Actual'] == 0) & (test_df['Pred'] == 1)].copy()

missed_risk_df['Risk_Score'] = missed_risk_df['보호구역도로폭_num'] - (missed_risk_df['CCTV설치대수'] * 0.5)
missed_risk_df = missed_risk_df.sort_values(by='Risk_Score', ascending=False)

false_alarm_df['Risk_Score'] = false_alarm_df['보호구역도로폭_num'] - (false_alarm_df['CCTV설치대수'] * 0.5)
false_alarm_df = false_alarm_df.sort_values(by='Risk_Score', ascending=False)

col_l, col_r = st.columns(2)

def render_error_list(sub_df, title, total_count_val, fn_fp_label):
    st.markdown(f"#### {title}")
    st.markdown(f"**전체 인원수(구역 수): {total_count_val:,}개**")
    
    display_df = sub_df.head(20).copy()
    
    display_table = pd.DataFrame({
        "구역명(시설명)": display_df['대상시설명'],
        "시도명": display_df['시도명'],
        "CCTV 설치 여부": display_df['CCTV설치여부'],
        "도로 폭(m)": display_df['보호구역도로폭_num']
    }).reset_index(drop=True)
    display_table.index = display_table.index + 1
    
    st.dataframe(display_table, use_container_width=True)
    st.caption(f"※ 위 목록의 구역 수는 혼동행렬의 **{fn_fp_label}** 칸 값과 정확히 일치합니다.")

with col_l:
    render_error_list(
        missed_risk_df, 
        "놓친 위험 구역 (실제 고위험 ➔ 예측 안전)", 
        len(missed_risk_df), 
        "FN"
    )

with col_r:
    render_error_list(
        false_alarm_df, 
        "헛짚은 구역 (실제 안전 ➔ 예측 고위험)", 
        len(false_alarm_df), 
        "FP"
    )

# 6. 페이지 하단: 인터랙티브 시뮬레이터 섹션
st.markdown("---")
st.subheader("🎛️ 인터랙티브 안전 점수 및 위험도 변동 시뮬레이터")
st.markdown("슬라이더를 통해 CCTV 설치 대수와 제한속도, 도로 폭 조건을 변경했을 때 해당 스쿨존의 **안전 점수(0~100점)**가 어떻게 변동되는지 Plotly 차트로 확인해보세요.")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    sim_cctv_count = st.slider("CCTV 설치 대수 (대)", min_value=0, max_value=20, value=2, step=1)
with col_s2:
    sim_speed_limit = st.slider("제한 속도 (km/h)", min_value=30, max_value=50, value=30, step=10)
with col_s3:
    sim_road_width = st.slider("보호구역 도로 폭 (m)", min_value=3.0, max_value=30.0, value=10.0, step=0.5)

base_score = 100
penalty_width = max(0, (sim_road_width - 8) * 2.5)
penalty_speed = (sim_speed_limit - 30) * 0.8
bonus_cctv = min(20, sim_cctv_count * 4)

safety_score = max(0, min(100, base_score - penalty_width - penalty_speed + bonus_cctv))

st.markdown(f"### 🎯 시뮬레이션 결과 산출된 안전 점수: **{safety_score:.1f} 점**")

if safety_score >= 80:
    st.success("✅ **안전 양호 구역:** 현재 시설 조건이 우수하여 사고 위험이 낮습니다.")
elif safety_score >= 50:
    st.warning("⚠️ **주의 필요 구역:** 일부 도로 환경 개선 및 추가적인 CCTV 설치가 권장됩니다.")
else:
    st.error("🚨 **고위험 구역:** 즉각적인 과속방지턱 설치, 방호울타리 확충 및 단속 카메라 증설이 필요합니다!")

fig_gauge = go.Figure(go.Indicator(
    mode = "gauge+number",
    value = safety_score,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': "스쿨존 종합 안전 점수"},
    gauge = {
        'axis': {'range': [None, 100]},
        'steps': [
            {'range': [0, 50], 'color': "rgba(231, 76, 60, 0.6)"},
            {'range': [50, 80], 'color': "rgba(241, 196, 15, 0.6)"},
            {'range': [80, 100], 'color': "rgba(46, 204, 113, 0.6)"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': safety_score
        }
    }
))
fig_gauge.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
st.plotly_chart(fig_gauge, use_container_width=True)
