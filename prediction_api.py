# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import sqlite3
import os
import pandas as pd
import numpy as np
import re
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

# --- 설정 --- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DB 경로 자동 감지: sfa2/incident_reports.db 우선, 상위 폴더의 incident_reports.db가 있으면 그쪽을 사용(dtr# --- 서버 실행 (개발용) --- #
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)와 동일 경로)
_LOCAL_DB = os.path.join(BASE_DIR, "incident_reports.db")
_PARENT_DB = os.path.abspath(os.path.join(BASE_DIR, os.pardir, "incident_reports.db"))
SQLITE_DB_PATH = _LOCAL_DB
if os.path.exists(_PARENT_DB):
        SQLITE_DB_PATH = _PARENT_DB
print(f"DEBUG: SQLITE_DB_PATH={SQLITE_DB_PATH}")
app = FastAPI(title="Prediction API", description="정량적 예측 서비스")

# --- Pydantic 모델 정의 --- #
class PredictRequest(BaseModel):
    fault_type: str
    target_year: Optional[int] = None # 예측할 연도 (기본값: 다음 해)

class PredictionResult(BaseModel):
    month: str
    predicted_count: int

class PredictResponse(BaseModel):
    fault_type: str
    predictions: List[PredictionResult]
    used_model: str
    debug_info: str
    total_predicted: int | None = None
    confidence: Optional[Dict[str, Optional[float]]] = None

# --- GRU 모델 정의 --- #
class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, output_size=1, num_layers=1):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # 마지막 타임스텝 예측
        return out

# --- 예측 함수 정의 (dtro.py에서 가져옴) --- #
# 데이터 개수 및 연속성 체크
# 신뢰도 계산 (B-모드와 동일 로직)
def calc_confidence(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Optional[float]]:
    try:
        y_pred = np.array(y_pred, dtype=float).flatten()
        std = float(np.std(y_pred)) if y_pred.size > 0 else None
        mean = float(np.mean(y_pred)) if y_pred.size > 0 else None
        mae = None
        rmse = None
        if y_true is not None and len(y_true) > 0 and y_pred.size > 0:
            y_true_tail = np.array(y_true[-len(y_pred):], dtype=float).flatten()
            if y_true_tail.size == y_pred.size:
                mae = float(np.mean(np.abs(y_true_tail - y_pred)))
                rmse = float(np.sqrt(np.mean((y_true_tail - y_pred) ** 2)))
        return {"std": std, "mean": mean, "mae": mae, "rmse": rmse}
    except Exception:
        return {"std": None, "mean": None, "mae": None, "rmse": None}
# LSTM 예측 (dtro.py와 동일한 조건)
def _improve_lstm_realism(predictions, data):
    """GRU 예측값을 현실적으로 조정 - 과도한 예측 방지"""
    try:
        if predictions is None or len(predictions) == 0:
            return predictions
        
        # 과거 데이터 분석
        past_values = data
        past_nonzero = past_values[past_values > 0]
        
        if len(past_nonzero) == 0:
            return np.zeros(len(predictions), dtype=int)
        
        # 과거 통계
        past_mean = np.mean(past_values)  # 0 포함한 전체 평균
        past_annual_total = np.sum(past_values) * (12 / len(past_values)) if len(past_values) > 0 else 0
        occurrence_rate = len(past_nonzero) / len(past_values)  # 발생 확률
        
        print(f"[DEBUG] 과거 연간 추정: {past_annual_total:.1f}건, 발생률: {occurrence_rate:.2f}")
        
        # 초기 예측값
        adjusted = np.array(predictions, dtype=float)
        initial_sum = np.sum(adjusted)
        
        # 1. 매우 보수적인 연간 총합 제한
        # 과거 연간 추정의 최대 80%로 제한
        max_annual = past_annual_total * 0.8
        
        if initial_sum > max_annual and initial_sum > 0:
            scale_factor = max_annual / initial_sum
            adjusted = adjusted * scale_factor
            print(f"[DEBUG] 연간 총합 조정: {initial_sum:.1f} -> {np.sum(adjusted):.1f}")
        
        # 2. 희소성 강화 - 발생률이 낮으면 더 많은 0 생성
        if occurrence_rate < 0.5:  # 50% 미만 발생시
            # 예측된 월 중에서 일부만 유지
            target_months = max(1, int(12 * occurrence_rate * 0.8))  # 더 보수적으로
            
            if np.count_nonzero(adjusted) > target_months:
                # 가장 큰 값들만 남기고 나머지는 0으로
                threshold_idx = np.argsort(adjusted)[-target_months]
                threshold = adjusted[threshold_idx] if target_months > 0 else np.max(adjusted)
                
                mask = adjusted >= threshold
                # 상위 target_months개만 유지
                top_indices = np.argsort(adjusted)[-target_months:]
                new_adjusted = np.zeros_like(adjusted)
                new_adjusted[top_indices] = adjusted[top_indices]
                adjusted = new_adjusted
                
                print(f"[DEBUG] 희소성 조정: {np.count_nonzero(predictions)}개월 -> {np.count_nonzero(adjusted)}개월")
        
        # 3. 과거 최대값으로 개별 월 제한
        past_max = np.max(past_values)
        adjusted = np.clip(adjusted, 0, past_max)
        
        # 4. 최종 연간 총합 재검증
        final_sum = np.sum(adjusted)
        if final_sum > past_annual_total * 0.6:  # 과거 추정의 60% 초과시 추가 감소
            excess_ratio = (final_sum - past_annual_total * 0.6) / final_sum
            if excess_ratio > 0:
                adjusted = adjusted * (1 - excess_ratio)
                print(f"[DEBUG] 최종 조정: {final_sum:.1f} -> {np.sum(adjusted):.1f}")
        
        # 5. 정수화 및 최종 검증
        result = np.round(adjusted).astype(int)
        result = np.clip(result, 0, past_max)
        
        # 6. 마지막 안전장치: 연간 총합이 여전히 과도하면 강제 감소
        final_total = np.sum(result)
        max_allowed_total = max(1, int(past_annual_total * 0.6))
        
        if final_total > max_allowed_total:
            # 무작위로 일부 월의 값을 0으로 만들기
            nonzero_indices = np.where(result > 0)[0]
            if len(nonzero_indices) > 0:
                reduce_count = int(final_total - max_allowed_total)  # 정수로 변환
                for _ in range(min(reduce_count, len(nonzero_indices))):
                    if len(nonzero_indices) > 0:
                        idx = np.random.choice(nonzero_indices)
                        if result[idx] > 0:
                            result[idx] -= 1
                            if result[idx] == 0:
                                nonzero_indices = nonzero_indices[nonzero_indices != idx]
        
        print(f"[DEBUG] 최종 결과: 연간 {np.sum(result)}건, {np.count_nonzero(result)}개월 발생")
        return result
        
    except Exception as e:
        print(f"[DEBUG] GRU 현실성 조정 오류: {e}")
        return np.round(predictions).astype(int) if predictions is not None else predictions

def predict_gru(data, future_units):
    min_required = 6  # 최소 데이터 개수
    if len(data) < min_required:
        return None, f"[GRU 스킵] 데이터 개수 부족: {len(data)}개 (최소 {min_required}개 필요)"
    
    zero_ratio = (data == 0).sum() / len(data)
    if zero_ratio >= 0.5:
        return None, f"[GRU 스킵] 결측(0) 비율이 높음: {zero_ratio:.1%}"
    
    try:
        # 데이터 전처리 (시퀀스 생성)
        seq_len = min(9, len(data) - 1) if len(data) > 1 else 1
        X, y = [], []
        for i in range(len(data) - seq_len):
            X.append(data[i:i+seq_len])
            y.append(data[i+seq_len])
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        
        model = GRUModel()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        # 학습
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output.squeeze(), y)
            loss.backward()
            optimizer.step()
        
        # 예측
        model.eval()
        with torch.no_grad():
            last_seq = torch.tensor(data[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            forecast = []
            for _ in range(len(future_units)):
                pred = model(last_seq).item()
                forecast.append(pred)
                last_seq = torch.cat([last_seq[:, 1:], torch.tensor([[pred]], dtype=torch.float32).unsqueeze(-1)], dim=1)
        
        raw_predictions = np.array(forecast)
        # 현실성 개선 적용
        realistic_predictions = _improve_lstm_realism(raw_predictions, data)
        
        return realistic_predictions, "GRU"
    except Exception as e:
        return None, f"[GRU 예측 오류] {type(e).__name__}: {e}"

# KNN 예측 (dtro.py와 동일한 조건)
def predict_knn(agg_df, group_key, future_units):
    try:
        y = agg_df.set_index(group_key)['count']
        min_required = 3  # dtro.py: 최소 12개
        if len(y) < min_required:
            return None, f"[KNN 스킵] 데이터 개수 부족: {len(y)}개 (최소 {min_required}개 필요)"
        
        zero_ratio = (y.values == 0).sum() / len(y) if len(y) > 0 else 0
        if zero_ratio >= 0.8:  # dtro.py 원본: 80% 이상이면 스킵
            return None, f"[KNN 스킵] 결측(0) 비율이 높음: {zero_ratio:.1%}"
        
        if not y.empty and len(y) > 1:
            # dtro.py와 동일한 규칙: 표본 < 6이면 1, 그 외 2
            n_neighbors = 1 if len(y) < 6 else 2
            model = KNeighborsRegressor(n_neighbors=n_neighbors)
            model.fit(np.arange(len(y)).reshape(-1, 1), y.values)
            x_future = np.arange(len(y), len(y) + len(future_units)).reshape(-1, 1)
            y_pred = model.predict(x_future)
            max_past = y.max()
            y_pred = np.clip(np.round(y_pred), 0, max_past).astype(int)
            
            # dtro.py와 동일: 모두 0 또는 1이면 평균으로 대체 시도
            if np.all((y_pred == 0) | (y_pred == 1)):
                temp, _ = predict_mean(agg_df)
                if temp is not None:
                    return np.array(temp, dtype=int), "KNN"
                else:
                    return y_pred, "KNN"
            
            if np.any(y_pred > 0):
                return y_pred, "KNN"
            else:
                return None, "[KNN] 예측값이 모두 0이거나 유효하지 않습니다."
        
        return None, "[KNN] 데이터가 부족하거나 유효하지 않습니다."
    except Exception as e:
        return None, f"[KNN 예측 오류] {type(e).__name__}: {e}"

# 선형회귀 예측 (dtro.py와 동일한 조건)
def predict_linear(agg_df, group_key, future_units):
    try:
        y = agg_df.set_index(group_key)['count']
        min_required = 3  # dtro.py: 최소 5개
        if len(y) < min_required:
            return None, f"[선형회귀 스킵] 데이터 개수 부족: {len(y)}개 (최소 {min_required}개 필요)"
        
        zero_ratio = (y.values == 0).sum() / len(y) if len(y) > 0 else 0
        if zero_ratio >= 0.90:  # dtro.py 원본: 90% 이상이면 스킵
            return None, f"[선형회귀 스킵] 결측(0) 비율이 높음: {zero_ratio:.1%}"
        
        if not y.empty and len(y) > 1:
            x = np.arange(len(y)).reshape(-1, 1)
            model = LinearRegression(fit_intercept=True, positive=True)
            model.fit(x, y.values)
            x_future = np.arange(len(y), len(y) + len(future_units)).reshape(-1, 1)
            y_pred = model.predict(x_future)
            max_past = y.max()
            mean_past = y.mean()
            y_pred = np.clip(np.round(y_pred), 0, max_past * 1.5).astype(int)  # 최대값 제한을 1.5배로 완화
            
            # 완화된 검증 조건들
            # 1. 표준편차 조건 완화: 0.1 → 0.01
            if np.all(y_pred == 0) or np.std(y_pred) < 0.01:
                return np.array([int(round(mean_past))] * len(future_units)), "선형회귀(평균대체-편차부족)"
            
            # 2. 과도한 예측값 조건 완화: 2배 → 3배
            if np.any(y_pred > mean_past * 3):
                # 3배 초과하는 값들만 평균으로 제한
                y_pred_adjusted = np.where(y_pred > mean_past * 3, int(round(mean_past)), y_pred)
                return y_pred_adjusted, "선형회귀(부분조정)"
            
            # 3. 0 이하 조건을 완화: 일부만 0이어도 허용
            if np.all(y_pred <= 0):
                # 모든 값이 0 이하일 때만 평균 대체
                return np.array([int(round(mean_past))] * len(future_units)), "선형회귀(평균대체-음수)"
            
            # 4. 예측값이 유효하면 반환 (일부가 0이어도 허용)
            return y_pred, "선형회귀"
        
        # 데이터가 1개만 있을 때는 평균으로 대체
        if len(y) == 1:
            return np.array([int(round(y.values[0]))] * len(future_units)), "선형회귀(단일값)"
        
        return None, "[선형회귀] 데이터가 부족하거나 유효하지 않습니다."
    except Exception as e:
        return None, f"[선형회귀 예측 오류] {type(e).__name__}: {e}"

# 평균 기반 예측(부트스트랩 샘플링: B-모드 규칙)
def predict_mean(agg_df):
    try:
        if agg_df.empty or 'count' not in agg_df.columns:
            return None, "[평균 스킵] 데이터 없음"
        # 정렬 및 최근 구간 선택
        df_sorted = agg_df.sort_values('month')
        win_n = int(os.getenv('MEAN_WINDOW_MONTHS', '36'))
        if win_n > 0 and len(df_sorted) > win_n:
            win_df = df_sorted.tail(win_n)
        else:
            win_df = df_sorted

        vals = np.array(win_df['count'].values, dtype=float)
        if vals.size == 0:
            return None, "[평균 스킵] 데이터 없음"
        # 0이 아닌 값이 1개 이상 필요
        if np.count_nonzero(vals) < 1:
            return None, "[평균 스킵] 0이 아닌 값 부족"
        # 1건만 있으면 예측 불가
        if np.count_nonzero(vals) == 1:
            return None, "[평균 스킵] 1건으로는 예측 불가"

        seed_env = os.getenv("MEAN_SEED")
        rng = np.random.default_rng(int(seed_env)) if seed_env and seed_env.isdigit() else np.random.default_rng()

        max_past = int(np.nanmax(vals)) if vals.size > 0 else 0
        mean_past = float(np.nanmean(vals)) if vals.size > 0 else 0.0

        # Case A: 희소(최대치<=1) 시계열 처리
        #   - dtro.py 스타일: 히스토리에서 복원추출 샘플링 후 12개월로 패딩(특히 히스토리<12일 때 매월 1 반복 방지에 효과)
        #   - 기본: Bernoulli(p) 샘플링 (과도 예측 방지 p cap 적용)
        if max_past <= 1:
            p = float(np.count_nonzero(vals) / len(vals))
            sparse_mode = os.getenv('MEAN_SPARSE_MODE', 'auto').lower()  # auto|dtro|bernoulli
            use_dtro = False
            if sparse_mode == 'dtro':
                use_dtro = True
            elif sparse_mode == 'auto':
                # 히스토리 길이가 12 미만이거나, p가 낮은(<=0.25) 경우 dtro 스타일이 더 보수적으로 작동
                use_dtro = (len(vals) < 12) or (p <= 0.25)
            if use_dtro:
                # size는 min(12, len(vals))로 샘플 후, 부족분은 0으로 패딩하여 12개월 구성
                k = min(12, len(vals))
                sampled = rng.choice(vals.astype(int), size=k, replace=True)
                pred = np.concatenate([sampled, np.zeros(12 - k, dtype=int)]) if k < 12 else sampled
                pred = np.clip(pred, 0, 1).astype(int)
                return pred, "평균(샘플·dtro)"
            # 기본 Bernoulli 경로
            p_cap = float(os.getenv('MEAN_BERNOULLI_CAP', '0.75'))
            if p_cap > 0:
                p = min(p, p_cap)
            bernoulli = rng.binomial(n=1, p=max(0.0, min(1.0, p)), size=12).astype(int)
            return bernoulli, "평균(확률·최근윈도우)"

        # Case B: 그 외에는 월별 계절 평균 사용 (과거 동일 월의 평균), 부족하면 전체 평균으로 보전
        months = pd.to_datetime(win_df['month']).dt.month
        monthly_mean = win_df.assign(_m=months).groupby('_m')['count'].mean().reindex(range(1,13), fill_value=mean_past)
        floats = monthly_mean.values.astype(float)
        # 연간 기대 합계를 유지하며 정수화(최대 나머지 방식)
        floors = np.floor(floats)
        rema = floats - floors
        target = int(np.round(np.sum(floats)))
        pred = floors.astype(int)
        remaining = int(target - np.sum(pred))
        if remaining > 0:
            idx_order = np.argsort(-rema)  # 큰 나머지부터 배분
            for i in idx_order:
                if remaining <= 0:
                    break
                pred[i] += 1
                remaining -= 1
        elif remaining < 0:
            idx_order = np.argsort(rema)  # 작은 나머지부터 회수
            for i in idx_order:
                if remaining >= 0:
                    break
                if pred[i] > 0:
                    pred[i] -= 1
                    remaining += 1
        # 범위 클리핑
        pred = np.clip(pred, 0, max_past).astype(int)
        return pred, "평균(계절·최근윈도우)"
    except Exception as e:
        return None, f"[평균 예측 오류] {type(e).__name__}: {e}"

# --- API 엔드포인트 구현 --- #
@app.post("/predict", response_model=PredictResponse)
async def predict_fault(request: PredictRequest):
    # 입력은 이미 UTF-8 문자열이므로 추가 인코딩 없이 공백만 정리한다
    request.fault_type = request.fault_type.strip()
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        conn.text_factory = lambda x: x.decode('utf-8')
        # 파라미터 바인딩으로 안전하게 조회
        query = "SELECT 장애일시 FROM incident_data WHERE 장애명 = ?"
        params = [request.fault_type]
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        df = pd.DataFrame(rows, columns=['장애일시'])
        print(f"DEBUG: fault_type={repr(request.fault_type)}, query={query}, params={params}")
        print(f"DEBUG: df.shape={df.shape}, df.head()={df.head() if not df.empty else 'empty'}")
        print(f"DEBUG: direct query len={len(rows)}")

        if df.empty:
            raise HTTPException(status_code=404, detail=f"장애 유형 '{request.fault_type}'에 대한 데이터가 없습니다.")

        # 월별 집계
        df['장애일시'] = pd.to_datetime(df['장애일시'], errors='coerce')
        df.dropna(subset=['장애일시'], inplace=True)
        df['month'] = df['장애일시'].dt.to_period('M').dt.to_timestamp('M')

        agg_df = df.groupby('month').size().reset_index(name='count')
        agg_df = agg_df.set_index('month').asfreq('ME', fill_value=0).reset_index()

        # 예측할 연도 설정 (기본값: 다음 해)
        target_year = request.target_year if request.target_year else pd.to_datetime('now').year + 1
        future_units = [f"{target_year}-{m:02}" for m in range(1, 13)]

        # 데이터 numpy array로 변환 (darts 제거)
        data = agg_df['count'].values

        # 4단계 순차 실행: 성공 시 즉시 종료
        predictions = None
        used_model = "미분류"
        debug_info = ""
        hist_total = int(agg_df['count'].sum()) if not agg_df.empty else 0

        # 1차: GRU 예측
        if predictions is None:
            gru_preds, gru_info = predict_gru(data, future_units)
            if gru_preds is not None and np.any(gru_preds > 0):
                predictions = gru_preds
                used_model = "GRU"
                debug_info += f"[1차 GRU 채택] {gru_info}\n"
            else:
                debug_info += f"[1차 GRU 실패] {gru_info}\n"

        # 2차: KNN 예측 (LSTM 실패 시에만)
        if predictions is None:
            knn_preds, knn_info = predict_knn(agg_df, 'month', future_units)
            if knn_preds is not None and np.any(knn_preds > 0):
                predictions = knn_preds
                used_model = "KNN"
                debug_info += f"[2차 KNN 채택] {knn_info}\n"
            else:
                debug_info += f"[2차 KNN 실패] {knn_info}\n"

        # 3차: 선형회귀 예측 (KNN 실패 시에만)
        if predictions is None:
            linear_preds, linear_info = predict_linear(agg_df, 'month', future_units)
            if linear_preds is not None and np.any(linear_preds > 0):
                predictions = linear_preds
                used_model = "선형회귀" if linear_info == "선형회귀" else linear_info
                debug_info += f"[3차 선형회귀 채택] {linear_info}\n"
            else:
                debug_info += f"[3차 선형회귀 실패] {linear_info}\n"

        # 4차: 평균 예측 (최종 보장)
        if predictions is None:
            mean_preds, mean_info = predict_mean(agg_df)
            if mean_preds is not None:
                predictions = mean_preds
                used_model = "평균" if mean_info.startswith("평균") else mean_info
                debug_info += f"[4차 평균 채택] {mean_info}\n"
            else:
                # 최종 보장: 모든 예측이 실패하면 0으로 채운 기본값 생성
                predictions = [0] * len(future_units)
                used_model = "예측 불가"
                debug_info += "[예측 불가] 모든 방법 실패, 0으로 채움\n"

        # 길이 보정: 항상 12개 반환
        if isinstance(predictions, (list, tuple, np.ndarray)):
            if len(predictions) != len(future_units):
                # 부족하면 마지막 값으로 패딩, 초과하면 잘라내기
                preds_list = list(map(int, list(np.array(predictions).astype(int).flatten())))
                if len(preds_list) < len(future_units):
                    pad_val = preds_list[-1] if preds_list else 0
                    preds_list += [pad_val] * (len(future_units) - len(preds_list))
                else:
                    preds_list = preds_list[:len(future_units)]
                predictions = preds_list

        response_predictions = []
        for i, month_str in enumerate(future_units):
            response_predictions.append(PredictionResult(month=month_str, predicted_count=int(predictions[i])))
        # 연간 합계 및 신뢰도 계산
        total_predicted = int(np.sum([p.predicted_count for p in response_predictions]))
        y_true_series = agg_df.set_index('month')['count'].values if 'month' in agg_df.columns else np.array([])
        confidence = calc_confidence(y_true_series, np.array([p.predicted_count for p in response_predictions]))

        return PredictResponse(
            fault_type=request.fault_type,
            predictions=response_predictions,
            used_model=used_model,
            debug_info=debug_info,
            total_predicted=total_predicted,
            confidence=confidence
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"예측 중 서버 오류 발생: {e}")
    finally:
        if conn:
            conn.close()

# --- 서버 실행 (개발용) --- #
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)