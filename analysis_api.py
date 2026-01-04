# -*- coding: utf-8 -*-
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import json
import os
import sqlite3
import pandas as pd
import requests
import torch
import re

# --- FastAPI 앱 인스턴스 생성 ---
app = FastAPI(title="Analysis API", description="AI 기반 분석 및 검색 서비스")

# --- 설정 로드 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, 'config.json')
SQLITE_DB_PATH = os.path.join(BASE_DIR, "incident_reports.db")

def load_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

CONFIG = load_config()
RISK_MATRIX = CONFIG['risk_matrix']

# --- LLM 설정 (Ollama) ---   기본은 hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_M
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip().rstrip("/")
ANALYSIS_LLM_MODEL = "hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_M"

# --- 전역 변수 ---
ollama_available: bool = False

# --- GPU 감지 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"사용할 디바이스: {DEVICE}")

# --- 서버 시작 이벤트 핸들러 ---
@app.on_event("startup")
async def startup_event():
    print("서버 시작 중: Ollama 가용성 체크...")

    # Ollama 가용성 체크
    global ollama_available
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        if resp.status_code == 200:
            tags = [t.get("name") for t in resp.json().get("models", [])]
            ollama_available = True
            print(f"Ollama 연결 성공. 사용 모델 기본값: {ANALYSIS_LLM_MODEL}. 설치됨: {', '.join(tags) if tags else '알 수 없음'}")
        else:
            ollama_available = False
            print(f"Ollama 연결 실패(status={resp.status_code}). URL={OLLAMA_BASE_URL}")
    except Exception as e:
        ollama_available = False
        print(f"Ollama 점검 실패: {e}. LLM 관련 기능이 제한됩니다.")
    
    print("서버 시작 완료.")

# --- Pydantic 모델 정의 ---
class DetectKeywordsRequest(BaseModel):
    text: str

class KeywordInfo(BaseModel):
    keyword: str

class DetectKeywordsResponse(BaseModel):
    detected_keywords: List[KeywordInfo]

class SQLBriefingRequest(BaseModel):
    fault_type: str
    keyword: Optional[str] = None
    year: Optional[int] = None

class CauseActionItem(BaseModel):
    text: str
    count: int

class SQLBriefingResponse(BaseModel):
    fault_type: str
    total_incidents: int
    importance_level: str
    yearly_frequency: float
    top_causes: List[CauseActionItem]
    top_actions: List[CauseActionItem]
    ai_recommendation: str
    mode: str

# --- 엔드포인트 구현 ---
@app.post("/detect_keywords", response_model=DetectKeywordsResponse)
async def detect_keywords(request: DetectKeywordsRequest):
    detected = []
    text_lower = request.text.lower()

    for level, keywords in RISK_MATRIX.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                detected.append(KeywordInfo(keyword=keyword))
    
    return DetectKeywordsResponse(detected_keywords=detected)

@app.post("/sql_based_briefing", response_model=SQLBriefingResponse)
async def sql_based_briefing(request: SQLBriefingRequest):
    """SQL DB 직접 조회로 구조화된 브리핑 생성"""
    conn = None
    try:
        conn = sqlite3.connect(SQLITE_DB_PATH)
        cursor = conn.cursor()
        
        # 1. 기본 통계 수집
        cursor.execute("SELECT COUNT(*) FROM incident_data WHERE `장애명` = ?", (request.fault_type,))
        total_incidents = cursor.fetchone()[0]
        
        # 2. 연평균 발생빈도 및 중요도 계산
        cursor.execute("SELECT MIN(`장애일시`), MAX(`장애일시`) FROM incident_data WHERE `장애명` = ?", (request.fault_type,))
        min_date_str, max_date_str = cursor.fetchone()

        yearly_frequency = 0
        if total_incidents > 0 and min_date_str and max_date_str:
            from datetime import datetime
            min_year = int(min_date_str.split()[0].split('-')[0].split('.')[0])
            max_year = int(max_date_str.split()[0].split('-')[0].split('.')[0])
            years_span = max(1, max_year - min_year + 1)
            yearly_frequency = total_incidents / years_span
        elif total_incidents > 0:
            yearly_frequency = float(total_incidents)

        if yearly_frequency >= 10:
            importance_level = "높음"
        elif yearly_frequency >= 5:
            importance_level = "중간"
        elif total_incidents > 0:
            importance_level = "낮음"
        else:
            importance_level = "정보없음"
            
        # 3. 원인 수집 및 집계
        cursor.execute("""
            SELECT `장애 원인`, COUNT(*) as cnt 
            FROM incident_data 
            WHERE `장애명` = ? AND `장애 원인` IS NOT NULL AND `장애 원인` != '' 
            GROUP BY `장애 원인` 
            ORDER BY cnt DESC LIMIT 8
        """, (request.fault_type,))
        top_causes_raw = cursor.fetchall()
        top_causes = [CauseActionItem(text=row[0], count=row[1]) for row in top_causes_raw]
        
        # 4. 조치방법 수집 및 집계
        cursor.execute("""
            SELECT `장애 발생 시 조치 방법`, COUNT(*) as cnt 
            FROM incident_data 
            WHERE `장애명` = ? AND `장애 발생 시 조치 방법` IS NOT NULL AND `장애 발생 시 조치 방법` != '' 
            GROUP BY `장애 발생 시 조치 방법` 
            ORDER BY cnt DESC LIMIT 8
        """, (request.fault_type,))
        top_actions_raw = cursor.fetchall()
        top_actions = [CauseActionItem(text=row[0], count=row[1]) for row in top_actions_raw]
        
        # 5. LLM 기반 AI 추천 조치방법 생성
        ai_recommendation = ""
        mode = "SQL_Basic"
        
        if ollama_available and top_causes and top_actions:
            mode = "SQL_LLM_Enhanced"
            causes_text = "\n".join([f"- {cause.text[:50]}... ({cause.count}건)" if len(cause.text) > 50 else f"- {cause.text} ({cause.count}건)" for cause in top_causes[:3]])  # 상위 3개, 길이 제한
            actions_text = "\n".join([f"- {action.text[:50]}... ({action.count}건)" if len(action.text) > 50 else f"- {action.text} ({action.count}건)" for action in top_actions[:3]])  # 상위 3개, 길이 제한
            
            # dataset 파일 로드 (우선 dataset_from_data_txt.json -> dataset.json)
            candidate_paths = [
                os.path.join(BASE_DIR, 'dataset_from_data_txt.json'),
                os.path.join(BASE_DIR, 'dataset.json')
            ]
            dataset = None
            for p in candidate_paths:
                if os.path.exists(p):
                    try:
                        with open(p, 'r', encoding='utf-8') as f:
                            dataset = json.load(f)
                        break
                    except Exception:
                        dataset = None

            qa_text = ""
            if dataset:
                # TF-IDF로 request.fault_type 관련 상위 문서 스니펫 추출 (sklearn 없으면 빈도 기반 폴백)
                selected = []
                try:
                    from sklearn.feature_extraction.text import TfidfVectorizer
                    import numpy as np

                    docs = []
                    items = []
                    for it in dataset:
                        txt = (it.get('instruction','') + ' ' + it.get('output','')).strip()
                        # 우선 장애명 포함 문서 우선 수집
                        if request.fault_type and request.fault_type in txt:
                            docs.append(txt)
                            items.append(it)
                    # 관련 문서가 없으면 전체를 대상으로 함
                    if not docs:
                        docs = [(it.get('instruction','') + ' ' + it.get('output','')).strip() for it in dataset]
                        items = list(dataset)

                    vect = TfidfVectorizer(max_df=0.9, min_df=1, ngram_range=(1,2))
                    X = vect.fit_transform(docs)
                    q_vec = vect.transform([request.fault_type or ''])
                    scores = (X @ q_vec.T).toarray().ravel()
                    idx = np.argsort(scores)[::-1][:5]
                    for i in idx:
                        if scores[i] > 0:
                            selected.append(items[i])
                except Exception:
                    # sklearn 미설치 등 문제 시 단순 빈도 기반 폴백
                    def score_item(it):
                        txt = (it.get('instruction','') + ' ' + it.get('output','')).lower()
                        q = (request.fault_type or '').lower()
                        return txt.count(q) if q else 0
                    scored = [(score_item(it), it) for it in dataset]
                    selected = [it for s,it in sorted(scored, key=lambda x: x[0], reverse=True) if s>0][:10]

                # 선택된 항목으로 qa_text 구성, 제어토큰 제거 및 길이 제한
                snippets = []
                for it in selected:
                    q = re.sub(r'(<end_of_turn>|<end_of_conversation>|<\|endoftext\|>)', '', it.get('instruction','')).strip()
                    a = re.sub(r'(<end_of_turn>|<end_of_conversation>|<\|endoftext\|>)', '', it.get('output','')).strip()
                    snippets.append(f"Q: {q}\nA: {a}")
                qa_text = "\n\n".join(snippets)
                if len(qa_text) > 5000:
                    qa_text = qa_text[:4000].rsplit("\n",1)[0] + "\n...[중략]"
            
            llm_prompt = f"""
{request.fault_type} 장애에 관해, 아래의 3가지 섹션을 한국어로 명확한 소제목과 목록으로 구체적으로 매우 자세히 작성하세요. 각 섹션은 주제별로 줄을 한 줄씩 띄어쓰기하여 시인성을 높이세요. 프롬프트의 지시를 엄격히 따르고, 다른 컨텍스트나 외부 정보를 무시하세요.

1. 원인 분석

2. 예방 조치

3. 결론

각 섹션은 최소 4-8문장으로 매우 자세히 서술하고 문장별로 줄바꿈을 하세요. 자세한 예시, 단계별 설명, 그리고 관련 데이터와 아래 Q&A 데이터를 기반으로 추론하세요.

{qa_text if qa_text else ''}

원인 목록 (요약):
{causes_text}

조치 목록 (요약):
{actions_text}
"""
            
            try:
                resp = requests.post(
                    f"{OLLAMA_BASE_URL}/api/generate",
                    json={
                        "model": ANALYSIS_LLM_MODEL,
                        "prompt": llm_prompt,
                        "stream": False,
                        "options": {
                            "num_thread": 4,
                            "temperature": 0.9
                        }
                    },
                    timeout=600,
                    headers={"Content-Type": "application/json"}
                )
                if resp.status_code == 200:
                    data = resp.json()
                    text = data.get("response", "") or ""
                    # 제어 토큰/종료 마커 제거
                    text = re.sub(r'(<end_of_turn>|<end_of_conversation>|<\|endoftext\|>)+', '', text)
                    # 연속 빈줄/과다 공백 정리
                    text = re.sub(r'\n{3,}', '\n\n', text).strip()

                    # 중복 라인 제거: 동일한 라인은 첫 등장만 남김
                    lines = text.splitlines()
                    seen = set()
                    new_lines = []
                    for line in lines:
                        s = line.strip()
                        if s == "":
                            new_lines.append(line)
                            continue
                        if s in seen:
                            continue
                        seen.add(s)
                        new_lines.append(line)
                    text = "\n".join(new_lines)

                    ai_recommendation = text if text else "추천 조치방법 생성 실패"
                else:
                    ai_recommendation = f"LLM 서비스 오류 (HTTP {resp.status_code})"
            except Exception as llm_error:
                ai_recommendation = f"추천 조치방법 생성 중 오류 발생: {str(llm_error)[:50]}"
        else:
            ai_recommendation = "LLM 서비스를 사용할 수 없거나 충분한 데이터가 없습니다."
        
        return SQLBriefingResponse(
            fault_type=request.fault_type,
            total_incidents=total_incidents,
            importance_level=importance_level,
            yearly_frequency=yearly_frequency,
            top_causes=top_causes,
            top_actions=top_actions,
            ai_recommendation=ai_recommendation,
            mode=mode
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"SQL 브리핑 생성 중 오류 발생: {e}"
        )
    finally:
        if conn:
            conn.close()
