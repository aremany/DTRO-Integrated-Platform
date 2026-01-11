# 🎯 대구교통공사(DTRO) 3호선 전력관제 장애 관리 통합 플랫폼

![DTRO Logo](ci.png)

![통합 플랫폼 메인화면](메인화면1.png)
*▲ 장애 관리 통합 플랫폼 메인화면 - 예측, 분석, 훈련, 검색 기능이 하나로 통합된 대시보드*

**보안상 훈련 시뮬레이터를 제외한, 응용 가능한 장애 관리 프레임워크**

이 프로그램은 **장애 예측, 분석, 지식 검색, 챗봇**을 하나의 플랫폼에 통합한 종합 솔루션의 **기본 뼈대(Base Framework)** 입니다. 
실제 전력 계통 시뮬레이터는 보안상의 이유로 제외되었으나, **다른 회사나 부서에서 본인의 환경에 맞게 커스터마이징하여 즉시 사용할 수 있도록 설계되었습니다.**

누구나 자유롭게 수정하고 배포하여 사용할 수 있습니다.

### 👤 개발자 (Developer)
- **소속:** 대구교통공사 3호선 경전철관제팀 전력관제
- **성명:** 강동우
- **역할:** 기획, 설계, 전체 개발 (Full Stack & AI), 시뮬레이터 디자인

> **Note:** 본 프로젝트는 개발자 개인의 연구 및 학습 결과물이며, 대구교통공사의 공식 입장이 아님을 밝힙니다.

---

## 📌 프로젝트 소개 및 활용 가이드

본 통합 플랫폼은 9개월간의 개발을 통해 완성된 시스템에서 **보안에 민감한 데이터와 시뮬레이터를 제거하고, 범용적으로 사용할 수 있도록 정제한 오픈소스 버전**입니다.

### 💡 활용 예시
이 플랫폼은 다음과 같은 조직에서 유용하게 활용할 수 있습니다:
- **제조 현장:** 설비 고장 예측 및 매뉴얼 검색 시스템 구축
- **IT 관제 센터:** 서버 장애 로그 분석 및 대응 가이드 챗봇
- **공공 시설:** 시설물 유지보수 이력 관리 및 교육 훈련 프레임워크
- **교육 기관:** AI 및 데이터 기반 유지보수(PdM) 시스템 실습용 교재

**"마음대로 수정하고 쓰셔도 됩니다!"** 
코드의 모든 부분은 수정 가능하며, 귀사의 로고, 데이터, 시뮬레이터로 교체하여 **나만의 통합 관제 시스템**을 만들어보세요.

### 🔗 개별 모듈 저장소
각 기능은 독립적인 모듈로도 사용 가능합니다:
- [장애 예측기](https://github.com/aremany/DTRO-Failure-Predictor)
- [장애 분석기](https://github.com/aremany/DTRO-Failure-Analyzer)
- [지식 검색기](https://github.com/aremany/DTRO-Knowledge-Searcher)
- [장애 보고서 뷰어](https://github.com/aremany/DTRO-Report-Viewer)
- [계통 시뮬레이터](https://github.com/aremany/DTRO-Power-Simulator)
- [사규 챗봇](https://github.com/aremany/DTRO-Legal-GraphRAG)

---

## 💡 핵심 철학

### "예측하고, 훈련하여, 대응한다"
전력관제에서 예방 정비만으로는 모든 장애를 막을 수 없습니다. 본 플랫폼의 핵심은:
- **예측**: 발생 가능한 장애를 AI로 예측
- **훈련**: 시뮬레이터로 반복 훈련
- **대응**: 실제 상황에서 빠르고 정확한 대응

---

## 🌟 주요 기능

### 1. 장애 예측 (Failure Prediction)
- **GRU 딥러닝**: 시계열 패턴 학습
- **KNN 머신러닝**: 유사 패턴 기반 예측
- **선형회귀**: 추세 기반 예측
- **평균법**: 데이터 부족 시 베이스라인
- **앙상블**: 데이터 양에 따라 최적 모델 자동 선택

### 2. 장애 분석 (Failure Analysis)
- **통계 분석**: 발생 빈도, 연간 추이, 주요 원인
- **AI 심층 분석**: LLM + RAG 기반 상세 분석
- **보고서 생성**: PDF, HTML, 인쇄 지원

### 3. 계통 시뮬레이터 (System Simulator)
- **AC 1/2계통**: 교류 전력 계통 훈련
- **AC 2/2계통**: 교류 전력 계통 훈련
- **본선 전차선**: 직류 전차선 계통 훈련
- **SVG 기반**: 확대/축소 가능한 인터랙티브 UI
- **실시간 조작**: 차단기, 단로기 개폐 시뮬레이션

### 4. 지식 검색 (Knowledge Search)
- **RAG 기술**: 사규 및 매뉴얼 검색
- **LLM 답변**: 문맥 기반 서술형 답변
- **출처 표시**: 답변 근거 투명성 확보

### 5. 사규 챗봇 (Regulation Chatbot)
- **대화형 인터페이스**: 자연어 질의응답
- **출처 원문 보기**: 클릭 한 번으로 원문 확인
- **프롬프트 커스터마이징**: 답변 스타일 조정

---

## ⚙️ 설치 및 실행

### 1. 필수 요구 사항
- **Python 3.8+**: 백엔드 API 서버
- **Node.js**: 프론트엔드 서버
- **Ollama**: Local LLM (분석 및 챗봇)

### 2. Ollama 모델 설치

#### 🎯 **옵션 A (최고 성능 - 전력계통 전문 파인튜닝 모델) ⭐ 강력 권장**
**2026년 1월 11일 공개된 전력계통 장애 분석 전문 파인튜닝 모델입니다.**
```bash
ollama pull bluejude10/smoothie-qwen3-8b-dtro
```
- **특징**: 전력계통, 급전계통, 전차선로, 수배전설비 도메인 특화
- **성능**: Gemma-3n 대비 우수한 장애 분석 정확도
- **허깅페이스**: [bluejude10/Smoothie-Qwen3-8B-DTRO-Edition](https://huggingface.co/bluejude10/Smoothie-Qwen3-8B-DTRO-Edition)
- **파인튜닝 + RAG 시너지**: 도메인 지식 내재화 + 실시간 컨텍스트 보강으로 최적의 분석 결과 제공

> **⚠️ 중요: 모델 적용을 위한 코드 수정**
> 모델 다운로드(Pull) 후, 반드시 `analysis_api.py` 파일을 열어 모델명을 수정해야 적용됩니다.
> ```python
> # analysis_api.py 내부
> # 수정 전
> model_name = "hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_M"
> 
> # 수정 후 (파인튜닝 모델 적용 시)
> model_name = "bluejude10/smoothie-qwen3-8b-dtro"
> ```

#### 옵션 B (범용 모델 - 고성능):
```bash
ollama pull hf.co/unsloth/gemma-3n-E4B-it-GGUF:Q4_K_M
```

#### 옵션 C (범용 모델 - 경량):
```bash
ollama pull hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_M
```

> **💡 Tip**: 전력계통 관련 장애 분석에는 **옵션 A (Smoothie-Qwen3-8B-DTRO)** 사용을 강력히 권장합니다. 다른 도메인에 적용 시에는 옵션 B 또는 C를 사용하세요.

### 3. 패키지 설치
```bash
# Python 패키지
pip install -r requirements.txt

# Node.js 패키지
npm install
```

### 4. 실행
```bash
# 원클릭 실행 (권장)
run_all.bat

# 또는 개별 실행
# 1. 분석 API
python analysis_api.py

# 2. 예측 API
python prediction_api.py

# 3. 메인 서버
node server.js
```

브라우저에서 `http://localhost:3000` 접속

---

## 📂 프로젝트 구조

```
통합 플랫폼/
├── index.html                  # 메인 대시보드 UI
├── server.js                   # Node.js 메인 서버
├── analysis_api.py             # 장애 분석 API (FastAPI)
├── prediction_api.py           # 장애 예측 API (Flask)
├── incident_reports.db         # 장애 이력 DB (SQLite)
├── dataset_from_data_txt.json  # 지식 데이터 (JSON)
├── requirements.txt            # Python 의존성
├── package.json                # Node.js 의존성
└── run_all.bat                 # 원클릭 실행 스크립트
```

---

## 🎨 커스터마이징 가이드

**이 플랫폼을 귀사/귀 기관의 시스템으로 쉽게 변환할 수 있습니다!**

### 1. 브랜딩 변경
- **로고**: `io.png` 파일을 귀사 로고로 교체
- **제목**: `index.html`에서 "3호선 전력관제" → "귀사명" 변경
- **색상**: CSS에서 `#8494a4` 등을 귀사 CI 색상으로 변경

### 2. 데이터 교체
- **장애 이력**: `incident_reports.db`를 귀사 데이터로 교체
  - 테이블 스키마: `id`, `fault_type`, `fault_datetime`, `location`, `cause`, `symptom`, `action`, `importance`, `original_text`
- **지식 데이터**: `dataset_from_data_txt.json`을 귀사 매뉴얼로 교체
  - 형식: `[{"category": "...", "question": "...", "answer": "..."}]`

### 3. 시뮬레이터 (보안상 미포함 & 커스터마이징)
- **중요**: 실제 전력 계통 시뮬레이터 소스 코드는 **보안 규정상 본 통합 플랫폼 배포본에는 포함되어 있지 않습니다.**
- **활용 방법 1 (다른 기능으로 사용)**: `index.html`의 버튼을 수정하여 **CCTV 뷰어, 대시보드 링크, 외부 시스템 연동** 등 원하는 다른 기능으로 재정의하여 사용할 수 있습니다.
- **활용 방법 2 (직접 제작 및 연동)**: 
    - `12계통`, `22계통`, `본선 시뮬레이션` 등의 폴더를 생성합니다.
    - SVG 이미지와 JS 코드를 해당 폴더에 배치하고 `index.html`에서 경로를 연결하면 즉시 연동됩니다.
    - 상세 구현 방법은 별도로 배포된 `DTRO-Power-Simulator` (샘플 시뮬레이터) 저장소를 참고하세요.

---

## 💻 최소 하드웨어 사양

이 플랫폼은 저사양 환경에서도 구동되도록 최적화되었습니다.

- **CPU:** Intel Core i3-13100 이상
- **RAM:** 16GB 이상
- **GPU:** 불필요 (CPU만으로 실행 가능)
- **저장공간:** 10GB 이상 (모델 포함)

---

## 🔧 기술 스택

| 구분 | 기술 | 비고 |
| :--- | :--- | :--- |
| **Frontend** | HTML5, Vanilla JS, Bootstrap | 의존성 최소화 |
| **Backend** | Node.js, Flask, FastAPI | 다중 API 서버 |
| **Database** | SQLite | 경량 DB |
| **AI/LLM** | Ollama, Gemma 3 | 로컬 실행 |
| **ML/DL** | PyTorch (GRU), Scikit-learn (KNN) | 저사양 최적화 |
| **RAG** | ChromaDB | 지식 검색 |
| **Simulator** | SVG, JavaScript | 인터랙티브 시각화 |

---

## ⚠️ 데이터 프라이버시 및 샘플 데이터 안내

**중요:** 본 배포 버전은 보안 및 저작권 이슈를 방지하기 위해 실제 전력관제 데이터를 **"가상의 산업용 로봇 팔 장애 데이터"**로 전면 대체하였습니다.

- **장애 데이터 (incident_reports.db)**: 
  - 실제 데이터 대신 **모터 과열, 엔코더 오류, 통신 두절 등 로봇 팔 관련 500건의 가상 장애 이력**이 탑재되어 있습니다.
  - 이를 통해 장애 예측 및 분석 기능을 테스트해 볼 수 있습니다.
  
- **지식 데이터 (dataset_from_data_txt.json)**: 
  - **로봇 팔 유지보수 매뉴얼, 안전 수칙, 문제 해결 가이드 등 가상의 지식 데이터**가 포함되어 있습니다.
  - 지식 검색 및 챗봇 기능을 체험하는 데 사용됩니다.

- **시뮬레이터**: 실제 전력 계통도는 국가 중요 시설 정보로 공개 불가하여, **본 통합 플랫폼 배포본에서는 제외되었습니다.** (별도 배포된 샘플 시뮬레이터 참조)

본 플랫폼을 실무에 적용하려면 위 데이터 파일들을 귀사의 실제 데이터로 교체하시기 바랍니다.

---

## 🚧 알려진 제한사항

1. **Ollama 필수**: 분석 및 챗봇 기능은 Ollama 실행 필요
2. **포트 충돌**: 3000, 8002, 8003 포트 사용 (변경 가능)
3. **브라우저 호환성**: Chrome, Edge 권장 (IE 미지원)

---

## 🔮 향후 계획

- [ ] Docker 컨테이너화
- [ ] 클라우드 배포 지원
- [ ] 모바일 앱 개발
- [ ] 다국어 지원 (영어, 일본어)
- [x] **고성능 LLM 파인튜닝** ✅ (2026-01-11 완료: [Smoothie-Qwen3-8B-DTRO-Edition](https://huggingface.co/bluejude10/Smoothie-Qwen3-8B-DTRO-Edition))

---

## 📜 라이선스

이 프로젝트는 교육 및 연구 목적으로 공개되었습니다.  
포함된 데이터는 가상의 데이터이며, 실제 대구교통공사의 운영 데이터와 무관합니다.

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트들의 도움으로 완성되었습니다:
- [Ollama](https://ollama.com/) - 로컬 LLM 실행 플랫폼
- [Flask](https://flask.palletsprojects.com/) - Python 웹 프레임워크
- [FastAPI](https://fastapi.tiangolo.com/) - 고성능 Python API 프레임워크
- [Node.js](https://nodejs.org/) - JavaScript 런타임
- [Bootstrap](https://getbootstrap.com/) - UI 프레임워크
- [Chart.js](https://www.chartjs.org/) - 차트 라이브러리

그리고 9개월간의 개발 여정을 함께한 **대구교통공사 3호선 경전철관제팀** 동료들께 감사드립니다.

---

**Powered by AI, Built with ❤️**  
**Developed by 강동우 @ DTRO**
