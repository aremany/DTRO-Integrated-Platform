// -*- coding: utf-8 -*-
const express = require('express');
const path = require('path');
const axios = require('axios');
const sqlite3 = require('sqlite3').verbose();

const app = express();
const PORT = 3000;
const { exec } = require('child_process');
const fs = require('fs');

// --- 지식검색 엔진 클래스 --- //
class KnowledgeSearchEngine {
    constructor() {
        this.knowledgeData = [];
        this.questionIndex = new Map(); // Q(질문)만 인덱싱
        this.loadKnowledgeData();
        this.buildQuestionIndex();
    }

    loadKnowledgeData() {
        try {
            const dataPath = path.join(__dirname, 'dataset_from_data_txt.json');
            console.log(`📁 데이터 파일 경로: ${dataPath}`);

            const rawData = fs.readFileSync(dataPath, 'utf8');
            console.log(`📊 파일 크기: ${(rawData.length / 1024 / 1024).toFixed(2)}MB`);

            this.knowledgeData = JSON.parse(rawData);
            console.log(`✅ 지식 데이터 로드 완료: ${this.knowledgeData.length}건`);

            // 첫 번째와 마지막 데이터 확인
            if (this.knowledgeData.length > 0) {
                console.log(`📋 첫 번째 데이터: ${this.knowledgeData[0].instruction?.substring(0, 50)}...`);
                console.log(`📋 마지막 데이터: ${this.knowledgeData[this.knowledgeData.length - 1].instruction?.substring(0, 50)}...`);
            }

            // 까치집 데이터 확인
            const magpieData = this.knowledgeData.find(item =>
                item.instruction && item.instruction.includes('까치집')
            );

            if (magpieData) {
                console.log(`🐦 까치집 데이터 발견: ${magpieData.instruction}`);
            } else {
                console.log(`❌ 까치집 데이터 없음`);
            }

        } catch (error) {
            console.error('❌ 지식 데이터 로드 실패:', error);
            this.knowledgeData = [];
        }
    }

    buildQuestionIndex() {
        console.log('🔍 질문 검색 인덱스 생성 중...');
        this.questionIndex.clear();

        let magpieFound = false;

        this.knowledgeData.forEach((item, index) => {
            const question = item.instruction || '';
            const questionLower = question.toLowerCase();

            // 까치집 데이터 특별 로그
            if (question.includes('까치집')) {
                console.log(`🐦 까치집 데이터 인덱싱 시작:`);
                console.log(`   인덱스: ${index}`);
                console.log(`   질문: "${question}"`);
                console.log(`   소문자: "${questionLower}"`);
                magpieFound = true;
            }

            // 질문에서만 키워드 추출
            const keywords = this.extractKeywords(questionLower);

            // 까치집 관련 키워드 로그
            if (question.includes('까치집')) {
                console.log(`   추출된 키워드: [${keywords.join(', ')}]`);
                console.log(`   까치집 포함 여부: ${keywords.includes('까치집')}`);
            }

            keywords.forEach(keyword => {
                if (!this.questionIndex.has(keyword)) {
                    this.questionIndex.set(keyword, []);
                }

                this.questionIndex.get(keyword).push({
                    index: index,
                    question: question,
                    positions: this.findKeywordPositions(questionLower, keyword)
                });

                // 까치집 키워드 인덱싱 로그
                if (keyword === '까치집') {
                    console.log(`🐦 까치집 키워드 인덱스에 추가됨!`);
                }
            });
        });

        if (!magpieFound) {
            console.log(`❌ 까치집 데이터가 인덱싱 과정에서 발견되지 않음`);
        }

        // 까치집 키워드 인덱스 확인
        const magpieIndex = this.questionIndex.get('까치집');
        if (magpieIndex && magpieIndex.length > 0) {
            console.log(`🐦 까치집 키워드 인덱스: ${magpieIndex.length}건`);
            console.log(`   첫 번째 항목: ${magpieIndex[0].question}`);
        } else {
            console.log(`❌ 까치집 키워드 인덱스 없음`);
            console.log(`   전체 키워드 수: ${this.questionIndex.size}`);
            console.log(`   샘플 키워드: [${Array.from(this.questionIndex.keys()).slice(0, 5).join(', ')}]`);
        }

        console.log(`✅ 질문 검색 인덱스 생성 완료: ${this.questionIndex.size}개 키워드`);
    }

    extractKeywords(text) {
        // 한글, 영문, 숫자 추출 (2글자 이상)
        const words = text.match(/[가-힣a-zA-Z0-9]+/g) || [];
        const filteredWords = words.filter(word => word.length >= 2);

        // 디버깅: 까치집 관련 로그
        if (text.includes('까치집')) {
            console.log(`🐦 까치집 텍스트 분석:`);
            console.log(`   원본: "${text}"`);
            console.log(`   매치된 단어들: [${words.join(', ')}]`);
            console.log(`   필터된 단어들: [${filteredWords.join(', ')}]`);
        }

        return [...new Set(filteredWords)];
    }

    findKeywordPositions(text, keyword) {
        const positions = [];
        let index = text.indexOf(keyword);
        while (index !== -1) {
            positions.push(index);
            index = text.indexOf(keyword, index + 1);
        }
        return positions;
    }

    searchWithAND(query, page = 1, limit = 10) {
        if (!query || query.trim().length === 0) {
            return { results: [], total: 0, page, limit, searchType: 'empty' };
        }

        // 검색 키워드 추출 (AND 조건용)
        const searchKeywords = this.extractKeywords(query.toLowerCase());

        if (searchKeywords.length === 0) {
            return { results: [], total: 0, page, limit, searchType: 'no_keywords' };
        }

        console.log(`🔍 AND 검색 시작: [${searchKeywords.join(', ')}]`);
        console.log(`📝 원본 쿼리: "${query}"`);

        // 전체 데이터에서 부분 문자열 매칭으로 후보 추출
        let candidates = [];

        this.knowledgeData.forEach((item, index) => {
            const question = item.instruction || '';
            const questionLower = question.toLowerCase();

            // 모든 키워드가 부분 문자열로 포함되는지 확인 (AND 조건)
            const allKeywordsMatch = searchKeywords.every(keyword => {
                return questionLower.includes(keyword);
            });

            if (allKeywordsMatch) {
                candidates.push({
                    index: index,
                    question: question,
                    positions: this.getAllKeywordPositions(questionLower, searchKeywords)
                });
            }
        });

        console.log(`📋 부분 매칭으로 찾은 후보: ${candidates.length}건`);

        // 관련도 점수 계산 및 정렬
        const scoredResults = candidates.map(candidate => {
            const data = this.knowledgeData[candidate.index];
            const question = data.instruction || '';

            return {
                index: candidate.index,
                question: question,
                answer: data.output || '',
                relevanceScore: this.calculateANDRelevance(question, searchKeywords),
                matchedKeywords: searchKeywords,
                keywordPositions: candidate.positions
            };
        }).sort((a, b) => b.relevanceScore - a.relevanceScore);

        // 페이징
        const total = scoredResults.length;
        const startIndex = (page - 1) * limit;
        const endIndex = startIndex + limit;
        const paginatedResults = scoredResults.slice(startIndex, endIndex);

        // 결과 포맷팅
        const formattedResults = paginatedResults.map((result, resultIndex) => ({
            id: result.index,
            rank: startIndex + resultIndex + 1,
            question: result.question,
            answer: result.answer,
            question_preview: this.truncateText(result.question, 80),
            answer_preview: this.truncateText(result.answer, 120),
            relevance_score: result.relevanceScore,
            matched_keywords: result.matchedKeywords,
            highlighted_question: this.highlightKeywords(result.question, searchKeywords),
            highlighted_answer_preview: this.highlightKeywords(
                this.truncateText(result.answer, 120),
                searchKeywords
            ),
            keyword_count: searchKeywords.length,
            search_type: 'AND_PARTIAL'
        }));

        return {
            results: formattedResults,
            total: total,
            page: page,
            limit: limit,
            query: query,
            keywords: searchKeywords,
            searchType: 'AND_PARTIAL',
            searchStats: {
                totalKeywords: searchKeywords.length,
                candidatesAfterPartialMatch: total,
                finalResults: total
            }
        };
    }

    calculateANDRelevance(question, keywords) {
        const questionLower = question.toLowerCase();
        let score = 0;

        keywords.forEach(keyword => {
            // 키워드가 질문에 포함된 횟수
            const occurrences = (questionLower.match(new RegExp(keyword, 'g')) || []).length;
            score += occurrences * 10;

            // 질문 시작 부분에 있으면 가산점
            if (questionLower.indexOf(keyword) < 20) {
                score += 5;
            }

            // 키워드 길이에 따른 가산점 (긴 키워드일수록 더 구체적)
            score += keyword.length;
        });

        // 질문 길이 대비 키워드 밀도
        const keywordDensity = keywords.reduce((sum, kw) => sum + kw.length, 0) / question.length;
        score += keywordDensity * 100;

        return Math.round(score);
    }

    getAllKeywordPositions(text, keywords) {
        const positions = {};
        keywords.forEach(keyword => {
            positions[keyword] = this.findKeywordPositions(text, keyword);
        });
        return positions;
    }

    highlightKeywords(text, keywords) {
        if (!text || !keywords || keywords.length === 0) return text;

        let highlighted = text;

        // 키워드를 길이 순으로 정렬 (긴 것부터 처리하여 중복 하이라이팅 방지)
        const sortedKeywords = keywords.sort((a, b) => b.length - a.length);

        sortedKeywords.forEach(keyword => {
            const regex = new RegExp(`(${this.escapeRegExp(keyword)})`, 'gi');
            highlighted = highlighted.replace(regex, '<mark class="keyword-highlight">$1</mark>');
        });

        return highlighted;
    }

    escapeRegExp(string) {
        return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    truncateText(text, maxLength) {
        if (!text || text.length <= maxLength) return text;
        return text.substring(0, maxLength) + '...';
    }

    // 검색 통계 정보
    getSearchStats() {
        return {
            totalQuestions: this.knowledgeData.length,
            totalKeywords: this.questionIndex.size,
            indexSize: this.questionIndex.size,
            averageKeywordsPerQuestion: this.calculateAverageKeywords()
        };
    }

    calculateAverageKeywords() {
        if (this.knowledgeData.length === 0) return 0;

        const totalKeywords = this.knowledgeData.reduce((sum, item) => {
            const keywords = this.extractKeywords((item.instruction || '').toLowerCase());
            return sum + keywords.length;
        }, 0);

        return Math.round(totalKeywords / this.knowledgeData.length * 10) / 10;
    }
}

// 전역 검색 엔진 인스턴스
const knowledgeSearch = new KnowledgeSearchEngine();

// --- 설정 --- //
const SQLITE_DB_PATH = path.join(__dirname, 'incident_reports.db');
const ANALYSIS_API_URL = 'http://localhost:8000';
const PREDICTION_API_URL = 'http://localhost:8002';

// --- AI 설정 --- //
const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL || 'http://localhost:11434';
const ANALYSIS_AI_MODEL = process.env.ANALYSIS_AI_MODEL || 'hf.co/unsloth/gemma-3n-E2B-it-GGUF:Q4_K_M';

// --- 미들웨어 --- //
app.use(express.json()); // JSON 요청 본문 파싱
// Simple CORS allow for local usage
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', '*');
    res.setHeader('Access-Control-Allow-Methods', 'GET,POST,OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
});
// 정적 파일 서빙 설정
app.use(express.static(path.join(__dirname, 'public'))); // 정적 파일 서빙 (index.html 등)
app.use('/ci.jpg', express.static(path.join(__dirname, 'ci.jpg'))); // CI 로고 파일 서빙
app.use('/io.png', express.static(path.join(__dirname, 'io.png'))); // io.gif 로고 파일 서빙

// --- API 게이트웨이 엔드포인트 --- //

// 1. 메인 페이지 서빙
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

// 2. 장애 데이터 조회 (SQL DB에서 직접)
app.get('/api/faults', (req, res) => {
    const db = new sqlite3.Database(SQLITE_DB_PATH, sqlite3.OPEN_READONLY, (err) => {
        if (err) {
            console.error('DB 연결 오류:', err.message);
            return res.status(500).json({ error: '데이터베이스 연결 오류' });
        }
    });

    // 실제 컬럼명 사용
    db.all("SELECT `순번` as id, `장애명` as fault_type, `장애일시` as fault_datetime, `장애 장소` as location, `장애 원인` as cause, `장애 발생 시 현상` as phenomenon, `장애 발생 시 조치 방법` as action_taken, `원본_추출텍스트` as raw_text, `risk_level`, `detected_keywords_json` FROM incident_data", [], (err, rows) => {
        if (err) {
            console.error('DB 쿼리 오류:', err.message);
            return res.status(500).json({ error: '데이터 조회 오류' });
        }
        res.json(rows);
    });

    db.close();
});

// 3. AI 브리핑 (새로운 SQL 기반)
app.post('/api/briefing', async (req, res) => {
    try {
        const { keyword, faultType, year } = req.body;
        console.log('Received faultType:', faultType);

        // 새로운 SQL 기반 브리핑 API 호출
        const analysisResponse = await axios.post(`${ANALYSIS_API_URL}/sql_based_briefing`, {
            fault_type: faultType || "전체",
            keyword: keyword,
            year: year ? parseInt(year) : null
        });

        const analysisData = analysisResponse.data; // Renamed to avoid conflict

        let predictionData = null;
        // Only attempt prediction if a specific faultType is selected (not "전체")
        if (faultType && faultType !== "전체") {
            try {
                const predictionResponse = await axios.post(`${PREDICTION_API_URL}/predict`, {
                    fault_type: faultType,
                    target_year: year ? parseInt(year) : (new Date().getFullYear() + 1) // Predict for current or next year
                }, { headers: { 'Content-Type': 'application/json; charset=UTF-8' } });
                if (predictionResponse.data && predictionResponse.data.predictions) {
                    predictionData = predictionResponse.data.predictions;
                }
            } catch (predictionError) {
                console.warn(`AI 브리핑: 장애 유형 '${faultType}'에 대한 예측 데이터 로드 실패: ${predictionError.message}`);
                // Continue without prediction data if there's an error
            }
        }

        // 구조화된 브리핑 텍스트 생성 (using analysisData)
        let briefingText = `📊 **${analysisData.fault_type} 장애 분석 브리핑**\n\n`;
        briefingText += `🔢 **발생건수**: 총 ${analysisData.total_incidents}건 (연평균 ${analysisData.yearly_frequency.toFixed(1)}건)\n`;
        briefingText += `⚠️ **중요도**: ${analysisData.importance_level}\n\n`;

        if (analysisData.top_causes.length > 0) {
            briefingText += `🔍 **주요 원인들**:\n`;
            analysisData.top_causes.forEach(cause => {
                briefingText += `• ${cause.text} (${cause.count}건)\n`;
            });
            briefingText += `\n`;
        }

        if (analysisData.top_actions.length > 0) {
            briefingText += `🛠️ **조치방법들**:\n`;
            analysisData.top_actions.forEach(action => {
                briefingText += `• ${action.text} (${action.count}건)\n`;
            });
            briefingText += `\n`;
        }

        if (analysisData.ai_recommendation && analysisData.ai_recommendation !== "AI 서비스를 사용할 수 없거나 충분한 데이터가 없습니다.") {
            briefingText += `🤖 **AI 추천 조치방법**:\n${analysisData.ai_recommendation}`;
        }

        res.json({
            ai_summary: briefingText,
            statistics: {
                total_incidents: analysisData.total_incidents,
                importance_level: analysisData.importance_level,
                yearly_frequency: analysisData.yearly_frequency,
                causes_count: analysisData.top_causes.length,
                actions_count: analysisData.top_actions.length,
                mode: analysisData.mode
            },
            prediction_trend: predictionData // Add prediction data here
        });

    } catch (error) {
        console.error('AI 브리핑 API 호출 오류:', error.message);
        res.status(500).json({
            error: 'AI 브리핑 데이터 로드 실패',
            ai_summary: '브리핑 데이터를 불러올 수 없습니다.',
            statistics: {
                total_incidents: 0,
                importance_level: '정보없음',
                yearly_frequency: 0,
                causes_count: 0,
                actions_count: 0,
                mode: 'error'
            },
            prediction_trend: null // Ensure prediction_trend is null on error
        });
    }
});

// 4. 상세 분석 (Analysis API 및 Prediction API 호출)
// 장애 상세 내용 조회 (원본_추출텍스트)
app.post('/api/fault_detail', async (req, res) => {
    try {
        const { fault_id } = req.body;

        const db = new sqlite3.Database(SQLITE_DB_PATH, sqlite3.OPEN_READONLY);
        const faultDetail = await new Promise((resolve, reject) => {
            db.get(
                "SELECT `순번`, `장애명`, `장애일시`, `장애 내용`, `장애 발생 시 현상`, `장애 발생 시 조치 방법`, `장애 장소`, `장애 원인`, `원본_추출텍스트` FROM incident_data WHERE `순번` = ?",
                [fault_id],
                (err, row) => {
                    if (err) reject(err);
                    else resolve(row);
                }
            );
        });
        db.close();

        if (!faultDetail) {
            return res.status(404).json({ error: '해당 장애 정보를 찾을 수 없습니다.' });
        }

        res.json({
            id: faultDetail['순번'],
            fault_type: faultDetail['장애명'],
            fault_datetime: faultDetail['장애일시'],
            fault_content: faultDetail['장애 내용'],
            fault_symptom: faultDetail['장애 발생 시 현상'],
            fault_action: faultDetail['장애 발생 시 조치 방법'],
            fault_location: faultDetail['장애 장소'],
            fault_cause: faultDetail['장애 원인'],
            original_text: faultDetail['원본_추출텍스트']
        });

    } catch (error) {
        console.error('장애 상세 정보 조회 오류:', error.message);
        res.status(500).json({ error: '장애 상세 정보 로드 실패' });
    }
});

app.post('/api/detailed_analysis', async (req, res) => {
    try {
        const { fault_id, fault_type } = req.body;

        // Analysis API 호출 (새로운 SQL 기반 브리핑)
        const analysisResponse = await axios.post(`${ANALYSIS_API_URL}/sql_based_briefing`, {
            fault_type: fault_type
        });

        // Prediction API 호출
        const predictionResponse = await axios.post(`${PREDICTION_API_URL}/predict`, {
            fault_type: fault_type
        }, { headers: { 'Content-Type': 'application/json; charset=UTF-8' } });

        // 유사 사례는 SQL DB에서 같은 장애명으로 검색
        const db = new sqlite3.Database(SQLITE_DB_PATH, sqlite3.OPEN_READONLY);
        const similarCases = await new Promise((resolve, reject) => {
            db.all(
                "SELECT `순번` as id, `장애명` as fault_type, `장애일시` as fault_datetime, `장애 원인` as cause, `장애 발생 시 조치 방법` as action FROM incident_data WHERE `장애명` = ? AND `순번` != ? LIMIT 3",
                [fault_type, fault_id],
                (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows.map(row => ({
                        id: row.id,
                        score: 0.95, // 같은 장애명이므로 높은 유사도
                        payload: {
                            text_chunk: `원인: ${row.cause || '정보없음'}, 조치: ${row.action || '정보없음'}`,
                            source_type: 'SQL_DB',
                            file_name: `장애보고서_${row.id}`,
                            fault_type: row.fault_type,
                            fault_datetime: row.fault_datetime
                        }
                    })));
                }
            );
        });
        db.close();

        res.json({
            qualitative_analysis: {
                summary: analysisResponse.data.ai_recommendation || `${fault_type} 장애에 대한 분석 결과입니다.`,
                sources: [{ content: `총 ${analysisResponse.data.total_incidents}건 발생, 중요도: ${analysisResponse.data.importance_level}` }],
                mode: analysisResponse.data.mode
            },
            predictions: predictionResponse.data.predictions,
            similar_cases: similarCases
        });

    } catch (error) {
        console.error('상세 분석 API 호출 오류:', error.message);
        res.status(500).json({ error: '상세 분석 데이터 로드 실패' });
    }
});



// 6. 사용자 피드백 (로그 기록)
app.post('/api/feedback', (req, res) => {
    const { type, data } = req.body; // type: 'like'/'dislike', data: 분석 결과 ID 등
    console.log(`[FEEDBACK] Type: ${type}, Data: ${JSON.stringify(data)}`);
    // 실제 구현에서는 파일에 저장하거나 별도 DB에 저장
    res.json({ status: 'success', message: '피드백이 기록되었습니다.' });
});

// 7. 종합 예측 데이터 조회 (Prediction API 호출 및 집계)
app.get('/api/overall_predictions', async (req, res) => {
    const selectedFaultType = (req.query && req.query.fault_type) ? decodeURIComponent(String(req.query.fault_type).trim()) : '';
    const keyword = (req.query && req.query.keyword) ? decodeURIComponent(String(req.query.keyword).trim()) : '';
    const year = (req.query && req.query.year) ? String(req.query.year).trim() : '';
    const db = new sqlite3.Database(SQLITE_DB_PATH, sqlite3.OPEN_READONLY, (err) => {
        if (err) {
            console.error('DB 연결 오류:', err.message);
            return res.status(500).json({ error: '데이터베이스 연결 오류' });
        }
    });

    try {
        // 모든 고유한 장애 유형 조회
        let faultTypes = [];
        if (selectedFaultType) {
            faultTypes = [selectedFaultType];
        } else {
            // 모든 장애 유형 조회 (연도 필터 제거 - 예측에만 연도 사용)
            const whereClauses = ["`장애명` IS NOT NULL AND `장애명` != ''"];
            const params = [];

            // keyword 필터는 유지 (장애 유형 검색용)
            if (keyword) {
                const like = `%${keyword}%`;
                whereClauses.push("( `장애명` LIKE ? OR `장애 장소` LIKE ? OR `장애 원인` LIKE ? OR `장애 발생 시 현상` LIKE ? OR `장애 발생 시 조치 방법` LIKE ? OR `원본_추출텍스트` LIKE ? )");
                params.push(like, like, like, like, like, like);
            }

            const sql = `SELECT DISTINCT \`장애명\` FROM incident_data WHERE ${whereClauses.join(' AND ')}`;
            faultTypes = await new Promise((resolve, reject) => {
                db.all(sql, params, (err, rows) => {
                    if (err) reject(err);
                    else resolve(rows.map(row => row['장애명'].trim()));
                });
            });
        }

        const allPredictions = [];
        for (const faultType of faultTypes) {
            try {
                const body = { fault_type: faultType };
                if (year) {
                    const y = parseInt(year, 10);
                    if (!Number.isNaN(y)) body.target_year = y;
                }
                const predictionResponse = await axios.post(`${PREDICTION_API_URL}/predict`, body, { headers: { 'Content-Type': 'application/json; charset=UTF-8' } });
                if (predictionResponse.data && predictionResponse.data.predictions) {
                    const usedModel = predictionResponse.data.used_model || 'N/A';
                    const preds = predictionResponse.data.predictions;
                    const annualSum = preds.reduce((acc, p) => acc + (p.predicted_count || 0), 0);
                    const base = {
                        fault_type: faultType,
                        year: (year ? parseInt(year, 10) : (new Date().getFullYear() + 1)),
                        total_predicted_count: annualSum,
                        used_model: usedModel
                    };
                    // 항상 월별 예측 포함 (프론트에서 스택 막대 렌더링에 사용)
                    base.monthly = preds; // [{month, predicted_count}]
                    allPredictions.push(base);
                }
            } catch (predictionError) {
                console.warn(`장애 유형 '${faultType}'에 대한 예측 데이터 로드 실패: ${predictionError.message}`);
                // 예측 실패 시에도 장애 유형을 포함하여 전체 목록 표시
                const base = {
                    fault_type: faultType,
                    year: (year ? parseInt(year, 10) : (new Date().getFullYear() + 1)),
                    total_predicted_count: 0,
                    used_model: '예측 불가'
                };
                base.monthly = []; // 빈 월별 예측
                allPredictions.push(base);
            }
        }
        res.json(allPredictions);

    } catch (error) {
        console.error('종합 예측 API 호출 오류:', error.message);
        res.status(500).json({ error: '종합 예측 데이터 로드 실패' });
    } finally {
        db.close();
    }
});

// --- 지식검색 API 엔드포인트 --- //

// 지식 검색 API (AND 조건)
app.post('/api/knowledge/search', (req, res) => {
    try {
        const { query, page = 1, limit = 10 } = req.body;

        if (!query || typeof query !== 'string') {
            return res.status(400).json({
                error: '검색어가 필요합니다.',
                code: 'INVALID_QUERY'
            });
        }

        const startTime = Date.now();
        const results = knowledgeSearch.searchWithAND(query.trim(), parseInt(page), parseInt(limit));
        const searchTime = Date.now() - startTime;

        res.json({
            success: true,
            data: {
                ...results,
                searchTime: `${searchTime}ms`,
                searchStrategy: 'AND',
                searchTarget: 'questions_only'
            },
            timestamp: new Date().toISOString()
        });

    } catch (error) {
        console.error('지식 검색 오류:', error);
        res.status(500).json({
            error: '검색 중 오류가 발생했습니다.',
            code: 'SEARCH_ERROR',
            details: error.message
        });
    }
});

// 지식 상세 정보 API
app.get('/api/knowledge/detail/:id', (req, res) => {
    try {
        const id = parseInt(req.params.id);

        if (isNaN(id) || id < 0 || id >= knowledgeSearch.knowledgeData.length) {
            return res.status(404).json({
                error: '해당 지식을 찾을 수 없습니다.',
                code: 'NOT_FOUND'
            });
        }

        const data = knowledgeSearch.knowledgeData[id];

        res.json({
            success: true,
            data: {
                id: id,
                question: data.instruction || '',
                answer: data.output || '',
                formatted_answer: formatKnowledgeAnswer(data.output || ''),
                category: extractCategory(data.instruction || ''),
                keywords: knowledgeSearch.extractKeywords(data.instruction || ''),
                created_at: new Date().toISOString()
            }
        });

    } catch (error) {
        console.error('지식 상세 조회 오류:', error);
        res.status(500).json({
            error: '상세 정보 조회 중 오류가 발생했습니다.',
            code: 'DETAIL_ERROR'
        });
    }
});

// 검색 통계 API
app.get('/api/knowledge/stats', (req, res) => {
    try {
        const stats = knowledgeSearch.getSearchStats();

        // 까치집 데이터 확인
        const magpieData = knowledgeSearch.knowledgeData.find(item =>
            item.instruction && item.instruction.includes('까치집')
        );

        // 까치집 키워드 인덱스 확인
        const magpieIndex = knowledgeSearch.questionIndex.get('까치집');

        res.json({
            success: true,
            data: {
                ...stats,
                magpieDataExists: !!magpieData,
                magpieQuestion: magpieData ? magpieData.instruction : null,
                magpieIndexCount: magpieIndex ? magpieIndex.length : 0,
                sampleKeywords: Array.from(knowledgeSearch.questionIndex.keys()).slice(0, 10)
            },
            timestamp: new Date().toISOString()
        });
    } catch (error) {
        console.error('통계 조회 오류:', error);
        res.status(500).json({
            error: '통계 조회 중 오류가 발생했습니다.',
            code: 'STATS_ERROR'
        });
    }
});

// 디버그용 까치집 검색 API
app.get('/api/knowledge/debug/magpie', (req, res) => {
    try {
        const magpieData = knowledgeSearch.knowledgeData.filter(item =>
            item.instruction && item.instruction.includes('까치집')
        );

        const keywords = knowledgeSearch.extractKeywords('까치집');
        const magpieIndex = knowledgeSearch.questionIndex.get('까치집');

        res.json({
            success: true,
            data: {
                totalData: knowledgeSearch.knowledgeData.length,
                magpieDataCount: magpieData.length,
                magpieData: magpieData,
                keywords: keywords,
                magpieIndex: magpieIndex,
                searchResult: knowledgeSearch.searchWithAND('까치집', 1, 10)
            }
        });
    } catch (error) {
        console.error('까치집 디버그 오류:', error);
        res.status(500).json({
            error: '디버그 중 오류가 발생했습니다.',
            code: 'DEBUG_ERROR'
        });
    }
});

// 챗봇 서버 상태 확인 API
app.get('/api/chatbot/status', async (req, res) => {
    try {
        const response = await axios.get('http://localhost:5000/', { timeout: 2000 });
        res.json({
            status: 'running',
            available: true,
            url: 'http://localhost:5000'
        });
    } catch (error) {
        res.json({
            status: 'stopped',
            available: false,
            message: '챗봇 서버가 실행되지 않았습니다.'
        });
    }
});

// --- 지식 데이터셋 관리 --- //
const KNOWLEDGE_DATASET_PATH = path.join(__dirname, 'dataset_from_data_txt.json');
const KNOWLEDGE_BACKUP_DIR = path.join(__dirname, 'knowledge_backups');
const KNOWLEDGE_LOG_PATH = path.join(__dirname, 'knowledge_edit_log.json');

// 백업 디렉토리 생성
if (!fs.existsSync(KNOWLEDGE_BACKUP_DIR)) {
    fs.mkdirSync(KNOWLEDGE_BACKUP_DIR, { recursive: true });
}

// 데이터셋 로드 함수
function loadKnowledgeDataset() {
    try {
        if (!fs.existsSync(KNOWLEDGE_DATASET_PATH)) {
            console.log('지식 데이터셋 파일이 없어 새로 생성합니다.');
            return [];
        }
        const data = fs.readFileSync(KNOWLEDGE_DATASET_PATH, 'utf8');
        return JSON.parse(data);
    } catch (error) {
        console.error('데이터셋 로드 실패:', error);
        return [];
    }
}

// 데이터셋 저장 함수
function saveKnowledgeDataset(data) {
    try {
        fs.writeFileSync(KNOWLEDGE_DATASET_PATH, JSON.stringify(data, null, 2), 'utf8');
        return true;
    } catch (error) {
        console.error('데이터셋 저장 실패:', error);
        return false;
    }
}

// 자동 백업 생성
function createKnowledgeBackup(reason = 'auto') {
    try {
        const data = loadKnowledgeDataset();
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `dataset_backup_${timestamp}_${reason}.json`;
        const filepath = path.join(KNOWLEDGE_BACKUP_DIR, filename);

        fs.writeFileSync(filepath, JSON.stringify(data, null, 2), 'utf8');
        console.log(`백업 생성: ${filename}`);
        return filename;
    } catch (error) {
        console.error('백업 생성 실패:', error);
        return null;
    }
}

// 변경 로그 기록
function logKnowledgeChange(action, details) {
    try {
        let logs = [];
        if (fs.existsSync(KNOWLEDGE_LOG_PATH)) {
            logs = JSON.parse(fs.readFileSync(KNOWLEDGE_LOG_PATH, 'utf8'));
        }

        logs.push({
            timestamp: new Date().toISOString(),
            action: action,
            details: details,
            user: 'system' // 추후 사용자 인증 추가 시 변경
        });

        // 최근 1000개 로그만 유지
        if (logs.length > 1000) {
            logs = logs.slice(-1000);
        }

        fs.writeFileSync(KNOWLEDGE_LOG_PATH, JSON.stringify(logs, null, 2), 'utf8');
    } catch (error) {
        console.error('로그 기록 실패:', error);
    }
}

// 유사 질문 체크 함수
function checkSimilarQuestions(newQuestion, excludeIndex = null) {
    const dataset = loadKnowledgeDataset();
    const similar = [];

    dataset.forEach((item, index) => {
        if (excludeIndex !== null && index === excludeIndex) return;

        const question = item.instruction || '';
        const similarity = calculateQuestionSimilarity(newQuestion, question);

        if (similarity > 0.6) { // 60% 이상 유사
            similar.push({
                index: index,
                question: question,
                similarity: similarity
            });
        }
    });

    return similar.sort((a, b) => b.similarity - a.similarity).slice(0, 5);
}

// 질문 유사도 계산 (간단한 Jaccard 유사도)
function calculateQuestionSimilarity(q1, q2) {
    const words1 = new Set(q1.toLowerCase().split(/\s+/).filter(w => w.length > 1));
    const words2 = new Set(q2.toLowerCase().split(/\s+/).filter(w => w.length > 1));

    const intersection = new Set([...words1].filter(x => words2.has(x)));
    const union = new Set([...words1, ...words2]);

    return intersection.size / union.size;
}

// --- 지식 편집 API --- //

// 지식 항목 추가 API
app.post('/api/knowledge/add', async (req, res) => {
    try {
        const { question, answer, force = false } = req.body;

        if (!question || !answer) {
            return res.status(400).json({
                error: '질문과 답변이 모두 필요합니다.',
                code: 'MISSING_FIELDS'
            });
        }

        // 유사 질문 체크
        const similar = checkSimilarQuestions(question);
        if (similar.length > 0 && !force) {
            return res.status(409).json({
                error: '유사한 질문이 이미 존재합니다.',
                code: 'SIMILAR_EXISTS',
                similar: similar
            });
        }

        // 데이터셋에 추가
        const dataset = loadKnowledgeDataset();
        const newItem = {
            instruction: question.trim(),
            output: answer.trim(),
            id: Date.now(),
            timestamp: new Date().toISOString(),
            created_by: 'system'
        };

        dataset.push(newItem);

        // 백업 생성
        createKnowledgeBackup('add');

        // 저장
        if (!saveKnowledgeDataset(dataset)) {
            return res.status(500).json({
                error: '데이터 저장 실패',
                code: 'SAVE_FAILED'
            });
        }

        // 로그 기록
        logKnowledgeChange('add', {
            question: question.trim(),
            answer: answer.trim(),
            itemId: newItem.id
        });

        // 검색 엔진 리로드 (선택적)
        if (knowledgeSearch) {
            knowledgeSearch.loadKnowledgeData();
            knowledgeSearch.buildQuestionIndex();
        }

        res.json({
            success: true,
            message: '지식이 성공적으로 추가되었습니다.',
            data: newItem
        });

    } catch (error) {
        console.error('지식 추가 오류:', error);
        res.status(500).json({
            error: '지식 추가 중 오류가 발생했습니다.',
            code: 'ADD_ERROR',
            details: error.message
        });
    }
});

// 지식 항목 수정 API
app.put('/api/knowledge/edit/:id', async (req, res) => {
    try {
        const id = parseInt(req.params.id);
        const { question, answer } = req.body;

        if (isNaN(id) || id < 0) {
            return res.status(400).json({
                error: '유효하지 않은 ID입니다.',
                code: 'INVALID_ID'
            });
        }

        if (!question || !answer) {
            return res.status(400).json({
                error: '질문과 답변이 모두 필요합니다.',
                code: 'MISSING_FIELDS'
            });
        }

        const dataset = loadKnowledgeDataset();

        if (id >= dataset.length) {
            return res.status(404).json({
                error: '해당 지식을 찾을 수 없습니다.',
                code: 'NOT_FOUND'
            });
        }

        const oldItem = { ...dataset[id] };
        const newItem = {
            ...dataset[id],
            instruction: question.trim(),
            output: answer.trim(),
            updated_at: new Date().toISOString(),
            updated_by: 'system'
        };

        dataset[id] = newItem;

        // 백업 생성
        createKnowledgeBackup('edit');

        // 저장
        if (!saveKnowledgeDataset(dataset)) {
            return res.status(500).json({
                error: '데이터 저장 실패',
                code: 'SAVE_FAILED'
            });
        }

        // 로그 기록
        logKnowledgeChange('edit', {
            itemId: id,
            oldQuestion: oldItem.instruction,
            newQuestion: question.trim(),
            oldAnswer: oldItem.output,
            newAnswer: answer.trim()
        });

        // 검색 엔진 리로드
        if (knowledgeSearch) {
            knowledgeSearch.loadKnowledgeData();
            knowledgeSearch.buildQuestionIndex();
        }

        res.json({
            success: true,
            message: '지식이 성공적으로 수정되었습니다.',
            data: newItem
        });

    } catch (error) {
        console.error('지식 수정 오류:', error);
        res.status(500).json({
            error: '지식 수정 중 오류가 발생했습니다.',
            code: 'EDIT_ERROR',
            details: error.message
        });
    }
});

// 지식 항목 삭제 API
app.delete('/api/knowledge/delete/:id', async (req, res) => {
    try {
        const id = parseInt(req.params.id);

        if (isNaN(id) || id < 0) {
            return res.status(400).json({
                error: '유효하지 않은 ID입니다.',
                code: 'INVALID_ID'
            });
        }

        const dataset = loadKnowledgeDataset();

        if (id >= dataset.length) {
            return res.status(404).json({
                error: '해당 지식을 찾을 수 없습니다.',
                code: 'NOT_FOUND'
            });
        }

        const deletedItem = dataset.splice(id, 1)[0];

        // 백업 생성
        createKnowledgeBackup('delete');

        // 저장
        if (!saveKnowledgeDataset(dataset)) {
            return res.status(500).json({
                error: '데이터 저장 실패',
                code: 'SAVE_FAILED'
            });
        }

        // 로그 기록
        logKnowledgeChange('delete', {
            itemId: id,
            question: deletedItem.instruction,
            answer: deletedItem.output
        });

        // 검색 엔진 리로드
        if (knowledgeSearch) {
            knowledgeSearch.loadKnowledgeData();
            knowledgeSearch.buildQuestionIndex();
        }

        res.json({
            success: true,
            message: '지식이 성공적으로 삭제되었습니다.',
            data: deletedItem
        });

    } catch (error) {
        console.error('지식 삭제 오류:', error);
        res.status(500).json({
            error: '지식 삭제 중 오류가 발생했습니다.',
            code: 'DELETE_ERROR',
            details: error.message
        });
    }
});

// 유사 질문 체크 API
app.post('/api/knowledge/check-similar', (req, res) => {
    try {
        const { question, excludeId } = req.body;

        if (!question) {
            return res.status(400).json({
                error: '질문이 필요합니다.',
                code: 'MISSING_QUESTION'
            });
        }

        const similar = checkSimilarQuestions(question, excludeId);

        res.json({
            success: true,
            similar: similar
        });

    } catch (error) {
        console.error('유사 질문 체크 오류:', error);
        res.status(500).json({
            error: '유사 질문 체크 중 오류가 발생했습니다.',
            code: 'SIMILAR_CHECK_ERROR',
            details: error.message
        });
    }
});

// 백업 관리 API
app.get('/api/knowledge/backups', (req, res) => {
    try {
        if (!fs.existsSync(KNOWLEDGE_BACKUP_DIR)) {
            return res.json({ success: true, backups: [] });
        }

        const files = fs.readdirSync(KNOWLEDGE_BACKUP_DIR)
            .filter(file => file.startsWith('dataset_backup_'))
            .sort()
            .reverse()
            .slice(0, 10); // 최근 10개만

        const backups = files.map(file => {
            const match = file.match(/dataset_backup_(.+)\.json/);
            return {
                filename: file,
                timestamp: match ? match[1] : file,
                path: file
            };
        });

        res.json({
            success: true,
            backups: backups
        });

    } catch (error) {
        console.error('백업 목록 조회 오류:', error);
        res.status(500).json({
            error: '백업 목록 조회 중 오류가 발생했습니다.',
            code: 'BACKUP_LIST_ERROR'
        });
    }
});

// 백업 복원 API
app.post('/api/knowledge/restore/:timestamp', async (req, res) => {
    try {
        const timestamp = req.params.timestamp;
        const backupPath = path.join(KNOWLEDGE_BACKUP_DIR, `dataset_backup_${timestamp}.json`);

        if (!fs.existsSync(backupPath)) {
            return res.status(404).json({
                error: '백업 파일을 찾을 수 없습니다.',
                code: 'BACKUP_NOT_FOUND'
            });
        }

        const backupData = JSON.parse(fs.readFileSync(backupPath, 'utf8'));

        // 현재 데이터 백업
        createKnowledgeBackup('before_restore');

        // 복원
        if (!saveKnowledgeDataset(backupData)) {
            return res.status(500).json({
                error: '데이터 복원 실패',
                code: 'RESTORE_FAILED'
            });
        }

        // 로그 기록
        logKnowledgeChange('restore', {
            timestamp: timestamp,
            itemCount: backupData.length
        });

        // 검색 엔진 리로드
        if (knowledgeSearch) {
            knowledgeSearch.loadKnowledgeData();
            knowledgeSearch.buildQuestionIndex();
        }

        res.json({
            success: true,
            message: '백업이 성공적으로 복원되었습니다.',
            data: {
                timestamp: timestamp,
                itemCount: backupData.length
            }
        });

    } catch (error) {
        console.error('백업 복원 오류:', error);
        res.status(500).json({
            error: '백업 복원 중 오류가 발생했습니다.',
            code: 'RESTORE_ERROR',
            details: error.message
        });
    }
});

// AI 브리핑 생성 API
app.post('/api/knowledge/generate-briefing', async (req, res) => {
    try {
        const { query, searchResults, selectedIds } = req.body;

        if (!query || !searchResults) {
            return res.status(400).json({
                error: '쿼리와 검색 결과가 필요합니다.',
                code: 'MISSING_PARAMS'
            });
        }

        // selectedIds가 있으면 해당 결과만 필터링
        let resultsToProcess = searchResults;
        if (selectedIds && Array.isArray(selectedIds) && selectedIds.length > 0) {
            resultsToProcess = searchResults.filter(r => selectedIds.includes(r.id));
            console.log(`📌 선택된 항목만 처리: ${resultsToProcess.length}개 / ${searchResults.length}개`);
        } else {
            console.log(`📌 전체 검색 결과 처리: ${searchResults.length}개`);
        }

        // 캐시 키 생성 (선택된 항목 기준)
        const cacheKey = `${query}_${JSON.stringify(resultsToProcess).length}`;

        // 캐시 확인 (간단한 메모리 캐시)
        if (global.aiCache && global.aiCache.has(cacheKey)) {
            const cached = global.aiCache.get(cacheKey);
            if (Date.now() - cached.timestamp < 24 * 60 * 60 * 1000) { // 24시간
                return res.json(cached.data);
            }
        }

        // 용어 종합 및 AI 프롬프트 생성 (선택된 결과 사용)
        const terms = extractUniqueTerms(resultsToProcess);
        const termData = terms.map(term => collectTermExplanations(term, resultsToProcess));

        const prompt = buildTermSynthesisPrompt(query, termData);

        // Ollama API 호출
        const aiResponse = await axios.post(`${OLLAMA_BASE_URL}/api/generate`, {
            model: ANALYSIS_AI_MODEL,
            prompt: prompt,
            stream: false,
            options: {
                temperature: 0.7,
                num_thread: 4
            }
        }, { timeout: 600000 }); // 10분 타임아웃 (analysis_api.py와 동일)

        const briefing = aiResponse.data.response || '';

        const result = {
            success: true,
            briefing: briefing,
            metadata: {
                query: query,
                termsAnalyzed: terms.length,
                sourcesUsed: searchResults.length,
                generatedAt: new Date().toISOString()
            }
        };

        // 캐시에 저장
        if (!global.aiCache) global.aiCache = new Map();
        global.aiCache.set(cacheKey, {
            data: result,
            timestamp: Date.now()
        });

        res.json(result);

    } catch (error) {
        console.error('AI 브리핑 생성 오류:', error);
        res.status(500).json({
            error: '브리핑 생성 중 오류가 발생했습니다.',
            code: 'BRIEFING_ERROR',
            details: error.message
        });
    }
});

// 용어 추출 헬퍼 함수
function extractUniqueTerms(results) {
    const allTerms = new Set();
    results.forEach(result => {
        const terms = (result.question || '').match(/[가-힣a-zA-Z]{2,10}/g) || [];
        terms.forEach(term => allTerms.add(term));
    });
    return Array.from(allTerms);
}

// 용어 설명 수집 헬퍼 함수
function collectTermExplanations(term, results) {
    const relevant = results.filter(result =>
        (result.question || '').includes(term) || (result.answer || '').includes(term)
    );

    // 품질 순으로 정렬하고 상위 2개만 사용
    const sortedExplanations = relevant
        .map(r => ({
            source: r.question || '',
            explanation: r.answer || '',
            quality: assessExplanationQuality(r.answer || '')
        }))
        .sort((a, b) => b.quality - a.quality)
        .slice(0, 2);

    return {
        term,
        explanations: sortedExplanations,
        summary: {
            totalSources: relevant.length,
            averageQuality: sortedExplanations.reduce((sum, r) => sum + r.quality, 0) / sortedExplanations.length
        }
    };
}

// 설명 품질 평가 헬퍼 함수
function assessExplanationQuality(text) {
    let score = 50;
    if (text.length > 50) score += 10;
    if (text.length > 200) score += 10;
    if (/\d+/.test(text)) score += 5;
    if (/예시|사례/.test(text)) score += 5;
    if (/이다|합니다/.test(text)) score += 10;
    return Math.min(100, score);
}

// AI 프롬프트 빌더 헬퍼 함수
function buildTermSynthesisPrompt(query, termData) {
    // 매우 간단한 프롬프트로 변경 (로컬 모델 최적화)
    const context = termData.slice(0, 40).map(td =>
        `${td.term}: ${td.explanations.slice(0, 1).map(e => e.explanation).join(' ')}`
    ).join('. ');

    return `${query}에 대해 ${context}를 바탕으로 정리:`;
}

// 답변 텍스트 포맷팅 함수
function formatKnowledgeAnswer(text) {
    if (!text) return '';

    return text
        .replace(/•/g, '\n• ')  // 불릿 포인트 정리
        .replace(/▪/g, '\n▪ ')  // 하위 불릿 정리
        .replace(/\n\s*\n/g, '\n')  // 빈 줄 정리
        .trim();
}

// 카테고리 추출 함수
function extractCategory(question) {
    const categories = {
        '관제': ['관제', '관제업무', '관제소', '관제시스템'],
        '전력': ['전력', '전기', '변전소', '전압', '정류'],
        '설비': ['설비', '기기', '장비', '시설'],
        '안전': ['안전', '사고', '위험', '보안'],
        '운영': ['운영', '운전', '제어', '모니터링'],
        '점검': ['점검', '검사', '시험', '확인'],
        '기타': []
    };

    for (const [category, keywords] of Object.entries(categories)) {
        if (keywords.some(keyword => question.includes(keyword))) {
            return category;
        }
    }

    return '기타';
}

// --- 서버 시작 --- //
app.listen(PORT, () => {
    console.log(`SFA 서버가 http://localhost:${PORT} 에서 실행 중입니다.`);
    console.log(`Analysis API: ${ANALYSIS_API_URL}`);
    console.log(`Prediction API: ${PREDICTION_API_URL}`);
});

// --- 안전한 시뮬레이터 실행 엔드포인트 --- //
// POST /api/launch_simulator { sim: '12' | '22' | 'dc' }
app.post('/api/launch_simulator', (req, res) => {
    try {
        const sim = (req.body && req.body.sim) ? String(req.body.sim) : '';
        const whitelist = {
            '12': { bat: path.join(__dirname, '12계통', '12 pratice.bat'), url: 'http://localhost:8111/index.html' },
            '22': { bat: path.join(__dirname, '22계통', '22 pratice.bat'), url: 'http://localhost:8222/index.html' },
            'dc': { bat: path.join(__dirname, '본선 시뮬레이션(최종)', 'dc pratice.bat'), url: 'http://localhost:8011/index.html' }
        };

        if (!whitelist[sim]) {
            return res.status(400).json({ error: '허용되지 않은 시뮬레이터 입니다.' });
        }

        const batPath = whitelist[sim].bat;
        const openUrl = whitelist[sim].url;

        if (!fs.existsSync(batPath)) {
            console.warn(`배치파일 미발견: ${batPath}`);
            return res.status(500).json({ error: '배치파일을 찾을 수 없습니다.', path: batPath });
        }

        // Use Windows start command to launch the .bat in a new window and return immediately
        // Quote paths to handle spaces and non-ascii characters
        const cmd = `start "" "${batPath}"`;
        exec(cmd, { windowsHide: true }, (err) => {
            if (err) {
                console.error('시뮬레이터 실행 실패:', err);
                return res.status(500).json({ error: '시뮬레이터 실행 실패', detail: String(err) });
            }
            // Return the URL (served by this server) the client should open
            return res.json({ status: 'started', url: openUrl });
        });

    } catch (error) {
        console.error('launch_simulator 오류:', error);
        res.status(500).json({ error: '서버 내부 오류' });
    }
});
