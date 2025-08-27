# 🤖 Next-Generation Hive-Mind Lottery Prediction System

차세대 분산형 AI 로또 예측 시스템 - 하이브리드 머신러닝과 분산 에이전트 아키텍처를 통한 지능형 번호 예측

## 🎯 시스템 개요

Claude-Flow의 한계를 극복한 혁신적 예측 시스템:
- **도메인 특화 지능**: 로또 전문 에이전트들의 협력적 분석
- **분산 합의 알고리즘**: 민주적 의사결정으로 중앙 집중화 병목 해결
- **실시간 성능**: <200ms 응답시간으로 즉각적 예측
- **인지적 추론**: 설명 가능한 AI로 예측 근거 제공

## 🏗️ 시스템 아키텍처

### 하이브-마인드 구조
```
👑 Queen Orchestrator
├── 🔍 Pattern Analyzer (PyTorch Transformer)
├── 📊 Statistical Predictor (scikit-learn Ensemble) 
├── 🧠 Cognitive Analyzer (Domain Expertise)
└── ⚖️ Ensemble Optimizer (Dynamic Weighting)
```

### 기술 스택
- **Backend**: FastAPI + Python 3.11
- **ML Frameworks**: PyTorch 2.1+ + scikit-learn 1.3+
- **Database**: SQLite + Redis (Hybrid Memory)
- **Communication**: Advanced Message Bus with Priority Queues
- **Deployment**: Docker + Docker Compose
- **Monitoring**: Prometheus + Grafana

## 🚀 빠른 시작

### 원클릭 설치 및 실행
```bash
# 저장소 클론
git clone <repository-url>
cd Lottery_2510

# 빠른 시작 (자동 설정 + 실행)
./scripts/quick_start.sh

# 개발 서버 수동 시작
python scripts/startup.py server --env development
```

### Docker로 전체 스택 실행
```bash
# 전체 시스템 시작 (앱 + Redis + 모니터링)
docker-compose up -d

# 로그 확인
docker-compose logs -f hive-mind
```

### API 접근
- **API 문서**: http://localhost:8000/docs
- **건강 상태**: http://localhost:8000/api/monitoring/health
- **Grafana 대시보드**: http://localhost:3000 (admin/hivemind2024)

## 📚 API 사용법

### 예측 생성
```bash
curl -X POST "http://localhost:8000/api/predictions/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_type": "2호기",
    "sets_count": 2,
    "algorithm": "hybrid",
    "ensemble_strategy": "dynamic"
  }'
```

### 시스템 상태 확인
```bash
# 전체 시스템 건강 상태
curl http://localhost:8000/api/monitoring/health

# 에이전트 상태
curl http://localhost:8000/api/agents/status

# 성능 메트릭
curl http://localhost:8000/api/monitoring/metrics
```

## 🎲 기계별 전략

### 1호기 - 신중한 전략가
- **특성**: 보수적, 안정성 중시
- **전략**: 고빈도 번호, 중간 범위 선호
- **신뢰도**: 높은 안정성, 낮은 변동성

### 2호기 - 완벽한 조화
- **특성**: 균형 최적화, 통계적 완성도
- **전략**: 홀짝 균형, AC값 최적화
- **신뢰도**: 가장 균형잡힌 예측

### 3호기 - 창조적 혁신
- **특성**: 창의적, 다양성 추구  
- **전략**: 홀수 선호, 패턴 다양성
- **신뢰도**: 높은 창의성, 돌발 변수 대응

## 🧠 AI 에이전트 상세

### Pattern Analyzer Agent 🔍
- **기술**: PyTorch Transformer with Multi-head Attention
- **특기**: 숨겨진 패턴 발견, 시퀀스 분석
- **출력**: 창의적 번호 조합, 패턴 기반 예측

### Statistical Predictor Agent 📊
- **기술**: RandomForest + GradientBoosting + MLP Ensemble
- **특기**: 통계적 특성 최적화, 안정적 예측
- **출력**: 확률론적 번호 선택, 기계별 특화

### Cognitive Analyzer Agent 🧠
- **기술**: 규칙 기반 추론 + 도메인 지식
- **특기**: 이상 탐지, 설명 생성, 검증
- **출력**: 예측 검증, 합리적 설명

### Ensemble Optimizer Agent ⚖️
- **기술**: 동적 가중치 최적화
- **특기**: 실시간 성능 기반 앙상블 조정
- **출력**: 최적 가중치, 메타 예측

## 🏃‍♂️ 개발자 워크플로우

### 로컬 개발 환경 설정
```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 데이터베이스 초기화
python scripts/init_database.py init

# 개발 서버 시작
python scripts/startup.py server --env development
```

### 테스트 실행
```bash
# 단위 테스트
pytest tests/

# 데모 예측
python scripts/startup.py demo

# 시스템 검증
python scripts/init_database.py verify
```

### 프로덕션 배포
```bash
# 프로덕션 데이터베이스 초기화
python scripts/init_database.py init --env production

# Docker 이미지 빌드 및 배포
docker-compose -f docker-compose.yml up -d

# 헬스체크
curl http://localhost/api/monitoring/health
```

## 📈 성능 및 모니터링

### 성능 목표
- **응답시간**: <200ms (95th percentile)
- **가용성**: 99.9% uptime
- **메모리사용량**: <2GB total
- **처리량**: >1,000 requests/minute

### 모니터링 대시보드
- **Grafana**: http://localhost:3000
- **Prometheus**: http://localhost:9090  
- **로그 수집**: Loki + Promtail 자동 연동

### 주요 메트릭
- 예측 정확도 및 신뢰도
- 에이전트별 성능 통계
- API 응답시간 및 오류율
- 시스템 리소스 사용량

## 🛠️ 고급 기능

### 실시간 학습
```bash
# 새로운 결과 데이터 추가
curl -X POST "http://localhost:8000/api/data/update" \
  -H "Content-Type: application/json" \
  -d '{"draw_no": 1187, "numbers": [1,2,3,4,5,6], "machine": "2호기"}'
```

### 에이전트 관리
```bash
# 새 에이전트 생성
curl -X POST "http://localhost:8000/api/agents/spawn/pattern_analyzer" \
  -H "Content-Type: application/json" \
  -d '{"config": {"diversity_threshold": 0.8}}'

# 에이전트 종료
curl -X DELETE "http://localhost:8000/api/agents/{agent_id}"
```

### 앙상블 전략 비교
```bash
curl -X POST "http://localhost:8000/api/ensemble/predictions/" \
  -H "Content-Type: application/json" \
  -d '{
    "machine_type": "2호기",
    "strategies": ["voting", "stacking", "blending", "dynamic"]
  }'
```

## 🔧 설정

### 환경 설정 파일
- `config/development.yaml`: 개발 환경 설정
- `config/production.yaml`: 프로덕션 환경 설정

### 주요 설정 항목
```yaml
# ML 모델 설정
ml:
  pytorch_device: "cpu"  # or "cuda"
  ensemble_weights:
    pattern_analyzer: 0.35
    statistical_predictor: 0.35
    cognitive_analyzer: 0.30

# API 설정  
api:
  host: "0.0.0.0"
  port: 8000
  api_key_required: true

# 성능 설정
performance:
  target_response_time: 200
  max_concurrent_predictions: 10
```

## 📋 문제 해결

### 일반적 문제들

**Q: 모델 초기화 실패**
```bash
# PyTorch 설치 확인
pip install torch torchvision

# CUDA 사용 불가 시 CPU 모드로 설정
export CUDA_VISIBLE_DEVICES=""
```

**Q: 데이터베이스 연결 오류**
```bash
# 데이터베이스 재초기화
python scripts/init_database.py reset

# 권한 확인
chmod 755 data/
```

**Q: 메모리 부족 오류**
```bash
# 설정에서 배치 크기 감소
# config/development.yaml
ml:
  batch_size: 16  # 기본값: 32
```

### 로그 확인
```bash
# 애플리케이션 로그
tail -f logs/hive_mind_dev.log

# Docker 로그
docker-compose logs -f hive-mind

# 특정 에이전트 로그
grep "PatternAnalyzer" logs/hive_mind_dev.log
```

## 🤝 기여하기

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 라이선스

이 프로젝트는 연구 및 교육 목적으로만 사용되어야 합니다.

## ⚠️ 면책조항

- 이 시스템은 연구 및 교육 목적으로 개발되었습니다
- 예측 결과는 보장되지 않습니다
- 투자 손실에 대한 책임을 지지 않습니다
- 과거 패턴 기반 분석이며 미래 결과를 보장하지 않습니다

## 📞 지원

- **문서**: `/docs` 엔드포인트에서 자세한 API 문서 확인
- **이슈 리포팅**: GitHub Issues 섹션 활용
- **시스템 상태**: `/api/monitoring/health` 엔드포인트로 실시간 확인

---

**🎉 차세대 하이브-마인드 시스템으로 더욱 정교하고 지능적인 예측을 경험하세요!**