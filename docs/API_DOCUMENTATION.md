# 🌐 API 상세 문서
# FastAPI 기반 로또 예측 시스템 API 가이드

## 📋 API 개요

### 기본 정보
- **Base URL**: `http://localhost:8000`
- **API Version**: v1.0.0
- **Protocol**: HTTP/HTTPS
- **Content Type**: `application/json`
- **Documentation**: `/docs` (Swagger UI), `/redoc` (ReDoc)

### 인증 방식
```http
Authorization: Bearer <JWT_TOKEN>
# 또는
X-API-Key: <API_KEY>
```

## 🎯 핵심 API 엔드포인트

### 1. 예측 API (`/api/predictions/`)

#### 1.1 기본 예측 생성

```http
POST /api/predictions/generate
Content-Type: application/json

{
  "machine_type": "1호기",
  "sets_count": 5,
  "algorithm": "enhanced",
  "analysis_window": 50
}
```

**응답 예시**:
```json
{
  "request_id": "req_20250827_001",
  "machine_type": "1호기",
  "predictions": [
    {
      "set_number": 1,
      "numbers": [7, 14, 21, 28, 35, 42],
      "confidence_score": 0.87,
      "algorithm_used": "enhanced_ml_premium",
      "explanation": [
        "🎯 신중한 전략가 1호기 ML 모델 기반 예측",
        "📊 고수 번호 가중치 적용 (빈도 상위 30%)",
        "⚡ 신뢰도: 87.0%"
      ]
    }
  ],
  "metadata": {
    "algorithm": "enhanced_ml_premium",
    "analysis_window": 50,
    "model_version": "4.0.0_differentiated",
    "features_used": [
      "machine_specific_patterns",
      "number_frequency",
      "weighted_frequency",
      "odd_even_pattern",
      "high_low_distribution"
    ]
  },
  "generated_at": "2025-08-27T14:30:00.000Z"
}
```

#### 1.2 호기별 차별화 전략

| 호기 | 전략 특성 | 가중치 조정 | 알고리즘 |
|------|-----------|-------------|----------|
| **1호기** | 신중한 전략가 | 고수 번호 +15% | `enhanced_ml_premium` |
| **2호기** | 완벽한 조화 | 균형점수 +20% | `enhanced_ml_balanced` |
| **3호기** | 창조적 혁신 | 홀수 가중 +12% | `enhanced_ml_creative` |

### 2. 앙상블 예측 API (`/api/ensemble/`)

#### 2.1 다중 모델 통합 예측

```http
POST /api/ensemble/predictions/
{
  "machine_type": "2호기",
  "sets_count": 10,
  "ensemble_method": "weighted_voting",
  "models": ["pytorch_transformer", "sklearn_ensemble", "enhanced_predictor"]
}
```

**Pydantic 스키마**:
```python
class EnsemblePredictionRequest(BaseModel):
    machine_type: Literal["1호기", "2호기", "3호기"]
    sets_count: int = Field(ge=1, le=20)
    ensemble_method: Literal["weighted_voting", "stacking", "blending"] = "weighted_voting"
    models: List[str] = ["pytorch_transformer", "sklearn_ensemble"]
    confidence_threshold: float = Field(ge=0.0, le=1.0, default=0.7)
    
    class Config:
        schema_extra = {
            "example": {
                "machine_type": "2호기",
                "sets_count": 5,
                "ensemble_method": "weighted_voting",
                "models": ["pytorch_transformer", "sklearn_ensemble"],
                "confidence_threshold": 0.8
            }
        }
```

### 3. 통계 분석 API (`/api/statistics/`)

#### 3.1 호기별 통계 분석

```http
GET /api/statistics/machine-analysis/{machine_type}?rounds=50

# 응답
{
  "machine_type": "1호기",
  "analysis_period": {
    "total_rounds": 50,
    "start_round": 1137,
    "end_round": 1186
  },
  "number_frequency": {
    "most_frequent": [7, 23, 31, 35, 43],
    "least_frequent": [2, 11, 18, 29, 44],
    "frequency_distribution": {
      "1": 0.08, "2": 0.02, "3": 0.12, "...": "..."
    }
  },
  "pattern_analysis": {
    "avg_odd_count": 3.2,
    "avg_even_count": 2.8,
    "avg_high_count": 3.1,
    "avg_low_count": 2.9,
    "avg_ac_value": 16.7,
    "avg_last_digit_sum": 18.4,
    "avg_total_sum": 134.2
  },
  "performance_metrics": {
    "prediction_accuracy": 0.742,
    "confidence_correlation": 0.821,
    "differentiation_score": 0.89
  }
}
```

### 4. 모니터링 API (`/api/monitoring/`)

#### 4.1 시스템 상태 확인

```http
GET /api/monitoring/health

# 응답
{
  "status": "healthy",
  "timestamp": "2025-08-27T14:30:00.000Z",
  "components": {
    "database": {
      "status": "healthy",
      "response_time_ms": 12,
      "last_check": "2025-08-27T14:29:50.000Z"
    },
    "ml_models": {
      "status": "healthy",
      "models_loaded": ["1호기", "2호기", "3호기"],
      "memory_usage_mb": 847,
      "last_prediction": "2025-08-27T14:28:32.000Z"
    },
    "cache": {
      "status": "healthy",
      "hit_rate": 0.84,
      "memory_usage_mb": 128
    }
  },
  "performance": {
    "avg_response_time_ms": 145,
    "requests_per_minute": 127,
    "error_rate": 0.002
  }
}
```

#### 4.2 성능 메트릭

```http
GET /api/monitoring/metrics?period=1h

# 응답
{
  "period": {
    "start": "2025-08-27T13:30:00.000Z",
    "end": "2025-08-27T14:30:00.000Z",
    "duration_minutes": 60
  },
  "request_metrics": {
    "total_requests": 3847,
    "successful_requests": 3831,
    "failed_requests": 16,
    "avg_response_time_ms": 147.3,
    "95th_percentile_ms": 289,
    "99th_percentile_ms": 458
  },
  "ml_metrics": {
    "predictions_generated": 15428,
    "avg_confidence_score": 0.834,
    "model_accuracy": 0.758,
    "cache_hit_rate": 0.82
  },
  "resource_usage": {
    "cpu_usage_percent": 23.7,
    "memory_usage_mb": 1247,
    "disk_usage_mb": 89,
    "gpu_usage_percent": 45.2
  }
}
```

## 📊 Pydantic 모델 스키마

### 핵심 데이터 모델

#### 1. 예측 요청 모델

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal
from datetime import datetime

class PredictionRequest(BaseModel):
    """예측 요청 스키마"""
    machine_type: Literal["1호기", "2호기", "3호기"] = Field(
        ..., 
        description="로또 추첨기 호기 선택"
    )
    sets_count: int = Field(
        ge=1, le=20, default=5,
        description="생성할 번호 세트 개수 (1-20)"
    )
    algorithm: Literal["basic", "enhanced", "premium"] = Field(
        default="enhanced",
        description="예측 알고리즘 선택"
    )
    analysis_window: int = Field(
        ge=10, le=100, default=50,
        description="분석 대상 회차 수 (10-100)"
    )
    
    @validator('machine_type')
    def validate_machine_type(cls, v):
        if v not in ["1호기", "2호기", "3호기"]:
            raise ValueError('올바른 호기를 선택하세요')
        return v
    
    class Config:
        schema_extra = {
            "example": {
                "machine_type": "1호기",
                "sets_count": 5,
                "algorithm": "enhanced",
                "analysis_window": 50
            }
        }
```

#### 2. 예측 응답 모델

```python
class PredictionSet(BaseModel):
    """개별 예측 세트"""
    set_number: int = Field(..., description="세트 번호")
    numbers: List[int] = Field(
        ..., 
        min_items=6, max_items=6,
        description="예측된 6개 번호 (1-45)"
    )
    confidence_score: float = Field(
        ..., ge=0.0, le=1.0,
        description="신뢰도 점수 (0.0-1.0)"
    )
    algorithm_used: str = Field(..., description="사용된 알고리즘")
    explanation: List[str] = Field(
        ..., min_items=1,
        description="예측 근거 설명"
    )
    
    @validator('numbers')
    def validate_numbers(cls, v):
        # 1-45 범위 검증
        if not all(1 <= num <= 45 for num in v):
            raise ValueError('번호는 1-45 범위여야 합니다')
        
        # 중복 검증
        if len(set(v)) != 6:
            raise ValueError('중복된 번호가 있습니다')
            
        return sorted(v)

class PredictionResponse(BaseModel):
    """예측 응답 스키마"""
    request_id: str = Field(..., description="요청 고유 ID")
    machine_type: str = Field(..., description="호기 타입")
    predictions: List[PredictionSet] = Field(
        ..., min_items=1,
        description="예측 결과 목록"
    )
    metadata: PredictionMetadata = Field(..., description="메타데이터")
    generated_at: datetime = Field(..., description="생성 시각")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
```

#### 3. 에러 응답 모델

```python
class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: str = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    field: Optional[str] = Field(None, description="문제 필드명")
    
class ErrorResponse(BaseModel):
    """에러 응답 스키마"""
    status: Literal["error"] = "error"
    error: ErrorDetail
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = None
```

## 🔐 인증 및 권한

### API 키 인증

```http
# 헤더 방식
GET /api/predictions/generate
X-API-Key: your-api-key-here

# 쿼리 파라미터 방식
GET /api/predictions/generate?api_key=your-api-key-here
```

### JWT 토큰 인증

```http
# 로그인
POST /auth/login
{
  "username": "user@example.com",
  "password": "secure_password"
}

# 응답
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}

# 인증된 요청
GET /api/predictions/generate
Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc...
```

## ⚡ Rate Limiting

### 사용 제한

| 사용자 티어 | 분당 요청 | 일간 요청 | 동시 연결 |
|-------------|-----------|-----------|-----------|
| **Free** | 10 req/min | 1,000 req/day | 2 concurrent |
| **Premium** | 100 req/min | 50,000 req/day | 10 concurrent |
| **Enterprise** | 1000 req/min | 무제한 | 100 concurrent |

### Rate Limit 헤더

```http
HTTP/1.1 200 OK
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1692345600
X-RateLimit-Retry-After: 60
```

## 💾 캐싱 전략

### 응답 캐싱

```http
# 캐시 제어 헤더
GET /api/statistics/machine-analysis/1호기
Cache-Control: public, max-age=3600
ETag: "a1b2c3d4e5f6"

# 조건부 요청
GET /api/statistics/machine-analysis/1호기
If-None-Match: "a1b2c3d4e5f6"

# 304 Not Modified (캐시 사용)
HTTP/1.1 304 Not Modified
```

### 캐시 무효화

```http
# 강제 갱신
GET /api/predictions/generate
Cache-Control: no-cache

# 응답
{
  "cache_status": "miss",
  "generated_fresh": true,
  "cache_expires_at": "2025-08-27T15:30:00.000Z"
}
```

## 🚨 에러 처리

### 표준 HTTP 상태 코드

| 코드 | 상태 | 설명 |
|------|------|------|
| **200** | OK | 요청 성공 |
| **201** | Created | 리소스 생성 성공 |
| **400** | Bad Request | 잘못된 요청 데이터 |
| **401** | Unauthorized | 인증 필요 |
| **403** | Forbidden | 권한 부족 |
| **404** | Not Found | 리소스 없음 |
| **422** | Unprocessable Entity | 유효성 검사 실패 |
| **429** | Too Many Requests | 요청 한도 초과 |
| **500** | Internal Server Error | 서버 내부 오류 |

### 에러 응답 예시

```json
// 400 Bad Request
{
  "status": "error",
  "error": {
    "code": "INVALID_MACHINE_TYPE",
    "message": "올바른 호기를 선택하세요. 가능한 값: 1호기, 2호기, 3호기",
    "field": "machine_type"
  },
  "timestamp": "2025-08-27T14:30:00.000Z",
  "request_id": "req_20250827_001"
}

// 422 Validation Error
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "입력 데이터 유효성 검사 실패",
    "details": [
      {
        "field": "sets_count",
        "message": "1 이상 20 이하의 값이어야 합니다",
        "value": 25
      }
    ]
  },
  "timestamp": "2025-08-27T14:30:00.000Z"
}

// 429 Rate Limit Exceeded  
{
  "status": "error",
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "요청 한도를 초과했습니다. 60초 후 다시 시도하세요",
    "retry_after_seconds": 60
  },
  "timestamp": "2025-08-27T14:30:00.000Z"
}

// 500 Internal Server Error
{
  "status": "error", 
  "error": {
    "code": "ML_MODEL_ERROR",
    "message": "ML 모델 로드에 실패했습니다. 잠시 후 다시 시도해 주세요",
    "support_id": "sup_20250827_001"
  },
  "timestamp": "2025-08-27T14:30:00.000Z"
}
```

## 📱 클라이언트 SDK 예시

### Python SDK

```python
import requests
from typing import List, Dict

class LotteryAPIClient:
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': api_key,
            'Content-Type': 'application/json'
        })
    
    def predict(self, 
                machine_type: str, 
                sets_count: int = 5,
                algorithm: str = "enhanced") -> Dict:
        """번호 예측 요청"""
        response = self.session.post(
            f"{self.base_url}/api/predictions/generate",
            json={
                "machine_type": machine_type,
                "sets_count": sets_count,
                "algorithm": algorithm
            }
        )
        response.raise_for_status()
        return response.json()
    
    def get_statistics(self, machine_type: str, rounds: int = 50) -> Dict:
        """통계 분석 조회"""
        response = self.session.get(
            f"{self.base_url}/api/statistics/machine-analysis/{machine_type}",
            params={"rounds": rounds}
        )
        response.raise_for_status()
        return response.json()

# 사용 예시
client = LotteryAPIClient("http://localhost:8000", "your-api-key")

# 1호기 예측 5줄 생성
result = client.predict("1호기", sets_count=5)
print(f"예측 결과: {result['predictions']}")

# 2호기 통계 분석
stats = client.get_statistics("2호기", rounds=100)
print(f"빈도 분석: {stats['number_frequency']}")
```

### JavaScript SDK

```javascript
class LotteryAPIClient {
    constructor(baseUrl, apiKey) {
        this.baseUrl = baseUrl.replace(/\/$/, '');
        this.apiKey = apiKey;
    }
    
    async predict(machineType, setsCount = 5, algorithm = 'enhanced') {
        const response = await fetch(`${this.baseUrl}/api/predictions/generate`, {
            method: 'POST',
            headers: {
                'X-API-Key': this.apiKey,
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                machine_type: machineType,
                sets_count: setsCount,
                algorithm: algorithm
            })
        });
        
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }
        
        return await response.json();
    }
    
    async getHealthCheck() {
        const response = await fetch(`${this.baseUrl}/api/monitoring/health`, {
            headers: { 'X-API-Key': this.apiKey }
        });
        return await response.json();
    }
}

// 사용 예시
const client = new LotteryAPIClient('http://localhost:8000', 'your-api-key');

// 예측 요청
try {
    const result = await client.predict('3호기', 10, 'premium');
    console.log('예측 결과:', result.predictions);
} catch (error) {
    console.error('예측 실패:', error.message);
}
```

## 🔧 개발자 도구

### OpenAPI 스펙 다운로드

```bash
# OpenAPI JSON 스펙
curl http://localhost:8000/openapi.json > lottery-api-spec.json

# Postman 컬렉션 생성
postman-collection-generator lottery-api-spec.json
```

### 테스트 환경

```bash
# 로컬 테스트 서버 실행
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API 테스트
curl -X POST http://localhost:8000/api/predictions/generate \
  -H "X-API-Key: test-key" \
  -H "Content-Type: application/json" \
  -d '{"machine_type": "1호기", "sets_count": 3}'

# 헬스체크
curl http://localhost:8000/api/monitoring/health
```

---

**API 버전**: v1.0.0  
**문서 업데이트**: 2025-08-27  
**지원**: api-support@lottery-prediction.com

> 💡 **Tip**: 더 자세한 예시와 실시간 테스트는 `/docs` 페이지를 활용하세요!