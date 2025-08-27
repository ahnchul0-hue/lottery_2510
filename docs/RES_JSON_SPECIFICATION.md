# RES_JSON_SPECIFICATION.md
# res.json 데이터 명세서 및 DB 연동 스키마

## 📄 res.json 개요

**파일 목적**: 한국 동행복권 로또 6/45 당첨 데이터의 표준화된 저장 형식  
**데이터 범위**: 1049회차 ~ 1186회차 (총 140회차)  
**업데이트 주기**: 매주 토요일 추첨 후  
**인코딩**: UTF-8  

---

## 🗂️ 전체 구조 개요

```json
{
  "metadata": { /* 메타데이터 */ },
  "lottery_data": [ /* 실제 추첨 데이터 배열 */ ]
}
```

---

## 📊 메타데이터 스키마

### metadata 필드 상세 명세
```json
{
  "metadata": {
    "description": "동행복권 로또 1049회차부터 1186회차까지 전체 추첨기별 데이터",
    "data_range": "1049-1186회차",
    "total_machines": 3,
    "total_rounds": 140,
    "data_source": "https://lottotapa.com/stat/result_hogi.php",
    "last_updated": "2025-08-27T10:00:00Z",
    "version": "1.2.1",
    "notes": [
      "홀짝 비율은 홀수:짝수 형태로 표시",
      "고저 비율은 고수(23-45):저수(1-22) 형태로 표시",
      "AC값은 Arithmetic Complexity(산술 복잡도)",
      "끝수합은 6개 당첨번호의 끝자리 수의 합",
      "총합은 6개 당첨번호의 총합"
    ]
  }
}
```

#### 메타데이터 필드 설명
| 필드명 | 타입 | 필수 | 설명 |
|--------|------|------|------|
| `description` | string | Y | 데이터셋 설명 |
| `data_range` | string | Y | 포함된 회차 범위 |
| `total_machines` | integer | Y | 추첨기 총 개수 (1,2,3호기) |
| `total_rounds` | integer | Y | 총 추첨 회차 수 |
| `data_source` | string | Y | 원본 데이터 소스 URL |
| `last_updated` | string | Y | 마지막 업데이트 시각 (ISO 8601) |
| `version` | string | Y | 데이터 스키마 버전 |
| `notes` | array | N | 데이터 해석을 위한 참고사항 |

---

## 🎰 로또 데이터 스키마

### lottery_data 배열 구조
```json
{
  "lottery_data": [
    {
      "회차": 1186,
      "호기": "1호기",
      "1등_당첨번호": [1, 15, 25, 31, 38, 43],
      "홀짝_비율": "3:3",
      "고저_비율": "3:3",
      "AC값": 8,
      "끝수합": 25,
      "총합": 153,
      "추첨일": "2025-08-24",
      "특별_메모": null
    }
  ]
}
```

#### 로또 데이터 필드 명세
| 필드명 | 타입 | 필수 | 제약조건 | 설명 |
|--------|------|------|----------|------|
| `회차` | integer | Y | >= 1049 | 로또 회차 번호 |
| `호기` | string | Y | "1호기", "2호기", "3호기" | 추첨기 식별자 |
| `1등_당첨번호` | array[int] | Y | 길이=6, 1-45 범위, 중복 없음 | 당첨번호 6개 |
| `홀짝_비율` | string | Y | "N:M" 형태, N+M=6 | 홀수:짝수 개수 비율 |
| `고저_비율` | string | Y | "N:M" 형태, N+M=6 | 고수(23-45):저수(1-22) 비율 |
| `AC값` | integer | Y | 0-15 | Arithmetic Complexity |
| `끝수합` | integer | Y | 0-54 | 6개 번호 끝자리 합 |
| `총합` | integer | Y | 21-270 | 6개 번호의 총합 |
| `추첨일` | string | N | YYYY-MM-DD | 추첨 날짜 |
| `특별_메모` | string/null | N | - | 특별한 패턴이나 메모 |

---

## 🔢 데이터 검증 규칙

### 1. 당첨번호 검증
```python
def validate_winning_numbers(numbers: List[int]) -> bool:
    """당첨번호 유효성 검증"""
    # 길이 검사
    if len(numbers) != 6:
        return False
    
    # 범위 검사 (1-45)
    if not all(1 <= num <= 45 for num in numbers):
        return False
    
    # 중복 검사
    if len(set(numbers)) != 6:
        return False
    
    # 정렬 검사
    if numbers != sorted(numbers):
        return False
        
    return True
```

### 2. 통계값 검증
```python
def validate_statistics(data: dict) -> bool:
    """통계 데이터 일관성 검증"""
    numbers = data['1등_당첨번호']
    
    # 홀짝비 검증
    odd_count = sum(1 for n in numbers if n % 2 == 1)
    even_count = 6 - odd_count
    expected_odd_even = f"{odd_count}:{even_count}"
    if data['홀짝_비율'] != expected_odd_even:
        return False
    
    # 고저비 검증
    high_count = sum(1 for n in numbers if n >= 23)
    low_count = 6 - high_count
    expected_high_low = f"{high_count}:{low_count}"
    if data['고저_비율'] != expected_high_low:
        return False
    
    # 총합 검증
    if data['총합'] != sum(numbers):
        return False
    
    # 끝수합 검증
    if data['끝수합'] != sum(n % 10 for n in numbers):
        return False
        
    return True
```

### 3. AC값 계산 알고리즘
```python
def calculate_ac_value(numbers: List[int]) -> int:
    """AC값(Arithmetic Complexity) 계산"""
    numbers = sorted(numbers)
    differences = []
    
    # 연속된 번호들의 차이 계산
    for i in range(len(numbers) - 1):
        diff = numbers[i + 1] - numbers[i]
        differences.append(diff)
    
    # 차이값들의 고유 개수가 AC값
    ac_value = len(set(differences))
    return ac_value
```

---

## 🗄️ 데이터베이스 연동 스키마

### SQLite 테이블 매핑

#### lottery_draws 테이블
```sql
CREATE TABLE lottery_draws (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    draw_round INTEGER NOT NULL,           -- res.json의 "회차"
    machine_type VARCHAR(10) NOT NULL,     -- res.json의 "호기"
    winning_numbers TEXT NOT NULL,         -- res.json의 "1등_당첨번호" (JSON 저장)
    draw_date DATE,                        -- res.json의 "추첨일"
    odd_even_ratio VARCHAR(10),            -- res.json의 "홀짝_비율"
    high_low_ratio VARCHAR(10),            -- res.json의 "고저_비율"
    ac_value INTEGER,                      -- res.json의 "AC값"
    last_digit_sum INTEGER,                -- res.json의 "끝수합"
    total_sum INTEGER,                     -- res.json의 "총합"
    special_memo TEXT,                     -- res.json의 "특별_메모"
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    
    UNIQUE(draw_round, machine_type)
);
```

#### 데이터 변환 로직
```python
class ResJsonToDbConverter:
    def convert_single_record(self, res_json_record: dict) -> dict:
        """res.json 단일 레코드를 DB 형식으로 변환"""
        return {
            'draw_round': res_json_record['회차'],
            'machine_type': res_json_record['호기'],
            'winning_numbers': json.dumps(res_json_record['1등_당첨번호']),
            'draw_date': res_json_record.get('추첨일'),
            'odd_even_ratio': res_json_record['홀짝_비율'],
            'high_low_ratio': res_json_record['고저_비율'],
            'ac_value': res_json_record['AC값'],
            'last_digit_sum': res_json_record['끝수합'],
            'total_sum': res_json_record['총합'],
            'special_memo': res_json_record.get('특별_메모')
        }
    
    def batch_convert(self, res_json_data: dict) -> List[dict]:
        """전체 res.json을 DB 레코드 리스트로 변환"""
        db_records = []
        for record in res_json_data['lottery_data']:
            db_record = self.convert_single_record(record)
            db_records.append(db_record)
        return db_records
```

---

## 🔄 데이터 동기화 전략

### 1. res.json → Database 동기화
```python
class DataSynchronizer:
    def sync_res_json_to_db(self):
        """res.json 데이터를 데이터베이스에 동기화"""
        
        # 1. res.json 읽기
        with open('res.json', 'r', encoding='utf-8') as f:
            res_data = json.load(f)
        
        # 2. 데이터 검증
        for record in res_data['lottery_data']:
            if not self.validate_record(record):
                raise ValueError(f"Invalid record: {record}")
        
        # 3. 변환 및 저장
        converter = ResJsonToDbConverter()
        db_records = converter.batch_convert(res_data)
        
        # 4. 배치 삽입 (UPSERT)
        for record in db_records:
            self.upsert_lottery_draw(record)
    
    def upsert_lottery_draw(self, record: dict):
        """데이터 삽입 또는 업데이트"""
        query = """
        INSERT INTO lottery_draws (
            draw_round, machine_type, winning_numbers, draw_date,
            odd_even_ratio, high_low_ratio, ac_value, 
            last_digit_sum, total_sum, special_memo
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(draw_round, machine_type) 
        DO UPDATE SET
            winning_numbers = excluded.winning_numbers,
            draw_date = excluded.draw_date,
            odd_even_ratio = excluded.odd_even_ratio,
            high_low_ratio = excluded.high_low_ratio,
            ac_value = excluded.ac_value,
            last_digit_sum = excluded.last_digit_sum,
            total_sum = excluded.total_sum,
            special_memo = excluded.special_memo,
            updated_at = CURRENT_TIMESTAMP
        """
        self.db.execute(query, tuple(record.values()))
```

### 2. Database → res.json 역동기화
```python
def export_db_to_res_json(self) -> dict:
    """데이터베이스에서 res.json 형식으로 내보내기"""
    
    # DB에서 모든 데이터 조회
    query = """
    SELECT draw_round, machine_type, winning_numbers, draw_date,
           odd_even_ratio, high_low_ratio, ac_value,
           last_digit_sum, total_sum, special_memo
    FROM lottery_draws 
    ORDER BY draw_round, machine_type
    """
    
    records = self.db.execute(query).fetchall()
    
    # res.json 형식으로 변환
    lottery_data = []
    for record in records:
        res_record = {
            "회차": record['draw_round'],
            "호기": record['machine_type'],
            "1등_당첨번호": json.loads(record['winning_numbers']),
            "홀짝_비율": record['odd_even_ratio'],
            "고저_비율": record['high_low_ratio'],
            "AC값": record['ac_value'],
            "끝수합": record['last_digit_sum'],
            "총합": record['total_sum'],
            "추첨일": record['draw_date'],
            "특별_메모": record['special_memo']
        }
        lottery_data.append(res_record)
    
    # 메타데이터 생성
    metadata = {
        "description": f"동행복권 로또 데이터 (총 {len(lottery_data)}개 레코드)",
        "data_range": f"{min(r['회차'] for r in lottery_data)}-{max(r['회차'] for r in lottery_data)}회차",
        "total_machines": len(set(r['호기'] for r in lottery_data)),
        "total_rounds": len(set(r['회차'] for r in lottery_data)),
        "last_updated": datetime.now().isoformat(),
        "version": "1.2.1"
    }
    
    return {
        "metadata": metadata,
        "lottery_data": lottery_data
    }
```

---

## 📋 데이터 품질 관리

### 1. 자동 검증 파이프라인
```python
class DataQualityChecker:
    def run_quality_checks(self, res_json_path: str) -> dict:
        """데이터 품질 종합 검사"""
        
        with open(res_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        results = {
            'schema_validation': self.validate_schema(data),
            'data_consistency': self.check_consistency(data),
            'statistical_validation': self.validate_statistics(data),
            'duplicate_detection': self.find_duplicates(data),
            'completeness_check': self.check_completeness(data)
        }
        
        return results
    
    def generate_quality_report(self, results: dict) -> str:
        """품질 검사 보고서 생성"""
        report = f"""
# 데이터 품질 검사 보고서
생성일시: {datetime.now().isoformat()}

## 검사 결과 요약
- 스키마 검증: {'✅ 통과' if results['schema_validation']['passed'] else '❌ 실패'}
- 데이터 일관성: {'✅ 통과' if results['data_consistency']['passed'] else '❌ 실패'}
- 통계 검증: {'✅ 통과' if results['statistical_validation']['passed'] else '❌ 실패'}
- 중복 데이터: {results['duplicate_detection']['count']}건 발견
- 데이터 완전성: {results['completeness_check']['completeness_rate']:.1%}

## 상세 내용
{self._format_detailed_results(results)}
        """
        return report
```

### 2. 실시간 모니터링
```python
class RealtimeDataMonitor:
    def __init__(self):
        self.alerts = []
        self.thresholds = {
            'max_ac_value': 15,
            'min_total_sum': 21,
            'max_total_sum': 270,
            'expected_number_count': 6
        }
    
    def monitor_new_data(self, new_record: dict):
        """신규 데이터 실시간 모니터링"""
        
        # 임계값 초과 검사
        if new_record['AC값'] > self.thresholds['max_ac_value']:
            self.alerts.append({
                'type': 'threshold_exceeded',
                'field': 'AC값',
                'value': new_record['AC값'],
                'threshold': self.thresholds['max_ac_value'],
                'timestamp': datetime.now().isoformat()
            })
        
        # 패턴 이상 감지
        if self.detect_anomaly(new_record):
            self.alerts.append({
                'type': 'pattern_anomaly',
                'record': new_record,
                'timestamp': datetime.now().isoformat()
            })
```

---

## 🔧 버전 관리 및 하위 호환성

### 스키마 버전 히스토리
| 버전 | 릴리스 날짜 | 주요 변경사항 |
|------|-------------|---------------|
| 1.0.0 | 2024-01-01 | 초기 스키마 정의 |
| 1.1.0 | 2024-06-01 | `추첨일` 필드 추가 |
| 1.2.0 | 2024-12-01 | `특별_메모` 필드 추가 |
| 1.2.1 | 2025-01-01 | 메타데이터에 `last_updated`, `version` 추가 |

### 하위 호환성 유지 전략
```python
class SchemaVersionHandler:
    def migrate_to_latest(self, data: dict) -> dict:
        """구버전 데이터를 최신 버전으로 마이그레이션"""
        
        version = data.get('metadata', {}).get('version', '1.0.0')
        
        if version == '1.0.0':
            data = self._migrate_1_0_to_1_1(data)
            version = '1.1.0'
        
        if version == '1.1.0':
            data = self._migrate_1_1_to_1_2(data)
            version = '1.2.0'
        
        if version == '1.2.0':
            data = self._migrate_1_2_to_1_2_1(data)
        
        return data
    
    def _migrate_1_0_to_1_1(self, data: dict) -> dict:
        """v1.0 -> v1.1 마이그레이션: 추첨일 필드 추가"""
        for record in data['lottery_data']:
            if '추첨일' not in record:
                # 회차 정보를 기반으로 추첨일 추정
                record['추첨일'] = self._estimate_draw_date(record['회차'])
        
        data['metadata']['version'] = '1.1.0'
        return data
```

---

*이 문서는 res.json 데이터의 정확한 구조와 검증 규칙을 정의하며, 데이터베이스 연동 및 품질 관리 방안을 포함합니다.*