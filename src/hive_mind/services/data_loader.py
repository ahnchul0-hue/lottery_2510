# 🤖 Data Loading Service - Historical Lottery Data Management
# Loads and processes res.json data for ML model training and prediction

import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class LotteryDraw:
    """개별 로또 추첨 데이터"""
    draw_no: int
    date: str
    numbers: List[int]
    bonus: int
    machine_type: str = "알수없음"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'draw_no': self.draw_no,
            'date': self.date,
            'numbers': self.numbers,
            'bonus': self.bonus,
            'machine_type': self.machine_type
        }
    
    @property
    def odd_count(self) -> int:
        """홀수 개수"""
        return len([n for n in self.numbers if n % 2 == 1])
    
    @property
    def even_count(self) -> int:
        """짝수 개수"""
        return len([n for n in self.numbers if n % 2 == 0])
    
    @property
    def high_count(self) -> int:
        """고번(23-45) 개수"""
        return len([n for n in self.numbers if n >= 23])
    
    @property
    def low_count(self) -> int:
        """저번(1-22) 개수"""
        return len([n for n in self.numbers if n < 23])
    
    @property
    def digit_sum(self) -> int:
        """번호 합계"""
        return sum(self.numbers)
    
    @property
    def ac_value(self) -> int:
        """AC값 (연속성 지수)"""
        sorted_nums = sorted(self.numbers)
        ac = 0
        for i in range(len(sorted_nums)):
            for j in range(i + 1, len(sorted_nums)):
                diff = abs(sorted_nums[i] - sorted_nums[j])
                if diff not in [abs(sorted_nums[k] - sorted_nums[l]) 
                               for k in range(len(sorted_nums)) 
                               for l in range(k + 1, len(sorted_nums)) 
                               if (k, l) != (i, j)]:
                    ac += 1
        return ac

class DataLoader:
    """
    데이터 로딩 및 관리 서비스
    
    Claude-Flow 개선사항:
    - 비동기 데이터 로딩
    - 실시간 데이터 검증
    - 머신러닝 특성 추출
    - 기계별 데이터 분류
    """
    
    def __init__(self, data_path: str = "data/res.json", config: Dict[str, Any] = None):
        self.data_path = Path(data_path)
        self.config = config or {}
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 캐시된 데이터
        self._raw_data: Optional[List[Dict]] = None
        self._processed_draws: Optional[List[LotteryDraw]] = None
        self._ml_features: Optional[pd.DataFrame] = None
        
        # 통계 캐시
        self._number_frequency: Optional[Dict[int, int]] = None
        self._recent_patterns: Optional[Dict[str, Any]] = None
        
        # 비동기 처리용 executor
        self.executor = ThreadPoolExecutor(max_workers=2)
        
    async def initialize(self) -> bool:
        """데이터 로더 초기화"""
        try:
            self.logger.info("🚀 Initializing data loader")
            
            # 데이터 파일 존재 확인
            if not self.data_path.exists():
                self.logger.error(f"❌ Data file not found: {self.data_path}")
                return False
                
            # 비동기 데이터 로드
            await self.load_data()
            
            # 기본 검증
            if not self._processed_draws:
                self.logger.error("❌ No valid lottery draws loaded")
                return False
                
            self.logger.info(f"✅ Data loader initialized with {len(self._processed_draws)} draws")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize data loader: {e}")
            return False
            
    async def load_data(self) -> List[LotteryDraw]:
        """비동기 데이터 로드"""
        loop = asyncio.get_event_loop()
        
        try:
            # 파일 읽기 (I/O 블로킹 작업을 executor에서 실행)
            self._raw_data = await loop.run_in_executor(
                self.executor, self._load_json_file
            )
            
            # 데이터 처리
            self._processed_draws = await loop.run_in_executor(
                self.executor, self._process_raw_data, self._raw_data
            )
            
            # ML 특성 생성
            self._ml_features = await loop.run_in_executor(
                self.executor, self._generate_ml_features
            )
            
            # 통계 생성
            await self._update_statistics()
            
            self.logger.info(f"📊 Loaded {len(self._processed_draws)} lottery draws")
            return self._processed_draws
            
        except Exception as e:
            self.logger.error(f"❌ Data loading failed: {e}")
            raise
            
    def _load_json_file(self) -> List[Dict]:
        """JSON 파일 로드 (동기)"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 데이터 구조 검증
        if not isinstance(data, list):
            raise ValueError("Invalid data format: expected list of draws")
            
        return data
        
    def _process_raw_data(self, raw_data: List[Dict]) -> List[LotteryDraw]:
        """원시 데이터를 LotteryDraw 객체로 변환"""
        processed_draws = []
        
        for item in raw_data:
            try:
                # 필수 필드 검증
                if not all(key in item for key in ['회차', '추첨일', '당첨번호']):
                    continue
                    
                # 번호 파싱
                numbers_str = item['당첨번호']
                if isinstance(numbers_str, str):
                    # "1,5,12,23,34,45" 형태 파싱
                    numbers = [int(x.strip()) for x in numbers_str.split(',')]
                elif isinstance(numbers_str, list):
                    numbers = [int(x) for x in numbers_str]
                else:
                    continue
                    
                # 보너스 번호
                bonus = int(item.get('보너스', 0))
                
                # 기계 유형 추정 (실제 데이터에 없는 경우 추정)
                machine_type = item.get('기계', self._estimate_machine_type(numbers))
                
                draw = LotteryDraw(
                    draw_no=int(item['회차']),
                    date=str(item['추첨일']),
                    numbers=sorted(numbers),
                    bonus=bonus,
                    machine_type=machine_type
                )
                
                # 데이터 유효성 검사
                if self._validate_draw(draw):
                    processed_draws.append(draw)
                    
            except (ValueError, KeyError, TypeError) as e:
                self.logger.warning(f"⚠️ Skipping invalid draw data: {e}")
                continue
                
        return processed_draws
        
    def _validate_draw(self, draw: LotteryDraw) -> bool:
        """추첨 데이터 유효성 검사"""
        # 번호 개수 확인
        if len(draw.numbers) != 6:
            return False
            
        # 번호 범위 확인 (1-45)
        if not all(1 <= num <= 45 for num in draw.numbers):
            return False
            
        # 중복 번호 확인
        if len(set(draw.numbers)) != 6:
            return False
            
        # 보너스 번호 범위 확인
        if draw.bonus < 1 or draw.bonus > 45:
            return False
            
        return True
        
    def _estimate_machine_type(self, numbers: List[int]) -> str:
        """번호 패턴으로 기계 유형 추정"""
        odd_count = len([n for n in numbers if n % 2 == 1])
        digit_sum = sum(numbers)
        
        # 간단한 휴리스틱으로 기계 타입 추정
        if odd_count <= 2 and digit_sum < 130:
            return "1호기"  # 보수적
        elif odd_count >= 4 and digit_sum > 150:
            return "3호기"  # 창의적
        else:
            return "2호기"  # 균형
            
    def _generate_ml_features(self) -> pd.DataFrame:
        """ML 모델용 특성 생성"""
        if not self._processed_draws:
            return pd.DataFrame()
            
        features = []
        
        for i, draw in enumerate(self._processed_draws):
            feature_row = {
                'draw_no': draw.draw_no,
                
                # 기본 번호 특성
                'num1': draw.numbers[0],
                'num2': draw.numbers[1], 
                'num3': draw.numbers[2],
                'num4': draw.numbers[3],
                'num5': draw.numbers[4],
                'num6': draw.numbers[5],
                'bonus': draw.bonus,
                
                # 통계적 특성
                'odd_count': draw.odd_count,
                'even_count': draw.even_count,
                'high_count': draw.high_count,
                'low_count': draw.low_count,
                'digit_sum': draw.digit_sum,
                'ac_value': draw.ac_value,
                
                # 간격 특성
                'gap_12': draw.numbers[1] - draw.numbers[0],
                'gap_23': draw.numbers[2] - draw.numbers[1],
                'gap_34': draw.numbers[3] - draw.numbers[2],
                'gap_45': draw.numbers[4] - draw.numbers[3],
                'gap_56': draw.numbers[5] - draw.numbers[4],
                
                # 기계 유형
                'machine_1ho': 1 if draw.machine_type == "1호기" else 0,
                'machine_2ho': 1 if draw.machine_type == "2호기" else 0,
                'machine_3ho': 1 if draw.machine_type == "3호기" else 0,
            }
            
            # 이전 추첨과의 관계
            if i > 0:
                prev_draw = self._processed_draws[i-1]
                common_nums = len(set(draw.numbers) & set(prev_draw.numbers))
                feature_row['common_with_prev'] = common_nums
            else:
                feature_row['common_with_prev'] = 0
                
            # 최근 빈도 (최근 10회)
            recent_draws = self._processed_draws[max(0, i-10):i]
            recent_numbers = []
            for rd in recent_draws:
                recent_numbers.extend(rd.numbers)
                
            for num in range(1, 46):
                feature_row[f'freq_recent_{num}'] = recent_numbers.count(num)
                
            features.append(feature_row)
            
        return pd.DataFrame(features)
        
    async def _update_statistics(self):
        """통계 정보 업데이트"""
        if not self._processed_draws:
            return
            
        loop = asyncio.get_event_loop()
        
        # 번호 빈도
        self._number_frequency = await loop.run_in_executor(
            self.executor, self._calculate_number_frequency
        )
        
        # 최근 패턴
        self._recent_patterns = await loop.run_in_executor(
            self.executor, self._analyze_recent_patterns
        )
        
    def _calculate_number_frequency(self) -> Dict[int, int]:
        """번호별 출현 빈도 계산"""
        frequency = {i: 0 for i in range(1, 46)}
        
        for draw in self._processed_draws:
            for num in draw.numbers:
                frequency[num] += 1
                
        return frequency
        
    def _analyze_recent_patterns(self, recent_count: int = 20) -> Dict[str, Any]:
        """최근 패턴 분석"""
        if len(self._processed_draws) < recent_count:
            recent_draws = self._processed_draws
        else:
            recent_draws = self._processed_draws[-recent_count:]
            
        patterns = {
            'avg_odd_count': np.mean([d.odd_count for d in recent_draws]),
            'avg_high_count': np.mean([d.high_count for d in recent_draws]),
            'avg_digit_sum': np.mean([d.digit_sum for d in recent_draws]),
            'avg_ac_value': np.mean([d.ac_value for d in recent_draws]),
            'machine_distribution': {}
        }
        
        # 기계별 분포
        for machine in ["1호기", "2호기", "3호기"]:
            count = len([d for d in recent_draws if d.machine_type == machine])
            patterns['machine_distribution'][machine] = count / len(recent_draws)
            
        return patterns
        
    # Public API 메소드들
    
    def get_all_draws(self) -> List[LotteryDraw]:
        """모든 추첨 데이터 반환"""
        return self._processed_draws or []
        
    def get_draws_by_machine(self, machine_type: str) -> List[LotteryDraw]:
        """기계별 추첨 데이터"""
        if not self._processed_draws:
            return []
        return [d for d in self._processed_draws if d.machine_type == machine_type]
        
    def get_recent_draws(self, count: int = 10) -> List[LotteryDraw]:
        """최근 N회 추첨 데이터"""
        if not self._processed_draws:
            return []
        return self._processed_draws[-count:]
        
    def get_ml_features(self) -> pd.DataFrame:
        """ML 특성 데이터프레임"""
        return self._ml_features or pd.DataFrame()
        
    def get_number_frequency(self) -> Dict[int, int]:
        """번호 빈도 통계"""
        return self._number_frequency or {}
        
    def get_recent_patterns(self) -> Dict[str, Any]:
        """최근 패턴 분석 결과"""
        return self._recent_patterns or {}
        
    def get_training_data(self, sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """시퀀스 학습용 데이터 생성"""
        if not self._processed_draws or len(self._processed_draws) < sequence_length + 1:
            return np.array([]), np.array([])
            
        sequences = []
        targets = []
        
        for i in range(len(self._processed_draws) - sequence_length):
            # 입력 시퀀스 (과거 sequence_length개 추첨)
            seq = []
            for j in range(sequence_length):
                draw = self._processed_draws[i + j]
                # 각 추첨을 벡터로 변환 (45차원 원핫 인코딩)
                vec = np.zeros(45)
                for num in draw.numbers:
                    vec[num - 1] = 1
                seq.append(vec)
            sequences.append(seq)
            
            # 타겟 (다음 추첨)
            next_draw = self._processed_draws[i + sequence_length]
            target = np.zeros(45)
            for num in next_draw.numbers:
                target[num - 1] = 1
            targets.append(target)
            
        return np.array(sequences), np.array(targets)
        
    async def refresh_data(self) -> bool:
        """데이터 새로고침"""
        try:
            await self.load_data()
            self.logger.info("✅ Data refreshed successfully")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to refresh data: {e}")
            return False
            
    def get_statistics_summary(self) -> Dict[str, Any]:
        """통계 요약 정보"""
        if not self._processed_draws:
            return {}
            
        return {
            'total_draws': len(self._processed_draws),
            'date_range': {
                'first': self._processed_draws[0].date,
                'last': self._processed_draws[-1].date
            },
            'machine_distribution': {
                machine: len([d for d in self._processed_draws if d.machine_type == machine])
                for machine in ["1호기", "2호기", "3호기", "알수없음"]
            },
            'number_frequency': self._number_frequency,
            'recent_patterns': self._recent_patterns
        }
        
    async def close(self):
        """리소스 정리"""
        if self.executor:
            self.executor.shutdown(wait=True)
            self.logger.info("✅ Data loader closed")

# 전역 데이터 로더 인스턴스 (싱글톤 패턴)
_data_loader = None

async def get_data_loader() -> DataLoader:
    """전역 데이터 로더 인스턴스 반환"""
    global _data_loader
    if _data_loader is None:
        _data_loader = DataLoader()
        await _data_loader.initialize()
    return _data_loader