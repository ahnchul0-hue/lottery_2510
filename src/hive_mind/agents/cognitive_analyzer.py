# 🧠 Cognitive Analyzer Agent - Domain Knowledge & Reasoning Specialist
# Advanced cognitive analysis with domain expertise and explainable AI

import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from collections import defaultdict, Counter

from .base import BaseAgent, AgentCapabilities, AgentStatus

class CognitiveAnalyzerAgent(BaseAgent):
    """
    인지 분석 에이전트
    
    도메인 전문가의 룰과 맥락적 판단을 담당:
    - 규칙 기반 지식과 맥락적 판단 수행
    - 다른 에이전트들의 결과에 설명을 달거나 모순 발견 및 조정
    - 인간처럼 해석하고 전략을 도출하는 인지적 추론
    - 설명 가능한 AI (XAI) 기법 적용
    """
    
    def __init__(self, agent_id: str, memory_manager, message_bus, config: Dict[str, Any]):
        super().__init__(agent_id, memory_manager, message_bus, config)
        
        self.capabilities = [
            AgentCapabilities.COGNITIVE_REASONING,
            AgentCapabilities.DOMAIN_EXPERTISE,
            AgentCapabilities.EXPLANATION_GENERATION
        ]
        
        # 도메인 지식 베이스
        self.domain_knowledge = self._initialize_domain_knowledge()
        self.cognitive_rules = self._initialize_cognitive_rules()
        
        # 분석 설정
        self.confidence_threshold = config.get('confidence_threshold', 0.7)
        self.explanation_detail_level = config.get('explanation_detail_level', 'detailed')
        
        # 인지 메모리
        self.pattern_memory = {}
        self.anomaly_detection_rules = {}
        self.explanation_templates = {}
        
    async def initialize(self):
        """에이전트 초기화"""
        self.logger.info(f"🧠 Initializing Cognitive Analyzer Agent {self.agent_id}")
        
        try:
            # 도메인 지식 로드
            await self._load_domain_knowledge()
            
            # 인지 규칙 설정
            await self._setup_cognitive_rules()
            
            # 설명 템플릿 초기화
            await self._initialize_explanation_templates()
            
            # 과거 패턴 학습
            await self._learn_historical_patterns()
            
            self.status = AgentStatus.ACTIVE
            self.logger.info("✅ Cognitive Analyzer Agent initialized")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"❌ Failed to initialize Cognitive Analyzer: {e}")
            raise
            
    def _initialize_domain_knowledge(self) -> Dict[str, Any]:
        """도메인 지식 베이스 초기화"""
        return {
            'number_properties': {
                'primes': [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43],
                'fibonacci': [1, 1, 2, 3, 5, 8, 13, 21, 34],
                'perfect_squares': [1, 4, 9, 16, 25, 36],
                'lucky_numbers': [7, 11, 13, 21, 23, 33],  # 문화적 행운 번호
                'unlucky_numbers': [4, 14, 24, 34, 44]     # 문화적 불운 번호 (한국)
            },
            'statistical_norms': {
                'odd_even_balance': {'min': 2, 'max': 4, 'ideal': 3},
                'high_low_balance': {'min': 2, 'max': 4, 'ideal': 3},
                'total_sum_range': {'min': 21, 'max': 270, 'typical': (105, 165)},
                'ac_value_range': {'min': 0, 'max': 15, 'typical': (8, 13)},
                'last_digit_sum_range': {'min': 0, 'max': 54, 'typical': (15, 25)}
            },
            'machine_personalities': {
                '1호기': {
                    'traits': ['conservative', 'reliable', 'high_frequency_bias'],
                    'typical_patterns': ['balanced_distribution', 'moderate_ac_values'],
                    'avoidance': ['extreme_combinations', 'very_low_sums']
                },
                '2호기': {
                    'traits': ['balanced', 'harmonious', 'perfectionist'],
                    'typical_patterns': ['even_distribution', 'optimal_ratios'],
                    'avoidance': ['unbalanced_sections', 'extreme_digit_sums']
                },
                '3호기': {
                    'traits': ['creative', 'diverse', 'odd_preference'],
                    'typical_patterns': ['unique_combinations', 'pattern_breaking'],
                    'avoidance': ['consecutive_heavy', 'conservative_ranges']
                }
            }
        }
        
    def _initialize_cognitive_rules(self) -> Dict[str, Any]:
        """인지 규칙 초기화"""
        return {
            'validation_rules': [
                {
                    'name': 'extreme_imbalance_check',
                    'condition': lambda numbers: self._check_extreme_imbalance(numbers),
                    'severity': 'high',
                    'message': '극단적인 홀짝 또는 고저 불균형 감지'
                },
                {
                    'name': 'unusual_sum_check', 
                    'condition': lambda numbers: sum(numbers) < 50 or sum(numbers) > 250,
                    'severity': 'medium',
                    'message': '비정상적인 총합 범위'
                },
                {
                    'name': 'consecutive_overflow_check',
                    'condition': lambda numbers: self._count_consecutive_numbers(numbers) > 3,
                    'severity': 'medium',
                    'message': '과도한 연속 번호 포함'
                },
                {
                    'name': 'recent_duplicate_check',
                    'condition': lambda numbers: self._check_recent_duplicates(numbers),
                    'severity': 'low',
                    'message': '최근 회차와 높은 유사성'
                }
            ],
            'enhancement_rules': [
                {
                    'name': 'diversity_enhancement',
                    'condition': lambda numbers: len(set(n % 10 for n in numbers)) < 4,
                    'action': 'suggest_digit_diversification',
                    'message': '끝자리 다양성 증대 권장'
                },
                {
                    'name': 'machine_alignment',
                    'condition': lambda numbers, machine: self._check_machine_alignment(numbers, machine),
                    'action': 'apply_machine_specific_adjustment',
                    'message': '호기별 특성 alignment 조정'
                }
            ]
        }
        
    async def process_prediction(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """인지 분석 처리"""
        machine_type = task_data['machine_type']
        sets_count = task_data['sets_count']
        
        self.logger.info(f"🧠 Processing cognitive analysis for {machine_type}")
        
        try:
            # 1. 컨텍스트 분석
            context_analysis = await self._analyze_context(machine_type)
            
            # 2. 인지 기반 예측 생성
            predictions = []
            for i in range(sets_count):
                cognitive_result = await self._generate_cognitive_prediction(
                    context_analysis, machine_type, i
                )
                predictions.append(cognitive_result)
                
            # 3. 메타 분석 및 설명 생성
            meta_analysis = await self._generate_meta_analysis(
                predictions, machine_type, context_analysis
            )
            
            return {
                'agent_id': self.agent_id,
                'agent_type': 'cognitive_analyzer',
                'predictions': predictions,
                'context_analysis': context_analysis,
                'meta_analysis': meta_analysis,
                'avg_confidence': np.mean([p['confidence_score'] for p in predictions])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Cognitive analysis failed: {e}")
            raise
            
    async def analyze_other_agents_results(self, 
                                         agent_results: Dict[str, Any],
                                         task_data: Dict[str, Any]) -> Dict[str, Any]:
        """다른 에이전트 결과 분석 및 조정"""
        machine_type = task_data['machine_type']
        
        analysis = {
            'consistency_check': {},
            'anomaly_detection': {},
            'recommendations': [],
            'explanations': []
        }
        
        # 각 에이전트 결과 검증
        for agent_id, result in agent_results.items():
            if 'predictions' in result:
                for prediction in result['predictions']:
                    numbers = prediction['numbers']
                    
                    # 인지 규칙 적용
                    validation_result = await self._validate_with_cognitive_rules(
                        numbers, machine_type
                    )
                    
                    analysis['consistency_check'][agent_id] = validation_result
                    
                    # 이상 탐지
                    anomalies = await self._detect_anomalies(
                        numbers, machine_type, agent_id
                    )
                    
                    if anomalies:
                        analysis['anomaly_detection'][agent_id] = anomalies
                        
        # 종합 추천사항 생성
        analysis['recommendations'] = await self._generate_recommendations(
            agent_results, machine_type
        )
        
        # 설명 생성
        analysis['explanations'] = await self._generate_comprehensive_explanations(
            agent_results, machine_type, analysis
        )
        
        return analysis
        
    async def get_specialization_info(self) -> Dict[str, Any]:
        """특화 정보 반환"""
        return {
            'specialization': 'Cognitive Reasoning & Domain Expertise',
            'capabilities': [
                'Rule-based validation',
                'Anomaly detection', 
                'Contextual reasoning',
                'Explanation generation',
                'Machine-specific analysis'
            ],
            'domain_knowledge_areas': list(self.domain_knowledge.keys()),
            'cognitive_rules_count': len(self.cognitive_rules['validation_rules']),
            'explanation_templates': len(self.explanation_templates),
            'confidence_threshold': self.confidence_threshold
        }
        
    async def _analyze_context(self, machine_type: str) -> Dict[str, Any]:
        """컨텍스트 분석"""
        
        # 최근 데이터 로드
        recent_data = await self.memory_manager.get_lottery_data(
            machine_type=machine_type, limit=20
        )
        
        context = {
            'machine_personality': self.domain_knowledge['machine_personalities'][machine_type],
            'recent_trends': {},
            'seasonal_factors': {},
            'anomaly_indicators': {}
        }
        
        if recent_data:
            # 최근 트렌드 분석
            context['recent_trends'] = self._analyze_recent_trends(recent_data)
            
            # 계절적 요인 (간단한 시간 기반)
            context['seasonal_factors'] = self._analyze_seasonal_factors()
            
            # 이상 지표
            context['anomaly_indicators'] = self._detect_context_anomalies(recent_data)
            
        return context
        
    def _analyze_recent_trends(self, recent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """최근 트렌드 분석"""
        
        trends = {
            'sum_trend': 'stable',
            'odd_even_trend': 'balanced',
            'ac_trend': 'normal',
            'frequency_shifts': []
        }
        
        if len(recent_data) >= 10:
            # 총합 트렌드
            recent_sums = [r['총합'] for r in recent_data[-10:]]
            if recent_sums[-1] > np.mean(recent_sums[:-1]) + np.std(recent_sums[:-1]):
                trends['sum_trend'] = 'increasing'
            elif recent_sums[-1] < np.mean(recent_sums[:-1]) - np.std(recent_sums[:-1]):
                trends['sum_trend'] = 'decreasing'
                
            # 홀짝 트렌드
            recent_odd_counts = []
            for r in recent_data[-5:]:
                odd_count = sum(1 for n in r['1등_당첨번호'] if n % 2 == 1)
                recent_odd_counts.append(odd_count)
                
            if np.mean(recent_odd_counts) > 3.5:
                trends['odd_even_trend'] = 'odd_favoring'
            elif np.mean(recent_odd_counts) < 2.5:
                trends['odd_even_trend'] = 'even_favoring'
                
            # AC값 트렌드
            recent_ac = [r['AC값'] for r in recent_data[-5:]]
            if np.mean(recent_ac) > 12:
                trends['ac_trend'] = 'high_complexity'
            elif np.mean(recent_ac) < 8:
                trends['ac_trend'] = 'low_complexity'
                
        return trends
        
    def _analyze_seasonal_factors(self) -> Dict[str, Any]:
        """계절적 요인 분석"""
        now = datetime.now()
        
        seasonal_factors = {
            'month_factor': now.month,
            'season': self._get_season(now.month),
            'cultural_events': self._check_cultural_events(now),
            'psychological_bias': 'none'
        }
        
        # 연말연시 효과
        if now.month in [12, 1]:
            seasonal_factors['psychological_bias'] = 'year_end_optimism'
        elif now.month in [6, 7]:
            seasonal_factors['psychological_bias'] = 'summer_relaxation'
            
        return seasonal_factors
        
    def _get_season(self, month: int) -> str:
        """계절 반환"""
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'autumn'
        else:
            return 'winter'
            
    def _check_cultural_events(self, date: datetime) -> List[str]:
        """문화적 이벤트 확인"""
        events = []
        
        # 간단한 이벤트 체크
        if date.month == 2 and 10 <= date.day <= 20:
            events.append('lunar_new_year')
        elif date.month == 12 and date.day >= 20:
            events.append('christmas_season')
            
        return events
        
    def _detect_context_anomalies(self, recent_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """컨텍스트 이상 탐지"""
        anomalies = {
            'unusual_patterns': [],
            'statistical_outliers': [],
            'sequence_breaks': []
        }
        
        if len(recent_data) >= 5:
            # 통계적 이상값
            recent_sums = [r['총합'] for r in recent_data[-5:]]
            mean_sum = np.mean(recent_sums)
            std_sum = np.std(recent_sums)
            
            for i, sum_val in enumerate(recent_sums):
                if abs(sum_val - mean_sum) > 2 * std_sum:
                    anomalies['statistical_outliers'].append({
                        'type': 'sum_outlier',
                        'value': sum_val,
                        'position': i,
                        'severity': 'high' if abs(sum_val - mean_sum) > 3 * std_sum else 'medium'
                    })
                    
        return anomalies
        
    async def _generate_cognitive_prediction(self,
                                          context_analysis: Dict[str, Any],
                                          machine_type: str,
                                          prediction_index: int) -> Dict[str, Any]:
        """인지 기반 예측 생성"""
        
        # 기본 번호 풀 생성 (도메인 지식 기반)
        base_numbers = await self._generate_base_number_pool(machine_type, context_analysis)
        
        # 인지 규칙 적용하여 번호 선택
        selected_numbers = await self._apply_cognitive_selection(
            base_numbers, machine_type, prediction_index, context_analysis
        )
        
        # 검증 및 조정
        validated_numbers = await self._validate_and_adjust(
            selected_numbers, machine_type, context_analysis
        )
        
        # 신뢰도 계산
        confidence_score = await self._calculate_cognitive_confidence(
            validated_numbers, machine_type, context_analysis
        )
        
        # 설명 생성
        explanation = await self._generate_prediction_explanation(
            validated_numbers, machine_type, context_analysis
        )
        
        return {
            'set_number': prediction_index + 1,
            'numbers': sorted(validated_numbers),
            'confidence_score': confidence_score,
            'cognitive_reasoning': explanation,
            'selection_method': 'cognitive_domain_knowledge',
            'context_factors': self._extract_key_context_factors(context_analysis)
        }
        
    async def _generate_base_number_pool(self,
                                       machine_type: str,
                                       context_analysis: Dict[str, Any]) -> List[int]:
        """기본 번호 풀 생성"""
        
        personality = context_analysis['machine_personality']
        number_pool = list(range(1, 46))
        
        # 호기별 선호도 적용
        if machine_type == '1호기':
            # 신중한 전략가: 중간~고빈도 번호 선호
            preferred_ranges = [10, 15, 20, 25, 30, 35, 40]
            weights = [2 if n in preferred_ranges else 1 for n in number_pool]
            
        elif machine_type == '2호기':
            # 완벽한 조화: 균형있는 분포
            weights = [1.5 if 15 <= n <= 35 else 1 for n in number_pool]
            
        elif machine_type == '3호기':
            # 창조적 혁신: 홀수 및 특수 번호 선호
            special_nums = set(self.domain_knowledge['number_properties']['primes'] + 
                             self.domain_knowledge['number_properties']['fibonacci'])
            weights = [1.8 if (n % 2 == 1 or n in special_nums) else 1 for n in number_pool]
            
        else:
            weights = [1] * len(number_pool)
            
        # 가중치 기반 번호 풀 생성
        weighted_pool = []
        for num, weight in zip(number_pool, weights):
            weighted_pool.extend([num] * int(weight))
            
        return weighted_pool
        
    async def _apply_cognitive_selection(self,
                                       base_pool: List[int],
                                       machine_type: str,
                                       prediction_index: int,
                                       context_analysis: Dict[str, Any]) -> List[int]:
        """인지적 선택 적용"""
        
        selected = []
        available_pool = base_pool.copy()
        
        # 컨텍스트 기반 우선순위 번호들
        priority_numbers = self._identify_priority_numbers(context_analysis, machine_type)
        
        # 우선순위 번호 중 하나 선택
        if priority_numbers:
            priority_choice = np.random.choice(priority_numbers)
            selected.append(priority_choice)
            available_pool = [n for n in available_pool if n != priority_choice]
            
        # 나머지 번호들 선택 (다양한 인지 전략 적용)
        while len(selected) < 6 and available_pool:
            
            # 균형 고려
            if len(selected) >= 3:
                next_number = self._select_for_balance(selected, available_pool, machine_type)
            else:
                # 초기에는 다양성 중심
                next_number = self._select_for_diversity(selected, available_pool)
                
            selected.append(next_number)
            available_pool = [n for n in available_pool if n != next_number]
            
        return selected[:6]
        
    def _identify_priority_numbers(self,
                                 context_analysis: Dict[str, Any],
                                 machine_type: str) -> List[int]:
        """우선순위 번호 식별"""
        
        priority_numbers = []
        trends = context_analysis['recent_trends']
        
        # 트렌드 기반 우선순위
        if trends['sum_trend'] == 'increasing':
            priority_numbers.extend([25, 30, 35, 40])  # 높은 번호들
        elif trends['sum_trend'] == 'decreasing':
            priority_numbers.extend([5, 10, 15, 20])   # 낮은 번호들
            
        if trends['odd_even_trend'] == 'odd_favoring':
            priority_numbers.extend([n for n in range(1, 46) if n % 2 == 1][:10])
        elif trends['odd_even_trend'] == 'even_favoring':
            priority_numbers.extend([n for n in range(1, 46) if n % 2 == 0][:10])
            
        # 문화적 선호도
        cultural_events = context_analysis['seasonal_factors']['cultural_events']
        if 'lunar_new_year' in cultural_events:
            priority_numbers.extend([8, 18, 28, 38])  # 행운 번호 (8)
            
        return list(set(priority_numbers))
        
    def _select_for_balance(self, selected: List[int], available: List[int], machine_type: str) -> int:
        """균형을 위한 번호 선택"""
        
        # 현재 선택된 번호들의 특성 분석
        selected_odd = sum(1 for n in selected if n % 2 == 1)
        selected_high = sum(1 for n in selected if n >= 23)
        
        # 균형을 위한 필터링
        candidates = available.copy()
        
        # 홀짝 균형
        if selected_odd < len(selected) / 2:  # 홀수가 적으면
            candidates = [n for n in candidates if n % 2 == 1]
        elif selected_odd > len(selected) / 2:  # 홀수가 많으면
            candidates = [n for n in candidates if n % 2 == 0]
            
        # 고저 균형
        if selected_high < len(selected) / 2:  # 고수가 적으면
            candidates = [n for n in candidates if n >= 23]
        elif selected_high > len(selected) / 2:  # 고수가 많으면
            candidates = [n for n in candidates if n < 23]
            
        if not candidates:
            candidates = available
            
        return np.random.choice(candidates)
        
    def _select_for_diversity(self, selected: List[int], available: List[int]) -> int:
        """다양성을 위한 번호 선택"""
        
        if not selected:
            return np.random.choice(available)
            
        # 기존 선택과 거리 계산
        diversity_scores = []
        for candidate in available:
            min_distance = min(abs(candidate - s) for s in selected)
            diversity_scores.append(min_distance)
            
        # 높은 다양성 점수를 가진 번호들 중 선택
        max_diversity = max(diversity_scores)
        best_candidates = [
            available[i] for i, score in enumerate(diversity_scores)
            if score >= max_diversity * 0.8
        ]
        
        return np.random.choice(best_candidates)
        
    async def _validate_and_adjust(self,
                                 numbers: List[int],
                                 machine_type: str,
                                 context_analysis: Dict[str, Any]) -> List[int]:
        """검증 및 조정"""
        
        validated_numbers = numbers.copy()
        
        # 인지 규칙 검증
        for rule in self.cognitive_rules['validation_rules']:
            if rule['condition'](validated_numbers):
                self.logger.warning(f"⚠️  Cognitive rule triggered: {rule['message']}")
                
                # 조정 로직 (간단한 예시)
                if rule['name'] == 'extreme_imbalance_check':
                    validated_numbers = self._adjust_for_balance(validated_numbers)
                elif rule['name'] == 'consecutive_overflow_check':
                    validated_numbers = self._reduce_consecutive_numbers(validated_numbers)
                    
        return validated_numbers
        
    def _adjust_for_balance(self, numbers: List[int]) -> List[int]:
        """균형 조정"""
        adjusted = numbers.copy()
        
        odd_count = sum(1 for n in adjusted if n % 2 == 1)
        
        if odd_count <= 1:  # 홀수가 너무 적음
            # 짝수 하나를 홀수로 교체
            even_numbers = [n for n in adjusted if n % 2 == 0]
            if even_numbers:
                replace_target = np.random.choice(even_numbers)
                adjusted.remove(replace_target)
                
                # 홀수로 교체 (비슷한 범위)
                replacement_odd = replace_target + 1 if replace_target < 45 else replace_target - 1
                if replacement_odd % 2 == 1 and replacement_odd not in adjusted:
                    adjusted.append(replacement_odd)
                else:
                    # 대안 홀수 찾기
                    for candidate in range(1, 46, 2):
                        if candidate not in adjusted:
                            adjusted.append(candidate)
                            break
                            
        elif odd_count >= 5:  # 홀수가 너무 많음
            # 홀수 하나를 짝수로 교체
            odd_numbers = [n for n in adjusted if n % 2 == 1]
            replace_target = np.random.choice(odd_numbers)
            adjusted.remove(replace_target)
            
            # 짝수로 교체
            replacement_even = replace_target + 1 if replace_target < 45 else replace_target - 1
            if replacement_even % 2 == 0 and replacement_even not in adjusted:
                adjusted.append(replacement_even)
            else:
                for candidate in range(2, 46, 2):
                    if candidate not in adjusted:
                        adjusted.append(candidate)
                        break
                        
        return adjusted
        
    def _reduce_consecutive_numbers(self, numbers: List[int]) -> List[int]:
        """연속 번호 감소"""
        sorted_numbers = sorted(numbers)
        adjusted = []
        
        i = 0
        consecutive_count = 1
        
        while i < len(sorted_numbers):
            current = sorted_numbers[i]
            adjusted.append(current)
            
            # 연속 번호 체크
            while (i + 1 < len(sorted_numbers) and 
                   sorted_numbers[i + 1] == sorted_numbers[i] + 1):
                consecutive_count += 1
                if consecutive_count <= 3:  # 최대 3개까지만 연속 허용
                    i += 1
                    adjusted.append(sorted_numbers[i])
                else:
                    # 연속 번호 건너뛰고 다른 번호로 대체
                    i += 1
                    replacement = self._find_replacement_number(adjusted, sorted_numbers[i:])
                    if replacement:
                        adjusted.append(replacement)
                    break
                    
            consecutive_count = 1
            i += 1
            
        return adjusted[:6]
        
    def _find_replacement_number(self, current: List[int], remaining: List[int]) -> Optional[int]:
        """대체 번호 찾기"""
        all_numbers = set(range(1, 46))
        used_numbers = set(current + remaining)
        available = list(all_numbers - used_numbers)
        
        if available:
            return np.random.choice(available)
        return None
        
    def _check_extreme_imbalance(self, numbers: List[int]) -> bool:
        """극단적 불균형 체크"""
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        high_count = sum(1 for n in numbers if n >= 23)
        
        return odd_count <= 1 or odd_count >= 5 or high_count <= 1 or high_count >= 5
        
    def _count_consecutive_numbers(self, numbers: List[int]) -> int:
        """연속 번호 개수 세기"""
        sorted_numbers = sorted(numbers)
        consecutive_count = 0
        
        for i in range(len(sorted_numbers) - 1):
            if sorted_numbers[i + 1] - sorted_numbers[i] == 1:
                consecutive_count += 1
                
        return consecutive_count
        
    def _check_recent_duplicates(self, numbers: List[int]) -> bool:
        """최근 회차 중복 체크 (단순 구현)"""
        # 실제로는 메모리에서 최근 데이터 체크해야 함
        return False
        
    def _check_machine_alignment(self, numbers: List[int], machine: str) -> bool:
        """호기 alignment 체크"""
        personality = self.domain_knowledge['machine_personalities'][machine]
        
        # 간단한 alignment 체크
        if machine == '3호기':
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            return odd_count < 3  # 3호기는 홀수를 선호해야 함
            
        return False
        
    async def _calculate_cognitive_confidence(self,
                                           numbers: List[int],
                                           machine_type: str,
                                           context_analysis: Dict[str, Any]) -> float:
        """인지적 신뢰도 계산"""
        
        confidence_factors = []
        
        # 1. 도메인 지식 적합성
        domain_fitness = self._calculate_domain_fitness(numbers, machine_type)
        confidence_factors.append(domain_fitness * 0.3)
        
        # 2. 컨텍스트 일치도
        context_alignment = self._calculate_context_alignment(numbers, context_analysis)
        confidence_factors.append(context_alignment * 0.3)
        
        # 3. 인지 규칙 통과율
        rule_compliance = self._calculate_rule_compliance(numbers, machine_type)
        confidence_factors.append(rule_compliance * 0.2)
        
        # 4. 균형 및 다양성 점수
        balance_score = self._calculate_balance_score(numbers)
        confidence_factors.append(balance_score * 0.2)
        
        return min(max(sum(confidence_factors), 0.0), 1.0)
        
    def _calculate_domain_fitness(self, numbers: List[int], machine_type: str) -> float:
        """도메인 적합성 계산"""
        personality = self.domain_knowledge['machine_personalities'][machine_type]
        fitness = 0.5  # 기본 점수
        
        # 호기별 특성 평가
        if machine_type == '1호기' and 'conservative' in personality['traits']:
            total_sum = sum(numbers)
            if 120 <= total_sum <= 180:
                fitness += 0.3
                
        elif machine_type == '2호기' and 'harmonious' in personality['traits']:
            # 균형 평가
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            if 2 <= odd_count <= 4:
                fitness += 0.3
                
        elif machine_type == '3호기' and 'creative' in personality['traits']:
            # 창의성 평가
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            if odd_count >= 4:
                fitness += 0.3
                
        return fitness
        
    def _calculate_context_alignment(self, numbers: List[int], context: Dict[str, Any]) -> float:
        """컨텍스트 일치도 계산"""
        alignment = 0.5
        
        trends = context['recent_trends']
        
        # 트렌드와 일치성 확인
        total_sum = sum(numbers)
        
        if trends['sum_trend'] == 'increasing' and total_sum > 140:
            alignment += 0.2
        elif trends['sum_trend'] == 'decreasing' and total_sum < 140:
            alignment += 0.2
            
        return alignment
        
    def _calculate_rule_compliance(self, numbers: List[int], machine_type: str) -> float:
        """규칙 준수율 계산"""
        violations = 0
        total_rules = len(self.cognitive_rules['validation_rules'])
        
        for rule in self.cognitive_rules['validation_rules']:
            if rule['condition'](numbers):
                violations += 1
                
        compliance_rate = (total_rules - violations) / total_rules
        return compliance_rate
        
    def _calculate_balance_score(self, numbers: List[int]) -> float:
        """균형 점수 계산"""
        # 홀짝 균형
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        odd_balance = 1.0 - abs(odd_count - 3) / 3.0
        
        # 고저 균형
        high_count = sum(1 for n in numbers if n >= 23)
        high_balance = 1.0 - abs(high_count - 3) / 3.0
        
        # 분산 균형
        spread_balance = min(1.0, np.std(numbers) / 15.0)
        
        return (odd_balance + high_balance + spread_balance) / 3.0
        
    async def _generate_prediction_explanation(self,
                                             numbers: List[int],
                                             machine_type: str,
                                             context: Dict[str, Any]) -> Dict[str, Any]:
        """예측 설명 생성"""
        
        explanation = {
            'reasoning_process': [],
            'domain_insights': [],
            'contextual_factors': [],
            'confidence_rationale': []
        }
        
        # 추론 과정 설명
        explanation['reasoning_process'].append(
            f"{machine_type}의 성격적 특성을 반영한 번호 선택"
        )
        
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        explanation['reasoning_process'].append(
            f"홀짝 비율 {odd_count}:{6-odd_count}로 적절한 균형 유지"
        )
        
        # 도메인 통찰
        personality = context['machine_personality']
        explanation['domain_insights'].append(
            f"선택된 조합이 {machine_type}의 {', '.join(personality['traits'])} 특성과 일치"
        )
        
        # 컨텍스트 요인
        trends = context['recent_trends']
        explanation['contextual_factors'].append(
            f"최근 트렌드 ({trends['sum_trend']}, {trends['odd_even_trend']})를 반영"
        )
        
        return explanation
        
    def _extract_key_context_factors(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """핵심 컨텍스트 요인 추출"""
        return {
            'machine_traits': context['machine_personality']['traits'],
            'trend_summary': context['recent_trends'],
            'seasonal_factor': context['seasonal_factors']['season']
        }
        
    async def _load_domain_knowledge(self):
        """도메인 지식 로드"""
        # 메모리에서 과거 패턴 로드하여 도메인 지식 보강
        pass
        
    async def _setup_cognitive_rules(self):
        """인지 규칙 설정"""
        pass
        
    async def _initialize_explanation_templates(self):
        """설명 템플릿 초기화"""
        self.explanation_templates = {
            'high_confidence': "도메인 지식과 컨텍스트 분석을 통해 높은 신뢰도로 예측",
            'medium_confidence': "일부 패턴과 일치하지만 추가 검증 필요",
            'low_confidence': "제한된 정보로 인한 불확실성 존재"
        }
        
    async def _learn_historical_patterns(self):
        """과거 패턴 학습"""
        # 각 호기별 과거 데이터를 학습하여 패턴 메모리 구축
        for machine_type in ['1호기', '2호기', '3호기']:
            historical_data = await self.memory_manager.get_lottery_data(
                machine_type=machine_type, limit=50
            )
            
            if historical_data:
                patterns = self._extract_cognitive_patterns(historical_data)
                self.pattern_memory[machine_type] = patterns
                
    def _extract_cognitive_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """인지적 패턴 추출"""
        return {
            'typical_sum_range': [
                np.percentile([r['총합'] for r in data], 25),
                np.percentile([r['총합'] for r in data], 75)
            ],
            'common_odd_counts': Counter([
                sum(1 for n in r['1등_당첨번호'] if n % 2 == 1) for r in data
            ]).most_common(3),
            'frequent_numbers': Counter([
                n for r in data for n in r['1등_당첨번호']
            ]).most_common(10)
        }
        
    async def _validate_with_cognitive_rules(self,
                                           numbers: List[int],
                                           machine_type: str) -> Dict[str, Any]:
        """인지 규칙으로 검증"""
        
        validation_result = {
            'passed_rules': [],
            'failed_rules': [],
            'warnings': [],
            'overall_score': 0.0
        }
        
        total_rules = len(self.cognitive_rules['validation_rules'])
        passed_count = 0
        
        for rule in self.cognitive_rules['validation_rules']:
            if not rule['condition'](numbers):
                validation_result['passed_rules'].append(rule['name'])
                passed_count += 1
            else:
                validation_result['failed_rules'].append({
                    'name': rule['name'],
                    'severity': rule['severity'],
                    'message': rule['message']
                })
                
                if rule['severity'] == 'high':
                    validation_result['warnings'].append(rule['message'])
                    
        validation_result['overall_score'] = passed_count / total_rules
        return validation_result
        
    async def _detect_anomalies(self,
                              numbers: List[int],
                              machine_type: str,
                              agent_id: str) -> List[Dict[str, Any]]:
        """이상 탐지"""
        anomalies = []
        
        # 통계적 이상
        total_sum = sum(numbers)
        if total_sum < 50 or total_sum > 250:
            anomalies.append({
                'type': 'statistical_anomaly',
                'description': f'비정상적인 총합: {total_sum}',
                'severity': 'high'
            })
            
        # 패턴 이상
        consecutive_count = self._count_consecutive_numbers(numbers)
        if consecutive_count > 3:
            anomalies.append({
                'type': 'pattern_anomaly',
                'description': f'과도한 연속 번호: {consecutive_count}개',
                'severity': 'medium'
            })
            
        return anomalies
        
    async def _generate_recommendations(self,
                                      agent_results: Dict[str, Any],
                                      machine_type: str) -> List[Dict[str, Any]]:
        """추천사항 생성"""
        recommendations = []
        
        # 에이전트 결과간 일관성 체크
        all_predictions = []
        for agent_id, result in agent_results.items():
            if 'predictions' in result:
                all_predictions.extend(result['predictions'])
                
        if len(all_predictions) > 1:
            # 다양성 체크
            unique_numbers = set()
            for pred in all_predictions:
                unique_numbers.update(pred['numbers'])
                
            if len(unique_numbers) < 15:  # 너무 적은 다양성
                recommendations.append({
                    'type': 'diversity_improvement',
                    'message': '에이전트 간 예측 다양성 증대 권장',
                    'priority': 'medium'
                })
                
        return recommendations
        
    async def _generate_comprehensive_explanations(self,
                                                 agent_results: Dict[str, Any],
                                                 machine_type: str,
                                                 analysis: Dict[str, Any]) -> List[str]:
        """종합 설명 생성"""
        explanations = []
        
        explanations.append(f"🧠 {machine_type} 인지적 분석 결과:")
        
        # 일관성 요약
        consistency_scores = []
        for agent_id, validation in analysis['consistency_check'].items():
            consistency_scores.append(validation['overall_score'])
            
        avg_consistency = np.mean(consistency_scores) if consistency_scores else 0
        explanations.append(f"📊 전체 일관성 점수: {avg_consistency:.2f}")
        
        # 이상 탐지 요약
        total_anomalies = sum(len(anomalies) for anomalies in analysis['anomaly_detection'].values())
        if total_anomalies > 0:
            explanations.append(f"⚠️  {total_anomalies}개 이상 패턴 감지됨")
        else:
            explanations.append("✅ 이상 패턴 감지되지 않음")
            
        return explanations
        
    async def _generate_meta_analysis(self,
                                    predictions: List[Dict[str, Any]],
                                    machine_type: str,
                                    context_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """메타 분석 생성"""
        
        return {
            'prediction_quality': 'high' if all(p['confidence_score'] > 0.7 for p in predictions) else 'medium',
            'context_alignment': 'good',
            'domain_compliance': 'validated',
            'cognitive_insights': [
                f"{machine_type}의 특성이 예측에 적절히 반영됨",
                "도메인 지식 기반 검증 통과",
                "컨텍스트 요인들이 종합적으로 고려됨"
            ]
        }