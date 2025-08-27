# 📊 Statistical Predictor Agent - Ensemble ML Statistics Specialist
# Advanced statistical analysis using scikit-learn ensembles

import numpy as np
import pandas as pd
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .base import BaseAgent, AgentCapabilities, AgentStatus
from ..models.ensemble_predictor import LotteryEnsemblePredictor
from ..services.data_loader import get_data_loader

class StatisticalPredictorAgent(BaseAgent):
    """
    통계 예측 에이전트
    
    scikit-learn 앙상블을 활용한 통계적 특성 기반 예측:
    - 홀짝비율, AC값, 끝수합 등 통계적 특성 최적화
    - RandomForest + GradientBoosting + MLP 결합
    - 호기별 특화된 가중치 적용
    - 안정적이고 해석 가능한 예측
    """
    
    def __init__(self, agent_id: str, memory_manager, message_bus, config: Dict[str, Any]):
        super().__init__(agent_id, memory_manager, message_bus, config)
        
        self.capabilities = [
            AgentCapabilities.STATISTICAL_ANALYSIS,
            AgentCapabilities.DOMAIN_EXPERTISE
        ]
        
        # 새로운 앙상블 예측 모델
        self.ensemble_predictor = None
        self.data_loader = None
        
        # 통계 분석 설정
        self.feature_importance_threshold = config.get('feature_importance_threshold', 0.05)
        self.ensemble_weights = config.get('ensemble_weights', [0.4, 0.4, 0.2])
        
        # 호기별 특화 전략
        self.machine_strategies = {
            '1호기': {
                'high_frequency_weight': 0.35,
                'ac_complexity_weight': 0.25, 
                'conservative_range_weight': 0.20,
                'description': '신중한 전략가 - 고수 번호 가중, AC값 복잡도 선호'
            },
            '2호기': {
                'balance_optimization_weight': 0.40,
                'last_digit_harmony_weight': 0.30,
                'even_distribution_weight': 0.20,
                'description': '완벽한 조화 - 균형 최적화, 끝수합 중점'
            },
            '3호기': {
                'odd_preference_weight': 0.32,
                'diversity_bonus_weight': 0.28,
                'creative_combinations_weight': 0.25,
                'description': '창조적 혁신 - 홀수 선호, 다양성 추구'
            }
        }
        
    async def initialize(self):
        """에이전트 초기화"""
        self.logger.info(f"📊 Initializing Statistical Predictor Agent {self.agent_id}")
        
        try:
            # 새로운 앙상블 예측기 초기화
            ensemble_config = {
                'n_estimators': self.config.get('n_estimators', 50),
                'max_depth': self.config.get('max_depth', 8),
                'learning_rate': self.config.get('learning_rate', 0.1)
            }
            
            self.ensemble_predictor = LotteryEnsemblePredictor(ensemble_config)
            success = self.ensemble_predictor.initialize()
            
            if not success:
                self.logger.warning("⚠️ Ensemble predictor initialization failed, using fallback")
            
            # 데이터 로더 연결
            self.data_loader = await get_data_loader()
            
            # 모델 훈련 (과거 데이터로)
            await self._train_models_with_historical_data()
            
            self.status = AgentStatus.ACTIVE
            self.logger.info("✅ Statistical Predictor Agent initialized")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"❌ Failed to initialize Statistical Predictor: {e}")
            raise
            
    async def process_prediction(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """통계 기반 예측 처리"""
        machine_type = task_data['machine_type']
        sets_count = task_data['sets_count']
        
        self.logger.info(f"📊 Processing statistical analysis for {machine_type}")
        
        try:
            # 1. 과거 데이터 가져오기
            historical_data = self._get_historical_numbers(machine_type)
            
            # 2. 앙상블 모델로 예측
            if self.ensemble_predictor and hasattr(self.ensemble_predictor, 'predict'):
                model_predictions = self.ensemble_predictor.predict(
                    historical_data,
                    sets_count,
                    machine_type=machine_type
                )
                
                # 결과 포맷 조정
                predictions = []
                for i, pred in enumerate(model_predictions):
                    statistical_result = {
                        'set_number': i + 1,
                        'numbers': pred['numbers'],
                        'confidence_score': pred['confidence'],
                        'statistical_analysis': self._analyze_statistical_features(pred['numbers'], machine_type),
                        'selection_method': pred.get('method', 'ensemble_ml'),
                        'machine_strategy': machine_type,
                        'metadata': pred.get('metadata', {})
                    }
                    predictions.append(statistical_result)
            else:
                # fallback 사용
                predictions = await self._fallback_statistical_predictions(historical_data, sets_count, machine_type)
                
            # 3. 메타데이터 생성
            metadata = await self._generate_statistical_metadata(machine_type, predictions)
            
            return {
                'success': True,
                'agent_id': self.agent_id,
                'agent_type': 'statistical_predictor',
                'predictions': predictions,
                'metadata': metadata,
                'avg_confidence': np.mean([p['confidence_score'] for p in predictions])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Statistical prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': self.agent_id,
                'agent_type': 'statistical_predictor'
            }
            
    async def get_specialization_info(self) -> Dict[str, Any]:
        """특화 정보 반환"""
        model_info = {}
        if self.ensemble_predictor:
            try:
                model_info = {
                    'is_trained': getattr(self.ensemble_predictor, 'is_trained', False),
                    'models': ['RandomForest', 'GradientBoosting', 'MLP'],
                    'ensemble_strategy': 'weighted_voting'
                }
            except:
                pass
                
        return {
            'specialization': 'Statistical Ensemble Analysis',
            'model_architecture': 'scikit-learn Ensemble',
            'capabilities': [
                'Multi-algorithm ensemble prediction',
                'Statistical feature optimization',
                'Machine-specific strategy application', 
                'Interpretable statistical analysis'
            ],
            'machine_strategies': list(self.machine_strategies.keys()),
            'model_info': model_info,
            'feature_importance_threshold': self.feature_importance_threshold
        }
        
    def _get_historical_numbers(self, machine_type: str) -> List[List[int]]:
        """데이터 로더에서 과거 번호 가져오기"""
        try:
            if hasattr(self, 'data_loader') and self.data_loader:
                if machine_type == "전체":
                    draws = self.data_loader.get_all_draws()
                else:
                    draws = self.data_loader.get_draws_by_machine(machine_type)
                
                # 번호만 추출
                historical_numbers = []
                for draw in draws[-100:]:  # 최근 100회 (통계에는 더 많은 데이터 필요)
                    historical_numbers.append(draw.numbers)
                    
                return historical_numbers
            else:
                return []
        except Exception as e:
            self.logger.error(f"❌ Failed to get historical data: {e}")
            return []
            
    async def _train_models_with_historical_data(self):
        """과거 데이터로 모델 훈련"""
        try:
            if not self.data_loader or not self.ensemble_predictor:
                return
                
            # 전체 데이터 가져오기
            all_draws = self.data_loader.get_all_draws()
            
            if len(all_draws) < 30:
                self.logger.warning("⚠️ Insufficient data for training")
                return
                
            # 번호 데이터 추출
            historical_numbers = []
            for draw in all_draws:
                historical_numbers.append(draw.numbers)
                
            # 모델 훈련
            self.logger.info(f"🎯 Training ensemble models with {len(historical_numbers)} draws")
            success = self.ensemble_predictor.train(historical_numbers)
            
            if success:
                self.logger.info("✅ Model training completed")
            else:
                self.logger.warning("⚠️ Model training failed, using fallback")
                
        except Exception as e:
            self.logger.error(f"❌ Model training error: {e}")
            
    def _analyze_statistical_features(self, numbers: List[int], machine_type: str) -> Dict[str, Any]:
        """통계적 특성 분석"""
        try:
            # 기본 통계
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            high_count = sum(1 for n in numbers if n >= 23)
            digit_sum = sum(numbers)
            
            # AC값 계산
            sorted_nums = sorted(numbers)
            ac_value = 0
            for i in range(len(sorted_nums)):
                for j in range(i + 1, len(sorted_nums)):
                    diff = abs(sorted_nums[i] - sorted_nums[j])
                    if diff not in [abs(sorted_nums[k] - sorted_nums[l]) 
                                   for k in range(len(sorted_nums)) 
                                   for l in range(k + 1, len(sorted_nums)) 
                                   if (k, l) != (i, j)]:
                        ac_value += 1
                        
            # 기계별 특화 점수
            machine_strategy = self.machine_strategies.get(machine_type, {})
            strategy_score = self._calculate_machine_strategy_score(numbers, machine_strategy)
            
            return {
                'statistical_strength': 0.8,  # 기본 통계 모델 강도
                'odd_count': odd_count,
                'even_count': 6 - odd_count,
                'high_count': high_count,
                'low_count': 6 - high_count,
                'digit_sum': digit_sum,
                'ac_value': ac_value,
                'machine_strategy_score': strategy_score,
                'balance_score': self._calculate_balance_score(numbers),
                'machine_specific_analysis': machine_strategy
            }
        except Exception as e:
            self.logger.error(f"❌ Statistical analysis failed: {e}")
            return {'statistical_strength': 0.5}
            
    def _calculate_machine_strategy_score(self, numbers: List[int], strategy: Dict[str, Any]) -> float:
        """기계별 전략 점수 계산"""
        try:
            if not strategy:
                return 0.5
                
            score = 0.0
            
            # 홀수 선호도 (3호기)
            if 'odd_preference_weight' in strategy:
                odd_count = sum(1 for n in numbers if n % 2 == 1)
                if odd_count >= 4:  # 홀수가 4개 이상이면 보너스
                    score += strategy['odd_preference_weight']
                    
            # 균형 최적화 (2호기)
            if 'balance_optimization_weight' in strategy:
                balance_score = self._calculate_balance_score(numbers)
                score += strategy['balance_optimization_weight'] * balance_score
                
            # 보수적 범위 (1호기)
            if 'conservative_range_weight' in strategy:
                middle_count = sum(1 for n in numbers if 15 <= n <= 35)
                if middle_count >= 4:
                    score += strategy['conservative_range_weight']
                    
            return min(1.0, max(0.0, score))
            
        except:
            return 0.5
            
    def _calculate_balance_score(self, numbers: List[int]) -> float:
        """균형 점수 계산"""
        try:
            # 홀짝 균형
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            odd_balance = 1.0 - abs(odd_count - 3) / 3
            
            # 고저 균형
            high_count = sum(1 for n in numbers if n >= 23)
            high_balance = 1.0 - abs(high_count - 3) / 3
            
            # 분산 점수
            variance_score = min(1.0, np.std(numbers) / 15.0)
            
            return (odd_balance + high_balance + variance_score) / 3
            
        except:
            return 0.5
            
    async def _fallback_statistical_predictions(self, historical_data: List[List[int]], sets_count: int, machine_type: str) -> List[Dict[str, Any]]:
        """통계 기반 fallback 예측"""
        predictions = []
        
        try:
            # 빈도 분석 기반 예측
            if historical_data:
                all_numbers = []
                for draw in historical_data[-30:]:  # 최근 30회
                    all_numbers.extend(draw)
                    
                # 빈도 계산
                frequency = {}
                for num in range(1, 46):
                    frequency[num] = all_numbers.count(num)
                    
                for i in range(sets_count):
                    # 기계별 전략 적용한 가중 선택
                    adjusted_weights = self._apply_machine_strategy_weights(frequency, machine_type)
                    
                    if adjusted_weights:
                        total_weight = sum(adjusted_weights.values())
                        probabilities = [adjusted_weights.get(num, 0) / max(total_weight, 1) for num in range(1, 46)]
                        
                        selected = np.random.choice(
                            range(1, 46), size=6, replace=False, p=probabilities
                        )
                        numbers = sorted(selected.tolist())
                    else:
                        numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
                    
                    predictions.append({
                        'set_number': i + 1,
                        'numbers': numbers,
                        'confidence_score': 0.6,
                        'statistical_analysis': self._analyze_statistical_features(numbers, machine_type),
                        'selection_method': 'frequency_analysis',
                        'machine_strategy': machine_type
                    })
            else:
                # 완전 랜덤 fallback
                for i in range(sets_count):
                    numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
                    predictions.append({
                        'set_number': i + 1,
                        'numbers': numbers.tolist(),
                        'confidence_score': 0.3,
                        'statistical_analysis': self._analyze_statistical_features(numbers, machine_type),
                        'selection_method': 'random_fallback',
                        'machine_strategy': machine_type
                    })
                    
        except Exception as e:
            self.logger.error(f"❌ Fallback prediction failed: {e}")
            
        return predictions
        
    def _apply_machine_strategy_weights(self, frequency: Dict[int, int], machine_type: str) -> Dict[int, float]:
        """기계별 전략 가중치 적용"""
        try:
            adjusted_weights = {}
            strategy = self.machine_strategies.get(machine_type, {})
            
            for num in range(1, 46):
                base_weight = frequency.get(num, 0) + 1  # +1 to avoid zero
                
                # 기계별 조정
                if machine_type == "1호기":
                    # 중간 번호 선호
                    if 15 <= num <= 35:
                        base_weight *= 1.3
                elif machine_type == "3호기":
                    # 홀수 선호
                    if num % 2 == 1:
                        base_weight *= 1.2
                # 2호기는 균형 유지
                
                adjusted_weights[num] = base_weight
                
            return adjusted_weights
            
        except:
            return frequency
            
    async def _generate_statistical_metadata(self, machine_type: str, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """통계 메타데이터 생성"""
        return {
            'model_type': 'scikit-learn Ensemble',
            'prediction_method': 'statistical_ensemble',
            'machine_specialization': machine_type,
            'feature_analysis': {
                'avg_odd_count': np.mean([p['statistical_analysis'].get('odd_count', 3) for p in predictions]),
                'avg_high_count': np.mean([p['statistical_analysis'].get('high_count', 3) for p in predictions]),
                'avg_digit_sum': np.mean([p['statistical_analysis'].get('digit_sum', 135) for p in predictions]),
                'avg_ac_value': np.mean([p['statistical_analysis'].get('ac_value', 10) for p in predictions])
            },
            'machine_strategy_applied': self.machine_strategies.get(machine_type, {}),
            'ensemble_weights': self.ensemble_weights,
            'model_status': 'trained' if (self.ensemble_predictor and getattr(self.ensemble_predictor, 'is_trained', False)) else 'fallback'
        }