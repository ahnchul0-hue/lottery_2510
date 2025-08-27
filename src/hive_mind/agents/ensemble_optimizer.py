# 🤖 Ensemble Optimizer Agent - Advanced Ensemble Coordination Specialist
# Optimizes prediction combination strategies with dynamic weight adjustment

import numpy as np
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import json

from .base import BaseAgent, AgentCapabilities, AgentStatus

class EnsembleOptimizer(BaseAgent):
    """
    앙상블 최적화 전문 에이전트
    
    Claude-Flow 개선사항:
    - 동적 가중치 조정
    - 성능 기반 모델 선택
    - 실시간 앙상블 최적화
    - 기계별 전략 통합
    """
    
    def __init__(self, agent_id: str, memory_manager, message_bus, config: Dict[str, Any]):
        super().__init__(agent_id, memory_manager, message_bus, config)
        
        self.capabilities = [
            AgentCapabilities.ENSEMBLE_OPTIMIZATION,
            AgentCapabilities.STATISTICAL_ANALYSIS,
            AgentCapabilities.EXPLANATION_GENERATION
        ]
        
        # 앙상블 전략 설정
        self.ensemble_strategies = {
            'voting': self._voting_ensemble,
            'stacking': self._stacking_ensemble,
            'blending': self._blending_ensemble,
            'dynamic': self._dynamic_ensemble
        }
        
        # 성능 추적
        self.model_performances = {}
        self.ensemble_history = []
        self.weight_optimization_history = []
        
        # 동적 가중치
        self.current_weights = {
            'pattern_analyzer': 0.35,
            'statistical_predictor': 0.35,
            'cognitive_analyzer': 0.30
        }
        
        # 기계별 최적화 전략
        self.machine_strategies = {
            '1호기': {
                'conservative_weight': 0.8,
                'diversity_penalty': 0.2,
                'frequent_number_boost': 0.15
            },
            '2호기': {
                'balance_weight': 0.7,
                'distribution_focus': 0.25,
                'ac_value_importance': 0.2
            },
            '3호기': {
                'creative_weight': 0.6,
                'pattern_diversity': 0.3,
                'odd_preference': 0.15
            }
        }
        
    async def initialize(self):
        """에이전트 초기화"""
        self.logger.info(f"🚀 Initializing EnsembleOptimizer {self.agent_id}")
        
        try:
            # 과거 성능 데이터 로드
            await self._load_performance_history()
            
            # 가중치 초기화
            await self._initialize_weights()
            
            # 메타-러너 모델 초기화
            self.meta_learner = LogisticRegression(random_state=42)
            
            self.status = AgentStatus.ACTIVE
            self.logger.info(f"✅ EnsembleOptimizer {self.agent_id} initialized successfully")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"❌ Failed to initialize EnsembleOptimizer: {e}")
            raise
            
    async def process_prediction(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """앙상블 예측 처리"""
        start_time = datetime.now()
        
        try:
            # 개별 모델 예측 수집
            agent_predictions = await self._collect_agent_predictions(task_data)
            
            # 앙상블 전략 선택
            strategy = task_data.get('ensemble_strategy', 'dynamic')
            machine_type = task_data.get('machine_type', '2호기')
            
            # 앙상블 실행
            ensemble_result = await self.ensemble_strategies[strategy](
                agent_predictions, machine_type
            )
            
            # 성능 모니터링
            await self._track_performance(ensemble_result, start_time)
            
            # 가중치 업데이트
            await self._update_weights(agent_predictions, ensemble_result)
            
            return {
                'success': True,
                'ensemble_predictions': ensemble_result['predictions'],
                'confidence_scores': ensemble_result['confidence'],
                'strategy_used': strategy,
                'model_contributions': ensemble_result['contributions'],
                'optimization_insights': await self._generate_optimization_insights(),
                'processing_time': (datetime.now() - start_time).total_seconds() * 1000
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ensemble prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_strategy': 'voting'
            }
            
    async def _collect_agent_predictions(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """개별 에이전트 예측 수집"""
        predictions = {}
        
        # 병렬로 예측 수집
        tasks = []
        agent_types = ['pattern_analyzer', 'statistical_predictor', 'cognitive_analyzer']
        
        for agent_type in agent_types:
            task = asyncio.create_task(
                self._request_agent_prediction(agent_type, task_data)
            )
            tasks.append((agent_type, task))
            
        # 결과 수집
        for agent_type, task in tasks:
            try:
                result = await asyncio.wait_for(task, timeout=30.0)
                if result.get('success'):
                    predictions[agent_type] = result
                else:
                    self.logger.warning(f"⚠️ {agent_type} prediction failed")
                    
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️ {agent_type} prediction timeout")
                
        return predictions
        
    async def _request_agent_prediction(self, agent_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """특정 에이전트에게 예측 요청"""
        try:
            # 메시지 버스를 통해 예측 요청
            await self.send_message(
                recipient=agent_type,
                message_type='prediction_request',
                content=task_data
            )
            
            # 응답 대기 (실제 구현에서는 메시지 버스에서 응답 수신)
            # 여기서는 임시로 더미 데이터 반환
            return await self._generate_dummy_prediction(agent_type, task_data)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to request prediction from {agent_type}: {e}")
            return {'success': False, 'error': str(e)}
            
    async def _generate_dummy_prediction(self, agent_type: str, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """더미 예측 생성 (실제 에이전트 통신 전까지 임시)"""
        np.random.seed(42 + hash(agent_type) % 1000)
        
        if agent_type == 'pattern_analyzer':
            # 패턴 기반 예측
            numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
            confidence = np.random.uniform(0.7, 0.9)
            
        elif agent_type == 'statistical_predictor':
            # 통계 기반 예측
            numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
            confidence = np.random.uniform(0.6, 0.8)
            
        else:  # cognitive_analyzer
            # 인지 기반 예측
            numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
            confidence = np.random.uniform(0.5, 0.7)
            
        return {
            'success': True,
            'predictions': [{'numbers': numbers, 'confidence': confidence}],
            'agent_type': agent_type,
            'metadata': {
                'processing_time': np.random.uniform(50, 150),
                'model_version': '1.0.0'
            }
        }
        
    async def _voting_ensemble(self, predictions: Dict[str, Any], machine_type: str) -> Dict[str, Any]:
        """투표 기반 앙상블"""
        if not predictions:
            raise ValueError("No predictions available for ensemble")
            
        # 모든 예측 수집
        all_predictions = []
        weights = []
        
        for agent_type, result in predictions.items():
            if result.get('success') and result.get('predictions'):
                for pred in result['predictions']:
                    all_predictions.append(pred['numbers'])
                    weights.append(self.current_weights.get(agent_type, 0.33))
                    
        if not all_predictions:
            raise ValueError("No valid predictions for ensemble")
            
        # 가중 투표
        number_votes = {}
        for i, prediction in enumerate(all_predictions):
            weight = weights[i]
            for num in prediction:
                number_votes[num] = number_votes.get(num, 0) + weight
                
        # 상위 6개 번호 선택
        sorted_numbers = sorted(number_votes.items(), key=lambda x: x[1], reverse=True)
        selected_numbers = [num for num, _ in sorted_numbers[:6]]
        
        # 기계별 전략 적용
        selected_numbers = await self._apply_machine_strategy(selected_numbers, machine_type)
        
        return {
            'predictions': [{'numbers': sorted(selected_numbers), 'confidence': 0.75}],
            'confidence': [0.75],
            'contributions': {
                agent: len([p for p in predictions[agent]['predictions']]) 
                for agent in predictions.keys()
            }
        }
        
    async def _stacking_ensemble(self, predictions: Dict[str, Any], machine_type: str) -> Dict[str, Any]:
        """스태킹 앙상블 (메타-러닝 기반)"""
        # 스태킹은 과거 데이터가 필요하므로 여기서는 간단한 구현
        return await self._voting_ensemble(predictions, machine_type)
        
    async def _blending_ensemble(self, predictions: Dict[str, Any], machine_type: str) -> Dict[str, Any]:
        """블렌딩 앙상블 (선형 조합)"""
        if not predictions:
            raise ValueError("No predictions available for blending")
            
        # 각 에이전트의 예측을 가중 평균
        blended_scores = np.zeros(45)  # 1-45번 번호
        total_weight = 0
        
        for agent_type, result in predictions.items():
            if result.get('success') and result.get('predictions'):
                weight = self.current_weights.get(agent_type, 0.33)
                
                # 예측 번호에 가중치 적용
                for pred in result['predictions']:
                    for num in pred['numbers']:
                        blended_scores[num - 1] += weight * pred.get('confidence', 0.5)
                        
                total_weight += weight
                
        if total_weight > 0:
            blended_scores /= total_weight
            
        # 상위 6개 선택
        top_indices = np.argsort(blended_scores)[-6:]
        selected_numbers = sorted([idx + 1 for idx in top_indices])
        
        # 기계별 전략 적용
        selected_numbers = await self._apply_machine_strategy(selected_numbers, machine_type)
        
        confidence = np.mean(blended_scores[top_indices])
        
        return {
            'predictions': [{'numbers': selected_numbers, 'confidence': confidence}],
            'confidence': [confidence],
            'contributions': {
                agent: self.current_weights.get(agent, 0.33) 
                for agent in predictions.keys()
            }
        }
        
    async def _dynamic_ensemble(self, predictions: Dict[str, Any], machine_type: str) -> Dict[str, Any]:
        """동적 앙상블 (성능 기반 가중치 조정)"""
        # 최근 성능에 따라 가중치 동적 조정
        await self._adjust_dynamic_weights()
        
        # 블렌딩 앙상블 실행
        return await self._blending_ensemble(predictions, machine_type)
        
    async def _apply_machine_strategy(self, numbers: List[int], machine_type: str) -> List[int]:
        """기계별 전략 적용"""
        strategy = self.machine_strategies.get(machine_type, self.machine_strategies['2호기'])
        
        if machine_type == '1호기':
            # 보수적 전략: 자주 나오는 번호 선호
            # (실제로는 frequency 데이터 필요)
            return sorted(numbers)
            
        elif machine_type == '2호기':
            # 균형 전략: 홀짝, 고저 균형
            odds = [n for n in numbers if n % 2 == 1]
            evens = [n for n in numbers if n % 2 == 0]
            
            # 홀짝 균형 조정
            while len(odds) < 3 and len(evens) > 3:
                # 짝수 중 하나를 홀수로 교체
                even_to_replace = evens.pop()
                for odd in range(1, 46, 2):
                    if odd not in numbers:
                        numbers[numbers.index(even_to_replace)] = odd
                        break
                        
            return sorted(numbers)
            
        elif machine_type == '3호기':
            # 창의적 전략: 홀수 선호, 다양성 중시
            odds = [n for n in numbers if n % 2 == 1]
            
            # 홀수 비율 증가
            if len(odds) < 4:
                evens = [n for n in numbers if n % 2 == 0]
                for even in evens[:1]:  # 짝수 하나를 홀수로
                    for odd in range(1, 46, 2):
                        if odd not in numbers:
                            numbers[numbers.index(even)] = odd
                            break
                            
            return sorted(numbers)
            
        return sorted(numbers)
        
    async def _adjust_dynamic_weights(self):
        """동적 가중치 조정"""
        if not self.model_performances:
            return
            
        # 최근 성능 기반 가중치 계산
        total_performance = sum(self.model_performances.values())
        if total_performance > 0:
            for agent in self.current_weights:
                if agent in self.model_performances:
                    self.current_weights[agent] = (
                        self.model_performances[agent] / total_performance
                    )
                    
        # 가중치 정규화
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for agent in self.current_weights:
                self.current_weights[agent] /= total_weight
                
    async def _track_performance(self, result: Dict[str, Any], start_time: datetime):
        """성능 추적"""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'processing_time_ms': processing_time,
            'prediction_count': len(result.get('predictions', [])),
            'confidence_avg': np.mean(result.get('confidence', [0])),
            'model_contributions': result.get('contributions', {})
        }
        
        self.ensemble_history.append(performance_data)
        
        # 최근 100개 기록만 유지
        if len(self.ensemble_history) > 100:
            self.ensemble_history = self.ensemble_history[-100:]
            
        # 메모리에 저장
        await self.memory_manager.store_agent_performance(
            self.agent_id, performance_data
        )
        
    async def _update_weights(self, predictions: Dict[str, Any], result: Dict[str, Any]):
        """가중치 업데이트"""
        # 실제 결과와 비교하여 가중치 업데이트 (실제 구현에서는 검증 데이터 필요)
        # 여기서는 예측 신뢰도 기반 임시 업데이트
        
        for agent_type in self.current_weights:
            if agent_type in predictions:
                pred_result = predictions[agent_type]
                if pred_result.get('success'):
                    # 임시 성능 점수 계산
                    confidence = np.mean([
                        p.get('confidence', 0.5) 
                        for p in pred_result.get('predictions', [])
                    ])
                    self.model_performances[agent_type] = confidence
                    
    async def _generate_optimization_insights(self) -> Dict[str, Any]:
        """최적화 인사이트 생성"""
        if not self.ensemble_history:
            return {'message': 'No performance history available'}
            
        recent_performance = self.ensemble_history[-10:] if len(self.ensemble_history) >= 10 else self.ensemble_history
        
        avg_processing_time = np.mean([p['processing_time_ms'] for p in recent_performance])
        avg_confidence = np.mean([p['confidence_avg'] for p in recent_performance])
        
        insights = {
            'current_weights': self.current_weights,
            'performance_metrics': {
                'avg_processing_time_ms': avg_processing_time,
                'avg_confidence': avg_confidence,
                'total_predictions': len(self.ensemble_history)
            },
            'recommendations': []
        }
        
        # 성능 기반 권장사항
        if avg_processing_time > 150:
            insights['recommendations'].append('Consider reducing model complexity for faster inference')
            
        if avg_confidence < 0.6:
            insights['recommendations'].append('Model retraining may be needed to improve confidence')
            
        return insights
        
    async def _load_performance_history(self):
        """과거 성능 데이터 로드"""
        try:
            # 메모리 관리자에서 성능 데이터 로드
            history = await self.memory_manager.get_agent_performance(self.agent_id)
            if history:
                self.ensemble_history = history[-100:]  # 최근 100개
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not load performance history: {e}")
            
    async def _initialize_weights(self):
        """가중치 초기화"""
        # 설정에서 초기 가중치 로드
        if 'initial_weights' in self.config:
            self.current_weights.update(self.config['initial_weights'])
            
        # 가중치 정규화
        total = sum(self.current_weights.values())
        if total > 0:
            for key in self.current_weights:
                self.current_weights[key] /= total
                
    async def get_specialization_info(self) -> Dict[str, Any]:
        """특화 정보 반환"""
        return {
            'specialization': 'Ensemble Optimization',
            'capabilities': [cap.value for cap in self.capabilities],
            'ensemble_strategies': list(self.ensemble_strategies.keys()),
            'current_weights': self.current_weights,
            'performance_history_length': len(self.ensemble_history),
            'machine_strategies': list(self.machine_strategies.keys()),
            'optimization_focus': [
                'Dynamic weight adjustment',
                'Performance-based model selection', 
                'Real-time ensemble optimization',
                'Machine-specific strategy integration'
            ]
        }
        
    async def get_performance_summary(self) -> Dict[str, Any]:
        """성능 요약 정보"""
        if not self.ensemble_history:
            return {'status': 'No performance data available'}
            
        recent = self.ensemble_history[-20:] if len(self.ensemble_history) >= 20 else self.ensemble_history
        
        return {
            'total_predictions': len(self.ensemble_history),
            'recent_performance': {
                'avg_processing_time_ms': np.mean([p['processing_time_ms'] for p in recent]),
                'avg_confidence': np.mean([p['confidence_avg'] for p in recent]),
                'prediction_count': len(recent)
            },
            'current_model_weights': self.current_weights,
            'model_performances': self.model_performances
        }