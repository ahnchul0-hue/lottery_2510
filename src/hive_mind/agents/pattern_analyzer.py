# 🔍 Pattern Analyzer Agent - Neural Pattern Recognition Specialist
# Advanced pattern recognition using PyTorch Transformers

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import json
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
import logging

from .base import BaseAgent, AgentCapabilities, AgentStatus
from ..models.lottery_transformer import LotteryPatternPredictor
from ..services.data_loader import get_data_loader

class PositionalEncoding(nn.Module):
    """Transformer용 위치 인코딩"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class LotteryTransformer(nn.Module):
    """로또 번호 패턴 학습용 Transformer 모델"""
    def __init__(self, 
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 dim_feedforward: int = 512,
                 max_seq_len: int = 100,
                 vocab_size: int = 46):  # 1-45 + padding
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 번호 임베딩 (1-45 번호 → d_model 차원)
        self.number_embedding = nn.Embedding(vocab_size, d_model)
        
        # 위치 인코딩
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # Transformer 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # 출력 헤드들
        self.probability_head = nn.Linear(d_model, 45)  # 번호별 확률
        self.pattern_head = nn.Linear(d_model, 64)      # 패턴 특성
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        # 임베딩 + 위치 인코딩
        embedded = self.number_embedding(x) * math.sqrt(self.d_model)
        embedded = self.positional_encoding(embedded)
        embedded = self.dropout(embedded)
        
        # Transformer 인코딩
        transformer_out = self.transformer(embedded)
        
        # 마지막 토큰의 출력 사용
        last_output = transformer_out[:, -1, :]
        
        # 각 헤드별 출력
        probabilities = F.softmax(self.probability_head(last_output), dim=-1)
        pattern_features = torch.tanh(self.pattern_head(last_output))
        
        return {
            'probabilities': probabilities,
            'pattern_features': pattern_features,
            'hidden_states': transformer_out
        }

class PatternAnalyzerAgent(BaseAgent):
    """
    패턴 분석 에이전트
    
    PyTorch Transformer를 활용한 딥러닝 기반 패턴 인식:
    - 시계열 번호 시퀀스의 숨겨진 패턴 발견
    - Multi-head Attention으로 번호간 상관관계 포착
    - 확률 분포 기반 정교한 예측
    - 창의적 번호 조합 생성
    """
    
    def __init__(self, agent_id: str, memory_manager, message_bus, config: Dict[str, Any]):
        super().__init__(agent_id, memory_manager, message_bus, config)
        
        self.capabilities = [
            AgentCapabilities.PATTERN_RECOGNITION,
            AgentCapabilities.NEURAL_PROCESSING
        ]
        
        # 모델 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[LotteryPatternPredictor] = None
        self.model_config = {
            'd_model': config.get('d_model', 64),
            'nhead': config.get('n_heads', 8),
            'num_layers': config.get('n_layers', 4),
            'dropout': config.get('dropout', 0.1)
        }
        
        # 패턴 분석 설정
        self.sequence_length = config.get('sequence_length', 20)
        self.prediction_diversity = config.get('prediction_diversity', 0.3)
        
        # 패턴 메모리
        self.discovered_patterns = {}
        self.pattern_confidence = {}
        
    async def initialize(self):
        """에이전트 초기화"""
        self.logger.info(f"🔍 Initializing Pattern Analyzer Agent {self.agent_id}")
        
        try:
            # 새로운 LotteryPatternPredictor 모델 초기화
            self.model = LotteryPatternPredictor(self.model_config)
            success = self.model.initialize()
            
            if not success:
                self.logger.warning("⚠️ Model initialization failed, using fallback")
            
            # 데이터 로더 연결
            self.data_loader = await get_data_loader()
            
            # 기본 패턴 분석 수행
            await self._analyze_base_patterns()
            
            self.status = AgentStatus.ACTIVE
            self.logger.info("✅ Pattern Analyzer Agent initialized")
            
        except Exception as e:
            self.status = AgentStatus.ERROR
            self.logger.error(f"❌ Failed to initialize Pattern Analyzer: {e}")
            raise
            
    async def process_prediction(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """패턴 기반 예측 처리"""
        machine_type = task_data['machine_type']
        sets_count = task_data['sets_count']
        
        self.logger.info(f"🧠 Processing pattern analysis for {machine_type}")
        
        try:
            # 1. 과거 데이터 로드 (새로운 데이터 로더 사용)
            historical_data = self._get_historical_numbers(machine_type)
            
            # 2. 모델을 사용한 예측 생성
            if self.model and hasattr(self.model, 'predict'):
                model_predictions = self.model.predict(
                    historical_data, 
                    sets_count,
                    diversity_threshold=self.prediction_diversity
                )
                
                # 결과 포맷 조정
                predictions = []
                for i, pred in enumerate(model_predictions):
                    pattern_result = {
                        'set_number': i + 1,
                        'numbers': pred['numbers'],
                        'confidence_score': pred['confidence'],
                        'pattern_analysis': self._analyze_pattern_for_numbers(pred['numbers'], machine_type),
                        'selection_method': pred.get('method', 'transformer_pattern'),
                        'diversity_factor': self.prediction_diversity,
                        'metadata': pred.get('metadata', {})
                    }
                    predictions.append(pattern_result)
            else:
                # 모델이 없으면 fallback 사용
                predictions = await self._fallback_predictions(historical_data, sets_count, machine_type)
                
            # 3. 메타 정보 생성
            metadata = await self._generate_metadata(machine_type, predictions)
            
            return {
                'success': True,
                'agent_id': self.agent_id,
                'agent_type': 'pattern_analyzer',
                'predictions': predictions,
                'metadata': metadata,
                'avg_confidence': np.mean([p['confidence_score'] for p in predictions])
            }
            
        except Exception as e:
            self.logger.error(f"❌ Pattern prediction failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'agent_id': self.agent_id,
                'agent_type': 'pattern_analyzer'
            }
            
    async def get_specialization_info(self) -> Dict[str, Any]:
        """특화 정보 반환"""
        return {
            'specialization': 'Deep Learning Pattern Recognition',
            'model_architecture': 'PyTorch Transformer',
            'capabilities': [
                'Sequential pattern analysis',
                'Multi-head attention correlation',
                'Probability distribution prediction',
                'Creative number combination generation'
            ],
            'model_config': self.model_config,
            'discovered_patterns': len(self.discovered_patterns),
            'device': str(self.device)
        }
        
    async def _load_historical_data(self, machine_type: str, limit: int = 50) -> List[Dict[str, Any]]:
        """과거 데이터 로드"""
        return await self.memory_manager.get_lottery_data(
            machine_type=machine_type,
            limit=limit
        )
        
    def _prepare_sequences(self, historical_data: List[Dict[str, Any]]) -> torch.Tensor:
        """시퀀스 데이터 준비"""
        # 시간 순서대로 정렬 (오래된 것부터)
        sorted_data = sorted(historical_data, key=lambda x: x['회차'])
        
        # 번호 시퀀스 생성
        sequences = []
        for record in sorted_data:
            numbers = record['1등_당첨번호']
            # 패딩 토큰 0으로 시작, 번호는 1-45
            sequence = [0] + numbers  # [0, num1, num2, num3, num4, num5, num6]
            sequences.append(sequence)
            
        # Tensor로 변환
        max_len = max(len(seq) for seq in sequences)
        padded_sequences = []
        
        for seq in sequences:
            if len(seq) < max_len:
                seq.extend([0] * (max_len - len(seq)))
            padded_sequences.append(seq)
            
        return torch.tensor(padded_sequences, dtype=torch.long).to(self.device)
        
    async def _predict_with_pattern_analysis(self, 
                                           sequences: torch.Tensor,
                                           machine_type: str,
                                           prediction_index: int) -> Dict[str, Any]:
        """패턴 분석 기반 예측"""
        
        with torch.no_grad():
            # 모델 추론
            outputs = self.model(sequences[-self.sequence_length:])
            
            probabilities = outputs['probabilities'][0]  # 배치 차원 제거
            pattern_features = outputs['pattern_features'][0]
            
            # 상위 확률 번호들 선택 (다양성 고려)
            selected_numbers = self._select_numbers_with_diversity(
                probabilities.cpu().numpy(),
                diversity_factor=self.prediction_diversity + (prediction_index * 0.1)
            )
            
            # 패턴 특성 분석
            pattern_analysis = self._analyze_pattern_features(
                pattern_features.cpu().numpy(),
                machine_type
            )
            
            # 신뢰도 계산
            confidence_score = self._calculate_pattern_confidence(
                selected_numbers, probabilities.cpu().numpy(), pattern_analysis
            )
            
            return {
                'set_number': prediction_index + 1,
                'numbers': sorted(selected_numbers),
                'confidence_score': confidence_score,
                'pattern_analysis': pattern_analysis,
                'selection_method': 'transformer_attention',
                'diversity_factor': self.prediction_diversity + (prediction_index * 0.1)
            }
            
    def _select_numbers_with_diversity(self, 
                                     probabilities: np.ndarray,
                                     diversity_factor: float = 0.3) -> List[int]:
        """다양성을 고려한 번호 선택"""
        
        # 상위 확률 번호들 (1-45 범위)
        top_indices = np.argsort(probabilities)[-15:]  # 상위 15개
        top_probs = probabilities[top_indices]
        
        selected = []
        
        # 첫 번째는 최고 확률 번호
        best_idx = top_indices[np.argmax(top_probs)]
        selected.append(best_idx + 1)  # 1-45 범위로 변환
        
        # 나머지 5개는 다양성 고려하여 선택
        remaining_indices = [idx for idx in top_indices if idx != best_idx]
        
        while len(selected) < 6 and remaining_indices:
            # 확률과 다양성의 균형
            scores = []
            for idx in remaining_indices:
                prob_score = probabilities[idx]
                
                # 이미 선택된 번호와의 다양성 점수
                diversity_score = self._calculate_diversity_score(
                    idx + 1, selected
                )
                
                combined_score = (
                    prob_score * (1 - diversity_factor) + 
                    diversity_score * diversity_factor
                )
                scores.append(combined_score)
                
            # 최고 점수 번호 선택
            best_remaining = remaining_indices[np.argmax(scores)]
            selected.append(best_remaining + 1)
            remaining_indices.remove(best_remaining)
            
        return selected[:6]
        
    def _calculate_diversity_score(self, candidate: int, selected: List[int]) -> float:
        """다양성 점수 계산"""
        if not selected:
            return 1.0
            
        diversity_score = 1.0
        
        for num in selected:
            # 숫자 간격 다양성
            distance = abs(candidate - num)
            if distance < 5:  # 너무 가까우면 점수 감소
                diversity_score *= 0.7
            elif distance > 30:  # 너무 멀어도 약간 감소
                diversity_score *= 0.9
                
        # 홀짝 균형
        selected_odd_count = sum(1 for n in selected if n % 2 == 1)
        if candidate % 2 == 1:  # 홀수
            if selected_odd_count >= 4:  # 이미 홀수가 많으면
                diversity_score *= 0.8
        else:  # 짝수
            if len(selected) - selected_odd_count >= 3:  # 이미 짝수가 많으면
                diversity_score *= 0.8
                
        return diversity_score
        
    def _analyze_pattern_features(self, 
                                pattern_features: np.ndarray,
                                machine_type: str) -> Dict[str, Any]:
        """패턴 특성 분석"""
        
        # 패턴 특성 벡터를 해석 가능한 정보로 변환
        feature_analysis = {
            'pattern_strength': float(np.mean(np.abs(pattern_features))),
            'pattern_complexity': float(np.std(pattern_features)),
            'dominant_features': pattern_features.argsort()[-5:].tolist(),
            'pattern_signature': self._generate_pattern_signature(pattern_features)
        }
        
        # 호기별 특성 적용
        machine_specific = self._apply_machine_specific_analysis(
            feature_analysis, machine_type
        )
        feature_analysis.update(machine_specific)
        
        return feature_analysis
        
    def _generate_pattern_signature(self, features: np.ndarray) -> str:
        """패턴 시그니처 생성"""
        # 특성 벡터를 해시로 변환
        feature_hash = hash(tuple(features.round(3)))
        return f"pattern_{abs(feature_hash) % 10000:04d}"
        
    def _apply_machine_specific_analysis(self, 
                                       base_analysis: Dict[str, Any],
                                       machine_type: str) -> Dict[str, Any]:
        """호기별 특화 분석"""
        
        machine_modifiers = {
            '1호기': {
                'conservative_factor': 0.8,  # 보수적 성향
                'high_frequency_bias': 0.3,  # 고빈도 번호 선호
                'explanation': '신중한 전략가 패턴 감지'
            },
            '2호기': {
                'balance_factor': 1.0,       # 균형 중시
                'harmony_preference': 0.4,   # 조화로운 조합
                'explanation': '완벽한 조화 패턴 추구'
            },
            '3호기': {
                'creativity_factor': 1.2,    # 창의적 성향
                'diversity_bonus': 0.5,      # 다양성 보너스
                'explanation': '창조적 혁신 패턴 탐지'
            }
        }
        
        modifier = machine_modifiers.get(machine_type, machine_modifiers['2호기'])
        
        return {
            'machine_specific_analysis': modifier,
            'pattern_adaptation': f"패턴이 {machine_type} 특성에 맞게 조정됨"
        }
        
    def _calculate_pattern_confidence(self,
                                    selected_numbers: List[int],
                                    probabilities: np.ndarray,
                                    pattern_analysis: Dict[str, Any]) -> float:
        """패턴 신뢰도 계산"""
        
        # 선택된 번호들의 평균 확률
        avg_prob = np.mean([probabilities[num-1] for num in selected_numbers])
        
        # 패턴 강도
        pattern_strength = pattern_analysis['pattern_strength']
        
        # 다양성 점수
        diversity_score = self._calculate_diversity_score_for_set(selected_numbers)
        
        # 종합 신뢰도
        confidence = (
            avg_prob * 0.5 +
            pattern_strength * 0.3 +
            diversity_score * 0.2
        )
        
        return min(max(confidence, 0.0), 1.0)
        
    def _calculate_diversity_score_for_set(self, numbers: List[int]) -> float:
        """번호 세트의 다양성 점수"""
        if len(numbers) != 6:
            return 0.0
            
        # 홀짝 균형
        odd_count = sum(1 for n in numbers if n % 2 == 1)
        odd_balance = 1.0 - abs(odd_count - 3) / 3.0
        
        # 고저 균형 (1-22 저수, 23-45 고수)
        high_count = sum(1 for n in numbers if n >= 23)
        high_balance = 1.0 - abs(high_count - 3) / 3.0
        
        # 번호 분산
        spread = np.std(numbers) / 15.0  # 정규화
        
        return (odd_balance + high_balance + spread) / 3.0
        
    async def _analyze_base_patterns(self):
        """기본 패턴 분석"""
        self.logger.info("🔍 Analyzing base patterns...")
        
        try:
            # 각 호기별 기본 패턴 분석
            for machine_type in ['1호기', '2호기', '3호기']:
                pattern_key = f"base_patterns_{machine_type}"
                
                # 캐시 확인
                cached = await self.memory_manager.get_pattern_cache(
                    pattern_key, machine_type
                )
                
                if not cached:
                    # 새로 분석
                    historical_data = await self._load_historical_data(machine_type, 100)
                    if historical_data:
                        patterns = self._extract_base_patterns(historical_data)
                        
                        # 캐시에 저장
                        await self.memory_manager.store_pattern_cache(
                            pattern_key, machine_type, patterns, 0.8, 48  # 48시간
                        )
                        
                        self.discovered_patterns[machine_type] = patterns
                        
        except Exception as e:
            self.logger.error(f"❌ Base pattern analysis failed: {e}")
            
    def _extract_base_patterns(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """기본 패턴 추출"""
        patterns = {
            'number_frequency': {},
            'pair_frequency': {},
            'sequence_patterns': [],
            'statistical_patterns': {}
        }
        
        # 번호 빈도 분석
        all_numbers = []
        for record in historical_data:
            all_numbers.extend(record['1등_당첨번호'])
            
        for num in range(1, 46):
            patterns['number_frequency'][num] = all_numbers.count(num)
            
        # 페어 빈도 분석
        for record in historical_data:
            numbers = record['1등_당첨번호']
            for i in range(len(numbers)):
                for j in range(i+1, len(numbers)):
                    pair = tuple(sorted([numbers[i], numbers[j]]))
                    if pair not in patterns['pair_frequency']:
                        patterns['pair_frequency'][pair] = 0
                    patterns['pair_frequency'][pair] += 1
                    
        # 통계적 패턴
        patterns['statistical_patterns'] = {
            'avg_odd_count': np.mean([
                sum(1 for n in r['1등_당첨번호'] if n % 2 == 1) 
                for r in historical_data
            ]),
            'avg_high_count': np.mean([
                sum(1 for n in r['1등_당첨번호'] if n >= 23)
                for r in historical_data
            ]),
            'avg_total_sum': np.mean([r['총합'] for r in historical_data]),
            'avg_ac_value': np.mean([r['AC값'] for r in historical_data])
        }
        
        return patterns
        
    async def _load_pretrained_weights(self):
        """사전 학습된 가중치 로드"""
        try:
            # 가중치 파일이 있다면 로드
            weights_path = f"models/pattern_analyzer_{self.agent_id}.pth"
            # 실제 구현에서는 파일 존재 확인 후 로드
            self.logger.info("🔄 No pretrained weights found, using random initialization")
        except Exception as e:
            self.logger.warning(f"⚠️  Could not load pretrained weights: {e}")
            
    async def _generate_metadata(self, 
                               machine_type: str,
                               predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """메타데이터 생성"""
        return {
            'model_type': 'PyTorch Transformer',
            'sequence_length': self.sequence_length,
            'prediction_method': 'neural_pattern_recognition',
            'machine_specialization': machine_type,
            'pattern_signatures': [p['pattern_analysis']['pattern_signature'] for p in predictions],
            'avg_pattern_strength': np.mean([
                p['pattern_analysis']['pattern_strength'] for p in predictions
            ]),
            'model_device': str(self.device),
            'discovered_patterns_count': len(self.discovered_patterns.get(machine_type, {}))
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
                for draw in draws[-50:]:  # 최근 50회
                    historical_numbers.append(draw.numbers)
                    
                return historical_numbers
            else:
                return []
        except Exception as e:
            self.logger.error(f"❌ Failed to get historical data: {e}")
            return []
    
    def _analyze_pattern_for_numbers(self, numbers: List[int], machine_type: str) -> Dict[str, Any]:
        """특정 번호 조합의 패턴 분석"""
        try:
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            high_count = sum(1 for n in numbers if n >= 23)
            digit_sum = sum(numbers)
            
            return {
                'pattern_strength': 0.7,  # 기본값
                'pattern_complexity': 0.5,
                'odd_count': odd_count,
                'high_count': high_count,
                'digit_sum': digit_sum,
                'machine_specific_analysis': {
                    'explanation': f'패턴이 {machine_type} 특성에 맞게 분석됨'
                },
                'pattern_signature': f"pattern_{hash(tuple(numbers)) % 10000:04d}"
            }
        except:
            return {'pattern_strength': 0.5, 'pattern_complexity': 0.5}
    
    async def _fallback_predictions(self, historical_data: List[List[int]], sets_count: int, machine_type: str) -> List[Dict[str, Any]]:
        """모델 없을 때 사용할 fallback 예측"""
        predictions = []
        
        try:
            # 간단한 통계 기반 예측
            all_numbers = []
            for draw in historical_data[-20:]:  # 최근 20회
                all_numbers.extend(draw)
                
            # 빈도 계산
            frequency = {}
            for num in range(1, 46):
                frequency[num] = all_numbers.count(num)
                
            for i in range(sets_count):
                # 빈도 기반 가중 선택
                if frequency:
                    weights = [frequency.get(num, 0) + 1 for num in range(1, 46)]
                    total_weight = sum(weights)
                    probabilities = [w / total_weight for w in weights]
                    
                    selected = np.random.choice(
                        range(1, 46), size=6, replace=False, p=probabilities
                    )
                    numbers = sorted(selected.tolist())
                else:
                    numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
                
                predictions.append({
                    'set_number': i + 1,
                    'numbers': numbers,
                    'confidence_score': 0.4,
                    'pattern_analysis': self._analyze_pattern_for_numbers(numbers, machine_type),
                    'selection_method': 'statistical_fallback',
                    'diversity_factor': self.prediction_diversity
                })
                
        except Exception as e:
            self.logger.error(f"❌ Fallback prediction failed: {e}")
            # 완전 랜덤 fallback
            for i in range(sets_count):
                numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
                predictions.append({
                    'set_number': i + 1,
                    'numbers': numbers.tolist(),
                    'confidence_score': 0.2,
                    'pattern_analysis': self._analyze_pattern_for_numbers(numbers, machine_type),
                    'selection_method': 'random_fallback',
                    'diversity_factor': self.prediction_diversity
                })
                
        return predictions