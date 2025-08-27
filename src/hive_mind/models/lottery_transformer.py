# 🤖 Lottery Pattern Transformer Model
# Simplified but functional transformer for lottery number pattern recognition

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any
import math

class PositionalEncoding(nn.Module):
    """위치 인코딩"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class LotteryTransformer(nn.Module):
    """
    로또 번호 패턴 인식용 트랜스포머 모델
    
    간소화된 구조로 실제 동작 가능한 모델
    """
    
    def __init__(self, 
                 vocab_size: int = 45,  # 1-45 번호
                 d_model: int = 64,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 max_seq_len: int = 20):
        
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # 임베딩 레이어 (번호를 벡터로 변환)
        self.number_embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 for padding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)
        
        # 트랜스포머 인코더
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # 출력 헤드 (45개 번호 확률)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, vocab_size)
        )
        
        # 가중치 초기화
        self._init_weights()
        
    def _init_weights(self):
        """가중치 초기화"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
                
    def forward(self, x, mask=None):
        """
        순전파
        x: (batch_size, seq_len) - 각 시퀀스는 6개 번호의 sequence
        """
        # 임베딩
        x = self.number_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        
        # 트랜스포머 인코딩
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # 평균 풀링 (시퀀스 차원에서)
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x = x.masked_fill(mask_expanded, 0)
            lengths = (~mask).sum(dim=1, keepdim=True).float()
            x = x.sum(dim=1) / lengths
        else:
            x = x.mean(dim=1)
            
        # 출력 확률
        logits = self.output_head(x)
        return logits
        
    def predict_numbers(self, 
                       history: List[List[int]], 
                       num_sets: int = 1,
                       diversity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        번호 예측
        
        Args:
            history: 과거 추첨 번호들 [[1,2,3,4,5,6], [7,8,9,10,11,12], ...]
            num_sets: 예측할 세트 수
            diversity_threshold: 다양성 임계값
        """
        self.eval()
        
        with torch.no_grad():
            # 입력 준비
            if len(history) == 0:
                # 과거 데이터 없으면 랜덤
                return self._generate_random_predictions(num_sets)
                
            # 최근 데이터만 사용 (메모리 효율)
            recent_history = history[-10:] if len(history) > 10 else history
            
            # 텐서 변환
            input_seqs = []
            for draw in recent_history:
                # 각 추첨을 6개 번호 시퀀스로 변환
                input_seqs.extend([num for num in draw])
                
            # 패딩 및 배치 처리
            max_len = min(len(input_seqs), 60)  # 최대 10회 * 6번호
            if len(input_seqs) < max_len:
                input_seqs.extend([0] * (max_len - len(input_seqs)))  # 패딩
            else:
                input_seqs = input_seqs[-max_len:]  # 최근 데이터만
                
            x = torch.tensor([input_seqs], dtype=torch.long)
            
            # 예측
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=-1).squeeze(0)  # (45,)
            
            # 다양한 예측 생성
            predictions = []
            used_combinations = set()
            
            for _ in range(num_sets):
                # Top-k 샘플링으로 다양성 확보
                top_k = min(15, len(probabilities))  # 상위 15개에서 선택
                top_probs, top_indices = torch.topk(probabilities, top_k)
                
                # 온도 기반 샘플링
                temperature = 1.0 + (_ * 0.3)  # 점진적으로 다양성 증가
                scaled_probs = F.softmax(top_probs / temperature, dim=-1)
                
                selected_numbers = []
                available_indices = list(range(top_k))
                
                # 6개 번호 선택
                for _ in range(6):
                    if not available_indices:
                        break
                        
                    # 확률 기반 샘플링
                    weights = [scaled_probs[i].item() for i in available_indices]
                    total_weight = sum(weights)
                    if total_weight == 0:
                        choice_idx = np.random.choice(available_indices)
                    else:
                        choice_idx = np.random.choice(
                            available_indices, 
                            p=[w/total_weight for w in weights]
                        )
                    
                    selected_idx = available_indices[choice_idx]
                    number = top_indices[selected_idx].item() + 1  # 1-based
                    selected_numbers.append(number)
                    
                    # 선택된 번호는 제거
                    available_indices.remove(selected_idx)
                
                # 6개가 안되면 랜덤으로 채우기
                while len(selected_numbers) < 6:
                    remaining = [i for i in range(1, 46) if i not in selected_numbers]
                    if remaining:
                        selected_numbers.append(np.random.choice(remaining))
                    else:
                        break
                
                selected_numbers = sorted(selected_numbers[:6])
                combo_key = tuple(selected_numbers)
                
                # 중복 체크 및 추가
                if combo_key not in used_combinations and len(selected_numbers) == 6:
                    used_combinations.add(combo_key)
                    
                    # 신뢰도 계산
                    confidence = self._calculate_confidence(
                        selected_numbers, probabilities, history
                    )
                    
                    predictions.append({
                        'numbers': selected_numbers,
                        'confidence': confidence,
                        'method': 'transformer_pattern',
                        'metadata': {
                            'temperature': temperature,
                            'top_probs': top_probs[:6].tolist()
                        }
                    })
                    
            # 예측이 부족하면 랜덤으로 채우기
            while len(predictions) < num_sets:
                random_pred = self._generate_random_predictions(1)[0]
                if tuple(random_pred['numbers']) not in used_combinations:
                    predictions.append(random_pred)
                    used_combinations.add(tuple(random_pred['numbers']))
                    
            return predictions[:num_sets]
            
    def _calculate_confidence(self, 
                            numbers: List[int], 
                            probabilities: torch.Tensor, 
                            history: List[List[int]]) -> float:
        """신뢰도 계산"""
        try:
            # 모델 확률 기반 신뢰도
            model_confidence = sum(probabilities[num-1].item() for num in numbers) / 6
            
            # 과거 패턴 일치도
            pattern_score = 0.0
            if history:
                recent_numbers = set()
                for draw in history[-5:]:  # 최근 5회
                    recent_numbers.update(draw)
                    
                overlap = len(set(numbers) & recent_numbers)
                pattern_score = min(overlap / 6, 0.5)  # 최대 0.5
                
            # 통계적 균형
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            balance_score = 1.0 - abs(odd_count - 3) / 3  # 홀짝 균형
            
            # 종합 신뢰도
            confidence = (model_confidence * 0.5 + 
                         pattern_score * 0.2 + 
                         balance_score * 0.3)
            
            return max(0.1, min(0.9, confidence))
            
        except:
            return 0.5
            
    def _generate_random_predictions(self, num_sets: int) -> List[Dict[str, Any]]:
        """랜덤 예측 생성 (fallback)"""
        predictions = []
        used_combinations = set()
        
        for _ in range(num_sets):
            while True:
                numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
                combo_key = tuple(numbers)
                
                if combo_key not in used_combinations:
                    used_combinations.add(combo_key)
                    predictions.append({
                        'numbers': numbers.tolist(),
                        'confidence': 0.3,
                        'method': 'random_fallback',
                        'metadata': {}
                    })
                    break
                    
        return predictions

class LotteryPatternPredictor:
    """
    트랜스포머 모델 래퍼 클래스
    """
    
    def __init__(self, model_config: Dict[str, Any] = None):
        self.config = model_config or {}
        self.model: Optional[LotteryTransformer] = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self):
        """모델 초기화"""
        try:
            self.model = LotteryTransformer(
                d_model=self.config.get('d_model', 64),
                nhead=self.config.get('nhead', 8),
                num_layers=self.config.get('num_layers', 4),
                dropout=self.config.get('dropout', 0.1)
            )
            
            self.model.to(self.device)
            self.logger.info(f"✅ Transformer model initialized on {self.device}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            return False
            
    def predict(self, 
                historical_data: List[List[int]], 
                num_predictions: int = 1,
                **kwargs) -> List[Dict[str, Any]]:
        """예측 실행"""
        if self.model is None:
            self.logger.warning("Model not initialized, using random predictions")
            return self._random_fallback(num_predictions)
            
        try:
            diversity_threshold = kwargs.get('diversity_threshold', 0.7)
            return self.model.predict_numbers(
                historical_data, 
                num_predictions, 
                diversity_threshold
            )
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return self._random_fallback(num_predictions)
            
    def _random_fallback(self, num_predictions: int) -> List[Dict[str, Any]]:
        """랜덤 예측 (fallback)"""
        predictions = []
        for _ in range(num_predictions):
            numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
            predictions.append({
                'numbers': numbers.tolist(),
                'confidence': 0.2,
                'method': 'random_fallback',
                'metadata': {}
            })
        return predictions
        
    def save_model(self, path: str):
        """모델 저장"""
        if self.model:
            torch.save(self.model.state_dict(), path)
            self.logger.info(f"Model saved to {path}")
            
    def load_model(self, path: str):
        """모델 로드"""
        if self.model:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.model.eval()
            self.logger.info(f"Model loaded from {path}")
            
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보"""
        if self.model is None:
            return {'status': 'not_initialized'}
            
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'status': 'initialized',
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_config': self.config
        }