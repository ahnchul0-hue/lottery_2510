# 🤖 Ensemble Statistical Predictor
# scikit-learn based ensemble models for lottery prediction

import numpy as np
import pandas as pd
import logging
from typing import List, Dict, Any, Optional, Tuple
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
import joblib
from pathlib import Path

class LotteryEnsemblePredictor:
    """
    scikit-learn 기반 앙상블 예측기
    
    RandomForest + GradientBoosting + MLP 조합
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # 모델 초기화
        self.models = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 특성 추출기
        self.feature_extractor = LotteryFeatureExtractor()
        
    def initialize(self):
        """모델 초기화"""
        try:
            # RandomForest 모델
            self.models['rf'] = RandomForestClassifier(
                n_estimators=self.config.get('n_estimators', 50),
                max_depth=self.config.get('max_depth', 8),
                random_state=42,
                n_jobs=-1
            )
            
            # GradientBoosting 모델
            self.models['gb'] = GradientBoostingClassifier(
                n_estimators=self.config.get('n_estimators', 50),
                max_depth=self.config.get('max_depth', 6),
                learning_rate=self.config.get('learning_rate', 0.1),
                random_state=42
            )
            
            # MLP 모델
            self.models['mlp'] = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            # MultiOutput 래퍼 (6개 번호 동시 예측)
            self.ensemble_models = {
                name: MultiOutputClassifier(model)
                for name, model in self.models.items()
            }
            
            self.logger.info("✅ Ensemble models initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Model initialization failed: {e}")
            return False
            
    def train(self, historical_data: List[List[int]]) -> bool:
        """모델 훈련"""
        try:
            if len(historical_data) < 20:
                self.logger.warning("Insufficient data for training")
                return False
                
            self.logger.info(f"Training with {len(historical_data)} historical draws")
            
            # 특성 추출
            X, y = self.feature_extractor.prepare_training_data(historical_data)
            
            if X is None or y is None or len(X) == 0:
                self.logger.error("Feature extraction failed")
                return False
                
            # 특성 스케일링
            X_scaled = self.scaler.fit_transform(X)
            
            # 각 모델 훈련
            for name, model in self.ensemble_models.items():
                self.logger.info(f"Training {name} model...")
                model.fit(X_scaled, y)
                
                # 교차 검증 점수
                try:
                    scores = cross_val_score(model, X_scaled, y, cv=3, scoring='accuracy')
                    self.logger.info(f"{name} CV accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
                except:
                    pass  # 교차 검증 실패해도 계속 진행
                    
            self.is_trained = True
            self.logger.info("✅ Model training completed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Training failed: {e}")
            return False
            
    def predict(self, 
                historical_data: List[List[int]], 
                num_predictions: int = 1,
                machine_type: str = "2호기",
                **kwargs) -> List[Dict[str, Any]]:
        """예측 실행"""
        try:
            if not self.is_trained:
                self.logger.warning("Model not trained, using statistical fallback")
                return self._statistical_fallback(historical_data, num_predictions, machine_type)
                
            # 특성 추출
            features = self.feature_extractor.extract_prediction_features(historical_data)
            if features is None:
                return self._statistical_fallback(historical_data, num_predictions, machine_type)
                
            # 스케일링
            features_scaled = self.scaler.transform([features])
            
            # 각 모델의 예측
            model_predictions = {}
            for name, model in self.ensemble_models.items():
                try:
                    # 확률 예측
                    probabilities = model.predict_proba(features_scaled)
                    model_predictions[name] = probabilities
                except:
                    # 확률 예측 실패 시 클래스 예측
                    pred = model.predict(features_scaled)
                    model_predictions[name] = pred
                    
            # 앙상블 예측 결합
            predictions = self._combine_predictions(
                model_predictions, num_predictions, machine_type
            )
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Prediction failed: {e}")
            return self._statistical_fallback(historical_data, num_predictions, machine_type)
            
    def _combine_predictions(self, 
                           model_predictions: Dict[str, Any], 
                           num_predictions: int,
                           machine_type: str) -> List[Dict[str, Any]]:
        """모델 예측 결합"""
        predictions = []
        
        try:
            # 각 모델의 가중치
            weights = {
                'rf': 0.4,
                'gb': 0.35, 
                'mlp': 0.25
            }
            
            # 앙상블 확률 계산
            ensemble_probs = np.zeros(45)  # 1-45번 확률
            
            for name, pred in model_predictions.items():
                weight = weights.get(name, 0.33)
                
                if isinstance(pred, (list, np.ndarray)):
                    # 다차원 배열 처리
                    if hasattr(pred[0], '__len__') and len(pred[0]) > 0:
                        # 각 위치별 확률의 평균
                        for pos_probs in pred:
                            if hasattr(pos_probs, '__len__') and len(pos_probs) >= 45:
                                ensemble_probs += weight * np.array(pos_probs[:45]) / 6
                    else:
                        # 단순 예측값
                        for num in pred[0]:
                            if 1 <= num <= 45:
                                ensemble_probs[num-1] += weight / 6
                                
            # 확률 정규화
            if ensemble_probs.sum() > 0:
                ensemble_probs /= ensemble_probs.sum()
            else:
                ensemble_probs = np.ones(45) / 45  # 균등 분포
                
            # 기계별 전략 적용
            ensemble_probs = self._apply_machine_strategy(ensemble_probs, machine_type)
            
            # 다양한 예측 생성
            used_combinations = set()
            
            for i in range(num_predictions):
                # Top-k 샘플링
                top_k = min(20, len(ensemble_probs))
                top_indices = np.argsort(ensemble_probs)[-top_k:]
                top_probs = ensemble_probs[top_indices]
                
                # 온도 스케일링으로 다양성 확보
                temperature = 1.0 + (i * 0.2)
                scaled_probs = np.exp(np.log(top_probs + 1e-8) / temperature)
                scaled_probs /= scaled_probs.sum()
                
                # 6개 번호 선택
                selected_numbers = []
                remaining_indices = list(top_indices)
                remaining_probs = scaled_probs.copy()
                
                for _ in range(6):
                    if len(remaining_indices) == 0:
                        break
                        
                    # 확률 기반 선택
                    if remaining_probs.sum() > 0:
                        choice_idx = np.random.choice(
                            len(remaining_indices),
                            p=remaining_probs / remaining_probs.sum()
                        )
                    else:
                        choice_idx = np.random.choice(len(remaining_indices))
                        
                    selected_idx = remaining_indices[choice_idx]
                    number = selected_idx + 1
                    selected_numbers.append(number)
                    
                    # 선택된 번호 제거
                    remaining_indices.pop(choice_idx)
                    remaining_probs = np.delete(remaining_probs, choice_idx)
                    
                # 부족한 번호 랜덤 채우기
                while len(selected_numbers) < 6:
                    available = [n for n in range(1, 46) if n not in selected_numbers]
                    if available:
                        selected_numbers.append(np.random.choice(available))
                    else:
                        break
                        
                selected_numbers = sorted(selected_numbers[:6])
                combo_key = tuple(selected_numbers)
                
                # 중복 체크
                if combo_key not in used_combinations and len(selected_numbers) == 6:
                    used_combinations.add(combo_key)
                    
                    # 신뢰도 계산
                    confidence = self._calculate_confidence(selected_numbers, ensemble_probs)
                    
                    predictions.append({
                        'numbers': selected_numbers,
                        'confidence': confidence,
                        'method': 'ensemble_ml',
                        'metadata': {
                            'models_used': list(model_predictions.keys()),
                            'machine_strategy': machine_type,
                            'temperature': temperature
                        }
                    })
                    
            return predictions
            
        except Exception as e:
            self.logger.error(f"❌ Prediction combination failed: {e}")
            return []
            
    def _apply_machine_strategy(self, probabilities: np.ndarray, machine_type: str) -> np.ndarray:
        """기계별 전략 적용"""
        adjusted_probs = probabilities.copy()
        
        try:
            if machine_type == "1호기":
                # 보수적: 중간 번호 선호, 극단 번호 회피
                for i in range(45):
                    num = i + 1
                    if 15 <= num <= 35:
                        adjusted_probs[i] *= 1.2
                    elif num <= 5 or num >= 40:
                        adjusted_probs[i] *= 0.8
                        
            elif machine_type == "3호기":
                # 창의적: 홀수 선호, 극단 번호 선호
                for i in range(45):
                    num = i + 1
                    if num % 2 == 1:  # 홀수
                        adjusted_probs[i] *= 1.15
                    if num <= 10 or num >= 35:  # 극단
                        adjusted_probs[i] *= 1.1
                        
            # 2호기는 균형이므로 그대로 유지
            
        except:
            pass  # 전략 적용 실패해도 원본 확률 사용
            
        # 확률 정규화
        if adjusted_probs.sum() > 0:
            adjusted_probs /= adjusted_probs.sum()
            
        return adjusted_probs
        
    def _calculate_confidence(self, numbers: List[int], probabilities: np.ndarray) -> float:
        """신뢰도 계산"""
        try:
            # 모델 확률 기반
            model_conf = sum(probabilities[n-1] for n in numbers) / 6
            
            # 통계적 균형 점수
            odd_count = sum(1 for n in numbers if n % 2 == 1)
            balance_score = 1.0 - abs(odd_count - 3) / 3
            
            # 범위 분산 점수
            range_score = (max(numbers) - min(numbers)) / 44  # 0-1 정규화
            
            # 종합 신뢰도
            confidence = (model_conf * 0.6 + 
                         balance_score * 0.25 + 
                         range_score * 0.15)
            
            return max(0.1, min(0.9, confidence))
            
        except:
            return 0.5
            
    def _statistical_fallback(self, 
                            historical_data: List[List[int]], 
                            num_predictions: int,
                            machine_type: str) -> List[Dict[str, Any]]:
        """통계 기반 fallback 예측"""
        try:
            if not historical_data:
                return self._random_fallback(num_predictions)
                
            # 번호 빈도 분석
            frequency = {}
            for draw in historical_data[-20:]:  # 최근 20회
                for num in draw:
                    frequency[num] = frequency.get(num, 0) + 1
                    
            # 빈도 기반 확률
            total_count = sum(frequency.values())
            probabilities = np.array([
                frequency.get(i+1, 0) / max(total_count, 1) 
                for i in range(45)
            ])
            
            # 기계 전략 적용
            probabilities = self._apply_machine_strategy(probabilities, machine_type)
            
            # 예측 생성
            predictions = []
            for _ in range(num_predictions):
                # 가중 랜덤 선택
                if probabilities.sum() > 0:
                    selected = np.random.choice(
                        45, size=6, replace=False,
                        p=probabilities / probabilities.sum()
                    )
                    numbers = sorted([int(x) + 1 for x in selected])
                else:
                    numbers = sorted(np.random.choice(range(1, 46), 6, replace=False))
                    
                predictions.append({
                    'numbers': numbers,
                    'confidence': 0.4,
                    'method': 'statistical_fallback',
                    'metadata': {'machine_type': machine_type}
                })
                
            return predictions
            
        except:
            return self._random_fallback(num_predictions)
            
    def _random_fallback(self, num_predictions: int) -> List[Dict[str, Any]]:
        """랜덤 fallback"""
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
        
    def save_models(self, model_dir: str):
        """모델 저장"""
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 앙상블 모델 저장
            joblib.dump(self.ensemble_models, model_path / 'ensemble_models.pkl')
            joblib.dump(self.scaler, model_path / 'scaler.pkl')
            
            # 훈련 상태 저장
            metadata = {
                'is_trained': self.is_trained,
                'config': self.config
            }
            joblib.dump(metadata, model_path / 'metadata.pkl')
            
            self.logger.info(f"Models saved to {model_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Model saving failed: {e}")
            
    def load_models(self, model_dir: str):
        """모델 로드"""
        model_path = Path(model_dir)
        
        try:
            if (model_path / 'ensemble_models.pkl').exists():
                self.ensemble_models = joblib.load(model_path / 'ensemble_models.pkl')
                self.scaler = joblib.load(model_path / 'scaler.pkl')
                
                metadata = joblib.load(model_path / 'metadata.pkl')
                self.is_trained = metadata.get('is_trained', False)
                
                self.logger.info(f"Models loaded from {model_path}")
                return True
            else:
                self.logger.warning(f"No saved models found in {model_path}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Model loading failed: {e}")
            return False

class LotteryFeatureExtractor:
    """로또 특성 추출기"""
    
    def prepare_training_data(self, historical_data: List[List[int]]) -> Tuple[np.ndarray, np.ndarray]:
        """훈련용 데이터 준비"""
        try:
            if len(historical_data) < 10:
                return None, None
                
            X, y = [], []
            
            for i in range(5, len(historical_data)):
                # 과거 5회 데이터로 특성 생성
                recent_draws = historical_data[i-5:i]
                features = self._extract_features(recent_draws)
                
                # 타겟은 다음 추첨
                target = historical_data[i]
                
                if features is not None and len(target) == 6:
                    X.append(features)
                    # 타겟을 6개 별도 분류 문제로 변환 (1-45 클래스)
                    y.append([num - 1 for num in target])  # 0-based
                    
            return np.array(X), np.array(y)
            
        except Exception as e:
            logging.error(f"Training data preparation failed: {e}")
            return None, None
            
    def extract_prediction_features(self, historical_data: List[List[int]]) -> Optional[np.ndarray]:
        """예측용 특성 추출"""
        try:
            if len(historical_data) < 5:
                return None
                
            recent_draws = historical_data[-5:]  # 최근 5회
            return self._extract_features(recent_draws)
            
        except Exception as e:
            logging.error(f"Feature extraction failed: {e}")
            return None
            
    def _extract_features(self, draws: List[List[int]]) -> Optional[np.ndarray]:
        """특성 추출 로직"""
        try:
            features = []
            
            # 1. 번호 빈도 (1-45)
            frequency = {i: 0 for i in range(1, 46)}
            for draw in draws:
                for num in draw:
                    if 1 <= num <= 45:
                        frequency[num] += 1
                        
            features.extend([frequency[i] for i in range(1, 46)])
            
            # 2. 통계적 특성
            for draw in draws:
                if len(draw) >= 6:
                    sorted_nums = sorted(draw[:6])
                    
                    # 홀짝 비율
                    odd_count = sum(1 for n in sorted_nums if n % 2 == 1)
                    features.append(odd_count / 6)
                    
                    # 고저 비율  
                    high_count = sum(1 for n in sorted_nums if n >= 23)
                    features.append(high_count / 6)
                    
                    # 번호 합
                    features.append(sum(sorted_nums))
                    
                    # 범위
                    features.append(max(sorted_nums) - min(sorted_nums))
                    
                    # 연속 번호 개수
                    consecutive = 0
                    for i in range(len(sorted_nums) - 1):
                        if sorted_nums[i+1] - sorted_nums[i] == 1:
                            consecutive += 1
                    features.append(consecutive)
                else:
                    features.extend([0, 0, 0, 0, 0])  # 기본값
                    
            # 3. 간격 패턴 (최근 추첨)
            if draws and len(draws[-1]) >= 6:
                last_draw = sorted(draws[-1][:6])
                gaps = [last_draw[i+1] - last_draw[i] for i in range(5)]
                features.extend(gaps)
            else:
                features.extend([0, 0, 0, 0, 0])
                
            return np.array(features)
            
        except Exception as e:
            logging.error(f"Feature extraction error: {e}")
            return None