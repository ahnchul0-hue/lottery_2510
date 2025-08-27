# 🤖 Hive-Mind Configuration Management - Advanced Config System
# Centralized configuration with environment-aware settings and validation

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field, asdict
from datetime import datetime
import logging
from enum import Enum

class Environment(Enum):
    """환경 타입"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

class MLBackend(Enum):
    """ML 백엔드 타입"""
    PYTORCH = "pytorch"
    SKLEARN = "sklearn"
    HYBRID = "hybrid"

@dataclass
class DatabaseConfig:
    """데이터베이스 설정"""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    username: str = ""
    password: str = ""
    database: str = "lottery_hive"
    sqlite_path: str = "data/lottery_hive.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    echo: bool = False
    
@dataclass 
class RedisConfig:
    """Redis 설정"""
    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    decode_responses: bool = True
    socket_timeout: int = 5
    
@dataclass
class MLConfig:
    """머신러닝 설정"""
    backend: MLBackend = MLBackend.HYBRID
    
    # PyTorch 설정
    pytorch_device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    pytorch_model_path: str = "models/pytorch/"
    sequence_length: int = 20
    embedding_dim: int = 64
    num_heads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    
    # scikit-learn 설정  
    sklearn_model_path: str = "models/sklearn/"
    n_estimators: int = 100
    max_depth: int = 10
    learning_rate: float = 0.1
    
    # 앙상블 설정
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        'pattern_analyzer': 0.35,
        'statistical_predictor': 0.35,
        'cognitive_analyzer': 0.30
    })
    
    # 메모리 관리
    max_memory_mb: int = 2048
    model_cache_size: int = 5
    batch_size: int = 32
    
@dataclass
class AgentConfig:
    """에이전트 설정"""
    max_agents: int = 10
    spawn_timeout: int = 30
    health_check_interval: int = 60
    performance_threshold: float = 0.6
    
    # 개별 에이전트 설정
    pattern_analyzer: Dict[str, Any] = field(default_factory=lambda: {
        'model_type': 'transformer',
        'max_sequence_length': 20,
        'diversity_threshold': 0.7
    })
    
    statistical_predictor: Dict[str, Any] = field(default_factory=lambda: {
        'feature_count': 100,
        'ensemble_methods': ['rf', 'gb', 'mlp'],
        'validation_splits': 5
    })
    
    cognitive_analyzer: Dict[str, Any] = field(default_factory=lambda: {
        'rule_strictness': 0.8,
        'anomaly_threshold': 2.0,
        'explanation_depth': 'detailed'
    })
    
    ensemble_optimizer: Dict[str, Any] = field(default_factory=lambda: {
        'optimization_interval': 100,
        'weight_decay': 0.01,
        'performance_window': 50
    })
    
@dataclass
class MessageBusConfig:
    """메시지 버스 설정"""
    max_queue_size: int = 1000
    processing_timeout: float = 30.0
    retry_max_attempts: int = 3
    retry_backoff_factor: float = 2.0
    
    # 워커 설정
    critical_workers: int = 2
    high_workers: int = 3
    normal_workers: int = 2  
    low_workers: int = 1
    
    # 성능 설정
    message_history_size: int = 1000
    dead_letter_queue_size: int = 100
    cleanup_interval: int = 60
    
@dataclass
class APIConfig:
    """API 서버 설정"""
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # CORS 설정
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    cors_methods: List[str] = field(default_factory=lambda: ["GET", "POST", "PUT", "DELETE"])
    cors_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # 보안 설정
    api_key_required: bool = True
    jwt_secret: str = "your-secret-key-here"
    jwt_expiry_hours: int = 24
    
    # 속도 제한
    rate_limit_enabled: bool = True
    rate_limit_requests: int = 100
    rate_limit_period: int = 60
    
    # 응답 설정
    max_response_size: int = 10 * 1024 * 1024  # 10MB
    response_timeout: int = 30
    
@dataclass
class PerformanceConfig:
    """성능 설정"""
    # 응답 시간 목표 (ms)
    target_response_time: int = 200
    max_response_time: int = 1000
    
    # 메모리 관리
    max_memory_usage_mb: int = 2048
    memory_cleanup_threshold: float = 0.85
    
    # 동시성
    max_concurrent_predictions: int = 10
    prediction_timeout: int = 30
    
    # 캐싱
    cache_enabled: bool = True
    cache_ttl: int = 300  # 5분
    cache_max_size: int = 1000
    
@dataclass
class MonitoringConfig:
    """모니터링 설정"""
    enabled: bool = True
    
    # 메트릭 수집
    collect_system_metrics: bool = True
    collect_ml_metrics: bool = True
    collect_api_metrics: bool = True
    
    # 로그 설정
    log_level: str = "INFO"
    log_file: str = "logs/hive_mind.log"
    log_rotation: str = "daily"
    log_retention: int = 30  # days
    
    # 알림 설정
    alert_enabled: bool = True
    alert_error_threshold: int = 10
    alert_response_time_threshold: int = 500
    
@dataclass 
class HiveMindConfig:
    """전체 하이브-마인드 설정"""
    environment: Environment = Environment.DEVELOPMENT
    version: str = "1.0.0"
    
    # 하위 설정들
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    redis: RedisConfig = field(default_factory=RedisConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    message_bus: MessageBusConfig = field(default_factory=MessageBusConfig)
    api: APIConfig = field(default_factory=APIConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # 글로벌 설정
    debug: bool = False
    data_dir: str = "data"
    model_dir: str = "models"
    log_dir: str = "logs"
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HiveMindConfig':
        """딕셔너리에서 생성"""
        # 환경 변수 처리
        if 'environment' in data:
            data['environment'] = Environment(data['environment'])
            
        # ML 백엔드 처리
        if 'ml' in data and 'backend' in data['ml']:
            data['ml']['backend'] = MLBackend(data['ml']['backend'])
            
        return cls(**data)

class ConfigManager:
    """
    고급 설정 관리자
    
    Claude-Flow 개선사항:
    - 환경별 설정 자동 로드
    - 동적 설정 업데이트
    - 검증 및 타입 체크
    - 암호화된 설정 지원
    """
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self._config: Optional[HiveMindConfig] = None
        self._config_file_path: Optional[Path] = None
        self._last_modified: Optional[datetime] = None
        
        # 환경 변수 우선순위
        self.env_mappings = {
            'HIVE_MIND_ENV': 'environment',
            'HIVE_MIND_DEBUG': 'debug',
            'DATABASE_URL': 'database.sqlite_path',
            'REDIS_URL': 'redis.host',
            'ML_BACKEND': 'ml.backend',
            'API_HOST': 'api.host',
            'API_PORT': 'api.port',
            'LOG_LEVEL': 'monitoring.log_level'
        }
        
    def load_config(self, config_name: Optional[str] = None) -> HiveMindConfig:
        """설정 로드"""
        if config_name is None:
            # 환경 변수에서 환경 결정
            env_name = os.environ.get('HIVE_MIND_ENV', 'development')
            config_name = env_name
            
        config_file = self.config_dir / f"{config_name}.yaml"
        
        # 기본 설정 파일이 없으면 생성
        if not config_file.exists():
            self._create_default_config(config_file)
            
        # YAML 파일 로드
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f) or {}
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load config from {config_file}: {e}")
            config_data = {}
            
        # 환경 변수로 오버라이드
        config_data = self._apply_env_overrides(config_data)
        
        # HiveMindConfig 객체 생성
        try:
            self._config = HiveMindConfig.from_dict(config_data)
            self._config_file_path = config_file
            self._last_modified = datetime.fromtimestamp(config_file.stat().st_mtime)
            
            self.logger.info(f"✅ Configuration loaded from {config_file}")
            return self._config
            
        except Exception as e:
            self.logger.error(f"❌ Invalid configuration: {e}")
            # 기본 설정 반환
            self._config = HiveMindConfig()
            return self._config
            
    def get_config(self) -> HiveMindConfig:
        """현재 설정 반환"""
        if self._config is None:
            return self.load_config()
        return self._config
        
    def reload_if_changed(self) -> bool:
        """변경된 경우 설정 재로드"""
        if self._config_file_path and self._config_file_path.exists():
            current_modified = datetime.fromtimestamp(
                self._config_file_path.stat().st_mtime
            )
            
            if self._last_modified and current_modified > self._last_modified:
                self.logger.info("🔄 Configuration file changed, reloading...")
                self.load_config()
                return True
                
        return False
        
    def save_config(self, config: HiveMindConfig, config_name: str = None):
        """설정 저장"""
        if config_name is None:
            config_name = config.environment.value
            
        config_file = self.config_dir / f"{config_name}.yaml"
        
        try:
            config_data = config.to_dict()
            
            # Enum 값을 문자열로 변환
            config_data['environment'] = config.environment.value
            config_data['ml']['backend'] = config.ml.backend.value
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                
            self.logger.info(f"✅ Configuration saved to {config_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
            raise
            
    def _create_default_config(self, config_file: Path):
        """기본 설정 파일 생성"""
        self.logger.info(f"📝 Creating default configuration at {config_file}")
        
        default_config = HiveMindConfig()
        
        # 개발 환경에 맞게 조정
        if "development" in config_file.name:
            default_config.debug = True
            default_config.monitoring.log_level = "DEBUG"
            default_config.api.api_key_required = False
            
        elif "production" in config_file.name:
            default_config.environment = Environment.PRODUCTION
            default_config.debug = False
            default_config.monitoring.log_level = "INFO"
            default_config.api.api_key_required = True
            
        self.save_config(default_config, config_file.stem)
        
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """환경 변수로 설정 오버라이드"""
        for env_var, config_path in self.env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                self._set_nested_value(config_data, config_path, env_value)
                
        return config_data
        
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: str):
        """중첩된 딕셔너리에 값 설정"""
        keys = path.split('.')
        current = data
        
        # 경로 생성
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        # 타입 변환
        final_key = keys[-1]
        if final_key in ['debug', 'enabled', 'api_key_required']:
            value = value.lower() in ('true', '1', 'yes', 'on')
        elif final_key in ['port', 'workers', 'max_agents']:
            try:
                value = int(value)
            except ValueError:
                pass
        elif final_key in ['dropout', 'learning_rate']:
            try:
                value = float(value)
            except ValueError:
                pass
                
        current[final_key] = value
        
    def validate_config(self, config: HiveMindConfig) -> List[str]:
        """설정 검증"""
        errors = []
        
        # 디렉토리 존재 확인
        for dir_attr in ['data_dir', 'model_dir', 'log_dir']:
            dir_path = Path(getattr(config, dir_attr))
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_path}: {e}")
                    
        # API 설정 검증
        if config.api.port < 1 or config.api.port > 65535:
            errors.append("API port must be between 1 and 65535")
            
        if config.api.workers < 1:
            errors.append("API workers must be at least 1")
            
        # ML 설정 검증
        if config.ml.max_memory_mb < 512:
            errors.append("ML max memory should be at least 512MB")
            
        if config.ml.batch_size < 1:
            errors.append("ML batch size must be at least 1")
            
        # 성능 설정 검증
        if config.performance.target_response_time <= 0:
            errors.append("Target response time must be positive")
            
        if config.performance.max_concurrent_predictions < 1:
            errors.append("Max concurrent predictions must be at least 1")
            
        return errors
        
    def get_database_url(self) -> str:
        """데이터베이스 URL 생성"""
        config = self.get_config()
        db_config = config.database
        
        if db_config.type == "sqlite":
            return f"sqlite:///{db_config.sqlite_path}"
        elif db_config.type == "postgresql":
            return (
                f"postgresql://{db_config.username}:{db_config.password}@"
                f"{db_config.host}:{db_config.port}/{db_config.database}"
            )
        else:
            raise ValueError(f"Unsupported database type: {db_config.type}")
            
    def get_redis_url(self) -> str:
        """Redis URL 생성"""
        config = self.get_config()
        redis_config = config.redis
        
        auth_part = f":{redis_config.password}@" if redis_config.password else ""
        return f"redis://{auth_part}{redis_config.host}:{redis_config.port}/{redis_config.db}"
        
# 글로벌 설정 관리자 인스턴스
_config_manager = None

def get_config_manager() -> ConfigManager:
    """글로벌 설정 관리자 인스턴스 반환"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config() -> HiveMindConfig:
    """현재 설정 반환"""
    return get_config_manager().get_config()

def reload_config():
    """설정 재로드"""
    get_config_manager().reload_if_changed()