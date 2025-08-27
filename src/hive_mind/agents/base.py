# 🤖 Base Agent - Next-Generation Hive-Mind Agent Foundation
# Unified agent architecture with capabilities-based specialization

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

class AgentCapabilities(Enum):
    """에이전트 능력 정의"""
    PATTERN_RECOGNITION = "pattern_recognition"
    STATISTICAL_ANALYSIS = "statistical_analysis"
    COGNITIVE_REASONING = "cognitive_reasoning"
    ENSEMBLE_OPTIMIZATION = "ensemble_optimization"
    NEURAL_PROCESSING = "neural_processing"
    DOMAIN_EXPERTISE = "domain_expertise"
    EXPLANATION_GENERATION = "explanation_generation"

class AgentStatus(Enum):
    """에이전트 상태"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    IDLE = "idle"
    ERROR = "error"
    SHUTDOWN = "shutdown"

@dataclass
class AgentMetrics:
    """에이전트 성능 메트릭"""
    tasks_processed: int = 0
    success_count: int = 0
    error_count: int = 0
    avg_response_time: float = 0.0
    last_activity: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.success_count / self.tasks_processed
        
    @property
    def error_rate(self) -> float:
        if self.tasks_processed == 0:
            return 0.0
        return self.error_count / self.tasks_processed

class BaseAgent(ABC):
    """
    기본 에이전트 클래스
    
    Claude-Flow 에이전트 시스템을 개선한 설계:
    - 자율적 작업 처리
    - 능력 기반 특화
    - 성능 모니터링
    - 장애 회복력
    """
    
    def __init__(self, 
                 agent_id: str,
                 memory_manager,
                 message_bus,
                 config: Dict[str, Any]):
        self.agent_id = agent_id
        self.memory_manager = memory_manager
        self.message_bus = message_bus
        self.config = config
        
        self.status = AgentStatus.INITIALIZING
        self.capabilities: List[AgentCapabilities] = []
        self.metrics = AgentMetrics()
        
        self.logger = logging.getLogger(f"{self.__class__.__name__}_{agent_id}")
        
        # 내부 상태
        self._current_task: Optional[str] = None
        self._last_health_check = datetime.now()
        self._task_history: List[Dict[str, Any]] = []
        
    @property
    def name(self) -> str:
        """에이전트 이름"""
        return f"{self.__class__.__name__}_{self.agent_id}"
        
    @abstractmethod
    async def initialize(self):
        """에이전트 초기화"""
        pass
        
    @abstractmethod
    async def process_prediction(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """예측 작업 처리"""
        pass
        
    @abstractmethod
    async def get_specialization_info(self) -> Dict[str, Any]:
        """특화 정보 반환"""
        pass
        
    async def shutdown(self):
        """에이전트 종료"""
        self.logger.info(f"🔄 Shutting down agent {self.agent_id}")
        self.status = AgentStatus.SHUTDOWN
        
        # 현재 작업 정리
        if self._current_task:
            await self._cleanup_current_task()
            
        # 메트릭 저장
        await self._save_metrics()
        
    def is_available(self) -> bool:
        """에이전트 사용 가능 여부"""
        return self.status in [AgentStatus.ACTIVE, AgentStatus.IDLE]
        
    def is_healthy(self) -> bool:
        """에이전트 건강 상태"""
        if self.status == AgentStatus.ERROR:
            return False
            
        # 최근 활동 시간 확인 (10분 이내)
        if self.metrics.last_activity:
            time_since_activity = datetime.now() - self.metrics.last_activity
            if time_since_activity.total_seconds() > 600:  # 10분
                return False
                
        return True
        
    async def execute_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """작업 실행 래퍼 - 메트릭 수집 및 에러 처리"""
        task_id = str(uuid.uuid4())
        start_time = time.time()
        
        self._current_task = task_id
        self.status = AgentStatus.BUSY
        
        try:
            self.logger.info(f"🚀 Starting task {task_id}")
            
            # 실제 처리 로직 호출
            result = await self.process_prediction(task_data)
            
            # 성공 메트릭 업데이트
            self.metrics.success_count += 1
            self.status = AgentStatus.IDLE
            
            self.logger.info(f"✅ Completed task {task_id}")
            return result
            
        except Exception as e:
            # 실패 메트릭 업데이트
            self.metrics.error_count += 1
            self.status = AgentStatus.ERROR
            
            self.logger.error(f"❌ Task {task_id} failed: {e}")
            raise
            
        finally:
            # 공통 메트릭 업데이트
            end_time = time.time()
            response_time = (end_time - start_time) * 1000  # ms
            
            self.metrics.tasks_processed += 1
            self.metrics.last_activity = datetime.now()
            self._update_avg_response_time(response_time)
            
            # 작업 기록 저장
            task_record = {
                'task_id': task_id,
                'start_time': start_time,
                'end_time': end_time,
                'response_time_ms': response_time,
                'status': 'success' if self.status != AgentStatus.ERROR else 'failed',
                'task_data_summary': self._summarize_task_data(task_data)
            }
            self._task_history.append(task_record)
            
            # 기록은 최근 100개만 유지
            if len(self._task_history) > 100:
                self._task_history = self._task_history[-100:]
                
            self._current_task = None
            
    def _update_avg_response_time(self, new_time: float):
        """평균 응답 시간 업데이트"""
        if self.metrics.tasks_processed == 1:
            self.metrics.avg_response_time = new_time
        else:
            # 이동 평균 계산
            total_processed = self.metrics.tasks_processed
            current_avg = self.metrics.avg_response_time
            
            self.metrics.avg_response_time = (
                (current_avg * (total_processed - 1) + new_time) / total_processed
            )
            
    def _summarize_task_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """작업 데이터 요약"""
        return {
            'machine_type': task_data.get('machine_type'),
            'sets_count': task_data.get('sets_count'),
            'algorithm': task_data.get('algorithm')
        }
        
    async def _cleanup_current_task(self):
        """현재 작업 정리"""
        if self._current_task:
            self.logger.warning(f"⚠️  Cleaning up interrupted task: {self._current_task}")
            # 필요한 정리 작업 수행
            
    async def _save_metrics(self):
        """메트릭을 메모리에 저장"""
        try:
            metrics_data = {
                'agent_id': self.agent_id,
                'agent_type': self.__class__.__name__,
                'metrics': {
                    'tasks_processed': self.metrics.tasks_processed,
                    'success_count': self.metrics.success_count,
                    'error_count': self.metrics.error_count,
                    'avg_response_time': self.metrics.avg_response_time,
                    'success_rate': self.metrics.success_rate,
                    'error_rate': self.metrics.error_rate
                },
                'capabilities': [cap.value for cap in self.capabilities],
                'last_updated': datetime.now().isoformat()
            }
            
            await self.memory_manager.store_agent_performance(
                self.agent_id, metrics_data
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save metrics: {e}")
            
    async def get_status(self) -> Dict[str, Any]:
        """에이전트 상태 정보 반환"""
        specialization = await self.get_specialization_info()
        
        return {
            'agent_id': self.agent_id,
            'name': self.name,
            'status': self.status.value,
            'capabilities': [cap.value for cap in self.capabilities],
            'metrics': {
                'tasks_processed': self.metrics.tasks_processed,
                'success_rate': self.metrics.success_rate,
                'error_rate': self.metrics.error_rate,
                'avg_response_time_ms': self.metrics.avg_response_time,
                'last_activity': self.metrics.last_activity.isoformat() 
                    if self.metrics.last_activity else None
            },
            'current_task': self._current_task,
            'specialization': specialization,
            'health_status': 'healthy' if self.is_healthy() else 'unhealthy'
        }
        
    async def send_message(self, 
                          recipient: str, 
                          message_type: str,
                          content: Dict[str, Any]):
        """다른 에이전트에게 메시지 전송"""
        try:
            message = {
                'sender': self.agent_id,
                'recipient': recipient,
                'type': message_type,
                'content': content,
                'timestamp': datetime.now().isoformat()
            }
            
            await self.message_bus.send_message(message)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send message: {e}")
            
    async def health_check(self) -> Dict[str, Any]:
        """에이전트 헬스 체크"""
        self._last_health_check = datetime.now()
        
        return {
            'agent_id': self.agent_id,
            'status': self.status.value,
            'is_healthy': self.is_healthy(),
            'is_available': self.is_available(),
            'current_task': self._current_task,
            'uptime_seconds': (
                datetime.now() - self.metrics.last_activity
            ).total_seconds() if self.metrics.last_activity else 0,
            'memory_usage_mb': self._get_memory_usage(),
            'check_timestamp': self._last_health_check.isoformat()
        }
        
    def _get_memory_usage(self) -> float:
        """메모리 사용량 추정 (MB)"""
        # 간단한 메모리 사용량 추정
        base_memory = 50  # 기본 50MB
        task_history_memory = len(self._task_history) * 0.1  # 작업당 0.1MB
        
        return base_memory + task_history_memory