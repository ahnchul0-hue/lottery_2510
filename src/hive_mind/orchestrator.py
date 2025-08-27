# 🧠 Queen Orchestrator - Next-Generation Hive-Mind Central Intelligence
# Enterprise-scale agent coordination system with swarm intelligence

import asyncio
import uuid
import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

from .memory import HiveMemoryManager
from .agents.base import BaseAgent, AgentCapabilities
from .agents.pattern_analyzer import PatternAnalyzerAgent
from .agents.statistical_predictor import StatisticalPredictorAgent
from .agents.cognitive_analyzer import CognitiveAnalyzerAgent
from .agents.ensemble_optimizer import EnsembleOptimizerAgent
from .communication import MessageBus, AgentMessage, MessageType
from .config import HiveConfig

class TaskPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """하이브-마인드 작업 정의"""
    id: str
    type: str
    priority: TaskPriority
    data: Dict[str, Any]
    required_capabilities: List[AgentCapabilities]
    assigned_agents: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'type': self.type,
            'priority': self.priority.value,
            'status': self.status.value,
            'assigned_agents': self.assigned_agents,
            'created_at': self.created_at.isoformat(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error': self.error
        }

class QueenOrchestrator:
    """
    차세대 하이브-마인드 퀸 오케스트레이터
    
    Claude-Flow의 한계를 극복한 분산 자율 협력 시스템:
    - 도메인 특화: 로또 예측에 특화된 전문 에이전트들
    - 분산 합의: 중앙 집중 없는 민주적 의사결정
    - 실시간 성능: 200ms 내 응답 최적화
    - 인지적 추론: 규칙 기반 지식과 설명 가능한 AI
    """
    
    def __init__(self, config: HiveConfig):
        self.config = config
        self.memory = HiveMemoryManager(config.memory_config)
        self.message_bus = MessageBus()
        
        # 에이전트 관리
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_pools: Dict[AgentCapabilities, List[str]] = {}
        
        # 작업 관리
        self.tasks: Dict[str, Task] = {}
        self.task_queue = asyncio.PriorityQueue()
        
        # 성능 메트릭
        self.performance_metrics = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'failed_tasks': 0,
            'avg_response_time': 0.0,
            'active_agents': 0
        }
        
        self.logger = logging.getLogger(__name__)
        self._running = False
        
    async def initialize(self):
        """퀸 오케스트레이터 초기화"""
        self.logger.info("🧠 Initializing Queen Orchestrator...")
        
        # 메모리 시스템 초기화
        await self.memory.initialize()
        
        # 메시지 버스 시작
        await self.message_bus.start()
        
        # 기본 에이전트 스폰
        await self._spawn_core_agents()
        
        # 백그라운드 태스크 시작
        asyncio.create_task(self._task_processor())
        asyncio.create_task(self._health_monitor())
        
        self._running = True
        self.logger.info("✅ Queen Orchestrator initialized successfully")
        
    async def shutdown(self):
        """시스템 종료"""
        self.logger.info("🔄 Shutting down Queen Orchestrator...")
        
        self._running = False
        
        # 모든 에이전트 종료
        for agent in self.agents.values():
            await agent.shutdown()
        
        # 메시지 버스 종료
        await self.message_bus.stop()
        
        # 메모리 시스템 종료
        await self.memory.shutdown()
        
        self.logger.info("✅ Queen Orchestrator shutdown complete")
        
    async def _spawn_core_agents(self):
        """핵심 에이전트들 스폰"""
        core_agents = [
            (PatternAnalyzerAgent, "pattern_analyzer_01"),
            (StatisticalPredictorAgent, "statistical_predictor_01"), 
            (CognitiveAnalyzerAgent, "cognitive_analyzer_01"),
            (EnsembleOptimizerAgent, "ensemble_optimizer_01")
        ]
        
        for agent_class, agent_id in core_agents:
            await self._spawn_agent(agent_class, agent_id)
            
    async def _spawn_agent(self, agent_class: Type[BaseAgent], agent_id: str) -> str:
        """새로운 에이전트 스폰"""
        agent = agent_class(
            agent_id=agent_id,
            memory_manager=self.memory,
            message_bus=self.message_bus,
            config=self.config.agent_config
        )
        
        await agent.initialize()
        
        self.agents[agent_id] = agent
        
        # 능력별 풀에 추가
        for capability in agent.capabilities:
            if capability not in self.agent_pools:
                self.agent_pools[capability] = []
            self.agent_pools[capability].append(agent_id)
            
        self.performance_metrics['active_agents'] += 1
        
        self.logger.info(f"🤖 Spawned agent: {agent_id} ({agent_class.__name__})")
        return agent_id
        
    async def submit_prediction_task(self, 
                                   machine_type: str,
                                   sets_count: int,
                                   algorithm: str = "enhanced") -> str:
        """예측 작업 제출"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type="prediction",
            priority=TaskPriority.HIGH,
            data={
                "machine_type": machine_type,
                "sets_count": sets_count,
                "algorithm": algorithm,
                "request_timestamp": datetime.now().isoformat()
            },
            required_capabilities=[
                AgentCapabilities.PATTERN_RECOGNITION,
                AgentCapabilities.STATISTICAL_ANALYSIS,
                AgentCapabilities.ENSEMBLE_OPTIMIZATION
            ]
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put((task.priority.value * -1, task))
        
        self.performance_metrics['total_tasks'] += 1
        
        self.logger.info(f"📊 Submitted prediction task: {task_id}")
        return task_id
        
    async def _task_processor(self):
        """작업 처리 백그라운드 프로세스"""
        while self._running:
            try:
                priority, task = await asyncio.wait_for(
                    self.task_queue.get(), 
                    timeout=1.0
                )
                
                await self._process_task(task)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"❌ Task processor error: {e}")
                
    async def _process_task(self, task: Task):
        """개별 작업 처리 - 분산 합의 방식"""
        task.status = TaskStatus.IN_PROGRESS
        task.started_at = datetime.now()
        
        try:
            if task.type == "prediction":
                result = await self._process_prediction_task(task)
                task.result = result
                task.status = TaskStatus.COMPLETED
                self.performance_metrics['completed_tasks'] += 1
            else:
                raise ValueError(f"Unknown task type: {task.type}")
                
            task.completed_at = datetime.now()
            
            # 성능 메트릭 업데이트
            response_time = (task.completed_at - task.started_at).total_seconds() * 1000
            self._update_response_time_metric(response_time)
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            self.performance_metrics['failed_tasks'] += 1
            self.logger.error(f"❌ Task {task.id} failed: {e}")
            
        # 결과를 메모리에 저장
        await self.memory.store_task_result(task.to_dict())
        
    async def _process_prediction_task(self, task: Task) -> Dict[str, Any]:
        """예측 작업 분산 처리"""
        machine_type = task.data['machine_type']
        sets_count = task.data['sets_count']
        
        # 1. 필요한 에이전트들 선택
        selected_agents = await self._select_agents_for_task(task)
        task.assigned_agents = selected_agents
        
        # 2. 병렬 예측 실행
        agent_results = await self._execute_parallel_predictions(
            selected_agents, task.data
        )
        
        # 3. 분산 합의를 통한 최종 결과 도출
        final_result = await self._reach_consensus(agent_results, task.data)
        
        # 4. 인지적 분석으로 설명 생성
        explanation = await self._generate_cognitive_explanation(
            final_result, agent_results
        )
        
        return {
            'request_id': task.id,
            'machine_type': machine_type,
            'predictions': final_result['predictions'],
            'agent_results': agent_results,
            'consensus_method': final_result['consensus_method'],
            'confidence_score': final_result['confidence_score'],
            'explanation': explanation,
            'generated_at': datetime.now().isoformat(),
            'performance_metrics': {
                'agents_used': len(selected_agents),
                'processing_time_ms': None  # 완료 시 계산
            }
        }
        
    async def _select_agents_for_task(self, task: Task) -> List[str]:
        """작업에 최적화된 에이전트 선택"""
        selected_agents = []
        
        for capability in task.required_capabilities:
            if capability in self.agent_pools:
                # 각 능력별로 가장 성능 좋은 에이전트 선택
                available_agents = self.agent_pools[capability]
                best_agent = await self._select_best_agent(
                    available_agents, capability
                )
                if best_agent and best_agent not in selected_agents:
                    selected_agents.append(best_agent)
                    
        return selected_agents
        
    async def _select_best_agent(self, 
                               agents: List[str], 
                               capability: AgentCapabilities) -> Optional[str]:
        """능력별 최적 에이전트 선택"""
        best_agent = None
        best_score = -1
        
        for agent_id in agents:
            agent = self.agents[agent_id]
            if agent.is_available():
                # 성능 기반 점수 계산
                score = await self._calculate_agent_performance_score(
                    agent_id, capability
                )
                if score > best_score:
                    best_score = score
                    best_agent = agent_id
                    
        return best_agent
        
    async def _calculate_agent_performance_score(self, 
                                               agent_id: str,
                                               capability: AgentCapabilities) -> float:
        """에이전트 성능 점수 계산"""
        # 메모리에서 과거 성능 데이터 조회
        performance_data = await self.memory.get_agent_performance(agent_id)
        
        if not performance_data:
            return 0.5  # 기본 점수
            
        # 성공률, 응답시간, 신뢰도 종합 점수
        success_rate = performance_data.get('success_rate', 0.5)
        avg_response_time = performance_data.get('avg_response_time_ms', 1000)
        confidence_score = performance_data.get('avg_confidence', 0.5)
        
        # 점수 계산 (낮은 응답시간이 높은 점수)
        time_score = max(0, 1 - (avg_response_time / 1000))  
        final_score = (success_rate * 0.4 + 
                      time_score * 0.3 + 
                      confidence_score * 0.3)
        
        return final_score
        
    async def _execute_parallel_predictions(self, 
                                          agents: List[str],
                                          task_data: Dict[str, Any]) -> Dict[str, Any]:
        """병렬 예측 실행"""
        tasks = []
        
        for agent_id in agents:
            agent = self.agents[agent_id]
            task_coro = agent.process_prediction(task_data)
            tasks.append((agent_id, task_coro))
            
        # 병렬 실행
        results = {}
        for agent_id, task_coro in tasks:
            try:
                result = await asyncio.wait_for(task_coro, timeout=5.0)
                results[agent_id] = result
            except asyncio.TimeoutError:
                self.logger.warning(f"⚠️  Agent {agent_id} timed out")
                results[agent_id] = {'error': 'timeout'}
            except Exception as e:
                self.logger.error(f"❌ Agent {agent_id} error: {e}")
                results[agent_id] = {'error': str(e)}
                
        return results
        
    async def _reach_consensus(self, 
                             agent_results: Dict[str, Any],
                             task_data: Dict[str, Any]) -> Dict[str, Any]:
        """분산 합의를 통한 최종 결과 도출"""
        valid_results = {k: v for k, v in agent_results.items() 
                        if 'error' not in v}
        
        if not valid_results:
            raise Exception("No valid agent results for consensus")
            
        sets_count = task_data['sets_count']
        
        # 가중 투표 방식으로 합의
        consensus_predictions = []
        
        for i in range(sets_count):
            # 각 에이전트의 i번째 예측 수집
            agent_predictions = []
            weights = []
            
            for agent_id, result in valid_results.items():
                if 'predictions' in result and len(result['predictions']) > i:
                    prediction = result['predictions'][i]
                    confidence = prediction.get('confidence_score', 0.5)
                    
                    agent_predictions.append(prediction['numbers'])
                    weights.append(confidence)
                    
            # 가중 앙상블로 최종 번호 선택
            if agent_predictions:
                final_numbers = await self._weighted_ensemble_selection(
                    agent_predictions, weights
                )
                
                consensus_predictions.append({
                    'set_number': i + 1,
                    'numbers': final_numbers,
                    'confidence_score': sum(weights) / len(weights),
                    'contributing_agents': len(agent_predictions)
                })
                
        return {
            'predictions': consensus_predictions,
            'consensus_method': 'weighted_voting',
            'confidence_score': (
                sum(p['confidence_score'] for p in consensus_predictions) / 
                len(consensus_predictions)
            ),
            'participating_agents': len(valid_results)
        }
        
    async def _weighted_ensemble_selection(self, 
                                         predictions: List[List[int]],
                                         weights: List[float]) -> List[int]:
        """가중 앙상블 번호 선택"""
        # 번호별 가중 점수 계산
        number_scores = {}
        
        for pred, weight in zip(predictions, weights):
            for num in pred:
                if num not in number_scores:
                    number_scores[num] = 0
                number_scores[num] += weight
                
        # 상위 6개 번호 선택
        sorted_numbers = sorted(
            number_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        selected_numbers = [num for num, score in sorted_numbers[:6]]
        return sorted(selected_numbers)
        
    async def _generate_cognitive_explanation(self, 
                                            final_result: Dict[str, Any],
                                            agent_results: Dict[str, Any]) -> List[str]:
        """인지적 분석을 통한 설명 생성"""
        explanations = []
        
        # 합의 과정 설명
        participating_agents = final_result['participating_agents']
        confidence = final_result['confidence_score']
        
        explanations.append(
            f"🧠 {participating_agents}개 전문 에이전트의 분산 합의 결과"
        )
        explanations.append(
            f"⚡ 평균 신뢰도: {confidence:.1%} (가중 투표 방식)"
        )
        
        # 에이전트별 기여도 분석
        for agent_id, result in agent_results.items():
            if 'error' not in result:
                agent_type = self.agents[agent_id].__class__.__name__
                agent_confidence = result.get('avg_confidence', 0.0)
                explanations.append(
                    f"🤖 {agent_type}: {agent_confidence:.1%} 신뢰도 기여"
                )
                
        return explanations
        
    async def _health_monitor(self):
        """시스템 헬스 모니터링"""
        while self._running:
            try:
                # 에이전트 상태 점검
                for agent_id, agent in self.agents.items():
                    if not agent.is_healthy():
                        self.logger.warning(f"⚠️  Agent {agent_id} unhealthy")
                        
                # 메모리 사용량 점검
                memory_usage = await self.memory.get_memory_usage()
                if memory_usage > 0.8:  # 80% 이상
                    self.logger.warning(f"⚠️  High memory usage: {memory_usage:.1%}")
                    
                await asyncio.sleep(30)  # 30초마다 점검
                
            except Exception as e:
                self.logger.error(f"❌ Health monitor error: {e}")
                
    def _update_response_time_metric(self, response_time_ms: float):
        """응답시간 메트릭 업데이트"""
        current_avg = self.performance_metrics['avg_response_time']
        completed_tasks = self.performance_metrics['completed_tasks']
        
        # 이동 평균 계산
        new_avg = ((current_avg * (completed_tasks - 1)) + response_time_ms) / completed_tasks
        self.performance_metrics['avg_response_time'] = new_avg
        
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """작업 상태 조회"""
        if task_id not in self.tasks:
            return None
            
        task = self.tasks[task_id]
        return task.to_dict()
        
    async def get_system_status(self) -> Dict[str, Any]:
        """시스템 전체 상태 조회"""
        return {
            'status': 'healthy' if self._running else 'shutdown',
            'agents': {
                'total': len(self.agents),
                'active': sum(1 for a in self.agents.values() if a.is_available()),
                'capabilities': {
                    cap.value: len(agents) 
                    for cap, agents in self.agent_pools.items()
                }
            },
            'tasks': {
                'pending': len([t for t in self.tasks.values() 
                              if t.status == TaskStatus.PENDING]),
                'in_progress': len([t for t in self.tasks.values() 
                                  if t.status == TaskStatus.IN_PROGRESS]),
                'completed': self.performance_metrics['completed_tasks'],
                'failed': self.performance_metrics['failed_tasks']
            },
            'performance': self.performance_metrics,
            'memory_usage': await self.memory.get_memory_usage()
        }