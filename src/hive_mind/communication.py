# 🤖 Hive-Mind Communication System - Advanced Message Bus
# Asynchronous communication infrastructure for distributed agent coordination

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from enum import Enum
import weakref

class MessageType(Enum):
    """메시지 타입 정의"""
    PREDICTION_REQUEST = "prediction_request"
    PREDICTION_RESPONSE = "prediction_response"
    TASK_ASSIGNMENT = "task_assignment" 
    TASK_COMPLETION = "task_completion"
    HEALTH_CHECK = "health_check"
    STATUS_UPDATE = "status_update"
    CONSENSUS_REQUEST = "consensus_request"
    CONSENSUS_RESPONSE = "consensus_response"
    PERFORMANCE_METRIC = "performance_metric"
    ERROR_NOTIFICATION = "error_notification"
    SHUTDOWN_SIGNAL = "shutdown_signal"

class MessagePriority(Enum):
    """메시지 우선순위"""
    CRITICAL = 0    # 시스템 중요 (셧다운, 에러)
    HIGH = 1        # 즉시 처리 (예측 요청, 합의)
    NORMAL = 2      # 일반 처리 (상태 업데이트)
    LOW = 3         # 지연 가능 (성능 메트릭)

@dataclass
class Message:
    """메시지 데이터 클래스"""
    id: str
    sender: str
    recipient: str  # 'broadcast' for all agents
    message_type: MessageType
    priority: MessagePriority
    content: Dict[str, Any]
    timestamp: datetime
    correlation_id: Optional[str] = None  # 관련 메시지 추적
    expires_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            'id': self.id,
            'sender': self.sender,
            'recipient': self.recipient,
            'message_type': self.message_type.value,
            'priority': self.priority.value,
            'content': self.content,
            'timestamp': self.timestamp.isoformat(),
            'correlation_id': self.correlation_id,
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'retry_count': self.retry_count,
            'max_retries': self.max_retries
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Message':
        """딕셔너리에서 복원"""
        return cls(
            id=data['id'],
            sender=data['sender'],
            recipient=data['recipient'],
            message_type=MessageType(data['message_type']),
            priority=MessagePriority(data['priority']),
            content=data['content'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            correlation_id=data.get('correlation_id'),
            expires_at=datetime.fromisoformat(data['expires_at']) if data.get('expires_at') else None,
            retry_count=data.get('retry_count', 0),
            max_retries=data.get('max_retries', 3)
        )

class MessageHandler:
    """메시지 핸들러 래퍼"""
    def __init__(self, handler: Callable, message_types: Set[MessageType], priority: int = 0):
        self.handler = handler
        self.message_types = message_types
        self.priority = priority
        self.call_count = 0
        self.error_count = 0
        
    async def handle(self, message: Message) -> Any:
        """메시지 처리"""
        try:
            self.call_count += 1
            return await self.handler(message)
        except Exception as e:
            self.error_count += 1
            raise e

class MessageBus:
    """
    고성능 비동기 메시지 버스
    
    Claude-Flow 개선사항:
    - 우선순위 기반 메시지 처리
    - 자동 재시도 및 데드레터 큐
    - 성능 모니터링 및 백프레셔 제어
    - 분산 합의 지원
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 메시지 큐 (우선순위별)
        self.message_queues = {
            priority: asyncio.PriorityQueue() 
            for priority in MessagePriority
        }
        
        # 핸들러 관리
        self.handlers: Dict[str, List[MessageHandler]] = defaultdict(list)
        self.global_handlers: List[MessageHandler] = []
        
        # 메시지 추적
        self.pending_messages: Dict[str, Message] = {}
        self.message_history = deque(maxlen=1000)  # 최근 1000개 메시지
        self.dead_letter_queue = deque(maxlen=100)
        
        # 성능 메트릭
        self.metrics = {
            'messages_sent': 0,
            'messages_processed': 0,
            'messages_failed': 0,
            'avg_processing_time': 0.0,
            'queue_sizes': {p.name: 0 for p in MessagePriority}
        }
        
        # 백프레셔 제어
        self.max_queue_size = config.get('max_queue_size', 1000)
        self.processing_timeout = config.get('processing_timeout', 30.0)
        
        # 처리 워커
        self.workers: List[asyncio.Task] = []
        self.is_running = False
        
        # 합의 추적
        self.consensus_requests: Dict[str, Dict] = {}
        
    async def start(self):
        """메시지 버스 시작"""
        if self.is_running:
            return
            
        self.is_running = True
        self.logger.info("🚀 Starting message bus")
        
        # 우선순위별 워커 시작
        worker_counts = {
            MessagePriority.CRITICAL: 2,
            MessagePriority.HIGH: 3,
            MessagePriority.NORMAL: 2,
            MessagePriority.LOW: 1
        }
        
        for priority, count in worker_counts.items():
            for i in range(count):
                worker = asyncio.create_task(
                    self._message_worker(priority, f"{priority.name}_{i}")
                )
                self.workers.append(worker)
                
        # 정리 작업 워커
        cleanup_worker = asyncio.create_task(self._cleanup_worker())
        self.workers.append(cleanup_worker)
        
        self.logger.info(f"✅ Message bus started with {len(self.workers)} workers")
        
    async def stop(self):
        """메시지 버스 중지"""
        if not self.is_running:
            return
            
        self.logger.info("🔄 Stopping message bus")
        self.is_running = False
        
        # 모든 워커 중지
        for worker in self.workers:
            worker.cancel()
            
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()
        
        self.logger.info("✅ Message bus stopped")
        
    async def send_message(self, message: Message) -> bool:
        """메시지 전송"""
        try:
            # 백프레셔 확인
            queue = self.message_queues[message.priority]
            if queue.qsize() >= self.max_queue_size:
                self.logger.warning(f"⚠️ Queue full for priority {message.priority.name}")
                return False
                
            # 큐에 추가 (우선순위, 타임스탬프 기준 정렬)
            priority_value = message.priority.value
            timestamp_ns = message.timestamp.timestamp() * 1e9
            await queue.put((priority_value, timestamp_ns, message))
            
            # 추적 정보 업데이트
            self.pending_messages[message.id] = message
            self.metrics['messages_sent'] += 1
            self.metrics['queue_sizes'][message.priority.name] = queue.qsize()
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to send message {message.id}: {e}")
            return False
            
    async def send(self, 
                  sender: str,
                  recipient: str, 
                  message_type: MessageType,
                  content: Dict[str, Any],
                  priority: MessagePriority = MessagePriority.NORMAL,
                  correlation_id: Optional[str] = None,
                  expires_in_seconds: Optional[int] = None) -> Optional[str]:
        """편의 메소드 - 메시지 생성 및 전송"""
        
        message_id = str(uuid.uuid4())
        expires_at = None
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
            
        message = Message(
            id=message_id,
            sender=sender,
            recipient=recipient,
            message_type=message_type,
            priority=priority,
            content=content,
            timestamp=datetime.now(),
            correlation_id=correlation_id,
            expires_at=expires_at
        )
        
        success = await self.send_message(message)
        return message_id if success else None
        
    def register_handler(self, 
                        agent_id: str,
                        handler: Callable,
                        message_types: Set[MessageType],
                        priority: int = 0):
        """메시지 핸들러 등록"""
        message_handler = MessageHandler(handler, message_types, priority)
        self.handlers[agent_id].append(message_handler)
        
        # 우선순위 순으로 정렬
        self.handlers[agent_id].sort(key=lambda h: h.priority, reverse=True)
        
        self.logger.info(f"📋 Registered handler for {agent_id}: {[t.value for t in message_types]}")
        
    def register_global_handler(self, 
                              handler: Callable, 
                              message_types: Set[MessageType],
                              priority: int = 0):
        """글로벌 핸들러 등록 (모든 메시지 처리)"""
        message_handler = MessageHandler(handler, message_types, priority)
        self.global_handlers.append(message_handler)
        
        # 우선순위 순으로 정렬
        self.global_handlers.sort(key=lambda h: h.priority, reverse=True)
        
    def unregister_handler(self, agent_id: str):
        """핸들러 등록 해제"""
        if agent_id in self.handlers:
            del self.handlers[agent_id]
            self.logger.info(f"🗑️ Unregistered handlers for {agent_id}")
            
    async def _message_worker(self, priority: MessagePriority, worker_name: str):
        """메시지 처리 워커"""
        self.logger.info(f"🔄 Message worker {worker_name} started")
        
        queue = self.message_queues[priority]
        
        while self.is_running:
            try:
                # 메시지 대기 (타임아웃 포함)
                try:
                    priority_val, timestamp_ns, message = await asyncio.wait_for(
                        queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue
                    
                # 메시지 만료 확인
                if message.expires_at and datetime.now() > message.expires_at:
                    self.logger.warning(f"⏰ Message {message.id} expired")
                    self._move_to_dead_letter(message, "expired")
                    continue
                    
                # 메시지 처리
                await self._process_message(message)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Worker {worker_name} error: {e}")
                
        self.logger.info(f"✅ Message worker {worker_name} stopped")
        
    async def _process_message(self, message: Message):
        """개별 메시지 처리"""
        start_time = datetime.now()
        
        try:
            # 핸들러 찾기
            handlers_found = False
            
            # 수신자별 핸들러
            if message.recipient in self.handlers:
                for handler in self.handlers[message.recipient]:
                    if message.message_type in handler.message_types:
                        await handler.handle(message)
                        handlers_found = True
                        
            # 브로드캐스트 메시지
            elif message.recipient == 'broadcast':
                for agent_id, agent_handlers in self.handlers.items():
                    for handler in agent_handlers:
                        if message.message_type in handler.message_types:
                            await handler.handle(message)
                            handlers_found = True
                            
            # 글로벌 핸들러
            for handler in self.global_handlers:
                if message.message_type in handler.message_types:
                    await handler.handle(message)
                    handlers_found = True
                    
            # 핸들러가 없는 경우
            if not handlers_found:
                self.logger.warning(
                    f"⚠️ No handler found for message {message.id} "
                    f"(type: {message.message_type.value}, recipient: {message.recipient})"
                )
                
            # 처리 완료
            self._complete_message(message, start_time)
            
        except Exception as e:
            self.logger.error(f"❌ Failed to process message {message.id}: {e}")
            await self._retry_message(message)
            
    def _complete_message(self, message: Message, start_time: datetime):
        """메시지 처리 완료"""
        # 추적에서 제거
        if message.id in self.pending_messages:
            del self.pending_messages[message.id]
            
        # 히스토리에 추가
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        history_entry = {
            'message_id': message.id,
            'sender': message.sender,
            'recipient': message.recipient,
            'message_type': message.message_type.value,
            'processing_time_ms': processing_time,
            'timestamp': start_time.isoformat(),
            'status': 'completed'
        }
        
        self.message_history.append(history_entry)
        
        # 메트릭 업데이트
        self.metrics['messages_processed'] += 1
        
        # 평균 처리 시간 업데이트 (이동 평균)
        current_avg = self.metrics['avg_processing_time']
        processed_count = self.metrics['messages_processed']
        
        self.metrics['avg_processing_time'] = (
            (current_avg * (processed_count - 1) + processing_time) / processed_count
        )
        
    async def _retry_message(self, message: Message):
        """메시지 재시도"""
        message.retry_count += 1
        
        if message.retry_count > message.max_retries:
            self.logger.error(f"❌ Message {message.id} exceeded max retries")
            self._move_to_dead_letter(message, "max_retries_exceeded")
            return
            
        # 지수 백오프로 재전송
        delay = 2 ** message.retry_count
        await asyncio.sleep(delay)
        
        # 우선순위를 높여서 재전송
        retry_priority = MessagePriority.HIGH if message.priority != MessagePriority.CRITICAL else MessagePriority.CRITICAL
        message.priority = retry_priority
        
        await self.send_message(message)
        
    def _move_to_dead_letter(self, message: Message, reason: str):
        """데드레터 큐로 이동"""
        dead_letter_entry = {
            'message': message.to_dict(),
            'reason': reason,
            'moved_at': datetime.now().isoformat()
        }
        
        self.dead_letter_queue.append(dead_letter_entry)
        self.metrics['messages_failed'] += 1
        
        # 추적에서 제거
        if message.id in self.pending_messages:
            del self.pending_messages[message.id]
            
        self.logger.warning(f"💀 Message {message.id} moved to dead letter queue: {reason}")
        
    async def _cleanup_worker(self):
        """정리 작업 워커"""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # 1분마다 실행
                
                # 만료된 메시지 정리
                current_time = datetime.now()
                expired_messages = []
                
                for message_id, message in list(self.pending_messages.items()):
                    if message.expires_at and current_time > message.expires_at:
                        expired_messages.append(message)
                        
                for message in expired_messages:
                    self._move_to_dead_letter(message, "expired_during_cleanup")
                    
                if expired_messages:
                    self.logger.info(f"🧹 Cleaned up {len(expired_messages)} expired messages")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"❌ Cleanup worker error: {e}")
                
    async def request_consensus(self, 
                              requester_id: str,
                              topic: str,
                              data: Dict[str, Any],
                              required_votes: int = None,
                              timeout_seconds: int = 30) -> Dict[str, Any]:
        """분산 합의 요청"""
        consensus_id = str(uuid.uuid4())
        
        # 필요 투표 수 계산 (과반수)
        if required_votes is None:
            agent_count = len(self.handlers)
            required_votes = (agent_count // 2) + 1
            
        # 합의 요청 추적
        self.consensus_requests[consensus_id] = {
            'requester_id': requester_id,
            'topic': topic,
            'data': data,
            'required_votes': required_votes,
            'votes': {},
            'start_time': datetime.now(),
            'timeout_seconds': timeout_seconds,
            'status': 'pending'
        }
        
        # 브로드캐스트 합의 요청
        await self.send(
            sender=requester_id,
            recipient='broadcast',
            message_type=MessageType.CONSENSUS_REQUEST,
            content={
                'consensus_id': consensus_id,
                'topic': topic,
                'data': data,
                'required_votes': required_votes
            },
            priority=MessagePriority.HIGH,
            expires_in_seconds=timeout_seconds
        )
        
        # 결과 대기
        return await self._wait_for_consensus(consensus_id)
        
    async def _wait_for_consensus(self, consensus_id: str) -> Dict[str, Any]:
        """합의 결과 대기"""
        request = self.consensus_requests[consensus_id]
        timeout_time = request['start_time'] + timedelta(seconds=request['timeout_seconds'])
        
        while datetime.now() < timeout_time:
            if len(request['votes']) >= request['required_votes']:
                # 투표 분석
                vote_counts = {}
                for vote in request['votes'].values():
                    decision = vote.get('decision', 'abstain')
                    vote_counts[decision] = vote_counts.get(decision, 0) + 1
                    
                # 과반수 결정
                majority_decision = max(vote_counts, key=vote_counts.get)
                
                request['status'] = 'completed'
                result = {
                    'consensus_id': consensus_id,
                    'status': 'completed',
                    'decision': majority_decision,
                    'vote_counts': vote_counts,
                    'votes': request['votes'],
                    'completion_time': datetime.now().isoformat()
                }
                
                # 정리
                del self.consensus_requests[consensus_id]
                return result
                
            await asyncio.sleep(0.1)
            
        # 타임아웃
        request['status'] = 'timeout'
        result = {
            'consensus_id': consensus_id,
            'status': 'timeout',
            'votes_received': len(request['votes']),
            'required_votes': request['required_votes']
        }
        
        del self.consensus_requests[consensus_id]
        return result
        
    async def handle_consensus_response(self, message: Message):
        """합의 응답 처리"""
        content = message.content
        consensus_id = content.get('consensus_id')
        
        if consensus_id not in self.consensus_requests:
            return
            
        request = self.consensus_requests[consensus_id]
        request['votes'][message.sender] = {
            'decision': content.get('decision'),
            'reasoning': content.get('reasoning'),
            'timestamp': datetime.now().isoformat()
        }
        
    def get_metrics(self) -> Dict[str, Any]:
        """성능 메트릭 반환"""
        # 큐 크기 업데이트
        for priority in MessagePriority:
            queue = self.message_queues[priority]
            self.metrics['queue_sizes'][priority.name] = queue.qsize()
            
        return {
            **self.metrics,
            'pending_messages': len(self.pending_messages),
            'dead_letter_queue_size': len(self.dead_letter_queue),
            'active_consensus_requests': len(self.consensus_requests),
            'handler_count': sum(len(handlers) for handlers in self.handlers.values())
        }
        
    def get_system_health(self) -> Dict[str, Any]:
        """시스템 건강 상태"""
        metrics = self.get_metrics()
        
        # 건강 상태 평가
        health_score = 100
        issues = []
        
        # 큐 크기 확인
        total_queue_size = sum(metrics['queue_sizes'].values())
        if total_queue_size > self.max_queue_size * 0.8:
            health_score -= 20
            issues.append("High queue utilization")
            
        # 실패율 확인
        if metrics['messages_sent'] > 0:
            failure_rate = metrics['messages_failed'] / metrics['messages_sent']
            if failure_rate > 0.05:  # 5% 이상
                health_score -= 30
                issues.append(f"High failure rate: {failure_rate:.2%}")
                
        # 처리 시간 확인
        if metrics['avg_processing_time'] > 1000:  # 1초 이상
            health_score -= 25
            issues.append("Slow message processing")
            
        # 데드레터 큐 확인
        if metrics['dead_letter_queue_size'] > 10:
            health_score -= 15
            issues.append("Many failed messages")
            
        return {
            'health_score': max(0, health_score),
            'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 60 else 'unhealthy',
            'issues': issues,
            'metrics': metrics,
            'is_running': self.is_running,
            'worker_count': len(self.workers)
        }