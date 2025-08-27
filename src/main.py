# 🤖 Next-Generation Hive-Mind Lottery Prediction System
# FastAPI backend with advanced agent spawning and orchestration

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json
import uvicorn

from pydantic import BaseModel, Field

# Hive-Mind Components
from hive_mind.orchestrator import QueenOrchestrator
from hive_mind.config import get_config, get_config_manager
from hive_mind.memory import MemoryManager
from hive_mind.communication import MessageBus

# API Models
class PredictionRequest(BaseModel):
    """예측 요청 모델"""
    machine_type: str = Field(..., description="로또 기계 유형 (1호기, 2호기, 3호기)")
    sets_count: int = Field(default=1, ge=1, le=5, description="예측 세트 수")
    algorithm: str = Field(default="hybrid", description="사용할 알고리즘")
    ensemble_strategy: str = Field(default="dynamic", description="앙상블 전략")
    historical_data: Optional[List[Dict]] = Field(default=None, description="추가 과거 데이터")
    
class PredictionResponse(BaseModel):
    """예측 응답 모델"""
    success: bool
    predictions: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time_ms: float
    machine_type: str
    algorithm_used: str
    system_info: Dict[str, Any]
    
class SystemHealthResponse(BaseModel):
    """시스템 건강 상태 응답"""
    status: str
    health_score: int
    uptime_seconds: float
    active_agents: int
    message_bus_health: Dict[str, Any]
    memory_usage: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    
class AgentStatusResponse(BaseModel):
    """에이전트 상태 응답"""
    agents: List[Dict[str, Any]]
    total_count: int
    active_count: int
    
# Global system components
app_state = {
    'orchestrator': None,
    'memory_manager': None,
    'message_bus': None,
    'startup_time': None
}

# Security
security = HTTPBearer(auto_error=False)

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """API 키 검증"""
    config = get_config()
    
    if not config.api.api_key_required:
        return None
        
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )
        
    # 실제 구현에서는 데이터베이스에서 API 키 검증
    # 여기서는 간단한 검증 로직
    if credentials.credentials != config.api.jwt_secret:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
        
    return credentials.credentials

@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 라이프사이클 관리"""
    # Startup
    logging.info("🚀 Starting Hive-Mind Lottery Prediction System")
    
    try:
        # 설정 로드
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        # 로그 설정
        logging.basicConfig(
            level=getattr(logging, config.monitoring.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # 메모리 관리자 초기화
        memory_manager = MemoryManager(config.to_dict())
        await memory_manager.initialize()
        app_state['memory_manager'] = memory_manager
        
        # 메시지 버스 초기화
        message_bus = MessageBus(config.message_bus.to_dict())
        await message_bus.start()
        app_state['message_bus'] = message_bus
        
        # Queen Orchestrator 초기화
        orchestrator = QueenOrchestrator(
            orchestrator_id="queen_main",
            config=config.to_dict(),
            memory_manager=memory_manager,
            message_bus=message_bus
        )
        await orchestrator.initialize()
        app_state['orchestrator'] = orchestrator
        
        app_state['startup_time'] = datetime.now()
        
        logging.info("✅ Hive-Mind system started successfully")
        
        yield
        
    except Exception as e:
        logging.error(f"❌ Failed to start Hive-Mind system: {e}")
        raise
        
    # Shutdown
    logging.info("🔄 Shutting down Hive-Mind system")
    
    try:
        if app_state['orchestrator']:
            await app_state['orchestrator'].shutdown()
            
        if app_state['message_bus']:
            await app_state['message_bus'].stop()
            
        if app_state['memory_manager']:
            await app_state['memory_manager'].close()
            
        logging.info("✅ Hive-Mind system shutdown completed")
        
    except Exception as e:
        logging.error(f"❌ Error during shutdown: {e}")

# FastAPI 애플리케이션 생성
app = FastAPI(
    title="Next-Generation Hive-Mind Lottery Prediction System",
    description="Advanced AI system for lottery prediction using hybrid ML and distributed agent architecture",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 설정
config = get_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.api.cors_origins,
    allow_credentials=True,
    allow_methods=config.api.cors_methods,
    allow_headers=config.api.cors_headers,
)

# API 엔드포인트들

@app.get("/", response_model=Dict[str, Any])
async def root():
    """루트 엔드포인트"""
    return {
        "service": "Next-Generation Hive-Mind Lottery Prediction System",
        "version": "1.0.0",
        "status": "operational",
        "documentation": "/docs",
        "health": "/api/monitoring/health"
    }

@app.post("/api/predictions/generate", response_model=PredictionResponse)
async def generate_predictions(
    request: PredictionRequest,
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """로또 번호 예측 생성"""
    start_time = datetime.now()
    
    try:
        orchestrator = app_state['orchestrator']
        if not orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized"
            )
            
        # 예측 요청 데이터 준비
        task_data = {
            'request_id': str(uuid.uuid4()),
            'machine_type': request.machine_type,
            'sets_count': request.sets_count,
            'algorithm': request.algorithm,
            'ensemble_strategy': request.ensemble_strategy,
            'historical_data': request.historical_data,
            'timestamp': datetime.now().isoformat()
        }
        
        # 예측 생성
        result = await orchestrator.generate_predictions(task_data)
        
        if not result.get('success'):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Prediction failed: {result.get('error', 'Unknown error')}"
            )
            
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 백그라운드에서 성능 메트릭 저장
        background_tasks.add_task(
            _save_prediction_metrics,
            task_data,
            result,
            processing_time
        )
        
        return PredictionResponse(
            success=True,
            predictions=result.get('predictions', []),
            confidence_scores=result.get('confidence_scores', []),
            processing_time_ms=processing_time,
            machine_type=request.machine_type,
            algorithm_used=result.get('algorithm_used', request.algorithm),
            system_info={
                'agents_used': result.get('agents_used', []),
                'consensus_score': result.get('consensus_score', 0),
                'model_versions': result.get('model_versions', {})
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Prediction generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )

@app.post("/api/ensemble/predictions/")
async def generate_ensemble_predictions(
    request: Dict[str, Any],
    background_tasks: BackgroundTasks,
    api_key: Optional[str] = Depends(get_api_key)
):
    """앙상블 예측 생성 (고급 옵션)"""
    try:
        orchestrator = app_state['orchestrator']
        
        # 다중 앙상블 전략으로 예측
        strategies = request.get('strategies', ['voting', 'stacking', 'blending', 'dynamic'])
        results = {}
        
        for strategy in strategies:
            task_data = {
                **request,
                'ensemble_strategy': strategy,
                'request_id': f"{str(uuid.uuid4())}_{strategy}"
            }
            
            result = await orchestrator.generate_predictions(task_data)
            results[strategy] = result
            
        return {
            'success': True,
            'ensemble_results': results,
            'recommended_strategy': await _determine_best_strategy(results),
            'processing_info': {
                'strategies_used': len(strategies),
                'total_predictions': sum(len(r.get('predictions', [])) for r in results.values())
            }
        }
        
    except Exception as e:
        logging.error(f"❌ Ensemble prediction failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ensemble prediction failed"
        )

@app.get("/api/agents/status", response_model=AgentStatusResponse)
async def get_agents_status(api_key: Optional[str] = Depends(get_api_key)):
    """에이전트 상태 조회"""
    try:
        orchestrator = app_state['orchestrator']
        if not orchestrator:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="System not initialized"
            )
            
        agents_info = await orchestrator.get_agents_status()
        
        return AgentStatusResponse(
            agents=agents_info.get('agents', []),
            total_count=agents_info.get('total_count', 0),
            active_count=agents_info.get('active_count', 0)
        )
        
    except Exception as e:
        logging.error(f"❌ Failed to get agents status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve agents status"
        )

@app.post("/api/agents/spawn/{agent_type}")
async def spawn_agent(
    agent_type: str,
    config: Optional[Dict[str, Any]] = None,
    api_key: Optional[str] = Depends(get_api_key)
):
    """새로운 에이전트 생성"""
    try:
        orchestrator = app_state['orchestrator']
        
        result = await orchestrator.spawn_agent(agent_type, config or {})
        
        if result.get('success'):
            return {
                'success': True,
                'agent_id': result['agent_id'],
                'agent_type': agent_type,
                'message': f"Agent {agent_type} spawned successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to spawn agent: {result.get('error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Agent spawning failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent spawning failed"
        )

@app.delete("/api/agents/{agent_id}")
async def terminate_agent(
    agent_id: str,
    api_key: Optional[str] = Depends(get_api_key)
):
    """에이전트 종료"""
    try:
        orchestrator = app_state['orchestrator']
        
        result = await orchestrator.terminate_agent(agent_id)
        
        if result.get('success'):
            return {
                'success': True,
                'message': f"Agent {agent_id} terminated successfully"
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Agent not found: {agent_id}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"❌ Agent termination failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Agent termination failed"
        )

@app.get("/api/statistics/machine-analysis/{machine_type}")
async def get_machine_analysis(
    machine_type: str,
    api_key: Optional[str] = Depends(get_api_key)
):
    """기계별 통계 분석"""
    try:
        memory_manager = app_state['memory_manager']
        
        # 과거 예측 데이터 조회
        historical_predictions = await memory_manager.get_prediction_history(
            filters={'machine_type': machine_type},
            limit=100
        )
        
        # 통계 계산
        stats = {
            'machine_type': machine_type,
            'total_predictions': len(historical_predictions),
            'avg_confidence': 0.0,
            'number_frequency': {},
            'pattern_analysis': {},
            'recent_performance': {}
        }
        
        if historical_predictions:
            # 신뢰도 평균
            confidences = [p.get('confidence', 0) for p in historical_predictions]
            stats['avg_confidence'] = sum(confidences) / len(confidences)
            
            # 번호 빈도
            all_numbers = []
            for pred in historical_predictions:
                for prediction_set in pred.get('predictions', []):
                    all_numbers.extend(prediction_set.get('numbers', []))
                    
            for num in range(1, 46):
                stats['number_frequency'][str(num)] = all_numbers.count(num)
                
        return stats
        
    except Exception as e:
        logging.error(f"❌ Machine analysis failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Machine analysis failed"
        )

@app.get("/api/monitoring/health", response_model=SystemHealthResponse)
async def get_system_health():
    """시스템 건강 상태 조회"""
    try:
        orchestrator = app_state['orchestrator']
        message_bus = app_state['message_bus']
        memory_manager = app_state['memory_manager']
        startup_time = app_state['startup_time']
        
        if not all([orchestrator, message_bus, memory_manager, startup_time]):
            return SystemHealthResponse(
                status="unhealthy",
                health_score=0,
                uptime_seconds=0,
                active_agents=0,
                message_bus_health={},
                memory_usage={},
                performance_metrics={}
            )
            
        # 업타임 계산
        uptime = (datetime.now() - startup_time).total_seconds()
        
        # 각 컴포넌트 상태
        orchestrator_health = await orchestrator.get_system_health()
        message_bus_health = message_bus.get_system_health()
        memory_health = await memory_manager.get_health_status()
        
        # 전체 건강 점수 계산
        health_scores = [
            orchestrator_health.get('health_score', 0),
            message_bus_health.get('health_score', 0),
            memory_health.get('health_score', 0)
        ]
        
        overall_health = sum(health_scores) / len(health_scores) if health_scores else 0
        
        status = "healthy" if overall_health >= 80 else "degraded" if overall_health >= 60 else "unhealthy"
        
        return SystemHealthResponse(
            status=status,
            health_score=int(overall_health),
            uptime_seconds=uptime,
            active_agents=orchestrator_health.get('active_agents', 0),
            message_bus_health=message_bus_health,
            memory_usage=memory_health,
            performance_metrics={
                'avg_prediction_time_ms': orchestrator_health.get('avg_prediction_time', 0),
                'total_predictions': orchestrator_health.get('total_predictions', 0),
                'success_rate': orchestrator_health.get('success_rate', 0)
            }
        )
        
    except Exception as e:
        logging.error(f"❌ Health check failed: {e}")
        return SystemHealthResponse(
            status="error",
            health_score=0,
            uptime_seconds=0,
            active_agents=0,
            message_bus_health={"error": str(e)},
            memory_usage={},
            performance_metrics={}
        )

@app.get("/api/monitoring/metrics")
async def get_system_metrics(api_key: Optional[str] = Depends(get_api_key)):
    """시스템 메트릭 조회"""
    try:
        orchestrator = app_state['orchestrator']
        message_bus = app_state['message_bus']
        
        metrics = {
            'orchestrator_metrics': await orchestrator.get_performance_metrics() if orchestrator else {},
            'message_bus_metrics': message_bus.get_metrics() if message_bus else {},
            'timestamp': datetime.now().isoformat()
        }
        
        return metrics
        
    except Exception as e:
        logging.error(f"❌ Metrics retrieval failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve metrics"
        )

# Helper functions

async def _save_prediction_metrics(task_data: Dict, result: Dict, processing_time: float):
    """예측 메트릭 저장 (백그라운드 작업)"""
    try:
        memory_manager = app_state['memory_manager']
        if memory_manager:
            await memory_manager.store_prediction_log({
                **task_data,
                'result': result,
                'processing_time_ms': processing_time,
                'completed_at': datetime.now().isoformat()
            })
    except Exception as e:
        logging.error(f"❌ Failed to save prediction metrics: {e}")

async def _determine_best_strategy(results: Dict[str, Any]) -> str:
    """최적 앙상블 전략 결정"""
    try:
        # 각 전략의 평균 신뢰도 계산
        strategy_scores = {}
        
        for strategy, result in results.items():
            if result.get('success'):
                confidences = result.get('confidence_scores', [])
                if confidences:
                    strategy_scores[strategy] = sum(confidences) / len(confidences)
                    
        # 가장 높은 신뢰도의 전략 반환
        if strategy_scores:
            return max(strategy_scores, key=strategy_scores.get)
        else:
            return "dynamic"  # 기본값
            
    except Exception:
        return "dynamic"

# 에러 핸들러
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not Found",
            "message": "The requested resource was not found",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": "An internal server error occurred",
            "timestamp": datetime.now().isoformat()
        }
    )

# 메인 실행
if __name__ == "__main__":
    config = get_config()
    
    uvicorn.run(
        "main:app",
        host=config.api.host,
        port=config.api.port,
        workers=1,  # 단일 워커 (멀티프로세싱 시 상태 공유 이슈 방지)
        reload=config.debug,
        log_level=config.monitoring.log_level.lower()
    )