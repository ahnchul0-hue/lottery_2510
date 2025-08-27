#!/usr/bin/env python3
# 🤖 System Startup Script
# Comprehensive startup sequence for Hive-Mind system

import asyncio
import logging
import sys
import os
import signal
from pathlib import Path
from typing import Optional

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hive_mind.config import get_config, get_config_manager
from hive_mind.memory import MemoryManager
from hive_mind.communication import MessageBus
from hive_mind.orchestrator import QueenOrchestrator
from hive_mind.services.data_loader import get_data_loader

class HiveMindSystem:
    """통합 시스템 관리자"""
    
    def __init__(self):
        self.config = None
        self.memory_manager: Optional[MemoryManager] = None
        self.message_bus: Optional[MessageBus] = None
        self.orchestrator: Optional[QueenOrchestrator] = None
        self.data_loader = None
        
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_running = False
        
    async def initialize(self):
        """시스템 전체 초기화"""
        try:
            print("🚀 Starting Hive-Mind System initialization...")
            
            # 1. 설정 로드
            config_manager = get_config_manager()
            self.config = config_manager.get_config()
            
            # 로깅 설정
            logging.basicConfig(
                level=getattr(logging, self.config.monitoring.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(self.config.monitoring.log_file)
                ]
            )
            
            self.logger.info(f"📁 Environment: {self.config.environment.value}")
            
            # 2. 필요한 디렉토리 생성
            self._create_directories()
            
            # 3. 데이터 로더 초기화
            print("📊 Initializing data loader...")
            self.data_loader = await get_data_loader()
            
            # 4. 메모리 관리자 초기화
            print("🧠 Initializing memory manager...")
            self.memory_manager = MemoryManager(self.config.to_dict())
            await self.memory_manager.initialize()
            
            # 5. 메시지 버스 초기화
            print("📡 Initializing message bus...")
            self.message_bus = MessageBus(self.config.message_bus.to_dict())
            await self.message_bus.start()
            
            # 6. Queen Orchestrator 초기화
            print("👑 Initializing Queen Orchestrator...")
            self.orchestrator = QueenOrchestrator(
                orchestrator_id="queen_main",
                config=self.config.to_dict(),
                memory_manager=self.memory_manager,
                message_bus=self.message_bus
            )
            await self.orchestrator.initialize()
            
            # 7. 시스템 건강 상태 확인
            await self._health_check()
            
            self.is_running = True
            self.logger.info("✅ Hive-Mind System initialized successfully")
            print("🎉 System ready!")
            
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            await self.shutdown()
            raise
            
    def _create_directories(self):
        """필요한 디렉토리 생성"""
        dirs = [
            self.config.data_dir,
            self.config.model_dir,
            self.config.log_dir,
            "models/pytorch",
            "models/sklearn"
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    async def _health_check(self):
        """초기 건강 상태 확인"""
        print("🏥 Performing health check...")
        
        # 메모리 관리자 상태
        memory_health = await self.memory_manager.get_health_status()
        self.logger.info(f"Memory manager health: {memory_health}")
        
        # 메시지 버스 상태
        bus_health = self.message_bus.get_system_health()
        self.logger.info(f"Message bus health: {bus_health}")
        
        # 오케스트레이터 상태
        orchestrator_health = await self.orchestrator.get_system_health()
        self.logger.info(f"Orchestrator health: {orchestrator_health}")
        
        print("✅ Health check completed")
        
    async def run_server(self):
        """FastAPI 서버 실행"""
        try:
            import uvicorn
            from main import app
            
            # 시스템 상태를 앱에 주입
            app.state.hive_mind_system = self
            
            print(f"🌐 Starting API server on {self.config.api.host}:{self.config.api.port}")
            
            config = uvicorn.Config(
                app=app,
                host=self.config.api.host,
                port=self.config.api.port,
                log_level=self.config.monitoring.log_level.lower(),
                reload=self.config.debug
            )
            
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"❌ Server failed: {e}")
            raise
            
    async def run_standalone(self):
        """서버 없이 독립 실행"""
        print("🔄 Running in standalone mode...")
        
        try:
            # 시그널 핸들러 설정
            loop = asyncio.get_event_loop()
            for sig in [signal.SIGINT, signal.SIGTERM]:
                loop.add_signal_handler(sig, lambda: asyncio.create_task(self.shutdown()))
            
            # 시스템 실행
            while self.is_running:
                await asyncio.sleep(1)
                
                # 주기적 건강 확인
                if hasattr(self, '_last_health_check'):
                    import time
                    if time.time() - self._last_health_check > 300:  # 5분마다
                        await self._periodic_health_check()
                        self._last_health_check = time.time()
                else:
                    import time
                    self._last_health_check = time.time()
                    
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested...")
            await self.shutdown()
            
    async def _periodic_health_check(self):
        """주기적 건강 상태 확인"""
        try:
            health = await self.orchestrator.get_system_health()
            self.logger.info(f"Periodic health check: {health.get('health_score', 0)}/100")
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            
    async def shutdown(self):
        """시스템 종료"""
        if not self.is_running:
            return
            
        print("🔄 Shutting down Hive-Mind System...")
        self.is_running = False
        
        try:
            # 오케스트레이터 종료
            if self.orchestrator:
                await self.orchestrator.shutdown()
                
            # 메시지 버스 종료
            if self.message_bus:
                await self.message_bus.stop()
                
            # 메모리 관리자 종료
            if self.memory_manager:
                await self.memory_manager.close()
                
            # 데이터 로더 종료
            if self.data_loader:
                await self.data_loader.close()
                
            self.logger.info("✅ System shutdown completed")
            print("👋 Goodbye!")
            
        except Exception as e:
            self.logger.error(f"❌ Error during shutdown: {e}")
            
    async def demo_predictions(self):
        """데모 예측 실행"""
        print("🎯 Running demo predictions...")
        
        try:
            # 각 기계 타입별 예측
            machine_types = ["1호기", "2호기", "3호기"]
            
            for machine_type in machine_types:
                print(f"\n🎰 Generating predictions for {machine_type}...")
                
                task_data = {
                    'machine_type': machine_type,
                    'sets_count': 2,
                    'algorithm': 'hybrid',
                    'ensemble_strategy': 'dynamic'
                }
                
                result = await self.orchestrator.generate_predictions(task_data)
                
                if result.get('success'):
                    predictions = result.get('predictions', [])
                    print(f"✅ {machine_type} predictions:")
                    for i, pred in enumerate(predictions, 1):
                        numbers = pred.get('numbers', [])
                        confidence = pred.get('confidence', 0)
                        print(f"  Set {i}: {numbers} (confidence: {confidence:.2f})")
                else:
                    print(f"❌ Failed to generate predictions for {machine_type}")
                    
                await asyncio.sleep(1)  # 간격 두기
                
        except Exception as e:
            self.logger.error(f"❌ Demo predictions failed: {e}")

async def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Hive-Mind System Startup')
    parser.add_argument('mode', choices=['server', 'standalone', 'demo'], 
                       help='Execution mode')
    parser.add_argument('--env', default='development',
                       help='Environment (development, production)')
    parser.add_argument('--host', help='Override API host')
    parser.add_argument('--port', type=int, help='Override API port')
    
    args = parser.parse_args()
    
    # 환경 설정
    os.environ['HIVE_MIND_ENV'] = args.env
    
    # 시스템 초기화
    system = HiveMindSystem()
    
    try:
        await system.initialize()
        
        # 설정 오버라이드
        if args.host:
            system.config.api.host = args.host
        if args.port:
            system.config.api.port = args.port
            
        # 모드별 실행
        if args.mode == 'server':
            await system.run_server()
        elif args.mode == 'standalone':
            await system.run_standalone()
        elif args.mode == 'demo':
            await system.demo_predictions()
            await system.shutdown()
            
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)
    finally:
        if system.is_running:
            await system.shutdown()

if __name__ == "__main__":
    asyncio.run(main())