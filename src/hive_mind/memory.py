# 🧠 Hive Memory Manager - Advanced Memory Management System
# SQLite + Redis hybrid memory architecture with intelligent caching

import asyncio
import sqlite3
import json
import logging
import aiosqlite
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import hashlib
import pickle
import redis.asyncio as redis

@dataclass
class MemoryConfig:
    """메모리 시스템 설정"""
    db_path: str = "data/hive_memory.db"
    cache_enabled: bool = True
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1시간
    max_cache_size_mb: int = 256
    compression_enabled: bool = True

class HiveMemoryManager:
    """
    차세대 하이브-마인드 메모리 관리자
    
    Claude-Flow의 memory.db를 확장한 고도화된 시스템:
    - 계층화된 메모리 (단기/장기)
    - 네임스페이스 기반 데이터 분류
    - 지능형 캐싱 및 압축
    - 실시간 분석 및 백업
    """
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.db_path = Path(config.db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.db_connection: Optional[aiosqlite.Connection] = None
        self.redis_client: Optional[redis.Redis] = None
        
        self.logger = logging.getLogger(__name__)
        
        # 메모리 사용량 추적
        self._memory_usage = {
            'cache_size_mb': 0,
            'db_size_mb': 0,
            'total_records': 0
        }
        
    async def initialize(self):
        """메모리 시스템 초기화"""
        self.logger.info("🧠 Initializing Hive Memory Manager...")
        
        # SQLite 데이터베이스 연결
        self.db_connection = await aiosqlite.connect(str(self.db_path))
        
        # 테이블 생성
        await self._create_tables()
        
        # Redis 캐시 초기화 (선택사항)
        if self.config.cache_enabled:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=False  # 바이너리 데이터 지원
                )
                await self.redis_client.ping()
                self.logger.info("✅ Redis cache connected")
            except Exception as e:
                self.logger.warning(f"⚠️  Redis unavailable, using in-memory cache: {e}")
                self.redis_client = None
                
        await self._update_memory_usage()
        self.logger.info("✅ Hive Memory Manager initialized")
        
    async def shutdown(self):
        """메모리 시스템 종료"""
        self.logger.info("🔄 Shutting down Hive Memory Manager...")
        
        if self.redis_client:
            await self.redis_client.close()
            
        if self.db_connection:
            await self.db_connection.close()
            
        self.logger.info("✅ Memory system shutdown complete")
        
    async def _create_tables(self):
        """데이터베이스 테이블 생성"""
        
        # 1. 기본 로또 추첨 데이터
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS lottery_draws (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                draw_round INTEGER NOT NULL,
                machine_type VARCHAR(10) NOT NULL,
                winning_numbers TEXT NOT NULL,
                draw_date DATE,
                odd_even_ratio VARCHAR(10),
                high_low_ratio VARCHAR(10),
                ac_value INTEGER,
                last_digit_sum INTEGER,
                total_sum INTEGER,
                special_memo TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(draw_round, machine_type)
            )
        """)
        
        # 2. 예측 결과 로그
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS prediction_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id VARCHAR(50) NOT NULL,
                machine_type VARCHAR(10) NOT NULL,
                algorithm VARCHAR(50),
                predictions TEXT NOT NULL,
                confidence_score REAL,
                agent_results TEXT,
                explanation TEXT,
                performance_metrics TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 3. 에이전트 성능 데이터
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS agent_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                agent_id VARCHAR(50) NOT NULL,
                agent_type VARCHAR(50),
                tasks_processed INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                error_count INTEGER DEFAULT 0,
                avg_response_time REAL DEFAULT 0.0,
                success_rate REAL DEFAULT 0.0,
                capabilities TEXT,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(agent_id)
            )
        """)
        
        # 4. 시스템 메트릭
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name VARCHAR(100) NOT NULL,
                metric_value REAL,
                metadata TEXT,
                recorded_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # 5. 작업 실행 기록
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS task_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id VARCHAR(50) NOT NULL,
                task_type VARCHAR(50),
                status VARCHAR(20),
                assigned_agents TEXT,
                result TEXT,
                error_message TEXT,
                processing_time_ms REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                completed_at DATETIME
            )
        """)
        
        # 6. 패턴 분석 캐시
        await self.db_connection.execute("""
            CREATE TABLE IF NOT EXISTS pattern_cache (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_key VARCHAR(100) NOT NULL,
                machine_type VARCHAR(10),
                pattern_data TEXT NOT NULL,
                confidence_score REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                expires_at DATETIME,
                UNIQUE(pattern_key, machine_type)
            )
        """)
        
        # 인덱스 생성
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_lottery_draws_round_machine ON lottery_draws(draw_round, machine_type)",
            "CREATE INDEX IF NOT EXISTS idx_prediction_logs_request ON prediction_logs(request_id)",
            "CREATE INDEX IF NOT EXISTS idx_agent_performance_id ON agent_performance(agent_id)",
            "CREATE INDEX IF NOT EXISTS idx_task_history_id ON task_history(task_id)",
            "CREATE INDEX IF NOT EXISTS idx_pattern_cache_key ON pattern_cache(pattern_key, machine_type)"
        ]
        
        for index in indexes:
            await self.db_connection.execute(index)
            
        await self.db_connection.commit()
        
    # === 로또 데이터 관리 ===
    
    async def store_lottery_data(self, lottery_data: List[Dict[str, Any]]):
        """로또 추첨 데이터 저장"""
        for record in lottery_data:
            await self.db_connection.execute("""
                INSERT OR REPLACE INTO lottery_draws 
                (draw_round, machine_type, winning_numbers, draw_date,
                 odd_even_ratio, high_low_ratio, ac_value, 
                 last_digit_sum, total_sum, special_memo, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                record['회차'],
                record['호기'],
                json.dumps(record['1등_당첨번호']),
                record.get('추첨일'),
                record['홀짝_비율'],
                record['고저_비율'],
                record['AC값'],
                record['끝수합'],
                record['총합'],
                record.get('특별_메모')
            ))
            
        await self.db_connection.commit()
        self.logger.info(f"💾 Stored {len(lottery_data)} lottery records")
        
    async def get_lottery_data(self, 
                              machine_type: Optional[str] = None,
                              limit: int = 50) -> List[Dict[str, Any]]:
        """로또 추첨 데이터 조회"""
        query = """
            SELECT draw_round, machine_type, winning_numbers, draw_date,
                   odd_even_ratio, high_low_ratio, ac_value,
                   last_digit_sum, total_sum, special_memo
            FROM lottery_draws
        """
        params = []
        
        if machine_type:
            query += " WHERE machine_type = ?"
            params.append(machine_type)
            
        query += " ORDER BY draw_round DESC LIMIT ?"
        params.append(limit)
        
        cursor = await self.db_connection.execute(query, params)
        rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                '회차': row[0],
                '호기': row[1],
                '1등_당첨번호': json.loads(row[2]),
                '추첨일': row[3],
                '홀짝_비율': row[4],
                '고저_비율': row[5],
                'AC값': row[6],
                '끝수합': row[7],
                '총합': row[8],
                '특별_메모': row[9]
            })
            
        return results
        
    # === 예측 결과 관리 ===
    
    async def store_prediction_result(self, result: Dict[str, Any]):
        """예측 결과 저장"""
        await self.db_connection.execute("""
            INSERT INTO prediction_logs 
            (request_id, machine_type, algorithm, predictions, 
             confidence_score, agent_results, explanation, performance_metrics)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result['request_id'],
            result['machine_type'],
            result.get('algorithm'),
            json.dumps(result['predictions']),
            result.get('confidence_score'),
            json.dumps(result.get('agent_results', {})),
            json.dumps(result.get('explanation', [])),
            json.dumps(result.get('performance_metrics', {}))
        ))
        
        await self.db_connection.commit()
        
    async def get_prediction_history(self, 
                                   machine_type: Optional[str] = None,
                                   limit: int = 20) -> List[Dict[str, Any]]:
        """예측 기록 조회"""
        query = """
            SELECT request_id, machine_type, algorithm, predictions,
                   confidence_score, created_at
            FROM prediction_logs
        """
        params = []
        
        if machine_type:
            query += " WHERE machine_type = ?"
            params.append(machine_type)
            
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        
        cursor = await self.db_connection.execute(query, params)
        rows = await cursor.fetchall()
        
        results = []
        for row in rows:
            results.append({
                'request_id': row[0],
                'machine_type': row[1],
                'algorithm': row[2],
                'predictions': json.loads(row[3]) if row[3] else [],
                'confidence_score': row[4],
                'created_at': row[5]
            })
            
        return results
        
    # === 에이전트 성능 관리 ===
    
    async def store_agent_performance(self, agent_id: str, data: Dict[str, Any]):
        """에이전트 성능 데이터 저장"""
        metrics = data['metrics']
        
        await self.db_connection.execute("""
            INSERT OR REPLACE INTO agent_performance 
            (agent_id, agent_type, tasks_processed, success_count, error_count,
             avg_response_time, success_rate, capabilities, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (
            agent_id,
            data['agent_type'],
            metrics['tasks_processed'],
            metrics['success_count'],
            metrics['error_count'],
            metrics['avg_response_time'],
            metrics['success_rate'],
            json.dumps(data['capabilities'])
        ))
        
        await self.db_connection.commit()
        
    async def get_agent_performance(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """에이전트 성능 데이터 조회"""
        
        # 캐시 확인
        if self.redis_client:
            cache_key = f"agent_perf:{agent_id}"
            cached = await self.redis_client.get(cache_key)
            if cached:
                return json.loads(cached.decode('utf-8'))
        
        cursor = await self.db_connection.execute("""
            SELECT tasks_processed, success_count, error_count,
                   avg_response_time, success_rate, capabilities, last_updated
            FROM agent_performance 
            WHERE agent_id = ?
        """, (agent_id,))
        
        row = await cursor.fetchone()
        if not row:
            return None
            
        result = {
            'tasks_processed': row[0],
            'success_count': row[1], 
            'error_count': row[2],
            'avg_response_time_ms': row[3],
            'success_rate': row[4],
            'avg_confidence': row[4],  # 임시로 success_rate 사용
            'capabilities': json.loads(row[5]) if row[5] else [],
            'last_updated': row[6]
        }
        
        # 캐시에 저장
        if self.redis_client:
            await self.redis_client.setex(
                cache_key, 
                self.config.cache_ttl,
                json.dumps(result)
            )
            
        return result
        
    # === 작업 기록 관리 ===
    
    async def store_task_result(self, task_data: Dict[str, Any]):
        """작업 결과 저장"""
        await self.db_connection.execute("""
            INSERT INTO task_history 
            (task_id, task_type, status, assigned_agents, result, 
             error_message, processing_time_ms, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_data['id'],
            task_data.get('type'),
            task_data['status'],
            json.dumps(task_data.get('assigned_agents', [])),
            json.dumps(task_data.get('result')),
            task_data.get('error'),
            self._calculate_processing_time(task_data),
            task_data.get('completed_at')
        ))
        
        await self.db_connection.commit()
        
    def _calculate_processing_time(self, task_data: Dict[str, Any]) -> Optional[float]:
        """작업 처리 시간 계산"""
        if task_data.get('started_at') and task_data.get('completed_at'):
            try:
                start = datetime.fromisoformat(task_data['started_at'])
                end = datetime.fromisoformat(task_data['completed_at'])
                return (end - start).total_seconds() * 1000
            except:
                pass
        return None
        
    # === 패턴 캐시 관리 ===
    
    async def store_pattern_cache(self,
                                 pattern_key: str,
                                 machine_type: str,
                                 pattern_data: Dict[str, Any],
                                 confidence_score: float,
                                 ttl_hours: int = 24):
        """패턴 분석 결과 캐시"""
        expires_at = datetime.now() + timedelta(hours=ttl_hours)
        
        await self.db_connection.execute("""
            INSERT OR REPLACE INTO pattern_cache 
            (pattern_key, machine_type, pattern_data, confidence_score, expires_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            pattern_key,
            machine_type,
            json.dumps(pattern_data),
            confidence_score,
            expires_at.isoformat()
        ))
        
        await self.db_connection.commit()
        
    async def get_pattern_cache(self,
                               pattern_key: str,
                               machine_type: str) -> Optional[Dict[str, Any]]:
        """패턴 캐시 조회"""
        cursor = await self.db_connection.execute("""
            SELECT pattern_data, confidence_score, created_at
            FROM pattern_cache 
            WHERE pattern_key = ? AND machine_type = ? 
            AND expires_at > CURRENT_TIMESTAMP
        """, (pattern_key, machine_type))
        
        row = await cursor.fetchone()
        if not row:
            return None
            
        return {
            'data': json.loads(row[0]),
            'confidence_score': row[1],
            'created_at': row[2]
        }
        
    # === 시스템 통계 및 모니터링 ===
    
    async def get_memory_usage(self) -> float:
        """메모리 사용률 반환 (0.0-1.0)"""
        await self._update_memory_usage()
        
        # 간단한 사용률 계산 (총 256MB 기준)
        total_mb = self._memory_usage['cache_size_mb'] + self._memory_usage['db_size_mb']
        max_mb = self.config.max_cache_size_mb + 512  # DB 최대 512MB
        
        return min(total_mb / max_mb, 1.0)
        
    async def _update_memory_usage(self):
        """메모리 사용량 업데이트"""
        try:
            # DB 파일 크기
            if self.db_path.exists():
                self._memory_usage['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
                
            # 전체 레코드 수
            cursor = await self.db_connection.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            )
            self._memory_usage['total_records'] = (await cursor.fetchone())[0]
            
            # Redis 메모리 사용량 (추정)
            if self.redis_client:
                try:
                    info = await self.redis_client.info('memory')
                    self._memory_usage['cache_size_mb'] = info.get('used_memory', 0) / (1024 * 1024)
                except:
                    pass
                    
        except Exception as e:
            self.logger.error(f"❌ Failed to update memory usage: {e}")
            
    async def get_system_statistics(self) -> Dict[str, Any]:
        """시스템 통계 정보"""
        await self._update_memory_usage()
        
        # 기본 통계 쿼리들
        stats = {}
        
        # 로또 데이터 통계
        cursor = await self.db_connection.execute(
            "SELECT COUNT(*) FROM lottery_draws"
        )
        stats['total_lottery_draws'] = (await cursor.fetchone())[0]
        
        # 예측 기록 통계
        cursor = await self.db_connection.execute(
            "SELECT COUNT(*) FROM prediction_logs"
        )
        stats['total_predictions'] = (await cursor.fetchone())[0]
        
        # 에이전트 수
        cursor = await self.db_connection.execute(
            "SELECT COUNT(*) FROM agent_performance"
        )
        stats['registered_agents'] = (await cursor.fetchone())[0]
        
        # 최근 24시간 활동
        yesterday = datetime.now() - timedelta(days=1)
        cursor = await self.db_connection.execute(
            "SELECT COUNT(*) FROM prediction_logs WHERE created_at >= ?",
            (yesterday.isoformat(),)
        )
        stats['predictions_24h'] = (await cursor.fetchone())[0]
        
        return {
            'statistics': stats,
            'memory_usage': self._memory_usage,
            'memory_usage_ratio': await self.get_memory_usage()
        }
        
    async def cleanup_old_data(self, days_to_keep: int = 30):
        """오래된 데이터 정리"""
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        cutoff_str = cutoff_date.isoformat()
        
        # 오래된 예측 로그 삭제
        await self.db_connection.execute(
            "DELETE FROM prediction_logs WHERE created_at < ?",
            (cutoff_str,)
        )
        
        # 만료된 패턴 캐시 삭제
        await self.db_connection.execute(
            "DELETE FROM pattern_cache WHERE expires_at < CURRENT_TIMESTAMP"
        )
        
        # 오래된 작업 기록 삭제
        await self.db_connection.execute(
            "DELETE FROM task_history WHERE created_at < ?",
            (cutoff_str,)
        )
        
        await self.db_connection.commit()
        self.logger.info(f"🧹 Cleaned up data older than {days_to_keep} days")
        
    async def backup_database(self, backup_path: str):
        """데이터베이스 백업"""
        backup_file = Path(backup_path)
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        
        # SQLite 백업
        async with aiosqlite.connect(str(backup_file)) as backup_db:
            await self.db_connection.backup(backup_db)
            
        self.logger.info(f"💾 Database backed up to {backup_path}")