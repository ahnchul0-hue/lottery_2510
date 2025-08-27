#!/usr/bin/env python3
# 🤖 Database Initialization Script
# Sets up SQLite database with proper schema and initial data

import asyncio
import logging
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hive_mind.config import get_config, get_config_manager
from hive_mind.memory import MemoryManager
from hive_mind.services.data_loader import DataLoader

async def initialize_database():
    """데이터베이스 초기화 메인 함수"""
    print("🚀 Starting database initialization...")
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # 1. 설정 로드
        config_manager = get_config_manager()
        config = config_manager.get_config()
        
        print(f"📁 Environment: {config.environment.value}")
        print(f"📁 Database path: {config.database.sqlite_path}")
        
        # 2. 데이터 디렉토리 생성
        data_dir = Path(config.data_dir)
        data_dir.mkdir(exist_ok=True)
        
        models_dir = Path(config.model_dir)
        models_dir.mkdir(exist_ok=True)
        
        logs_dir = Path(config.log_dir)
        logs_dir.mkdir(exist_ok=True)
        
        print("✅ Created necessary directories")
        
        # 3. 메모리 관리자 초기화
        memory_manager = MemoryManager(config.to_dict())
        await memory_manager.initialize()
        
        print("✅ Database schema created")
        
        # 4. 데이터 로더 초기화 및 데이터 로드
        data_loader = DataLoader()
        success = await data_loader.initialize()
        
        if not success:
            print("❌ Failed to initialize data loader")
            return False
            
        # 5. 과거 데이터를 데이터베이스에 저장
        draws = data_loader.get_all_draws()
        
        if draws:
            print(f"📊 Loading {len(draws)} historical draws into database...")
            
            for i, draw in enumerate(draws):
                await memory_manager.store_lottery_draw({
                    'draw_no': draw.draw_no,
                    'date': draw.date,
                    'numbers': draw.numbers,
                    'bonus': draw.bonus,
                    'machine_type': draw.machine_type,
                    'created_at': f"2024-01-01T00:00:00"  # Historical data
                })
                
                if (i + 1) % 100 == 0:
                    print(f"  ... loaded {i + 1}/{len(draws)} draws")
                    
            print("✅ Historical data loaded")
        else:
            print("⚠️  No historical data found")
            
        # 6. 시스템 초기 설정 저장
        initial_settings = {
            'system_initialized': True,
            'initialization_date': '2024-01-01T00:00:00',
            'data_version': '1.0.0',
            'total_draws_loaded': len(draws)
        }
        
        await memory_manager.store_system_config(initial_settings)
        print("✅ System configuration stored")
        
        # 7. 정리
        await data_loader.close()
        await memory_manager.close()
        
        print("🎉 Database initialization completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        logging.error(f"Database initialization failed: {e}", exc_info=True)
        return False

async def reset_database():
    """데이터베이스 재설정 (모든 데이터 삭제)"""
    print("🔄 Resetting database...")
    
    config = get_config()
    db_path = Path(config.database.sqlite_path)
    
    if db_path.exists():
        db_path.unlink()
        print(f"🗑️  Deleted existing database: {db_path}")
    
    # 재초기화
    return await initialize_database()

async def verify_database():
    """데이터베이스 검증"""
    print("🔍 Verifying database...")
    
    try:
        config = get_config()
        memory_manager = MemoryManager(config.to_dict())
        
        await memory_manager.initialize()
        
        # 테이블 존재 확인
        async with memory_manager.db_pool.acquire() as conn:
            tables = await conn.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name NOT LIKE 'sqlite_%'
            """)
            table_names = [row[0] for row in await tables.fetchall()]
            
        expected_tables = [
            'lottery_draws', 'prediction_logs', 'agent_performance',
            'system_metrics', 'task_history', 'pattern_cache'
        ]
        
        print(f"📋 Found tables: {table_names}")
        
        missing_tables = [t for t in expected_tables if t not in table_names]
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False
            
        # 데이터 개수 확인
        draw_count = await memory_manager.get_total_draws()
        print(f"📊 Total lottery draws: {draw_count}")
        
        await memory_manager.close()
        print("✅ Database verification completed")
        return True
        
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

def main():
    """메인 실행 함수"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Database initialization script')
    parser.add_argument('command', choices=['init', 'reset', 'verify'], 
                       help='Command to execute')
    parser.add_argument('--env', default='development',
                       help='Environment (development, production)')
    
    args = parser.parse_args()
    
    # 환경 설정
    os.environ['HIVE_MIND_ENV'] = args.env
    
    # 명령어 실행
    if args.command == 'init':
        success = asyncio.run(initialize_database())
    elif args.command == 'reset':
        success = asyncio.run(reset_database())
    elif args.command == 'verify':
        success = asyncio.run(verify_database())
    else:
        print(f"Unknown command: {args.command}")
        success = False
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()