# 🤖 Hive-Mind Core Package
# Core components for distributed lottery prediction system

from .config import get_config, get_config_manager
from .memory import MemoryManager
from .communication import MessageBus
from .orchestrator import QueenOrchestrator

__version__ = "1.0.0"
__author__ = "Hive-Mind AI Team"