"""
ReelForge Configuration System

Unified configuration management with Pydantic validation.

Usage:
    from reelforge.config import config_manager
    
    # Access config (type-safe)
    api_key = config_manager.config.llm.api_key
    
    # Update config
    config_manager.update({"llm": {"api_key": "xxx"}})
    config_manager.save()
    
    # Validate
    if config_manager.validate():
        print("Config is valid!")
"""
from .schema import ReelForgeConfig, LLMConfig, ComfyUIConfig, TTSSubConfig, ImageSubConfig
from .manager import ConfigManager
from .loader import load_config_dict, save_config_dict

# Global singleton instance
config_manager = ConfigManager()

__all__ = [
    "ReelForgeConfig",
    "LLMConfig", 
    "ComfyUIConfig",
    "TTSSubConfig",
    "ImageSubConfig",
    "ConfigManager",
    "config_manager",
    "load_config_dict",
    "save_config_dict",
]

