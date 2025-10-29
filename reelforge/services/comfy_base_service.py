"""
ComfyUI Base Service - Common logic for ComfyUI-based services
"""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict, Any

from comfykit import ComfyKit
from loguru import logger


class ComfyBaseService:
    """
    Base service for ComfyUI workflow-based capabilities
    
    Provides common functionality for TTS, Image, and other ComfyUI-based services.
    
    Subclasses should define:
    - WORKFLOW_PREFIX: Prefix for workflow files (e.g., "image_", "tts_")
    - DEFAULT_WORKFLOW: Default workflow filename (e.g., "image_flux.json")
    - WORKFLOWS_DIR: Directory containing workflows (default: "workflows")
    """
    
    WORKFLOW_PREFIX: str = ""  # Must be overridden by subclass
    DEFAULT_WORKFLOW: str = ""  # Must be overridden by subclass
    WORKFLOWS_DIR: str = "workflows"
    
    def __init__(self, config: dict, service_name: str):
        """
        Initialize ComfyUI base service
        
        Args:
            config: Full application config dict
            service_name: Service name in config (e.g., "tts", "image")
        """
        # Service-specific config (e.g., config["comfyui"]["tts"])
        comfyui_config = config.get("comfyui", {})
        self.config = comfyui_config.get(service_name, {})
        
        # Global ComfyUI config (for comfyui_url and runninghub_api_key)
        self.global_config = comfyui_config
        
        self.service_name = service_name
        self._workflows_cache: Optional[List[str]] = None
    
    def _scan_workflows(self) -> List[Dict[str, Any]]:
        """
        Scan workflows/source/*.json files from all source directories
        
        Returns:
            List of workflow info dicts
            Example: [
                {
                    "name": "image_flux.json",
                    "display_name": "image_flux.json - Selfhost",
                    "source": "selfhost",
                    "path": "workflows/selfhost/image_flux.json",
                    "key": "selfhost/image_flux.json"
                },
                {
                    "name": "image_flux.json",
                    "display_name": "image_flux.json - Runninghub", 
                    "source": "runninghub",
                    "path": "workflows/runninghub/image_flux.json",
                    "key": "runninghub/image_flux.json",
                    "workflow_id": "123456"
                }
            ]
        """
        workflows = []
        workflows_dir = Path(self.WORKFLOWS_DIR)
        
        if not workflows_dir.exists():
            logger.warning(f"Workflows directory not found: {workflows_dir}")
            return workflows
        
        # Scan subdirectories (selfhost, runninghub, etc.)
        for source_dir in workflows_dir.iterdir():
            if not source_dir.is_dir():
                logger.debug(f"Skipping non-directory: {source_dir}")
                continue
            
            source_name = source_dir.name
            
            # Scan workflow files in this source directory
            for file_path in source_dir.glob(f"{self.WORKFLOW_PREFIX}*.json"):
                try:
                    workflow_info = self._parse_workflow_file(file_path, source_name)
                    workflows.append(workflow_info)
                    logger.debug(f"Found workflow: {workflow_info['key']}")
                except Exception as e:
                    logger.error(f"Failed to parse workflow {file_path}: {e}")
        
        # Sort by key (source/name)
        return sorted(workflows, key=lambda w: w["key"])
    
    def _parse_workflow_file(self, file_path: Path, source: str) -> Dict[str, Any]:
        """
        Parse workflow file and extract metadata
        
        Args:
            file_path: Path to workflow JSON file
            source: Source directory name (e.g., "selfhost", "runninghub")
        
        Returns:
            Workflow info dict with structure:
            {
                "name": "image_flux.json",
                "display_name": "image_flux.json - Runninghub",
                "source": "runninghub",
                "path": "workflows/runninghub/image_flux.json",
                "key": "runninghub/image_flux.json",
                "workflow_id": "123456"  # Only for RunningHub
            }
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = json.load(f)
        
        # Build base info
        workflow_info = {
            "name": file_path.name,
            "display_name": f"{file_path.name} - {source.title()}",
            "source": source,
            "path": str(file_path),
            "key": f"{source}/{file_path.name}"
        }
        
        # Check if it's a wrapper format (RunningHub, etc.)
        if "source" in content:
            # Wrapper format: {"source": "runninghub", "workflow_id": "xxx", ...}
            if "workflow_id" in content:
                workflow_info["workflow_id"] = content["workflow_id"]
        
        return workflow_info
    
    def _get_default_workflow(self) -> str:
        """
        Get default workflow from config (required, no fallback)
        
        Returns:
            Default workflow key (e.g., "runninghub/image_flux.json")
        
        Raises:
            ValueError: If default_workflow not configured
        """
        default_workflow = self.config.get("default_workflow")
        
        if not default_workflow:
            raise ValueError(
                f"No default workflow configured for {self.service_name}. "
                f"Please set 'default_workflow' in config.yaml under '{self.service_name}' section. "
                f"Available workflows: {', '.join(self.available)}"
            )
        
        return default_workflow
    
    def _resolve_workflow(self, workflow: Optional[str] = None) -> Dict[str, Any]:
        """
        Resolve workflow key to workflow info
        
        Args:
            workflow: Workflow key (e.g., "runninghub/image_flux.json")
                     If None, uses default from config
        
        Returns:
            Workflow info dict with structure:
            {
                "name": "image_flux.json",
                "display_name": "image_flux.json - Runninghub",
                "source": "runninghub",
                "path": "workflows/runninghub/image_flux.json",
                "key": "runninghub/image_flux.json",
                "workflow_id": "123456"  # Only for RunningHub
            }
        
        Raises:
            ValueError: If workflow not found
        """
        # 1. If not specified, use default from config
        if workflow is None:
            workflow = self._get_default_workflow()
        
        # 2. Scan available workflows
        available_workflows = self._scan_workflows()
        
        # 3. Find matching workflow by key
        for wf_info in available_workflows:
            if wf_info["key"] == workflow:
                logger.info(f"🎬 Using {self.service_name} workflow: {workflow}")
                return wf_info
        
        # 4. Not found - generate error message
        available_keys = [wf["key"] for wf in available_workflows]
        available_str = ", ".join(available_keys) if available_keys else "none"
        raise ValueError(
            f"Workflow '{workflow}' not found. "
            f"Available workflows: {available_str}"
        )
    
    def _prepare_comfykit_config(
        self,
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Prepare ComfyKit configuration
        
        Args:
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
        
        Returns:
            ComfyKit configuration dict
        """
        kit_config = {}
        
        # ComfyUI URL (priority: param > global config > env > default)
        final_comfyui_url = (
            comfyui_url 
            or self.global_config.get("comfyui_url")
            or os.getenv("COMFYUI_BASE_URL")
            or "http://127.0.0.1:8188"
        )
        kit_config["comfyui_url"] = final_comfyui_url
        
        # RunningHub API key (priority: param > global config > env)
        final_rh_key = (
            runninghub_api_key
            or self.global_config.get("runninghub_api_key")
            or os.getenv("RUNNINGHUB_API_KEY")
        )
        if final_rh_key:
            kit_config["runninghub_api_key"] = final_rh_key
        
        logger.debug(f"ComfyKit config: {kit_config}")
        return kit_config
    
    def list_workflows(self) -> List[Dict[str, Any]]:
        """
        List all available workflows with full metadata
        
        Returns:
            List of workflow info dicts (sorted by key)
        
        Example:
            workflows = service.list_workflows()
            # [
            #     {
            #         "name": "image_flux.json",
            #         "display_name": "image_flux.json - Runninghub",
            #         "source": "runninghub",
            #         "path": "workflows/runninghub/image_flux.json",
            #         "key": "runninghub/image_flux.json",
            #         "workflow_id": "123456"
            #     },
            #     ...
            # ]
        """
        return self._scan_workflows()
    
    @property
    def available(self) -> List[str]:
        """
        List available workflow keys
        
        Returns:
            List of available workflow keys (e.g., ["runninghub/image_flux.json", ...])
        
        Example:
            print(f"Available workflows: {service.available}")
        """
        workflows = self.list_workflows()
        return [wf["key"] for wf in workflows]
    
    def __repr__(self) -> str:
        """String representation"""
        default = self._get_default_workflow()
        available = ", ".join(self.available) if self.available else "none"
        return (
            f"<{self.__class__.__name__} "
            f"default={default!r} "
            f"available=[{available}]>"
        )

