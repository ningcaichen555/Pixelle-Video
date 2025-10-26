"""
Image Generation Service - Workflow-based, no capability layer

This service directly uses ComfyKit to execute workflows without going through
the capability abstraction layer. This is because workflow files themselves
already provide sufficient abstraction and flexibility.
"""

import os
from pathlib import Path
from typing import Optional, List, Dict

from comfykit import ComfyKit
from loguru import logger


class ImageService:
    """
    Image generation service - Workflow-based
    
    Directly uses ComfyKit to execute workflows. No capability abstraction needed
    since workflow itself is already the abstraction.
    
    Usage:
        # Use default preset (workflows/image_default.json)
        image_url = await reelforge.image(prompt="a cat")
        
        # Use specific preset
        image_url = await reelforge.image(preset="flux", prompt="a cat")
        
        # List available presets
        presets = reelforge.image.list_presets()
        
        # Get preset path
        path = reelforge.image.get_preset_path("flux")
    """
    
    PRESET_PREFIX = "image_"
    DEFAULT_PRESET = "default"
    WORKFLOWS_DIR = "workflows"
    
    def __init__(self, config: dict):
        """
        Initialize image service
        
        Args:
            config: Full application config dict
        """
        self.config = config.get("image", {})
        self._presets_cache: Optional[Dict[str, str]] = None
    
    def _scan_presets(self) -> Dict[str, str]:
        """
        Scan workflows/image_*.json files
        
        Returns:
            Dict mapping preset name to workflow path
            Example: {"default": "workflows/image_default.json", "flux": "workflows/image_flux.json"}
        """
        if self._presets_cache is not None:
            return self._presets_cache
        
        presets = {}
        workflows_dir = Path(self.WORKFLOWS_DIR)
        
        if not workflows_dir.exists():
            logger.warning(f"Workflows directory not found: {workflows_dir}")
            return presets
        
        # Scan for image_*.json files
        for file in workflows_dir.glob(f"{self.PRESET_PREFIX}*.json"):
            # Extract preset name: "image_flux.json" -> "flux"
            preset_name = file.stem.replace(self.PRESET_PREFIX, "")
            presets[preset_name] = str(file)
            logger.debug(f"Found image preset: {preset_name} -> {file}")
        
        self._presets_cache = presets
        return presets
    
    def _get_default_preset(self) -> str:
        """
        Get default preset name from config or use "default"
        
        Priority:
        1. config.yaml: image.default
        2. "default"
        """
        return self.config.get("default", self.DEFAULT_PRESET)
    
    def _resolve_workflow(
        self, 
        preset: Optional[str] = None, 
        workflow: Optional[str] = None
    ) -> str:
        """
        Resolve preset/workflow to actual workflow path
        
        Args:
            preset: Preset name (e.g., "flux", "default")
            workflow: Full workflow path (for backward compatibility)
        
        Returns:
            Workflow file path
        
        Raises:
            ValueError: If preset not found or no workflows available
        """
        # 1. If explicit workflow path provided, use it
        if workflow:
            logger.debug(f"Using explicit workflow: {workflow}")
            return workflow
        
        # 2. Scan available presets
        presets = self._scan_presets()
        
        if not presets:
            raise ValueError(
                f"No workflow presets found in {self.WORKFLOWS_DIR}/ directory. "
                f"Please create at least one workflow file: {self.WORKFLOWS_DIR}/{self.PRESET_PREFIX}default.json"
            )
        
        # 3. Determine which preset to use
        if preset:
            # Use specified preset
            target_preset = preset
        else:
            # Use default preset
            target_preset = self._get_default_preset()
        
        # 4. Lookup preset
        if target_preset not in presets:
            available = ", ".join(sorted(presets.keys()))
            raise ValueError(
                f"Preset '{target_preset}' not found. "
                f"Available presets: {available}\n"
                f"Please create: {self.WORKFLOWS_DIR}/{self.PRESET_PREFIX}{target_preset}.json"
            )
        
        workflow_path = presets[target_preset]
        logger.info(f"🎨 Using image preset: {target_preset} ({workflow_path})")
        
        return workflow_path
    
    async def __call__(
        self,
        prompt: str,
        preset: Optional[str] = None,
        workflow: Optional[str] = None,
        # ComfyUI connection (optional overrides)
        comfyui_url: Optional[str] = None,
        runninghub_api_key: Optional[str] = None,
        # Common workflow parameters
        width: Optional[int] = None,
        height: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        steps: Optional[int] = None,
        seed: Optional[int] = None,
        cfg: Optional[float] = None,
        sampler: Optional[str] = None,
        **params
    ) -> str:
        """
        Generate image using workflow
        
        Args:
            prompt: Image generation prompt
            preset: Preset name (default: from config or "default")
            workflow: Full workflow path (backward compatible)
            comfyui_url: ComfyUI URL (optional, overrides config)
            runninghub_api_key: RunningHub API key (optional, overrides config)
            width: Image width
            height: Image height
            negative_prompt: Negative prompt
            steps: Sampling steps
            seed: Random seed
            cfg: CFG scale
            sampler: Sampler name
            **params: Additional workflow parameters
        
        Returns:
            Generated image URL/path
        
        Examples:
            # Simplest: use default preset (workflows/image_default.json)
            image_url = await reelforge.image(prompt="a beautiful cat")
            
            # Use specific preset
            image_url = await reelforge.image(preset="flux", prompt="a cat")
            
            # With additional parameters
            image_url = await reelforge.image(
                preset="flux",
                prompt="a cat",
                width=1024,
                height=1024,
                steps=20,
                seed=42
            )
            
            # Backward compatible: direct workflow path
            image_url = await reelforge.image(
                workflow="workflows/custom.json",
                prompt="a cat"
            )
            
            # With custom ComfyUI server
            image_url = await reelforge.image(
                prompt="a cat",
                comfyui_url="http://192.168.1.100:8188"
            )
        """
        # 1. Resolve workflow path
        workflow_path = self._resolve_workflow(preset=preset, workflow=workflow)
        
        # 2. Prepare ComfyKit config
        kit_config = {}
        
        # ComfyUI URL (priority: param > config > env > default)
        final_comfyui_url = (
            comfyui_url 
            or self.config.get("comfyui_url")
            or os.getenv("COMFYUI_BASE_URL")
            or "http://127.0.0.1:8188"
        )
        kit_config["comfyui_url"] = final_comfyui_url
        
        # RunningHub API key (priority: param > config > env)
        final_rh_key = (
            runninghub_api_key
            or self.config.get("runninghub_api_key")
            or os.getenv("RUNNINGHUB_API_KEY")
        )
        if final_rh_key:
            kit_config["runninghub_api_key"] = final_rh_key
        
        logger.debug(f"ComfyKit config: {kit_config}")
        
        # 3. Build workflow parameters
        workflow_params = {"prompt": prompt}
        
        # Add optional parameters
        if width is not None:
            workflow_params["width"] = width
        if height is not None:
            workflow_params["height"] = height
        if negative_prompt is not None:
            workflow_params["negative_prompt"] = negative_prompt
        if steps is not None:
            workflow_params["steps"] = steps
        if seed is not None:
            workflow_params["seed"] = seed
        if cfg is not None:
            workflow_params["cfg"] = cfg
        if sampler is not None:
            workflow_params["sampler"] = sampler
        
        # Add any additional parameters
        workflow_params.update(params)
        
        logger.debug(f"Workflow parameters: {workflow_params}")
        
        # 4. Execute workflow
        try:
            kit = ComfyKit(**kit_config)
            
            logger.info(f"Executing workflow: {workflow_path}")
            result = await kit.execute(workflow_path, workflow_params)
            
            # 5. Handle result
            if result.status != "completed":
                error_msg = result.msg or "Unknown error"
                logger.error(f"Image generation failed: {error_msg}")
                raise Exception(f"Image generation failed: {error_msg}")
            
            if not result.images:
                logger.error("No images generated")
                raise Exception("No images generated")
            
            image_url = result.images[0]
            logger.info(f"✅ Generated image: {image_url}")
            return image_url
        
        except Exception as e:
            logger.error(f"Image generation error: {e}")
            raise
    
    def list_presets(self) -> List[str]:
        """
        List all available image presets
        
        Returns:
            List of preset names (sorted alphabetically)
        
        Example:
            presets = reelforge.image.list_presets()
            # ['anime', 'default', 'flux', 'sd15']
        """
        return sorted(self._scan_presets().keys())
    
    def get_preset_path(self, preset: str) -> Optional[str]:
        """
        Get workflow path for a preset
        
        Args:
            preset: Preset name
        
        Returns:
            Workflow file path, or None if not found
        
        Example:
            path = reelforge.image.get_preset_path("flux")
            # 'workflows/image_flux.json'
        """
        return self._scan_presets().get(preset)
    
    @property
    def active(self) -> str:
        """
        Get active preset name
        
        This property is provided for compatibility with other services
        that use the capability layer.
        
        Returns:
            Active preset name
        
        Example:
            print(f"Using preset: {reelforge.image.active}")
        """
        return self._get_default_preset()
    
    @property
    def available(self) -> List[str]:
        """
        List available presets
        
        This property is provided for compatibility with other services
        that use the capability layer.
        
        Returns:
            List of available preset names
        
        Example:
            print(f"Available presets: {reelforge.image.available}")
        """
        return self.list_presets()
    
    def __repr__(self) -> str:
        """String representation"""
        active = self.active
        available = ", ".join(self.available) if self.available else "none"
        return (
            f"<ImageService "
            f"active={active!r} "
            f"available=[{available}]>"
        )
