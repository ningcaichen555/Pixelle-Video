# 🎬 Pixelle-Video AI视频生成流程详解

## 📋 目录
- [整体架构](#整体架构)
- [核心流程](#核心流程)
- [技术栈详解](#技术栈详解)
- [工作流配置](#工作流配置)
- [部署方案](#部署方案)
- [使用建议](#使用建议)

---

## 🏗️ 整体架构

Pixelle-Video 采用**模块化管道设计**，基于 ComfyUI 架构，支持灵活的AI能力组合。

### 架构图
```
用户输入 → 管道处理 → AI服务 → 媒体生成 → 视频输出
    ↓         ↓         ↓         ↓         ↓
  主题文本   线性管道   LLM服务   ComfyUI   最终视频
            标准管道   TTS服务   RunningHub
            资产管道   媒体服务
```

### 核心组件
- **管道系统** (`pipelines/`): 流程编排和状态管理
- **服务层** (`services/`): AI服务抽象和实现
- **工作流** (`workflows/`): ComfyUI配置文件
- **模板系统** (`templates/`): HTML视频模板
- **Web界面** (`web/`): Streamlit用户界面

---

## 🔄 核心流程

### 流程概览
```
输入主题 → 文案生成 → 配图规划 → 逐帧处理 → 视频合成
```

### 详细步骤

#### 1️⃣ 环境准备阶段 (Setup Environment)
**文件**: `pixelle_video/pipelines/standard.py` - `setup_environment()`

**功能**:
- 创建独立任务目录: `output/task_{timestamp}/`
- 生成唯一任务ID
- 初始化输出路径

**完整代码实现**:
```python
async def setup_environment(self, ctx: PipelineContext):
    """Step 1: Setup task directory and environment."""
    text = ctx.input_text
    mode = ctx.params.get("mode", "generate")
    
    logger.info(f"🚀 Starting StandardPipeline in '{mode}' mode")
    logger.info(f"   Text length: {len(text)} chars")
    
    # 创建独立任务目录
    task_dir, task_id = create_task_output_dir()
    ctx.task_id = task_id
    ctx.task_dir = task_dir
    
    logger.info(f"📁 Task directory created: {task_dir}")
    logger.info(f"   Task ID: {task_id}")
    
    # 确定最终视频路径
    output_path = ctx.params.get("output_path")
    if output_path is None:
        ctx.final_video_path = get_task_final_video_path(task_id)
    else:
        ctx.final_video_path = get_task_final_video_path(task_id)
        logger.info(f"   Will copy final video to: {output_path}")
```

**工具函数实现**:
```python
# pixelle_video/utils/os_util.py
def create_task_output_dir():
    """创建任务输出目录"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_id = f"task_{timestamp}_{random.randint(1000, 9999)}"
    task_dir = Path("output") / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    return str(task_dir), task_id

def get_task_final_video_path(task_id: str) -> str:
    """获取任务最终视频路径"""
    return f"output/{task_id}/final_video.mp4"

def get_task_frame_path(task_id: str, frame_index: int, file_type: str) -> str:
    """获取帧文件路径"""
    extensions = {
        "audio": ".mp3",
        "image": ".png", 
        "video": ".mp4",
        "composed": ".png",
        "segment": ".mp4"
    }
    ext = extensions.get(file_type, ".tmp")
    return f"output/{task_id}/frame_{frame_index:03d}_{file_type}{ext}"
```

---

#### 2️⃣ 内容生成阶段 (Generate Content)
**文件**: `pixelle_video/pipelines/standard.py` - `generate_content()`

**两种模式**:

**完整代码实现**:
```python
async def generate_content(self, ctx: PipelineContext):
    """Step 2: Generate or process script/narrations."""
    mode = ctx.params.get("mode", "generate")
    text = ctx.input_text
    n_scenes = ctx.params.get("n_scenes", 5)
    min_words = ctx.params.get("min_narration_words", 5)
    max_words = ctx.params.get("max_narration_words", 20)
    
    if mode == "generate":
        # AI生成模式: 使用LLM根据主题生成分镜文案
        self._report_progress(ctx.progress_callback, "generating_narrations", 0.05)
        ctx.narrations = await generate_narrations_from_topic(
            self.llm,
            topic=text,
            n_scenes=n_scenes,
            min_words=min_words,
            max_words=max_words
        )
        logger.info(f"✅ Generated {len(ctx.narrations)} narrations")
    else:  # fixed
        # 固定文案模式: 使用用户提供的完整脚本
        self._report_progress(ctx.progress_callback, "splitting_script", 0.05)
        split_mode = ctx.params.get("split_mode", "paragraph")
        ctx.narrations = await split_narration_script(text, split_mode=split_mode)
        logger.info(f"✅ Split script into {len(ctx.narrations)} segments (mode={split_mode})")
```

**核心函数实现**:
```python
# pixelle_video/utils/content_generators.py
async def generate_narrations_from_topic(
    llm_service,
    topic: str,
    n_scenes: int = 5,
    min_words: int = 5,
    max_words: int = 20
) -> List[str]:
    """根据主题生成分镜文案"""
    
    prompt = f"""
请根据主题"{topic}"创作一个短视频的分镜脚本。

要求:
1. 总共{n_scenes}个分镜
2. 每个分镜{min_words}-{max_words}个字
3. 内容要有逻辑性和连贯性
4. 适合短视频传播
5. 每行一个分镜，不要编号

主题: {topic}
"""
    
    response = await llm_service.generate(prompt)
    
    # 解析响应，按行分割
    narrations = []
    for line in response.strip().split('\n'):
        line = line.strip()
        if line and not line.startswith('#'):
            # 清理可能的编号
            line = re.sub(r'^\d+[.\s]*', '', line)
            if len(line) >= min_words:
                narrations.append(line)
    
    return narrations[:n_scenes]

async def split_narration_script(
    script: str, 
    split_mode: str = "paragraph"
) -> List[str]:
    """分割固定脚本"""
    
    if split_mode == "paragraph":
        # 按段落分割
        narrations = [p.strip() for p in script.split('\n\n') if p.strip()]
    elif split_mode == "sentence":
        # 按句子分割
        import re
        sentences = re.split(r'[。！？.!?]', script)
        narrations = [s.strip() for s in sentences if s.strip()]
    else:  # line
        # 按行分割
        narrations = [line.strip() for line in script.split('\n') if line.strip()]
    
    return narrations
```

---

#### 3️⃣ 标题确定阶段 (Determine Title)
**文件**: `pixelle_video/pipelines/standard.py` - `determine_title()`

**策略**:
- 用户指定标题: 直接使用
- AI生成模式: 调用 `generate_title()` 自动生成
- 固定文案模式: LLM基于内容生成标题

---

#### 4️⃣ 视觉规划阶段 (Plan Visuals)
**文件**: `pixelle_video/pipelines/standard.py` - `plan_visuals()`

**完整代码实现**:
```python
async def plan_visuals(self, ctx: PipelineContext):
    """Step 4: Generate image prompts or visual descriptions."""
    # 检测模板类型决定是否需要媒体生成
    frame_template = ctx.params.get("frame_template") or "1080x1920/default.html"
    
    template_name = Path(frame_template).name
    template_type = get_template_type(template_name)
    template_requires_media = (template_type in ["image", "video"])
    
    if template_type == "image":
        logger.info(f"📸 Template requires image generation")
    elif template_type == "video":
        logger.info(f"🎬 Template requires video generation")
    else:  # static
        logger.info(f"⚡ Static template - skipping media generation pipeline")
        logger.info(f"   💡 Benefits: Faster generation + Lower cost + No ComfyUI dependency")
    
    # 只有模板需要媒体时才生成图像提示词
    if template_requires_media:
        self._report_progress(ctx.progress_callback, "generating_image_prompts", 0.15)
        
        prompt_prefix = ctx.params.get("prompt_prefix")
        min_words = ctx.params.get("min_image_prompt_words", 30)
        max_words = ctx.params.get("max_image_prompt_words", 60)
        
        # 临时覆盖prompt_prefix配置
        original_prefix = None
        if prompt_prefix is not None:
            image_config = self.core.config.get("comfyui", {}).get("image", {})
            original_prefix = image_config.get("prompt_prefix")
            image_config["prompt_prefix"] = prompt_prefix
            logger.info(f"Using custom prompt_prefix: '{prompt_prefix}'")
        
        try:
            # 创建进度回调包装器
            def image_prompt_progress(completed: int, total: int, message: str):
                batch_progress = completed / total if total > 0 else 0
                overall_progress = 0.15 + (batch_progress * 0.15)
                self._report_progress(
                    ctx.progress_callback,
                    "generating_image_prompts",
                    overall_progress,
                    extra_info=message
                )
            
            # 生成基础图像提示词
            base_image_prompts = await generate_image_prompts(
                self.llm,
                narrations=ctx.narrations,
                min_words=min_words,
                max_words=max_words,
                progress_callback=image_prompt_progress
            )
            
            # 应用提示词前缀
            image_config = self.core.config.get("comfyui", {}).get("image", {})
            prompt_prefix_to_use = prompt_prefix if prompt_prefix is not None else image_config.get("prompt_prefix", "")
            
            ctx.image_prompts = []
            for base_prompt in base_image_prompts:
                final_prompt = build_image_prompt(base_prompt, prompt_prefix_to_use)
                ctx.image_prompts.append(final_prompt)
        
        finally:
            # 恢复原始prompt_prefix
            if original_prefix is not None:
                image_config["prompt_prefix"] = original_prefix
        
        logger.info(f"✅ Generated {len(ctx.image_prompts)} image prompts")
    else:
        # 静态模板 - 完全跳过图像提示词生成
        ctx.image_prompts = [None] * len(ctx.narrations)
        logger.info(f"⚡ Skipped image prompt generation (static template)")
        logger.info(f"   💡 Savings: {len(ctx.narrations)} LLM calls + {len(ctx.narrations)} media generations")
```

**工具函数实现**:
```python
# pixelle_video/utils/template_util.py
def get_template_type(template_name: str) -> str:
    """根据模板名称确定类型"""
    if template_name.startswith("static_"):
        return "static"
    elif template_name.startswith("image_"):
        return "image"
    elif template_name.startswith("video_"):
        return "video"
    else:
        # 默认为图片模板
        return "image"

# pixelle_video/utils/content_generators.py
async def generate_image_prompts(
    llm_service,
    narrations: List[str],
    min_words: int = 30,
    max_words: int = 60,
    progress_callback=None
) -> List[str]:
    """为每个分镜生成图像提示词"""
    
    image_prompts = []
    total = len(narrations)
    
    for i, narration in enumerate(narrations):
        if progress_callback:
            progress_callback(i, total, f"Generating prompt for scene {i+1}")
        
        prompt = f"""
请为以下视频分镜生成一个详细的图像描述提示词，用于AI图像生成。

分镜内容: {narration}

要求:
1. 描述要具体生动，{min_words}-{max_words}个词
2. 包含场景、人物、动作、环境等细节
3. 适合AI图像生成模型理解
4. 使用英文输出
5. 不要包含文字、字幕等元素

请直接输出图像描述，不要其他内容:
"""
        
        response = await llm_service.generate(prompt)
        image_prompt = response.strip()
        image_prompts.append(image_prompt)
    
    if progress_callback:
        progress_callback(total, total, "All prompts generated")
    
    return image_prompts

# pixelle_video/utils/prompt_helper.py
def build_image_prompt(base_prompt: str, prompt_prefix: str = "") -> str:
    """构建最终的图像提示词"""
    if not prompt_prefix:
        return base_prompt
    
    # 组合前缀和基础提示词
    if prompt_prefix.endswith(",") or base_prompt.startswith(","):
        final_prompt = f"{prompt_prefix} {base_prompt}".strip()
    else:
        final_prompt = f"{prompt_prefix}, {base_prompt}"
    
    return final_prompt
```

---

#### 5️⃣ 故事板初始化 (Initialize Storyboard)
**文件**: `pixelle_video/pipelines/standard.py` - `initialize_storyboard()`

**配置对象创建**:
```python
ctx.config = StoryboardConfig(
    task_id=ctx.task_id,
    n_storyboard=len(ctx.narrations),
    tts_inference_mode="local" | "comfyui",
    voice_id="zh-CN-YunjianNeural",
    tts_workflow="tts_edge.json",
    media_workflow="image_flux.json",
    frame_template="1080x1920/default.html"
)
```

**故事板创建**:
- 创建 `Storyboard` 对象包含标题和配置
- 为每个分镜创建 `StoryboardFrame` 对象
- 关联文案和图像提示词

---

#### 6️⃣ 资产生产阶段 (Produce Assets) ⭐ 核心
**文件**: `pixelle_video/pipelines/standard.py` - `produce_assets()`

这是最重要的阶段，对每一帧进行完整处理。

**并行处理优化**:
```python
# RunningHub工作流支持并行处理
if is_runninghub and runninghub_concurrent_limit > 1:
    semaphore = asyncio.Semaphore(runninghub_concurrent_limit)
    tasks = [process_frame_with_semaphore(i, frame) for i, frame in enumerate(frames)]
    results = await asyncio.gather(*tasks)
```

**单帧处理流程** (`services/frame_processor.py`):

##### 6.1 🎤 TTS音频生成
**文件**: `services/frame_processor.py` - `_step_generate_audio()`

**完整代码实现**:
```python
async def _step_generate_audio(
    self,
    frame: StoryboardFrame,
    config: StoryboardConfig
):
    """Step 1: Generate audio using TTS"""
    logger.debug(f"  1/4: Generating audio for frame {frame.index}...")
    
    # 生成输出路径
    from pixelle_video.utils.os_util import get_task_frame_path
    output_path = get_task_frame_path(config.task_id, frame.index, "audio")
    
    # 根据推理模式构建TTS参数
    tts_params = {
        "text": frame.narration,
        "inference_mode": config.tts_inference_mode,
        "output_path": output_path,
        "index": frame.index + 1,  # 1-based index for workflow
    }
    
    if config.tts_inference_mode == "local":
        # 本地模式: 传递语音和速度参数
        if config.voice_id:
            tts_params["voice"] = config.voice_id
        if config.tts_speed is not None:
            tts_params["speed"] = config.tts_speed
    else:  # comfyui
        # ComfyUI模式: 传递工作流、语音、速度和参考音频
        if config.tts_workflow:
            tts_params["workflow"] = config.tts_workflow
        if config.voice_id:
            tts_params["voice"] = config.voice_id
        if config.tts_speed is not None:
            tts_params["speed"] = config.tts_speed
        if config.ref_audio:
            tts_params["ref_audio"] = config.ref_audio
    
    # 调用TTS服务
    audio_path = await self.core.tts(**tts_params)
    
    frame.audio_path = audio_path
    
    # 获取音频时长
    frame.duration = await self._get_audio_duration(audio_path)
    
    logger.debug(f"  ✓ Audio generated: {audio_path} ({frame.duration:.2f}s)")

async def _get_audio_duration(self, audio_path: str) -> float:
    """获取音频时长（秒）"""
    try:
        # 使用ffmpeg-python获取时长
        import ffmpeg
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.warning(f"Failed to get audio duration: {e}, using estimate")
        # 备用方案: 根据文件大小估算（非常粗略）
        import os
        file_size = os.path.getsize(audio_path)
        # 假设MP3约16kbps，即每秒2KB
        estimated_duration = file_size / 2000
        return max(1.0, estimated_duration)  # 至少1秒
```

**TTS服务实现**:
```python
# pixelle_video/services/tts_service.py
class TTSService:
    async def __call__(
        self,
        text: str,
        inference_mode: str = "local",
        voice: str = "zh-CN-YunjianNeural",
        speed: float = 1.2,
        workflow: str = None,
        ref_audio: str = None,
        output_path: str = None,
        **kwargs
    ) -> str:
        """TTS音频生成统一接口"""
        
        if inference_mode == "local":
            return await self._generate_local_tts(
                text=text,
                voice=voice,
                speed=speed,
                output_path=output_path
            )
        else:  # comfyui
            return await self._generate_comfyui_tts(
                text=text,
                workflow=workflow,
                voice=voice,
                speed=speed,
                ref_audio=ref_audio,
                output_path=output_path,
                **kwargs
            )
    
    async def _generate_local_tts(
        self,
        text: str,
        voice: str,
        speed: float,
        output_path: str
    ) -> str:
        """本地Edge-TTS生成"""
        import edge_tts
        
        # 调整语速格式
        rate = f"{int((speed - 1) * 100):+d}%"
        
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        await communicate.save(output_path)
        
        return output_path
    
    async def _generate_comfyui_tts(
        self,
        text: str,
        workflow: str,
        voice: str = None,
        speed: float = None,
        ref_audio: str = None,
        output_path: str = None,
        **kwargs
    ) -> str:
        """ComfyUI工作流TTS生成"""
        from pixelle_video.services.comfy_base_service import ComfyBaseService
        
        # 构建工作流参数
        workflow_params = {
            "text": text,
        }
        
        if voice:
            workflow_params["voice"] = voice
        if speed:
            workflow_params["speed"] = speed
        if ref_audio:
            workflow_params["ref_audio"] = ref_audio
        
        # 执行ComfyUI工作流
        comfy_service = ComfyBaseService(self.core.config)
        result = await comfy_service.execute_workflow(
            workflow_path=workflow,
            params=workflow_params,
            **kwargs
        )
        
        # 下载音频文件到本地
        if result.audio_url:
            await self._download_file(result.audio_url, output_path)
            return output_path
        else:
            raise ValueError("No audio generated from ComfyUI workflow")
```

##### 6.2 🎨 媒体生成
**文件**: `services/frame_processor.py` - `_step_generate_media()`

**完整代码实现**:
```python
async def _step_generate_media(
    self,
    frame: StoryboardFrame,
    config: StoryboardConfig
):
    """Step 2: Generate media (image or video) using ComfyKit"""
    logger.debug(f"  2/4: Generating media for frame {frame.index}...")
    
    # 根据工作流确定媒体类型
    workflow_name = config.media_workflow or ""
    is_video_workflow = "video_" in workflow_name.lower()
    media_type = "video" if is_video_workflow else "image"
    
    logger.debug(f"  → Media type: {media_type} (workflow: {workflow_name})")
    
    # 构建媒体生成参数
    media_params = {
        "prompt": frame.image_prompt,
        "workflow": config.media_workflow,  # 从配置传递工作流
        "media_type": media_type,
        "width": config.media_width,
        "height": config.media_height,
        "index": frame.index + 1,  # 工作流使用1基索引
    }
    
    # 视频工作流: 传递音频时长作为目标视频时长
    # 这确保视频长度与音频长度匹配
    if is_video_workflow and frame.duration:
        media_params["duration"] = frame.duration
        logger.info(f"  → Generating video with target duration: {frame.duration:.2f}s (from TTS audio)")
    
    # 调用媒体生成服务
    media_result = await self.core.media(**media_params)
    
    # 存储媒体类型
    frame.media_type = media_result.media_type
    
    if media_result.is_image:
        # 下载图片到本地
        local_path = await self._download_media(
            media_result.url,
            frame.index,
            config.task_id,
            media_type="image"
        )
        frame.image_path = local_path
        logger.debug(f"  ✓ Image generated: {local_path}")
    
    elif media_result.is_video:
        # 下载视频到本地
        local_path = await self._download_media(
            media_result.url,
            frame.index,
            config.task_id,
            media_type="video"
        )
        frame.video_path = local_path
        
        # 从视频结果更新时长
        if media_result.duration:
            frame.duration = media_result.duration
            logger.debug(f"  ✓ Video generated: {local_path} (duration: {frame.duration:.2f}s)")
        else:
            # 从文件获取视频时长
            frame.duration = await self._get_video_duration(local_path)
            logger.debug(f"  ✓ Video generated: {local_path} (duration: {frame.duration:.2f}s)")
    
    else:
        raise ValueError(f"Unknown media type: {media_result.media_type}")

async def _download_media(
    self,
    url: str,
    frame_index: int,
    task_id: str,
    media_type: str
) -> str:
    """从URL下载媒体文件到本地"""
    from pixelle_video.utils.os_util import get_task_frame_path
    output_path = get_task_frame_path(task_id, frame_index, media_type)
    
    timeout = httpx.Timeout(connect=10.0, read=60, write=60, pool=60)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
    
    return output_path

async def _get_video_duration(self, video_path: str) -> float:
    """获取视频时长（秒）"""
    try:
        import ffmpeg
        probe = ffmpeg.probe(video_path)
        duration = float(probe['format']['duration'])
        return duration
    except Exception as e:
        logger.warning(f"Failed to get video duration: {e}, using audio duration")
        return 1.0  # 默认1秒
```

**媒体服务实现**:
```python
# pixelle_video/services/media.py
class MediaService:
    async def __call__(
        self,
        prompt: str,
        workflow: str = None,
        media_type: str = "image",
        width: int = 1024,
        height: int = 1024,
        duration: float = None,
        **kwargs
    ) -> MediaResult:
        """媒体生成统一接口"""
        
        # 确定工作流路径
        if not workflow:
            workflow = "image_flux.json" if media_type == "image" else "video_wan2.1.json"
        
        # 构建工作流参数
        workflow_params = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        
        # 视频特定参数
        if media_type == "video" and duration:
            workflow_params["duration"] = duration
        
        # 执行ComfyUI工作流
        from pixelle_video.services.comfy_base_service import ComfyBaseService
        comfy_service = ComfyBaseService(self.core.config)
        
        result = await comfy_service.execute_workflow(
            workflow_path=workflow,
            params=workflow_params,
            **kwargs
        )
        
        return MediaResult(
            media_type=media_type,
            url=result.output_url,
            duration=result.duration if media_type == "video" else None,
            width=width,
            height=height
        )

@dataclass
class MediaResult:
    """媒体生成结果"""
    media_type: str  # "image" or "video"
    url: str
    duration: Optional[float] = None
    width: int = 1024
    height: int = 1024
    
    @property
    def is_image(self) -> bool:
        return self.media_type == "image"
    
    @property
    def is_video(self) -> bool:
        return self.media_type == "video"
```

##### 6.3 🖼️ 帧合成
**文件**: `services/frame_processor.py` - `_step_compose_frame()`

**HTML模板渲染**:
- 使用 `HTMLFrameGenerator` 渲染最终画面
- 添加字幕、标题等文字元素
- 支持多种尺寸（竖屏1080x1920/横屏1920x1080/方形1080x1080）

**模板系统**:
```python
generator = HTMLFrameGenerator(template_path)
composed_path = await generator.generate_frame(
    title=storyboard.title,
    text=frame.narration,
    image=media_path,  # 支持图片和视频
    ext=custom_params
)
```

##### 6.4 🎬 视频片段创建
**文件**: `services/frame_processor.py` - `_step_create_video_segment()`

**完整代码实现**:
```python
async def _step_create_video_segment(
    self,
    frame: StoryboardFrame,
    config: StoryboardConfig
):
    """Step 4: Create video segment from media + audio"""
    logger.debug(f"  4/4: Creating video segment for frame {frame.index}...")
    
    # 生成输出路径
    from pixelle_video.utils.os_util import get_task_frame_path
    output_path = get_task_frame_path(config.task_id, frame.index, "segment")
    
    from pixelle_video.services.video import VideoService
    video_service = VideoService()
    
    # 根据媒体类型分支处理
    if frame.media_type == "video":
        # 视频工作流: 在视频上叠加HTML模板，然后添加音频
        logger.debug(f"  → Using video-based composition with HTML overlay")
        
        # 步骤1: 在视频上叠加透明HTML图像
        # composed_image_path包含带透明背景的渲染HTML
        temp_video_with_overlay = get_task_frame_path(config.task_id, frame.index, "video") + "_overlay.mp4"
        
        video_service.overlay_image_on_video(
            video=frame.video_path,
            overlay_image=frame.composed_image_path,
            output=temp_video_with_overlay,
            scale_mode="contain"  # 缩放视频以适应模板尺寸
        )
        
        # 步骤2: 为叠加后的视频添加旁白音频
        # 注意: 视频可能有音频（被替换）或静音（添加音频）
        segment_path = video_service.merge_audio_video(
            video=temp_video_with_overlay,
            audio=frame.audio_path,
            output=output_path,
            replace_audio=True,  # 用旁白替换视频音频
            audio_volume=1.0
        )
        
        # 清理临时文件
        import os
        if os.path.exists(temp_video_with_overlay):
            os.unlink(temp_video_with_overlay)
    
    elif frame.media_type == "image" or frame.media_type is None:
        # 图像工作流: 直接使用合成图像
        # asset_default.html模板在合成中包含图像
        logger.debug(f"  → Using image-based composition")
        
        segment_path = video_service.create_video_from_image(
            image=frame.composed_image_path,
            audio=frame.audio_path,
            output=output_path,
            fps=config.video_fps
        )
    
    else:
        raise ValueError(f"Unknown media type: {frame.media_type}")
    
    frame.video_segment_path = segment_path
    
    logger.debug(f"  ✓ Video segment created: {segment_path}")
```

**视频服务实现**:
```python
# pixelle_video/services/video.py
class VideoService:
    def overlay_image_on_video(
        self,
        video: str,
        overlay_image: str,
        output: str,
        scale_mode: str = "contain"
    ):
        """在视频上叠加图像"""
        import ffmpeg
        
        # 获取视频信息
        probe = ffmpeg.probe(video)
        video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
        video_width = int(video_info['width'])
        video_height = int(video_info['height'])
        
        # 构建ffmpeg命令
        video_input = ffmpeg.input(video)
        overlay_input = ffmpeg.input(overlay_image)
        
        if scale_mode == "contain":
            # 保持宽高比，适应视频尺寸
            overlay_scaled = ffmpeg.filter(
                overlay_input,
                'scale',
                f'{video_width}:{video_height}:force_original_aspect_ratio=decrease'
            )
            # 居中叠加
            output_stream = ffmpeg.overlay(
                video_input,
                overlay_scaled,
                x='(W-w)/2',
                y='(H-h)/2'
            )
        else:  # stretch
            # 拉伸到视频尺寸
            overlay_scaled = ffmpeg.filter(
                overlay_input,
                'scale',
                f'{video_width}:{video_height}'
            )
            output_stream = ffmpeg.overlay(video_input, overlay_scaled)
        
        # 输出视频
        output_stream = ffmpeg.output(output_stream, output)
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
    
    def merge_audio_video(
        self,
        video: str,
        audio: str,
        output: str,
        replace_audio: bool = True,
        audio_volume: float = 1.0
    ) -> str:
        """合并音频和视频"""
        import ffmpeg
        
        video_input = ffmpeg.input(video)
        audio_input = ffmpeg.input(audio)
        
        if replace_audio:
            # 替换视频音频
            if audio_volume != 1.0:
                audio_filtered = ffmpeg.filter(audio_input, 'volume', audio_volume)
            else:
                audio_filtered = audio_input
            
            output_stream = ffmpeg.output(
                video_input,
                audio_filtered,
                output,
                vcodec='copy',  # 复制视频流
                acodec='aac',   # 重新编码音频
                shortest=True   # 以最短流为准
            )
        else:
            # 混合音频
            audio_filtered = ffmpeg.filter(audio_input, 'volume', audio_volume)
            mixed_audio = ffmpeg.filter([video_input['a'], audio_filtered], 'amix')
            
            output_stream = ffmpeg.output(
                video_input['v'],
                mixed_audio,
                output,
                vcodec='copy',
                acodec='aac'
            )
        
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
        return output
    
    def create_video_from_image(
        self,
        image: str,
        audio: str,
        output: str,
        fps: int = 30
    ) -> str:
        """从图像和音频创建视频"""
        import ffmpeg
        
        # 获取音频时长
        probe = ffmpeg.probe(audio)
        duration = float(probe['format']['duration'])
        
        # 创建视频
        image_input = ffmpeg.input(image, loop=1, t=duration, framerate=fps)
        audio_input = ffmpeg.input(audio)
        
        output_stream = ffmpeg.output(
            image_input,
            audio_input,
            output,
            vcodec='libx264',
            acodec='aac',
            pix_fmt='yuv420p',
            shortest=True
        )
        
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
        return output
    
    def concat_videos(
        self,
        videos: List[str],
        output: str,
        bgm_path: str = None,
        bgm_volume: float = 0.2,
        bgm_mode: str = "loop"
    ) -> str:
        """拼接多个视频片段"""
        import ffmpeg
        
        # 创建输入流
        inputs = [ffmpeg.input(video) for video in videos]
        
        # 拼接视频
        if len(inputs) == 1:
            concatenated = inputs[0]
        else:
            concatenated = ffmpeg.concat(*inputs, v=1, a=1)
        
        # 添加背景音乐
        if bgm_path:
            bgm_input = ffmpeg.input(bgm_path)
            
            if bgm_mode == "loop":
                # 循环背景音乐
                bgm_looped = ffmpeg.filter(bgm_input, 'aloop', loop=-1, size=2**31-1)
            else:
                bgm_looped = bgm_input
            
            # 调整BGM音量
            bgm_adjusted = ffmpeg.filter(bgm_looped, 'volume', bgm_volume)
            
            # 混合音频
            mixed_audio = ffmpeg.filter(
                [concatenated['a'], bgm_adjusted],
                'amix',
                inputs=2,
                duration='first'
            )
            
            output_stream = ffmpeg.output(
                concatenated['v'],
                mixed_audio,
                output,
                vcodec='libx264',
                acodec='aac'
            )
        else:
            # 无背景音乐
            output_stream = ffmpeg.output(
                concatenated,
                output,
                vcodec='libx264',
                acodec='aac'
            )
        
        ffmpeg.run(output_stream, overwrite_output=True, quiet=True)
        return output
```

---

#### 7️⃣ 后期制作阶段 (Post Production)
**文件**: `pixelle_video/pipelines/standard.py` - `post_production()`

**完整代码实现**:
```python
async def post_production(self, ctx: PipelineContext):
    """Step 7: Concatenate videos and add BGM."""
    self._report_progress(ctx.progress_callback, "concatenating", 0.85)
    
    storyboard = ctx.storyboard
    segment_paths = [frame.video_segment_path for frame in storyboard.frames]
    
    video_service = VideoService()
    
    final_video_path = video_service.concat_videos(
        videos=segment_paths,
        output=ctx.final_video_path,
        bgm_path=ctx.params.get("bgm_path"),
        bgm_volume=ctx.params.get("bgm_volume", 0.2),
        bgm_mode=ctx.params.get("bgm_mode", "loop")
    )
    
    storyboard.final_video_path = final_video_path
    storyboard.completed_at = datetime.now()
    
    # 复制到用户指定路径（如果提供）
    user_specified_output = ctx.params.get("output_path")
    if user_specified_output:
        Path(user_specified_output).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(final_video_path, user_specified_output)
        logger.info(f"📹 Final video copied to: {user_specified_output}")
        ctx.final_video_path = user_specified_output
        storyboard.final_video_path = user_specified_output
    
    logger.success(f"🎬 Video generation completed: {ctx.final_video_path}")
```

**BGM处理详细实现**:
```python
# 在VideoService.concat_videos()中的BGM处理逻辑
def _add_background_music(
    self,
    video_input,
    bgm_path: str,
    bgm_volume: float = 0.2,
    bgm_mode: str = "loop"
):
    """添加背景音乐到视频"""
    import ffmpeg
    
    # 获取视频时长
    probe = ffmpeg.probe(video_input)
    video_duration = float(probe['format']['duration'])
    
    bgm_input = ffmpeg.input(bgm_path)
    
    if bgm_mode == "loop":
        # 循环模式: BGM循环播放直到视频结束
        bgm_looped = ffmpeg.filter(
            bgm_input,
            'aloop',
            loop=-1,  # 无限循环
            size=2**31-1  # 最大样本数
        )
        # 截取到视频时长
        bgm_trimmed = ffmpeg.filter(bgm_looped, 'atrim', duration=video_duration)
    elif bgm_mode == "fade":
        # 淡入淡出模式
        bgm_faded = ffmpeg.filter(
            bgm_input,
            'afade',
            type='in',
            start_time=0,
            duration=2
        )
        bgm_faded = ffmpeg.filter(
            bgm_faded,
            'afade',
            type='out',
            start_time=video_duration-2,
            duration=2
        )
        bgm_trimmed = ffmpeg.filter(bgm_faded, 'atrim', duration=video_duration)
    else:  # once
        # 单次播放模式
        bgm_trimmed = ffmpeg.filter(bgm_input, 'atrim', duration=video_duration)
    
    # 调整音量
    bgm_adjusted = ffmpeg.filter(bgm_trimmed, 'volume', bgm_volume)
    
    return bgm_adjusted
```

---

#### 8️⃣ 完成阶段 (Finalize)
**文件**: `pixelle_video/pipelines/standard.py` - `finalize()`

**完整代码实现**:
```python
async def finalize(self, ctx: PipelineContext) -> VideoGenerationResult:
    """Step 8: Create result object and persist metadata."""
    self._report_progress(ctx.progress_callback, "completed", 1.0)
    
    video_path_obj = Path(ctx.final_video_path)
    file_size = video_path_obj.stat().st_size
    
    result = VideoGenerationResult(
        video_path=ctx.final_video_path,
        storyboard=ctx.storyboard,
        duration=ctx.storyboard.total_duration,
        file_size=file_size
    )
    
    ctx.result = result
    
    logger.info(f"✅ Generated video: {ctx.final_video_path}")
    logger.info(f"   Duration: {ctx.storyboard.total_duration:.2f}s")
    logger.info(f"   Size: {file_size / (1024*1024):.2f} MB")
    logger.info(f"   Frames: {len(ctx.storyboard.frames)}")
    
    # 持久化元数据
    await self._persist_task_data(ctx)
    
    return result

async def _persist_task_data(self, ctx: PipelineContext):
    """持久化任务元数据和故事板到文件系统"""
    try:
        storyboard = ctx.storyboard
        result = ctx.result
        task_id = storyboard.config.task_id
        
        if not task_id:
            logger.warning("No task_id in storyboard, skipping persistence")
            return
        
        # 构建元数据
        input_with_title = ctx.params.copy()
        input_with_title["text"] = ctx.input_text  # 确保包含文本
        if not input_with_title.get("title"):
            input_with_title["title"] = storyboard.title
        
        metadata = {
            "task_id": task_id,
            "created_at": storyboard.created_at.isoformat() if storyboard.created_at else None,
            "completed_at": storyboard.completed_at.isoformat() if storyboard.completed_at else None,
            "status": "completed",
            
            "input": input_with_title,
            
            "result": {
                "video_path": result.video_path,
                "duration": result.duration,
                "file_size": result.file_size,
                "n_frames": len(storyboard.frames)
            },
            
            "config": {
                "llm_model": self.core.config.get("llm", {}).get("model", "unknown"),
                "llm_base_url": self.core.config.get("llm", {}).get("base_url", "unknown"),
                "comfyui_url": self.core.config.get("comfyui", {}).get("comfyui_url", "unknown"),
                "runninghub_enabled": bool(self.core.config.get("comfyui", {}).get("runninghub_api_key")),
            }
        }
        
        # 保存元数据
        await self.core.persistence.save_task_metadata(task_id, metadata)
        logger.info(f"💾 Saved task metadata: {task_id}")
        
        # 保存故事板
        await self.core.persistence.save_storyboard(task_id, storyboard)
        logger.info(f"💾 Saved storyboard: {task_id}")
        
    except Exception as e:
        logger.error(f"Failed to persist task data: {e}")
        # 不抛出异常 - 持久化失败不应该破坏视频生成
```

**数据模型定义**:
```python
# pixelle_video/models/storyboard.py
@dataclass
class VideoGenerationResult:
    """视频生成结果"""
    video_path: str
    storyboard: 'Storyboard'
    duration: float
    file_size: int
    
    @property
    def size_mb(self) -> float:
        """文件大小（MB）"""
        return self.file_size / (1024 * 1024)
    
    @property
    def frames_count(self) -> int:
        """帧数"""
        return len(self.storyboard.frames)
    
    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "video_path": self.video_path,
            "duration": self.duration,
            "file_size": self.file_size,
            "size_mb": self.size_mb,
            "frames_count": self.frames_count,
            "title": self.storyboard.title,
            "created_at": self.storyboard.created_at.isoformat() if self.storyboard.created_at else None,
            "completed_at": self.storyboard.completed_at.isoformat() if self.storyboard.completed_at else None
        }

# 持久化服务实现
class PersistenceService:
    async def save_task_metadata(self, task_id: str, metadata: dict):
        """保存任务元数据"""
        metadata_path = Path(f"output/{task_id}/metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    async def save_storyboard(self, task_id: str, storyboard: Storyboard):
        """保存故事板"""
        storyboard_path = Path(f"output/{task_id}/storyboard.json")
        
        # 序列化故事板
        storyboard_data = {
            "title": storyboard.title,
            "total_duration": storyboard.total_duration,
            "created_at": storyboard.created_at.isoformat() if storyboard.created_at else None,
            "completed_at": storyboard.completed_at.isoformat() if storyboard.completed_at else None,
            "config": asdict(storyboard.config),
            "frames": [asdict(frame) for frame in storyboard.frames],
            "content_metadata": storyboard.content_metadata
        }
        
        with open(storyboard_path, 'w', encoding='utf-8') as f:
            json.dump(storyboard_data, f, indent=2, ensure_ascii=False)
    
    async def load_task_metadata(self, task_id: str) -> dict:
        """加载任务元数据"""
        metadata_path = Path(f"output/{task_id}/metadata.json")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found for task {task_id}")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
```

---

## 🛠️ 技术栈详解

### AI模型支持

#### 大语言模型 (LLM)
**文件**: `pixelle_video/llm_presets.py`
- **GPT系列**: GPT-4o, GPT-4o-mini
- **国产模型**: 通义千问, DeepSeek, 智谱GLM
- **开源模型**: Ollama (本地部署)

#### 图像生成模型
**工作流**: `workflows/*/image_*.json`
- **FLUX**: 高质量图像生成
- **SDXL**: Stable Diffusion XL
- **通义万相**: 阿里云图像生成
- **Qwen**: 通义千问图像模型

#### 视频生成模型  
**工作流**: `workflows/*/video_*.json`
- **WAN 2.1**: 文本到视频生成
- **WAN 2.2**: 升级版视频生成
- **FusionX**: 融合增强模型

#### 语音合成 (TTS)
**工作流**: `workflows/*/tts_*.json`
- **Edge-TTS**: 微软免费TTS
- **Index-TTS**: 支持声音克隆
- **Spark TTS**: 讯飞语音合成

### 核心服务

#### LLM服务 (`services/llm_service.py`)
- 统一的LLM调用接口
- 支持多种API格式
- 自动重试和错误处理

#### TTS服务 (`services/tts_service.py`)
- 本地TTS和ComfyUI工作流
- 声音克隆支持
- 音频格式转换

#### 媒体服务 (`services/media.py`)
- 图像和视频生成
- ComfyUI工作流执行
- RunningHub云端调用

#### 视频服务 (`services/video.py`)
- FFmpeg视频处理
- 音视频合成
- 格式转换和压缩

---

## ⚙️ 工作流配置

### 目录结构
```
workflows/
├── selfhost/          # 本地部署工作流
│   ├── image_flux.json      # FLUX图像生成
│   ├── video_wan2.1.json    # WAN 2.1视频生成
│   └── tts_edge.json        # Edge-TTS语音合成
└── runninghub/        # 云端部署工作流
    ├── image_flux.json      # 云端FLUX
    ├── video_wan2.2.json    # 云端WAN 2.2
    └── tts_spark.json       # 云端讯飞TTS
```

### 工作流格式
基于ComfyUI的JSON配置文件:

```json
{
  "节点ID": {
    "inputs": {
      "参数名": "参数值",
      "连接": ["源节点ID", 输出索引]
    },
    "class_type": "节点类型",
    "_meta": {"title": "节点标题"}
  }
}
```

### 参数替换
工作流支持动态参数替换:
- `$prompt.value`: 图像提示词
- `$width.value`: 图像宽度  
- `$height.value`: 图像高度
- `$duration.value`: 视频时长

---

## 🚀 部署方案

### 方案对比

| 方案 | 成本 | 速度 | 质量 | 技术要求 |
|------|------|------|------|----------|
| 完全免费 | 0元 | 慢 | 中等 | 高 |
| 推荐方案 | 极低 | 快 | 高 | 中等 |
| 云端方案 | 较高 | 最快 | 最高 | 低 |

### 1. 完全免费方案
**配置**: Ollama (本地LLM) + ComfyUI (本地部署)
- **LLM**: Ollama运行Qwen等开源模型
- **图像**: 本地ComfyUI + FLUX/SDXL
- **TTS**: Edge-TTS (免费)
- **成本**: 0元
- **要求**: 显卡8GB+ VRAM

### 2. 推荐方案 ⭐
**配置**: 通义千问 (云端LLM) + ComfyUI (本地部署)
- **LLM**: 通义千问API (极低成本)
- **图像**: 本地ComfyUI + FLUX
- **TTS**: Edge-TTS或本地Index-TTS
- **成本**: 每个视频约0.1-0.5元
- **要求**: 显卡4GB+ VRAM

### 3. 云端方案
**配置**: OpenAI (云端LLM) + RunningHub (云端媒体)
- **LLM**: GPT-4o API
- **图像**: RunningHub FLUX/SDXL
- **TTS**: RunningHub Spark TTS
- **成本**: 每个视频约2-5元
- **要求**: 仅需网络连接

---

## 💡 使用建议

### 新手入门
1. **使用Windows一键包**: 免安装，开箱即用
2. **选择推荐方案**: 通义千问 + 本地ComfyUI
3. **从默认模板开始**: 使用内置模板和工作流
4. **逐步自定义**: 熟悉后再调整参数

### 进阶用户
1. **本地部署ComfyUI**: 更好的控制和自定义
2. **自定义工作流**: 根据需求调整AI模型
3. **批量生成**: 利用并行处理提升效率
4. **模板定制**: 创建专属视频风格

### 专业用户
1. **混合部署**: 关键步骤本地，辅助步骤云端
2. **工作流优化**: 深度定制ComfyUI节点
3. **API集成**: 集成到现有工作流
4. **性能调优**: 优化并发和缓存策略

### 成本优化建议
1. **静态模板**: 纯文字视频，跳过媒体生成
2. **批量处理**: 一次生成多个视频分摊成本
3. **本地缓存**: 复用相似的图像和音频
4. **参数调优**: 平衡质量和成本

---

## 🔧 故障排除

### 常见问题

#### 1. 环境问题
- **Python版本**: 需要3.11+
- **依赖安装**: 使用`uv`管理依赖
- **FFmpeg**: 视频处理必需

#### 2. ComfyUI连接
- **检查服务**: `http://127.0.0.1:8188`
- **工作流兼容**: 确保节点版本匹配
- **显存不足**: 降低分辨率或批次大小

#### 3. API配置
- **密钥有效性**: 检查API Key是否正确
- **网络连接**: 确保能访问API服务
- **配额限制**: 检查API使用额度

#### 4. 生成质量
- **提示词优化**: 调整prompt_prefix
- **模型选择**: 尝试不同的AI模型
- **参数调整**: 修改steps、cfg等参数

---

## 📚 扩展开发

### 自定义管道
继承`LinearVideoPipeline`创建自定义流程:

```python
# custom_pipeline.py
from pixelle_video.pipelines.linear import LinearVideoPipeline, PipelineContext
from pixelle_video.models.storyboard import VideoGenerationResult

class CustomPipeline(LinearVideoPipeline):
    """自定义视频生成管道"""
    
    async def plan_visuals(self, ctx: PipelineContext):
        """自定义视觉规划逻辑"""
        # 例: 使用固定的图像风格
        ctx.image_prompts = []
        for narration in ctx.narrations:
            # 自定义提示词生成逻辑
            custom_prompt = f"anime style, {narration}, high quality, detailed"
            ctx.image_prompts.append(custom_prompt)
        
        logger.info(f"✅ Generated {len(ctx.image_prompts)} custom prompts")
    
    async def produce_assets(self, ctx: PipelineContext):
        """自定义资产生产流程"""
        # 例: 添加自定义处理步骤
        await super().produce_assets(ctx)
        
        # 后处理: 为每个帧添加水印
        for frame in ctx.storyboard.frames:
            if frame.composed_image_path:
                await self._add_watermark(frame.composed_image_path)
    
    async def _add_watermark(self, image_path: str):
        """添加水印"""
        from PIL import Image, ImageDraw, ImageFont
        
        with Image.open(image_path) as img:
            draw = ImageDraw.Draw(img)
            
            # 添加水印文字
            font = ImageFont.load_default()
            watermark_text = "Generated by Pixelle-Video"
            
            # 计算位置（右下角）
            bbox = draw.textbbox((0, 0), watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            x = img.width - text_width - 10
            y = img.height - text_height - 10
            
            # 绘制水印
            draw.text((x, y), watermark_text, fill=(255, 255, 255, 128), font=font)
            
            # 保存
            img.save(image_path)

# 使用自定义管道
async def generate_custom_video(topic: str):
    from pixelle_video.service import PixelleVideoCore
    
    core = PixelleVideoCore()
    pipeline = CustomPipeline(core)
    
    result = await pipeline(
        text=topic,
        mode="generate",
        n_scenes=5,
        frame_template="1080x1920/anime.html"  # 自定义模板
    )
    
    return result
```

### 自定义服务
实现新的AI服务:

```python
# custom_tts_service.py
from pixelle_video.services.tts_service import TTSService

class CustomTTSService(TTSService):
    """自定义TTS服务"""
    
    async def _generate_custom_tts(
        self,
        text: str,
        voice: str = "custom_voice",
        output_path: str = None,
        **kwargs
    ) -> str:
        """自定义TTS实现"""
        
        # 例: 调用第三方TTS API
        import httpx
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://api.custom-tts.com/synthesize",
                json={
                    "text": text,
                    "voice": voice,
                    "format": "mp3"
                },
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
            
            response.raise_for_status()
            
            # 保存音频文件
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            return output_path
    
    async def __call__(self, **kwargs):
        """重写调用方法以支持自定义TTS"""
        inference_mode = kwargs.get("inference_mode", "local")
        
        if inference_mode == "custom":
            return await self._generate_custom_tts(**kwargs)
        else:
            return await super().__call__(**kwargs)

# 注册自定义服务
from pixelle_video.service import PixelleVideoCore

class CustomPixelleVideoCore(PixelleVideoCore):
    def __init__(self, config_path: str = None):
        super().__init__(config_path)
        # 替换TTS服务
        self.tts = CustomTTSService(self.config)
```

### 自定义模板
创建HTML模板:

```html
<!-- templates/custom/my_template.html -->
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <style>
        body {
            margin: 0;
            padding: 0;
            width: 1080px;
            height: 1920px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            font-family: 'Arial', sans-serif;
            color: white;
        }
        
        .title {
            font-size: 48px;
            font-weight: bold;
            text-align: center;
            margin-bottom: 40px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }
        
        .content-area {
            width: 90%;
            height: 60%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .media-container {
            width: 100%;
            height: 70%;
            border-radius: 20px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-bottom: 30px;
        }
        
        .media-container img,
        .media-container video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .text-overlay {
            background: rgba(0,0,0,0.7);
            padding: 20px 30px;
            border-radius: 15px;
            font-size: 32px;
            text-align: center;
            line-height: 1.4;
            max-width: 90%;
            backdrop-filter: blur(10px);
        }
        
        .watermark {
            position: absolute;
            bottom: 20px;
            right: 20px;
            font-size: 16px;
            opacity: 0.7;
        }
    </style>
</head>
<body>
    <div class="title">{{title}}</div>
    
    <div class="content-area">
        <div class="media-container">
            {% if image %}
                <img src="{{image}}" alt="Generated content">
            {% else %}
                <div style="background: rgba(255,255,255,0.1); display: flex; align-items: center; justify-content: center; font-size: 24px;">
                    No Media
                </div>
            {% endif %}
        </div>
        
        <div class="text-overlay">
            {{text}}
        </div>
    </div>
    
    <div class="watermark">
        Frame {{ext.index}} | Pixelle-Video
    </div>
</body>
</html>
```

### 自定义工作流
创建ComfyUI工作流:

```json
{
  "workflow_name": "custom_image_generation",
  "description": "自定义图像生成工作流",
  "nodes": {
    "1": {
      "inputs": {
        "text": "$prompt.value",
        "width": "$width.value",
        "height": "$height.value"
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "Prompt Input"
      }
    },
    "2": {
      "inputs": {
        "seed": 42,
        "steps": 20,
        "cfg": 7.5,
        "sampler_name": "euler",
        "scheduler": "normal",
        "denoise": 1,
        "model": ["3", 0],
        "positive": ["1", 0],
        "negative": ["4", 0],
        "latent_image": ["5", 0]
      },
      "class_type": "KSampler",
      "_meta": {
        "title": "Sampler"
      }
    },
    "3": {
      "inputs": {
        "ckpt_name": "sd_xl_base_1.0.safetensors"
      },
      "class_type": "CheckpointLoaderSimple",
      "_meta": {
        "title": "Load Checkpoint"
      }
    },
    "4": {
      "inputs": {
        "text": "low quality, blurry, distorted",
        "clip": ["3", 1]
      },
      "class_type": "CLIPTextEncode",
      "_meta": {
        "title": "Negative Prompt"
      }
    },
    "5": {
      "inputs": {
        "width": "$width.value",
        "height": "$height.value",
        "batch_size": 1
      },
      "class_type": "EmptyLatentImage",
      "_meta": {
        "title": "Empty Latent"
      }
    },
    "6": {
      "inputs": {
        "samples": ["2", 0],
        "vae": ["3", 2]
      },
      "class_type": "VAEDecode",
      "_meta": {
        "title": "VAE Decode"
      }
    },
    "7": {
      "inputs": {
        "filename_prefix": "custom_output",
        "images": ["6", 0]
      },
      "class_type": "SaveImage",
      "_meta": {
        "title": "Save Image"
      }
    }
  }
}
```

### 完整示例: 自定义视频生成器

```python
# complete_custom_example.py
import asyncio
from pathlib import Path
from pixelle_video.service import PixelleVideoCore
from pixelle_video.pipelines.linear import LinearVideoPipeline, PipelineContext
from pixelle_video.models.storyboard import VideoGenerationResult

class MovieTrailerPipeline(LinearVideoPipeline):
    """电影预告片风格的视频生成管道"""
    
    async def generate_content(self, ctx: PipelineContext):
        """生成电影预告片风格的分镜"""
        topic = ctx.input_text
        
        # 预告片固定结构
        trailer_structure = [
            f"在一个充满{topic}的世界里...",
            f"一个关于{topic}的故事即将展开",
            f"当{topic}遇到前所未有的挑战",
            f"英雄必须面对{topic}的考验",
            f"这将是一场关于{topic}的史诗冒险"
        ]
        
        ctx.narrations = trailer_structure
        logger.info(f"✅ Generated movie trailer structure with {len(ctx.narrations)} scenes")
    
    async def plan_visuals(self, ctx: PipelineContext):
        """生成电影风格的视觉提示词"""
        cinematic_prompts = []
        
        for i, narration in enumerate(ctx.narrations):
            if i == 0:
                # 开场: 宽阔的景观镜头
                prompt = f"cinematic wide shot, epic landscape, {narration}, dramatic lighting, film grain"
            elif i == len(ctx.narrations) - 1:
                # 结尾: 动作镜头
                prompt = f"dynamic action shot, {narration}, intense lighting, motion blur, cinematic"
            else:
                # 中间: 角色特写
                prompt = f"cinematic close-up, character portrait, {narration}, dramatic shadows, film noir style"
            
            cinematic_prompts.append(prompt)
        
        ctx.image_prompts = cinematic_prompts
        logger.info(f"✅ Generated {len(ctx.image_prompts)} cinematic prompts")

async def main():
    """主函数示例"""
    # 初始化核心服务
    core = PixelleVideoCore()
    
    # 创建自定义管道
    pipeline = MovieTrailerPipeline(core)
    
    # 生成视频
    result = await pipeline(
        text="人工智能",
        mode="generate",
        frame_template="1920x1080/movie_trailer.html",
        media_workflow="image_flux.json",
        tts_inference_mode="local",
        voice_id="zh-CN-YunjianNeural",
        tts_speed=1.0,
        bgm_path="bgm/epic_trailer.mp3",
        bgm_volume=0.3
    )
    
    print(f"🎬 视频生成完成!")
    print(f"   路径: {result.video_path}")
    print(f"   时长: {result.duration:.2f}秒")
    print(f"   大小: {result.size_mb:.2f}MB")
    print(f"   帧数: {result.frames_count}")

if __name__ == "__main__":
    asyncio.run(main())
```

### API调用示例

```python
# api_usage_example.py
import asyncio
from pixelle_video.service import PixelleVideoCore

async def simple_video_generation():
    """简单的视频生成示例"""
    
    # 初始化服务
    core = PixelleVideoCore()
    
    # 使用标准管道生成视频
    from pixelle_video.pipelines.standard import StandardPipeline
    pipeline = StandardPipeline(core)
    
    # 定义进度回调
    def progress_callback(event):
        print(f"Progress: {event.progress:.1%} - {event.event_type}")
        if hasattr(event, 'frame_current') and event.frame_current:
            print(f"  Frame {event.frame_current}/{event.frame_total}")
    
    # 生成视频
    result = await pipeline(
        text="如何学习人工智能",
        mode="generate",
        n_scenes=3,
        frame_template="1080x1920/default.html",
        tts_inference_mode="local",
        voice_id="zh-CN-YunjianNeural",
        progress_callback=progress_callback
    )
    
    return result

async def batch_video_generation():
    """批量视频生成示例"""
    
    topics = [
        "人工智能的发展历程",
        "机器学习基础知识", 
        "深度学习应用案例"
    ]
    
    core = PixelleVideoCore()
    pipeline = StandardPipeline(core)
    
    results = []
    for i, topic in enumerate(topics):
        print(f"Generating video {i+1}/{len(topics)}: {topic}")
        
        result = await pipeline(
            text=topic,
            mode="generate",
            n_scenes=4,
            output_path=f"output/batch_video_{i+1}.mp4"
        )
        
        results.append(result)
        print(f"✅ Video {i+1} completed: {result.video_path}")
    
    return results

# 运行示例
if __name__ == "__main__":
    # 单个视频生成
    result = asyncio.run(simple_video_generation())
    print(f"Generated: {result.video_path}")
    
    # 批量视频生成
    # results = asyncio.run(batch_video_generation())
    # print(f"Generated {len(results)} videos")
```

---

## 📖 参考资源

- **项目主页**: https://github.com/AIDC-AI/Pixelle-Video
- **使用文档**: https://aidc-ai.github.io/Pixelle-Video/zh
- **ComfyUI**: https://github.com/comfyanonymous/ComfyUI
- **视频教程**: https://www.bilibili.com/video/BV1WzyGBnEVp

---

*本文档基于Pixelle-Video v0.1.11版本编写，如有更新请参考最新版本。*