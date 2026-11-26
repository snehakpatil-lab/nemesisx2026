import platform
import psutil
from pathlib import Path
from importlib.metadata import version
from . import pipeline
from .config import ALLOWED, CONF, IMGSZ, MAX_ANALYSIS_FRAMES, MODEL, STRIDE

LIVE = pipeline.LIVE

def snapshot():
    vm = psutil.virtual_memory()
    device = pipeline.device()
    cuda_available = device != "cpu"
    processing_fps = LIVE["fps"]
    d = {
        "cpu_percent": psutil.cpu_percent(interval=0.1),
        "ram_used_gb": round(vm.used / 1e9, 2),
        "ram_total_gb": round(vm.total / 1e9, 2),
        "device": f"cuda:{device}" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "processing_fps": processing_fps,
        "processing_state": "running" if processing_fps is not None else "starting" if LIVE["video_id"] is not None else "idle",
        "active_video_id": LIVE["video_id"],
        "analysis_stride": LIVE["stride"],
        "gpu": None,
        "gpu_telemetry_status": "cpu_only" if not cuda_available else "checking",
    }
    if cuda_available:
        try:
            import pynvml
        except ImportError as exc:
            d["gpu_telemetry_status"] = "unavailable"
            d["gpu_telemetry_error"] = str(exc)
        else:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(device)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                name = pynvml.nvmlDeviceGetName(handle)
                d["gpu"] = {
                    "name": name.decode() if isinstance(name, bytes) else name,
                    "util_percent": utilization.gpu,
                    "vram_used_gb": round(memory.used / 1e9, 2),
                    "vram_total_gb": round(memory.total / 1e9, 2),
                }
                d["gpu_telemetry_status"] = "available"
            except pynvml.NVMLError as exc:
                d["gpu_telemetry_status"] = "unavailable"
                d["gpu_telemetry_error"] = str(exc)
    return d

def configuration():
    import torch

    device = pipeline.device()
    cuda_available = device != "cpu"
    model_path = Path(MODEL).expanduser()
    return {
        "device": {
            "selected": f"cuda:{device}" if cuda_available else "cpu",
            "name": torch.cuda.get_device_name(device) if cuda_available else "CPU",
            "cuda_available": cuda_available,
            "status": "CUDA inference enabled" if cuda_available else "PyTorch CUDA is unavailable; inference runs on CPU.",
            "logical_cpu_cores": psutil.cpu_count(logical=True),
        },
        "detection_model": {
            "framework": "Ultralytics YOLO",
            "model": MODEL,
            "weights_file_available": model_path.is_file(),
            "task": "Object detection; detections with ByteTrack IDs are stored.",
        },
        "tracking_model": {
            "name": "ByteTrack",
            "config": "bytetrack.yaml",
            "status": "Configured by the video analysis pipeline.",
            "cross_video_reidentification": False,
        },
        "pipeline": {
            "confidence_threshold": CONF,
            "minimum_frame_stride": STRIDE,
            "max_analysis_frames": MAX_ANALYSIS_FRAMES,
            "image_size": IMGSZ,
            "supported_extensions": sorted(ALLOWED),
        },
        "runtime": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": version("ultralytics"),
            "scikit_learn": version("scikit-learn"),
        },
        "processing": {
            "active_video_id": pipeline.LIVE["video_id"],
            "inference_fps": pipeline.LIVE["fps"],
            "effective_frame_stride": pipeline.LIVE["stride"],
            "sampled_frame_count": pipeline.LIVE["total_frames"],
        },
    }
