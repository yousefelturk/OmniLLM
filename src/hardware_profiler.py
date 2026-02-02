"""
Hardware-Aware Auto-Tuner (HAAT) - Component 5

Profiles the user's hardware ONCE and auto-configures everything.
This is NOVEL because:
- One-time profiling vs manual configuration
- Automatically selects optimal quantization, batch size, models
- Adapts to YOUR specific hardware
"""

import torch
import psutil
import platform
import json
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
import time


@dataclass
class HardwareProfile:
    """Complete hardware capability profile"""
    # CPU
    cpu_count: int
    cpu_freq_mhz: float
    cpu_brand: str
    ram_gb: float
    
    # GPU
    has_cuda: bool
    has_mps: bool  # Apple Silicon
    gpu_name: Optional[str]
    gpu_vram_gb: Optional[float]
    gpu_compute_capability: Optional[float]
    
    # Disk
    disk_free_gb: float
    
    # Derived capabilities
    recommended_quantization: str  # Q4, Q3, Q8
    recommended_max_model_size: int  # in billions
    recommended_batch_size: int
    can_run_70b: bool
    can_run_7b: bool
    can_run_1b: bool
    
    # Performance estimates
    est_tokens_per_sec_1b: float
    est_tokens_per_sec_7b: float
    est_tokens_per_sec_70b: Optional[float]


class HardwareProfiler:
    """
    Profiles hardware once and generates optimal configuration.
    
    Usage:
        profiler = HardwareProfiler()
        profile = profiler.profile()
        config = profiler.get_recommended_config()
    """
    
    def __init__(self):
        self.profile: Optional[HardwareProfile] = None
        
    def profile(self) -> HardwareProfile:
        """
        Perform complete hardware profiling.
        Takes ~1-2 seconds, should be done once at startup.
        """
        print("🔍 Profiling hardware...")
        start_time = time.time()
        
        # CPU Info
        cpu_count = psutil.cpu_count(logical=True)
        cpu_freq = psutil.cpu_freq()
        cpu_freq_mhz = cpu_freq.max if cpu_freq else 0
        cpu_brand = platform.processor() or "Unknown"
        ram_gb = psutil.virtual_memory().total / (1024**3)
        
        # GPU Info
        has_cuda = torch.cuda.is_available()
        has_mps = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        
        gpu_name = None
        gpu_vram_gb = None
        gpu_compute_capability = None
        
        if has_cuda:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_compute_capability = torch.cuda.get_device_capability(0)[0] + \
                                    torch.cuda.get_device_capability(0)[1] / 10
        elif has_mps:
            gpu_name = "Apple Silicon MPS"
            # Can't easily get unified memory, estimate from system RAM
            gpu_vram_gb = ram_gb
            gpu_compute_capability = None
        
        # Disk
        disk_free_gb = psutil.disk_usage('/').free / (1024**3)
        
        # Calculate recommendations
        profile = self._calculate_capabilities(
            cpu_count, cpu_freq_mhz, ram_gb,
            has_cuda, has_mps, gpu_name, gpu_vram_gb
        )
        
        self.profile = profile
        
        elapsed = time.time() - start_time
        print(f"✓ Profiling complete ({elapsed:.2f}s)")
        
        return profile
    
    def _calculate_capabilities(
        self,
        cpu_count: int,
        cpu_freq_mhz: float,
        ram_gb: float,
        has_cuda: bool,
        has_mps: bool,
        gpu_name: Optional[str],
        gpu_vram_gb: Optional[float]
    ) -> HardwareProfile:
        """Calculate what this hardware can actually do"""
        
        # Determine GPU VRAM (use system RAM as proxy if no GPU)
        effective_vram = gpu_vram_gb if gpu_vram_gb else ram_gb * 0.5
        
        # Determine quantization based on VRAM
        if effective_vram >= 20:
            recommended_quantization = "Q4"  # Good quality, enough VRAM
            recommended_max_model_size = 70
            can_run_70b = True
            can_run_7b = True
            can_run_1b = True
        elif effective_vram >= 10:
            recommended_quantization = "Q4"
            recommended_max_model_size = 13
            can_run_70b = False
            can_run_7b = True
            can_run_1b = True
        elif effective_vram >= 6:
            recommended_quantization = "Q4"
            recommended_max_model_size = 7
            can_run_70b = False
            can_run_7b = True
            can_run_1b = True
        elif effective_vram >= 4:
            recommended_quantization = "Q3"  # Aggressive compression
            recommended_max_model_size = 7
            can_run_70b = False
            can_run_7b = True  # Might be slow
            can_run_1b = True
        else:
            recommended_quantization = "Q2"  # Extreme compression
            recommended_max_model_size = 3
            can_run_70b = False
            can_run_7b = False
            can_run_1b = True
        
        # Batch size based on VRAM
        if effective_vram >= 20:
            recommended_batch_size = 4
        elif effective_vram >= 10:
            recommended_batch_size = 2
        else:
            recommended_batch_size = 1
        
        # Performance estimates (rough)
        if has_cuda:
            est_1b = 50.0  # tokens/sec
            est_7b = 20.0
            est_70b = 5.0 if can_run_70b else None
        elif has_mps:
            est_1b = 30.0
            est_7b = 10.0
            est_70b = None
        else:  # CPU only
            est_1b = 5.0
            est_7b = 1.0
            est_70b = None
        
        return HardwareProfile(
            cpu_count=cpu_count,
            cpu_freq_mhz=cpu_freq_mhz,
            cpu_brand=platform.processor() or "Unknown",
            ram_gb=ram_gb,
            has_cuda=has_cuda,
            has_mps=has_mps,
            gpu_name=gpu_name,
            gpu_vram_gb=gpu_vram_gb,
            gpu_compute_capability=gpu_compute_capability,
            disk_free_gb=disk_free_gb,
            recommended_quantization=recommended_quantization,
            recommended_max_model_size=recommended_max_model_size,
            recommended_batch_size=recommended_batch_size,
            can_run_70b=can_run_70b,
            can_run_7b=can_run_7b,
            can_run_1b=can_run_1b,
            est_tokens_per_sec_1b=est_1b,
            est_tokens_per_sec_7b=est_7b,
            est_tokens_per_sec_70b=est_70b
        )
    
    def get_recommended_config(self) -> Dict[str, Any]:
        """Get auto-generated configuration for OmniLLM"""
        if self.profile is None:
            self.profile()
        
        p = self.profile
        
        # Map to actual models
        if p.can_run_70b:
            tier3_model = "meta-llama/Llama-2-70b-chat-hf"
        elif p.can_run_7b:
            tier3_model = "mistralai/Mistral-7B-Instruct-v0.2"
        else:
            tier3_model = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        return {
            "hardware_tier": self._get_tier_name(),
            "models": {
                "tier1": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "tier2": "microsoft/Phi-3-mini-4k-instruct" if p.can_run_7b else "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "tier3": tier3_model
            },
            "quantization": {
                "tier1": "Q4",
                "tier2": p.recommended_quantization,
                "tier3": p.recommended_quantization
            },
            "inference": {
                "batch_size": p.recommended_batch_size,
                "max_seq_length": 2048,
                "use_fp16": p.has_cuda or p.has_mps,
                "device": "cuda" if p.has_cuda else ("mps" if p.has_mps else "cpu")
            },
            "cache": {
                "similarity_threshold": 0.92,
                "max_entries": 10000
            },
            "performance": {
                "est_tok_per_sec_tier1": p.est_tokens_per_sec_1b,
                "est_tok_per_sec_tier2": p.est_tokens_per_sec_7b if p.can_run_7b else p.est_tokens_per_sec_1b,
                "est_tok_per_sec_tier3": p.est_tokens_per_sec_70b if p.can_run_70b else p.est_tokens_per_sec_7b
            }
        }
    
    def _get_tier_name(self) -> str:
        """Get human-readable tier name"""
        if self.profile is None:
            return "Unknown"
        
        p = self.profile
        if p.can_run_70b:
            return "High-End (70B capable)"
        elif p.can_run_7b and (p.has_cuda or p.has_mps):
            return "Mid-Range (7B GPU)"
        elif p.can_run_7b:
            return "Mid-Range (7B CPU)"
        else:
            return "Entry-Level (1B only)"
    
    def save_profile(self, path: str = "./configs/hardware_profile.json"):
        """Save profile to disk"""
        if self.profile is None:
            self.profile()
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self.profile), f, indent=2)
        print(f"✓ Profile saved to {path}")
    
    def load_profile(self, path: str = "./configs/hardware_profile.json") -> bool:
        """Load profile from disk"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            self.profile = HardwareProfile(**data)
            print(f"✓ Profile loaded from {path}")
            return True
        except FileNotFoundError:
            return False
    
    def print_summary(self):
        """Print nice summary of hardware"""
        if self.profile is None:
            self.profile()
        
        p = self.profile
        
        print("\n" + "=" * 60)
        print("HARDWARE PROFILE SUMMARY")
        print("=" * 60)
        
        print(f"\n🖥️  CPU: {p.cpu_brand[:50]}")
        print(f"   Cores: {p.cpu_count} | RAM: {p.ram_gb:.1f} GB")
        
        if p.has_cuda:
            print(f"\n🎮 GPU: {p.gpu_name}")
            print(f"   VRAM: {p.gpu_vram_gb:.1f} GB | CC: {p.gpu_compute_capability}")
        elif p.has_mps:
            print(f"\n🍎 Apple Silicon: {p.gpu_name}")
        else:
            print(f"\n⚠️  No GPU detected - CPU only mode")
        
        print(f"\n📊 Capabilities:")
        print(f"   Can run 1B models: {'✓' if p.can_run_1b else '✗'}")
        print(f"   Can run 7B models: {'✓' if p.can_run_7b else '✗'}")
        print(f"   Can run 70B models: {'✓' if p.can_run_70b else '✗'}")
        
        print(f"\n⚙️  Recommendations:")
        print(f"   Quantization: {p.recommended_quantization}")
        print(f"   Max model size: {p.recommended_max_model_size}B")
        print(f"   Hardware tier: {self._get_tier_name()}")
        
        print(f"\n🚀 Estimated Performance:")
        print(f"   1B model: ~{p.est_tokens_per_sec_1b:.0f} tok/s")
        if p.can_run_7b:
            print(f"   7B model: ~{p.est_tokens_per_sec_7b:.0f} tok/s")
        if p.can_run_70b:
            print(f"   70B model: ~{p.est_tokens_per_sec_70b:.0f} tok/s")
        
        print("=" * 60)


# Quick test
if __name__ == "__main__":
    profiler = HardwareProfiler()
    
    # Try to load existing profile
    if not profiler.load_profile():
        # Profile fresh
        profiler.profile()
        profiler.save_profile()
    
    profiler.print_summary()
    
    print("\n📋 Recommended Config:")
    config = profiler.get_recommended_config()
    print(json.dumps(config, indent=2))
