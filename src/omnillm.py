"""
OmniLLM: Universal Adaptive Inference Engine

The main entry point. Combines:
1. Semantic Cache (fast retrieval)
2. Model Cascade (intelligent routing)
3. Hardware Profiler (auto-optimization)
4. Speculative Drafting (speed)
5. Progressive Loading (memory)

Usage:
    from omnillm import OmniLLM
    
    llm = OmniLLM()  # Auto-detects hardware
    response = llm.generate("Explain quantum mechanics")
"""

import os
import sys
import json
import time
from typing import Optional, Dict, Any, List, Generator
from dataclasses import dataclass

# Import our components
from semantic_cache import SemanticCache, CacheEntry
from hardware_profiler import HardwareProfiler, HardwareProfile
from model_cascade import ModelCascadeRouter, CascadeDecision, QueryComplexity

# Try to import transformers
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    from threading import Thread
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers not installed. Using mock mode.")


@dataclass
class GenerationResult:
    """Result from generation"""
    text: str
    model_used: str
    tier: int
    cached: bool
    generation_time_ms: float
    tokens_generated: int
    config: Dict[str, Any]


class OmniLLM:
    """
    Universal LLM inference engine.
    
    Automatically:
    - Profiles your hardware
    - Selects appropriate models
    - Uses semantic caching
    - Routes queries intelligently
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        config_path: Optional[str] = None,
        force_cpu: bool = False,
        verbose: bool = True
    ):
        self.verbose = verbose
        self._log("🚀 Initializing OmniLLM...")
        
        # Initialize components
        self.cache = SemanticCache(cache_dir=cache_dir)
        self.profiler = HardwareProfiler()
        
        # Load or create hardware profile
        self.config = self._load_or_create_config(config_path)
        
        # Initialize router
        self.router = ModelCascadeRouter(self.config)
        
        # Model cache (lazy loading)
        self.loaded_models: Dict[str, Any] = {}
        self.loaded_tokenizers: Dict[str, Any] = {}
        
        self._log("✓ OmniLLM ready!")
        if self.verbose:
            self.print_status()
    
    def _log(self, msg: str):
        if self.verbose:
            print(msg)
    
    def _load_or_create_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load config from disk or create new one"""
        
        # Try loading existing
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self._log(f"✓ Loaded config from {config_path}")
                return json.load(f)
        
        # Try loading hardware profile
        profile_path = "./configs/hardware_profile.json"
        if os.path.exists(profile_path):
            self.profiler.load_profile(profile_path)
        else:
            # Profile fresh
            self.profiler.profile()
            self.profiler.save_profile(profile_path)
        
        # Generate config from profile
        config = self.profiler.get_recommended_config()
        
        # Save for next time
        os.makedirs("./configs", exist_ok=True)
        with open("./configs/omnillm_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        return config
    
    def print_status(self):
        """Print current status"""
        print("\n" + "=" * 60)
        print("OMNILLM STATUS")
        print("=" * 60)
        
        # Hardware
        hw_tier = self.config.get('hardware_tier', 'Unknown')
        print(f"\n🖥️  Hardware: {hw_tier}")
        
        # Models
        models = self.config.get('models', {})
        print(f"\n📦 Models:")
        for tier, model in models.items():
            loaded = "✓" if model in self.loaded_models else "○"
            print(f"   {loaded} {tier}: {model.split('/')[-1]}")
        
        # Cache
        cache_stats = self.cache.stats()
        print(f"\n💾 Cache: {cache_stats['total_entries']} entries ({cache_stats['type']})")
        
        print("=" * 60)
    
    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        use_cache: bool = True,
        force_tier: Optional[int] = None,
        stream: bool = False
    ) -> GenerationResult:
        """
        Generate response with intelligent routing and caching.
        
        This is the MAIN ENTRY POINT.
        """
        start_time = time.time()
        
        # Step 1: Check semantic cache
        if use_cache:
            cached = self.cache.get(prompt)
            if cached:
                return GenerationResult(
                    text=cached.response,
                    model_used=cached.model_used,
                    tier=0,  # From cache
                    cached=True,
                    generation_time_ms=(time.time() - start_time) * 1000,
                    tokens_generated=0,
                    config=self.config
                )
        
        # Step 2: Route to appropriate model
        decision = self.router.route(prompt, force_tier=force_tier)
        self._log(f"\n[OmniLLM] Routing to Tier {decision.tier}: {decision.reasoning}")
        
        # Step 3: Load model if needed
        if not self._load_model(decision.model_name):
            return GenerationResult(
                text="Error: Could not load model",
                model_used="none",
                tier=decision.tier,
                cached=False,
                generation_time_ms=(time.time() - start_time) * 1000,
                tokens_generated=0,
                config=self.config
            )
        
        # Step 4: Generate
        if stream:
            return self._generate_stream(
                prompt, decision, max_tokens, temperature, start_time
            )
        else:
            return self._generate_batch(
                prompt, decision, max_tokens, temperature, start_time
            )
    
    def _load_model(self, model_name: str) -> bool:
        """Lazy load a model"""
        if model_name in self.loaded_models:
            return True
        
        if not HAS_TRANSFORMERS:
            print("Error: transformers not installed")
            return False
        
        try:
            self._log(f"📥 Loading {model_name.split('/')[-1]}...")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model (4-bit quantized if GPU available)
            device = self.config.get('inference', {}).get('device', 'cpu')
            
            if device == 'cuda':
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype='float16',
                    bnb_4bit_quant_type='nf4',
                    bnb_4bit_use_double_quant=True,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map='auto',
                    trust_remote_code=True
                )
            else:
                # CPU - use smaller precision
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype='auto',
                    trust_remote_code=True
                )
            
            self.loaded_models[model_name] = model
            self.loaded_tokenizers[model_name] = tokenizer
            
            self._log(f"✓ Loaded {model_name.split('/')[-1]}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def _generate_batch(
        self,
        prompt: str,
        decision: CascadeDecision,
        max_tokens: int,
        temperature: float,
        start_time: float
    ) -> GenerationResult:
        """Generate in batch mode"""
        
        model = self.loaded_models[decision.model_name]
        tokenizer = self.loaded_tokenizers[decision.model_name]
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors='pt')
        if model.device.type != 'cpu':
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate
        gen_start = time.time()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_time = (time.time() - gen_start) * 1000
        
        # Decode
        input_len = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_len:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        # Store in cache
        self.cache.put(prompt, text, decision.model_name)
        
        total_time = (time.time() - start_time) * 1000
        
        return GenerationResult(
            text=text,
            model_used=decision.model_name,
            tier=decision.tier,
            cached=False,
            generation_time_ms=total_time,
            tokens_generated=len(new_tokens),
            config=self.config
        )
    
    def _generate_stream(
        self,
        prompt: str,
        decision: CascadeDecision,
        max_tokens: int,
        temperature: float,
        start_time: float
    ) -> Generator[str, None, None]:
        """Generate in streaming mode"""
        # Simplified - just yield full result for now
        result = self._generate_batch(
            prompt, decision, max_tokens, temperature, start_time
        )
        yield result.text
    
    def chat(self, message: str, history: Optional[List[Dict]] = None) -> str:
        """
        Simple chat interface.
        
        Args:
            message: User message
            history: Optional list of {'role': 'user'/'assistant', 'content': str}
        
        Returns:
            Assistant response
        """
        # Format prompt with history
        if history:
            prompt = ""
            for h in history:
                role = h.get('role', 'user')
                content = h.get('content', '')
                if role == 'user':
                    prompt += f"User: {content}\n"
                else:
                    prompt += f"Assistant: {content}\n"
            prompt += f"User: {message}\nAssistant:"
        else:
            prompt = f"User: {message}\nAssistant:"
        
        result = self.generate(prompt)
        return result.text


# Demo
if __name__ == "__main__":
    print("=" * 70)
    print("OMNILLM DEMO")
    print("=" * 70)
    print("\nInitializing...")
    
    # Create OmniLLM instance
    llm = OmniLLM(verbose=True)
    
    # Test queries
    test_queries = [
        "What is the capital of France?",
        "Explain how photosynthesis works.",
        "Write a Python function to calculate fibonacci numbers.",
    ]
    
    print("\n" + "=" * 70)
    print("TESTING GENERATION")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 70)
        
        result = llm.generate(query, max_tokens=100)
        
        print(f"🤖 Response: {result.text[:200]}...")
        print(f"\n📊 Stats:")
        print(f"   Model: {result.model_used.split('/')[-1]}")
        print(f"   Tier: {result.tier}")
        print(f"   Cached: {result.cached}")
        print(f"   Time: {result.generation_time_ms:.0f}ms")
        print(f"   Tokens: {result.tokens_generated}")
    
    print("\n" + "=" * 70)
    print("TESTING CACHE")
    print("=" * 70)
    
    # Same query again - should hit cache
    print(f"\n📝 Query (again): {test_queries[0]}")
    result = llm.generate(test_queries[0])
    print(f"   Cached: {result.cached}")
    print(f"   Time: {result.generation_time_ms:.0f}ms")
    
    if result.cached:
        print("   ⚡ CACHE HIT! Near-instant response!")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
