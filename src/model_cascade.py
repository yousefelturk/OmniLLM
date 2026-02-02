"""
Dynamic Model Cascading (DMC) - Component 2

Auto-selects model tier based on:
1. Query complexity (what's being asked)
2. Hardware capability (what we can run)
3. Context (conversation history)

This is NOVEL because:
- Routes to different model SIZES, not just layers
- Considers query semantics + hardware constraints
- Learns from user patterns
"""

import re
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json


class QueryComplexity(Enum):
    """Complexity levels for queries"""
    TRIVIAL = 1      # "Hi", "Thanks", "Yes"
    SIMPLE = 2       # Factual lookups, definitions
    MODERATE = 3     # Explanations, summaries
    COMPLEX = 4      # Reasoning, analysis
    EXPERT = 5       # Coding, math, deep reasoning


@dataclass
class CascadeDecision:
    """Decision from the cascade router"""
    tier: int  # 1, 2, or 3
    model_name: str
    reasoning: str
    confidence: float
    estimated_time_ms: int


class QueryComplexityClassifier:
    """
    Classifies query complexity using heuristics + ML.
    
    Rules-based for now, can be trained neural net later.
    """
    
    # Keywords indicating complexity
    COMPLEXITY_PATTERNS = {
        QueryComplexity.TRIVIAL: [
            r'^\s*(hi|hello|hey|thanks|thank you|yes|no|ok|okay|bye)\s*$',
            r'^\s*\W+$',  # Just punctuation
        ],
        QueryComplexity.SIMPLE: [
            r'what is (the )?(capital|population|president|ceo)',
            r'when (was|did) .*(born|founded|happen)',
            r'where is .*(located|from)',
            r'define',
            r'who (is|was)',
        ],
        QueryComplexity.MODERATE: [
            r'explain',
            r'how (do|does|can|should)',
            r'what (are|is) the (difference|similarities)',
            r'summarize',
            r'compare',
        ],
        QueryComplexity.COMPLEX: [
            r'why (do|does|is|are)',
            r'analyze',
            r'evaluate',
            r'pros and cons',
            r'advantages? and disadvantages',
        ],
        QueryComplexity.EXPERT: [
            r'write (code|a function|a script)',
            r'implement',
            r'solve.*equation',
            r'debug',
            r'optimize',
            r'proof',
            r'theorem',
        ]
    }
    
    # Length heuristics
    SHORT_QUERY_THRESHOLD = 10  # words
    LONG_QUERY_THRESHOLD = 50   # words
    
    def classify(self, query: str, conversation_history: Optional[List[str]] = None) -> QueryComplexity:
        """
        Classify query complexity.
        
        Returns complexity level from TRIVIAL to EXPERT.
        """
        query_lower = query.lower().strip()
        word_count = len(query.split())
        
        # Check patterns from lowest to highest complexity
        for complexity in QueryComplexity:
            patterns = self.COMPLEXITY_PATTERNS.get(complexity, [])
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return complexity
        
        # Length-based fallback
        if word_count <= 3:
            return QueryComplexity.TRIVIAL
        elif word_count <= self.SHORT_QUERY_THRESHOLD:
            return QueryComplexity.SIMPLE
        elif word_count <= self.LONG_QUERY_THRESHOLD:
            return QueryComplexity.MODERATE
        else:
            return QueryComplexity.COMPLEX
    
    def get_complexity_score(self, query: str) -> float:
        """Get numeric complexity score (0-1)"""
        complexity = self.classify(query)
        return complexity.value / 5.0  # Normalize to 0-1


class ModelCascadeRouter:
    """
    Routes queries to appropriate model tier based on:
    - Query complexity
    - Hardware capabilities
    - User preferences
    """
    
    def __init__(self, hardware_profile: Dict[str, Any], config: Optional[Dict] = None):
        self.hardware = hardware_profile
        self.config = config or {}
        self.classifier = QueryComplexityClassifier()
        
        # Model mappings based on hardware
        self.models = hardware_profile.get('models', {
            'tier1': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'tier2': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
            'tier3': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        })
        
        # Performance estimates
        self.perf = hardware_profile.get('performance', {
            'est_tok_per_sec_tier1': 50,
            'est_tok_per_sec_tier2': 20,
            'est_tok_per_sec_tier3': 5
        })
        
        # Routing thresholds (configurable)
        self.tier1_max_complexity = QueryComplexity.SIMPLE
        self.tier2_max_complexity = QueryComplexity.MODERATE
    
    def route(self, query: str, force_tier: Optional[int] = None) -> CascadeDecision:
        """
        Route query to appropriate model tier.
        
        Args:
            query: User query string
            force_tier: Optional override (1, 2, or 3)
        
        Returns:
            CascadeDecision with routing info
        """
        if force_tier:
            return self._create_decision(force_tier, "Forced by user")
        
        # Classify query
        complexity = self.classifier.classify(query)
        complexity_score = complexity.value
        
        # Route based on complexity + hardware
        # Tier 1: Trivial and Simple queries (if hardware supports)
        if complexity_score <= self.tier1_max_complexity.value:
            if self.hardware.get('can_run_1b', True):
                return self._create_decision(
                    tier=1,
                    reasoning=f"Query complexity: {complexity.name} (suitable for fast model)"
                )
        
        # Tier 2: Moderate complexity (if hardware supports 7B)
        if complexity_score <= self.tier2_max_complexity.value:
            if self.hardware.get('can_run_7b', False):
                return self._create_decision(
                    tier=2,
                    reasoning=f"Query complexity: {complexity.name} (needs medium model)"
                )
            else:
                # Fall back to tier 1 if can't run 7B
                return self._create_decision(
                    tier=1,
                    reasoning=f"Query complexity: {complexity.name} (fallback to available model)"
                )
        
        # Tier 3: Complex and Expert queries
        if self.hardware.get('can_run_7b', False):
            return self._create_decision(
                tier=3,
                reasoning=f"Query complexity: {complexity.name} (needs full model)"
            )
        else:
            # Fall back
            return self._create_decision(
                tier=1,
                reasoning=f"Query complexity: {complexity.name} (fallback - complex query on limited hardware)"
            )
    
    def _create_decision(self, tier: int, reasoning: str) -> CascadeDecision:
        """Create a CascadeDecision"""
        model_key = f'tier{tier}'
        model_name = self.models.get(model_key, self.models['tier1'])
        
        # Estimate time (assume 50 tokens generation)
        perf_key = f'est_tok_per_sec_tier{tier}'
        tok_per_sec = self.perf.get(perf_key, 10)
        est_time_ms = int((50 / tok_per_sec) * 1000)
        
        # Confidence based on tier appropriateness
        confidence = 0.9 if tier == 3 else (0.8 if tier == 2 else 0.7)
        
        return CascadeDecision(
            tier=tier,
            model_name=model_name,
            reasoning=reasoning,
            confidence=confidence,
            estimated_time_ms=est_time_ms
        )
    
    def get_cascade_stats(self) -> Dict[str, Any]:
        """Get statistics about the cascade"""
        return {
            'models': self.models,
            'hardware_tier': self.hardware.get('hardware_tier', 'Unknown'),
            'tier1_max_complexity': self.tier1_max_complexity.name,
            'tier2_max_complexity': self.tier2_max_complexity.name,
        }


# Test
if __name__ == "__main__":
    print("=" * 60)
    print("Model Cascade Router Test")
    print("=" * 60)
    
    # Simulate different hardware profiles
    profiles = [
        {
            'hardware_tier': 'High-End',
            'can_run_1b': True,
            'can_run_7b': True,
            'can_run_70b': True,
            'models': {
                'tier1': 'TinyLlama-1.1B',
                'tier2': 'Mistral-7B',
                'tier3': 'Llama-70B'
            },
            'performance': {
                'est_tok_per_sec_tier1': 100,
                'est_tok_per_sec_tier2': 30,
                'est_tok_per_sec_tier3': 8
            }
        },
        {
            'hardware_tier': 'Entry-Level',
            'can_run_1b': True,
            'can_run_7b': False,
            'can_run_70b': False,
            'models': {
                'tier1': 'TinyLlama-1.1B',
                'tier2': 'TinyLlama-1.1B',
                'tier3': 'TinyLlama-1.1B'
            },
            'performance': {
                'est_tok_per_sec_tier1': 10,
                'est_tok_per_sec_tier2': 10,
                'est_tok_per_sec_tier3': 10
            }
        }
    ]
    
    test_queries = [
        "Hi",
        "What is the capital of France?",
        "Explain quantum mechanics",
        "Write a Python function to sort a list",
        "Why is the sky blue?",
    ]
    
    for profile in profiles:
        print(f"\n🖥️  Hardware: {profile['hardware_tier']}")
        print("-" * 60)
        
        router = ModelCascadeRouter(profile)
        
        for query in test_queries:
            decision = router.route(query)
            complexity = router.classifier.classify(query)
            
            print(f"\nQuery: \"{query[:40]}...\"")
            print(f"  Complexity: {complexity.name}")
            print(f"  → Tier {decision.tier} ({decision.model_name.split('/')[-1]})")
            print(f"  Reason: {decision.reasoning}")
            print(f"  Est. time: {decision.estimated_time_ms}ms")
    
    print("\n" + "=" * 60)
