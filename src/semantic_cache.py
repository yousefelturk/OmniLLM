"""
Semantic Response Cache (SRC) - Component 1

Stores past Q&A pairs in a vector database and retrieves
semantically similar responses instead of recomputing.

This is NOVEL because:
- Not just KV caching (which is per-token)
- Full response caching based on semantic similarity
- Learns from your usage patterns
"""

import hashlib
import json
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    print("Warning: chromadb not installed. Using fallback dict cache.")

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    HAS_SBERT = False
    print("Warning: sentence-transformers not installed.")


@dataclass
class CacheEntry:
    """Single cached response"""
    query: str
    response: str
    model_used: str
    timestamp: float
    access_count: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'query': self.query,
            'response': self.response,
            'model_used': self.model_used,
            'timestamp': self.timestamp,
            'access_count': self.access_count
        }


class SemanticCache:
    """
    Semantic Response Cache using vector similarity.
    
    Key innovation: Cache full responses, not just KV tensors.
    Retrieve if semantic similarity > threshold.
    """
    
    def __init__(
        self,
        cache_dir: str = "./cache",
        similarity_threshold: float = 0.92,
        max_entries: int = 10000,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        self.cache_dir = cache_dir
        self.similarity_threshold = similarity_threshold
        self.max_entries = max_entries
        
        # Initialize embedding model
        self.embedding_model = None
        if HAS_SBERT:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                print(f"✓ Loaded embedding model: {embedding_model}")
            except Exception as e:
                print(f"⚠ Failed to load embedding model: {e}")
        
        # Initialize vector DB
        self.collection = None
        self.fallback_cache: Dict[str, CacheEntry] = {}
        
        if HAS_CHROMADB and self.embedding_model:
            try:
                self.client = chromadb.PersistentClient(
                    path=cache_dir,
                    settings=Settings(anonymized_telemetry=False)
                )
                self.collection = self.client.get_or_create_collection(
                    name="omnillm_cache",
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"✓ Initialized ChromaDB cache at {cache_dir}")
            except Exception as e:
                print(f"⚠ ChromaDB init failed: {e}, using fallback")
                self.collection = None
    
    def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding vector for text"""
        if self.embedding_model is None:
            return None
        try:
            return self.embedding_model.encode(text, convert_to_numpy=True)
        except Exception as e:
            print(f"Embedding error: {e}")
            return None
    
    def get(self, query: str) -> Optional[CacheEntry]:
        """
        Check cache for semantically similar query.
        
        Returns cached response if similarity > threshold.
        """
        start_time = time.time()
        
        # Fallback mode: simple exact match
        if self.collection is None:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self.fallback_cache:
                entry = self.fallback_cache[query_hash]
                entry.access_count += 1
                print(f"[Cache] Exact match hit! ({(time.time()-start_time)*1000:.1f}ms)")
                return entry
            return None
        
        # Vector search mode
        try:
            embedding = self._get_embedding(query)
            if embedding is None:
                return None
            
            results = self.collection.query(
                query_embeddings=[embedding.tolist()],
                n_results=1,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['ids'][0]:
                return None
            
            # Check similarity (Chroma returns distance, convert to similarity)
            distance = results['distances'][0][0]
            similarity = 1 - distance  # Cosine similarity
            
            if similarity >= self.similarity_threshold:
                metadata = results['metadatas'][0][0]
                cached_query = results['documents'][0][0]
                
                entry = CacheEntry(
                    query=cached_query,
                    response=metadata['response'],
                    model_used=metadata['model_used'],
                    timestamp=metadata['timestamp'],
                    access_count=metadata.get('access_count', 1) + 1
                )
                
                # Update access count
                self.collection.update(
                    ids=results['ids'][0],
                    metadatas=[{
                        **metadata,
                        'access_count': entry.access_count,
                        'last_accessed': time.time()
                    }]
                )
                
                print(f"[Cache] Semantic hit! (sim={similarity:.3f}, {(time.time()-start_time)*1000:.1f}ms)")
                return entry
            else:
                print(f"[Cache] Miss (sim={similarity:.3f} < {self.similarity_threshold})")
                return None
                
        except Exception as e:
            print(f"[Cache] Error: {e}")
            return None
    
    def put(self, query: str, response: str, model_used: str):
        """Store query-response pair in cache"""
        entry = CacheEntry(
            query=query,
            response=response,
            model_used=model_used,
            timestamp=time.time()
        )
        
        # Fallback mode
        if self.collection is None:
            query_hash = hashlib.md5(query.encode()).hexdigest()
            self.fallback_cache[query_hash] = entry
            
            # Simple eviction
            if len(self.fallback_cache) > self.max_entries:
                oldest = min(self.fallback_cache.items(), key=lambda x: x[1].timestamp)
                del self.fallback_cache[oldest[0]]
            return
        
        # Vector DB mode
        try:
            embedding = self._get_embedding(query)
            if embedding is None:
                return
            
            doc_id = hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()
            
            self.collection.add(
                ids=[doc_id],
                embeddings=[embedding.tolist()],
                documents=[query],
                metadatas=[{
                    'response': response,
                    'model_used': model_used,
                    'timestamp': time.time(),
                    'access_count': 1,
                    'last_accessed': time.time()
                }]
            )
            print(f"[Cache] Stored (total: {self.collection.count()})")
            
        except Exception as e:
            print(f"[Cache] Store error: {e}")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.collection:
            return {
                'total_entries': self.collection.count(),
                'type': 'semantic_vector',
                'threshold': self.similarity_threshold
            }
        else:
            return {
                'total_entries': len(self.fallback_cache),
                'type': 'exact_match_fallback',
                'threshold': 1.0
            }
    
    def clear(self):
        """Clear all cached entries"""
        if self.collection:
            self.client.delete_collection("omnillm_cache")
            self.collection = self.client.create_collection(
                name="omnillm_cache",
                metadata={"hnsw:space": "cosine"}
            )
        self.fallback_cache.clear()
        print("[Cache] Cleared all entries")


# Simple test
if __name__ == "__main__":
    print("=" * 60)
    print("Semantic Cache Test")
    print("=" * 60)
    
    cache = SemanticCache()
    
    # Test 1: Store and retrieve
    print("\n1. Storing response...")
    cache.put(
        "What is the capital of France?",
        "The capital of France is Paris.",
        "TinyLlama-1.1B"
    )
    
    # Test 2: Exact match
    print("\n2. Testing exact match...")
    result = cache.get("What is the capital of France?")
    if result:
        print(f"✓ Hit: {result.response[:50]}...")
    else:
        print("✗ Miss")
    
    # Test 3: Semantic match
    print("\n3. Testing semantic match...")
    result = cache.get("What's France's capital city?")
    if result:
        print(f"✓ Semantic hit: {result.response[:50]}...")
    else:
        print("✗ Miss (similarity too low)")
    
    # Test 4: Different query (should miss)
    print("\n4. Testing different query...")
    result = cache.get("How does quantum mechanics work?")
    if result:
        print(f"✗ Unexpected hit: {result.response[:50]}...")
    else:
        print("✓ Correctly missed (different topic)")
    
    print("\n" + "=" * 60)
    print(f"Cache stats: {cache.stats()}")
    print("=" * 60)
