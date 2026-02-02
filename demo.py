#!/usr/bin/env python3
"""
OmniLLM Demo Script

Run this to see OmniLLM in action!

Usage:
    python demo.py
    
Requirements:
    pip install -r requirements.txt
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from omnillm import OmniLLM


def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   ██████╗ ███╗   ███╗███╗   ██╗██╗██╗     ██████╗ ███╗   ███╗       ║
║  ██╔═══██╗████╗ ████║████╗  ██║██║██║     ██╔══██╗████╗ ████║       ║
║  ██║   ██║██╔████╔██║██╔██╗ ██║██║██║     ██║  ██║██╔████╔██║       ║
║  ██║   ██║██║╚██╔╝██║██║╚██╗██║██║██║     ██║  ██║██║╚██╔╝██║       ║
║  ╚██████╔╝██║ ╚═╝ ██║██║ ╚████║██║███████╗██████╔╝██║ ╚═╝ ██║       ║
║   ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═══╝╚═╝╚══════╝╚═════╝ ╚═╝     ╚═╝       ║
║                                                                      ║
║   Universal Adaptive Inference Engine                                ║
║   Run LLMs on ANY hardware intelligently                            ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    """)
    
    print("🚀 Starting OmniLLM...")
    print("   This will:")
    print("   1. Profile your hardware")
    print("   2. Configure optimal settings")
    print("   3. Load appropriate models")
    print("   4. Enable semantic caching")
    print()
    
    try:
        # Initialize
        llm = OmniLLM(verbose=True)
        
        print("\n" + "=" * 70)
        print("INTERACTIVE MODE")
        print("=" * 70)
        print("\nType your questions (or 'quit' to exit, 'status' for info)\n")
        
        history = []
        
        while True:
            try:
                user_input = input("\n👤 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye!")
                    break
                
                if user_input.lower() == 'status':
                    llm.print_status()
                    continue
                
                if not user_input:
                    continue
                
                # Generate response
                print("\n🤖 OmniLLM: ", end='', flush=True)
                
                result = llm.generate(
                    user_input,
                    max_tokens=256,
                    use_cache=True
                )
                
                print(result.text)
                
                # Show stats
                print(f"\n   ⚡ {result.generation_time_ms:.0f}ms | ", end='')
                print(f"Model: {result.model_used.split('/')[-1]} | ", end='')
                print(f"Tier: {result.tier} | ", end='')
                print(f"Cached: {'✓' if result.cached else '✗'}")
                
                # Add to history
                history.append({'role': 'user', 'content': user_input})
                history.append({'role': 'assistant', 'content': result.text})
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    except Exception as e:
        print(f"\n❌ Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
