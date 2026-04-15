#!/usr/bin/env python
"""
Simple script to verify Ollama is running and accessible
Run: python check_ollama.py
"""

import sys
sys.path.insert(0, 'src')

from orlock.providers.llm_client import LocalLLMClient

def check_ollama():
    print("\n" + "=" * 60)
    print("🔍 Checking Ollama Connection...")
    print("=" * 60 + "\n")

    try:
        client = LocalLLMClient()

        print(f"✅ Ollama Base URL: {client.base_url}")
        print(f"✅ Model: {client.model}\n")

        # Try a simple request
        print("⏳ Testing connection to Ollama...\n")
        response = client.generate("Hello")

        print("✅ SUCCESS! Ollama is running and responding!\n")
        print(f"Test Response: {response[:100]}...\n")
        print("=" * 60)
        print("Your project is ready to use!")
        print("=" * 60 + "\n")

    except ConnectionError as e:
        print("❌ ERROR: Cannot connect to Ollama")
        print(f"   {e}\n")
        print("Make sure Ollama is running with: ollama serve\n")

    except Exception as e:
        print(f"❌ ERROR: {e}\n")

if __name__ == "__main__":
    check_ollama()
