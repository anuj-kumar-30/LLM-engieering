#!/usr/bin/env python3
"""
Quick Start Ollama + Mem0 Chatbot
Simple version that gets you up and running fast
"""

import requests
import json
from datetime import datetime

# Simple memory storage without heavy dependencies
class SimpleMemory:
    def __init__(self, user_id):
        self.user_id = user_id
        self.memories = []
        self.load_memories()
    
    def load_memories(self):
        """Load memories from file"""
        try:
            with open(f"memory_{self.user_id}.json", "r") as f:
                self.memories = json.load(f)
        except FileNotFoundError:
            self.memories = []
    
    def save_memories(self):
        """Save memories to file"""
        with open(f"memory_{self.user_id}.json", "w") as f:
            json.dump(self.memories, f, indent=2)
    
    def add_memory(self, user_msg, bot_msg):
        """Add conversation to memory"""
        memory = {
            "timestamp": datetime.now().isoformat(),
            "user": user_msg,
            "bot": bot_msg
        }
        self.memories.append(memory)
        self.save_memories()
    
    def search_memories(self, query, limit=3):
        """Simple keyword search in memories"""
        query_words = query.lower().split()
        relevant = []
        
        for memory in self.memories:
            text = (memory["user"] + " " + memory["bot"]).lower()
            score = sum(1 for word in query_words if word in text)
            if score > 0:
                relevant.append((score, memory))
        
        # Sort by relevance and return top results
        relevant.sort(reverse=True, key=lambda x: x[0])
        return [mem[1] for mem in relevant[:limit]]

class QuickOllamaBot:
    def __init__(self, user_id="user", model="llama2"):
        self.user_id = user_id
        self.model = model
        self.memory = SimpleMemory(user_id)
        self.ollama_url = "http://localhost:11434/api/generate"
        
        # Check if Ollama is running
        if not self.check_ollama():
            print("❌ Ollama is not running!")
            print("Please start Ollama with: ollama serve")
            print("And download a model with: ollama pull llama2")
            exit(1)
    
    def check_ollama(self):
        """Quick check if Ollama is available"""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=3)
            return response.status_code == 200
        except:
            return False
    
    def get_response(self, message):
        """Get response from Ollama"""
        # Get relevant memories
        relevant_memories = self.memory.search_memories(message)
        
        # Build context
        context = ""
        if relevant_memories:
            context = "Previous conversation context:\n"
            for mem in relevant_memories:
                context += f"User: {mem['user']}\nBot: {mem['bot']}\n"
            context += "\nCurrent conversation:\n"
        
        # Create prompt
        prompt = f"""{context}User: {message}
Bot: """
        
        # Call Ollama
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 150
            }
        }
        
        try:
            response = requests.post(self.ollama_url, json=payload, timeout=30)
            if response.status_code == 200:
                bot_response = response.json()["response"].strip()
                
                # Store in memory
                self.memory.add_memory(message, bot_response)
                
                return bot_response
            else:
                return f"Error: {response.status_code}"
        except Exception as e:
            return f"Error communicating with Ollama: {e}"
    
    def chat_loop(self):
        """Main chat loop"""
        print(f"🤖 Quick Ollama Chatbot (Model: {self.model})")
        print(f"👤 User: {self.user_id}")
        print("=" * 50)
        print("Type 'quit' to exit, 'memory' to see memories")
        print("=" * 50)
        
        while True:
            try:
                user_input = input(f"\n{self.user_id}: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'memory':
                    print(f"\n🧠 Memory ({len(self.memory.memories)} conversations):")
                    for i, mem in enumerate(self.memory.memories[-5:], 1):
                        print(f"{i}. User: {mem['user'][:50]}...")
                        print(f"   Bot: {mem['bot'][:50]}...")
                    continue
                
                if not user_input:
                    continue
                
                print("🤔 Thinking...", end="", flush=True)
                response = self.get_response(user_input)
                print(f"\r🤖: {response}")
                
            except KeyboardInterrupt:
                print("\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")

def quick_setup_check():
    """Quick setup verification"""
    print("🔍 Quick Setup Check")
    print("=" * 30)
    
    # Check Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            models = response.json().get("models", [])
            print(f"✅ Ollama is running")
            print(f"📦 Models available: {len(models)}")
            for model in models:
                print(f"   • {model['name']}")
            return True
        else:
            print("❌ Ollama responded with error")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to Ollama: {e}")
        print("\n🔧 To fix this:")
        print("1. Install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        print("2. Start Ollama: ollama serve")
        print("3. Download a model: ollama pull llama2")
        return False

def main():
    print("🚀 Quick Start Ollama Chatbot")
    print("=" * 40)
    
    # Quick setup check
    if not quick_setup_check():
        return
    
    # Get user preferences
    user_id = input("\nEnter your name: ").strip() or "user"
    
    # Get available models
    try:
        response = requests.get("http://localhost:11434/api/tags")
        models = [m['name'] for m in response.json().get("models", [])]
        
        if len(models) > 1:
            print(f"\nAvailable models: {', '.join(models)}")
            model = input("Choose model (or press Enter for first available): ").strip()
            if not model or model not in models:
                model = models[0]
        else:
            model = models[0] if models else "llama2"
        
        print(f"Using model: {model}")
        
    except:
        model = "llama2"
    
    # Start chatbot
    bot = QuickOllamaBot(user_id, model)
    bot.chat_loop()

if __name__ == "__main__":
    main()