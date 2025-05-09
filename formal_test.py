import openai
import time
import asyncio
import threading
from typing import List, Dict
import statistics
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import random
import re

# Configure OpenAI client for Ollama
client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="no-key-required"
)

MODEL_NAME = "Qwen3-0.6B"
OUTPUT_DIR = "results/" +  MODEL_NAME + "/" + "LM_studio_test_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class PerformanceMetrics:
    def __init__(self, scenario_name: str):
        self.scenario_name = scenario_name
        self.ttft_list = []    # Time-To-First-Token
        self.tbt_list = []     # Time-Between-Tokens
        self.tps_list = []     # Tokens-Per-Second
        self.total_tokens = 0
        self.concurrency = 0
        self.prompts_used = []
        self.lock = threading.Lock()

def generate_dynamic_prompt(scenario_type: str) -> str:
    """Generate test prompts dynamically based on scenario type"""
    prompt_map = {
        "short_text": "Generate a very short (5-10 tokens)  question about {}/no_think",
        "qa_medium": "Create a medium-length (50-100 tokens) question about {}/no_think",
        "long_form": "Make a detailed question with the length of 20-30 tokens about {} and this prompt ask llm to give a long answer/no_think",
        "stress_test": "Create a (30-50 tokens) question about {}/no_think",
        "mixed_load": "Provide a question about {} with variable length(10-200 tokens)/no_think"
    }
    
    topics = ["artificial intelligence", "quantum computing", 
             "climate change", "machine learning", "neuroscience"]
    
    topic = random.choice(topics)
    
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{
            "role": "system",
            "content": "You are a prompt engineering assistant. Generate the requested test questions."
        }, {
            "role": "user",
            "content": prompt_map[scenario_type].format(topic)
        }],
        temperature=0.7
    )
    
    # Remove all content between <think> and </think> tags
    generated_prompt = re.sub(r'<think>.*?</think>', '', response.choices[0].message.content, flags=re.DOTALL)
    
    # Clean up any resulting extra whitespace
    generated_prompt = ' '.join(generated_prompt.split()).strip()
    
    return generated_prompt

def process_stream(stream, start_time: float, metrics: PerformanceMetrics):
    """Process synchronous stream response"""
    first_token_time = None
    token_count = 0
    tbt_list = []
    last_time = start_time
    
    for chunk in stream:
        current_time = time.time()
        if first_token_time is None:
            first_token_time = current_time - start_time
        else:
            tbt_list.append(current_time - last_time)
        token_count += 1
        last_time = current_time
    
    # Thread-safe metrics update
    with metrics.lock:
        if first_token_time is not None:
            metrics.ttft_list.append(first_token_time)
            metrics.tbt_list.extend(tbt_list)
            metrics.total_tokens += token_count
            duration = current_time - start_time
            if duration > 0:
                metrics.tps_list.append(token_count / duration)

def make_request(prompt: str, max_tokens: int, metrics: PerformanceMetrics):
    """Make a single API request and process the response"""
    start_time = time.time()
    
    try:
        # Create the stream (synchronous API)
        stream = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt + "/no_think"}],
            max_tokens=max_tokens,
            stream=True
        )
        
        # Process the stream
        process_stream(stream, start_time, metrics)
                
    except Exception as e:
        print(f"Error processing request: {str(e)}")

def run_scenario(scenario_type: str, concurrency: int, max_tokens: int, num_requests: int) -> PerformanceMetrics:
    """Run a test scenario with threading"""
    metrics = PerformanceMetrics(scenario_type)
    metrics.concurrency = concurrency
    
    # Generate unique prompts for this test run
    metrics.prompts_used = [generate_dynamic_prompt(scenario_type) for _ in range(num_requests)]

    time.sleep(5)  # Allow some time for the model to reset

    # Create and run all threads
    threads = []
    for prompt in metrics.prompts_used:
        thread = threading.Thread(
            target=make_request,
            args=(prompt, max_tokens, metrics)
        )
        threads.append(thread)
        thread.start()
        
        # Control concurrency
        while threading.active_count() > concurrency + 1:  # +1 for main thread
            time.sleep(0.1)
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    return metrics

def save_results(metrics: PerformanceMetrics):
    """Save test results and generate plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{OUTPUT_DIR}/{metrics.scenario_name}_{timestamp}"
    
    # Save raw data
    with open(f"{base_filename}.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "concurrency": metrics.concurrency,
            "ttft": metrics.ttft_list,
            "tbt": metrics.tbt_list,
            "tps": metrics.tps_list,
            "total_tokens": metrics.total_tokens,
            "prompts": metrics.prompts_used
        }, f, indent=2)
    
    # Generate plots
    plt.figure(figsize=(15, 5))
    plt.suptitle(f"{MODEL_NAME} - {metrics.scenario_name} (Concurrency: {metrics.concurrency})", 
                fontsize=12, y=1.05)
    
    # TTFT Plot
    plt.subplot(1, 3, 1)
    plt.hist(metrics.ttft_list, bins=10, color='skyblue', edgecolor='black')
    plt.title("TTFT Distribution", pad=10)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # TBT Plot
    plt.subplot(1, 3, 2)
    plt.hist(metrics.tbt_list, bins=10, color='lightgreen', edgecolor='black')
    plt.title("TBT Distribution", pad=10)
    plt.xlabel("Time between tokens (seconds)")
    plt.ylabel("Frequency")
    plt.grid(True, alpha=0.3)
    
    # TPS Plot
    plt.subplot(1, 3, 3)
    plt.plot(metrics.tps_list, marker='o', linestyle='-', color='salmon')
    plt.title("TPS Trend", pad=10)
    plt.xlabel("Request sequence")
    plt.ylabel("Tokens per second")
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{base_filename}.png", bbox_inches='tight', dpi=300)
    plt.close()
    
    # Print summary
    print(f"\n{metrics.scenario_name.upper()} RESULTS ({MODEL_NAME})")
    print(f"Concurrency: {metrics.concurrency}")
    print(f"Avg TTFT: {statistics.mean(metrics.ttft_list):.4f}s")
    print(f"Avg TBT: {statistics.mean(metrics.tbt_list):.4f}s")
    print(f"Avg TPS: {statistics.mean(metrics.tps_list):.2f} tokens/s")
    print(f"Total tokens: {metrics.total_tokens}")
    print(f"Results saved to {base_filename}.[json/png]")

    # Print summary To File
    summary_filename = f"{base_filename}_summary.txt"
    with open(summary_filename, "w") as f:
        f.write(f"{metrics.scenario_name.upper()} RESULTS ({MODEL_NAME})\n")
        f.write(f"Concurrency: {metrics.concurrency}\n")
        f.write(f"Avg TTFT: {statistics.mean(metrics.ttft_list):.4f}s\n")
        f.write(f"Avg TBT: {statistics.mean(metrics.tbt_list):.4f}s\n")
        f.write(f"Avg TPS: {statistics.mean(metrics.tps_list):.2f} tokens/s\n")
        f.write(f"Total tokens: {metrics.total_tokens}\n")
        f.write(f"Results saved to {summary_filename}\n")

def main():

    requests_num = 40

    # Define test scenarios
    scenarios = [
        {"type": "short_text", "concurrency": 5, "max_tokens": 20, "requests": requests_num},
        {"type": "qa_medium", "concurrency": 3, "max_tokens": 100, "requests": requests_num},
        {"type": "long_form", "concurrency": 1, "max_tokens": 300, "requests": requests_num},
        {"type": "stress_test", "concurrency": 10, "max_tokens": 50, "requests": requests_num},
        {"type": "mixed_load", "concurrency": 4, "max_tokens": 200, "requests": requests_num}
    ]

    # Run all scenarios
    for scenario in scenarios:
        print(f"\nStarting {scenario['type']} scenario...")
        metrics = run_scenario(
            scenario_type=scenario["type"],
            concurrency=scenario["concurrency"],
            max_tokens=scenario["max_tokens"],
            num_requests=scenario["requests"]
        )
        save_results(metrics)

if __name__ == "__main__":
    main()