import ollama
import time

print("STEP 1: Python started")

print("STEP 2: Sending request to Mistral...")
start = time.time()

response = ollama.chat(
    model="mistral",
    messages=[
        {"role": "user", "content": "Say hello in one sentence"}
    ]
)

print("STEP 3: Response received")
print("Time taken:", round(time.time() - start, 2), "seconds")

print("\nMISTRAL OUTPUT:")
print(response["message"]["content"])
