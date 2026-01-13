import os
import json
import asyncio
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
from prompts import LLM_AS_A_JUDGE_PROMPT

load_dotenv()

# Use Portkey as the API gateway
portkey_client = AsyncOpenAI(
    api_key=os.getenv("PORTKEY_API_KEY"),
    base_url="https://api.portkey.ai/v1",
)

# Rate limiter: 100 requests per minute = 0.6 seconds between requests
class RateLimiter:
    def __init__(self, requests_per_minute):
        self.min_interval = 60.0 / requests_per_minute
        self.last_request_time = 0

    async def wait(self):
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_interval:
            await asyncio.sleep(self.min_interval - time_since_last)
        self.last_request_time = time.time()

rate_limiter = RateLimiter(100)  # 100 requests per minute

async def send_request(prompt: str, model: str, client: AsyncOpenAI):
    await rate_limiter.wait()  # Apply rate limiting
    response = await client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt},
        ],
        model=model,
    )

    return response.choices[0].message.content

async def main():
    llm_as_a_judge_results_dir = "llm_as_a_judge_results"
    os.makedirs(llm_as_a_judge_results_dir, exist_ok=True)

    models = [
        # Models to test
        # "gpt-4.1",
        # "claude-haiku-4-5-20250514",
        "gemini-3-flash-preview",
    ]

    for model in models:
        print(f"Processing model: {model}")

        prompt_list = []
        with open("../dataset/crawl/prompts.jsonl", "r") as f:
            for line in f:
                result = json.loads(line)
                prompt = result['prompt']
                prompt_list.append(prompt)

        evaluation_prompt_list = []
        with open(f"generation_results/{model}_generation_results.jsonl", "r") as f:
            for index, line in enumerate(f):
                result = json.loads(line)
                # prompt = prompt_list[index]
                prompt = result['prompt']
                ground_truth = result['ground_truth']
                model_output = result['model_output']

                evaluation_prompt = LLM_AS_A_JUDGE_PROMPT.format(
                    prompt=prompt,
                    ground_truth=ground_truth,
                    model_output=model_output
                )

                evaluation_prompt_list.append(evaluation_prompt)

        responses = []
        for i, prompt in enumerate(evaluation_prompt_list):
            response = await send_request(prompt, "gpt-4.1", portkey_client)
            responses.append(response)
            if (i + 1) % 10 == 0:  # Progress update every 10 requests
                print(f"Processed {i + 1} / {len(evaluation_prompt_list)} requests")

        correct_predictions = 0
        for i, response in enumerate(responses):
            if response == "yes":
                correct_predictions += 1
        accuracy = round((correct_predictions / len(responses)) * 100, 2)
        print(f"Model: {model}, LLM-as-a-Judge Accuracy: {accuracy}")

        with open(f"llm_as_a_judge_results/{model}_llm_as_a_judge_results.jsonl", "w") as f:
            for i, response in enumerate(responses):
                f.write(json.dumps({"response": response}) + "\n")

        print(f"Evaluation results saved to llm_as_a_judge_results/{model}_llm_as_a_judge_results.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
