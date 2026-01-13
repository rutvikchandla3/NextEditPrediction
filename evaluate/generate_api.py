import os
import json
import difflib
import asyncio
import time
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
import httpx
from google.genai.client import Client
from google.genai.types import GenerateContentConfig, HttpOptions
from tqdm.asyncio import tqdm_asyncio
from prompts import SYSTEM_INSTRUCTION, PROMPT_TEMPLATE

load_dotenv()

openai_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

anthropic_client = AsyncAnthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY"),
)

gemini_client = Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=HttpOptions(
        asyncClientArgs={
            "limits": httpx.Limits(max_connections=100, max_keepalive_connections=20)
        }
    ),
)

deepseek_client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
)

async def send_request(client, model_name, model_group, prompt):
    MAX_OUTPUT_TOKENS = 8192
    metrics = {}

    print("Sending request")

    start_time = time.perf_counter()

    if model_group == "openai" or model_group == "deepseek":
        completion = await client.chat.completions.create(
            model=model_name,
            max_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": prompt},
            ],
        )
        end_time = time.perf_counter()
        print(completion)
        response = completion.choices[0].message.content
        metrics = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
    elif model_group == "anthropic":
        completion = await client.messages.create(
            model=model_name,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=SYSTEM_INSTRUCTION,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        end_time = time.perf_counter()
        print(completion)
        response = completion.content[0].text
        metrics = {
            "input_tokens": completion.usage.input_tokens,
            "output_tokens": completion.usage.output_tokens,
            "total_tokens": completion.usage.input_tokens + completion.usage.output_tokens,
        }
    elif model_group == "gemini":
        completion = await client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
        )
        end_time = time.perf_counter()
        # print(completion.text)
        response = completion.text
        metrics = {
            "input_tokens": completion.usage_metadata.prompt_token_count,
            "output_tokens": completion.usage_metadata.candidates_token_count,
            "total_tokens": completion.usage_metadata.total_token_count,
        }

    latency_ms = (end_time - start_time) * 1000
    metrics["latency_ms"] = latency_ms
    metrics["tokens_per_second"] = metrics["output_tokens"] / (latency_ms / 1000) if latency_ms > 0 else 0

    if response is None or response.strip() == "":
        return None, False, metrics

    if response.startswith("```"):
        first_newline = response.find("\n")
        response = response[first_newline + 1:-4]

    if response.endswith("\n"):
        response = response[:-1]

    start_tag = "<next_version>\n"
    end_tag = "\n</next_version>"

    if response.startswith(start_tag) and response.endswith(end_tag):
        response = response[len(start_tag):-len(end_tag)]
        return response, True, metrics
    else:
        return None, False, metrics

async def process_line(client, line, model_name, model_group, semaphore, output_file, file_lock):
    async with semaphore:
        commit = json.loads(line)
        diff = difflib.unified_diff(
            commit['old_contents'].splitlines(),
            commit['current_contents'].splitlines(),
            lineterm='',
            fromfile=commit['old_file'],
            tofile=commit['new_file'],
        )
        prompt = PROMPT_TEMPLATE.format(
            original_code=commit['old_contents'],
            edits="\n".join(diff),
            current_version=commit['current_contents'],
        )
        ground_truth = commit['new_contents']

        valid = False
        retry_count = 0
        while not valid:
            response, valid, metrics = await send_request(client, model_name, model_group, prompt)
            retry_count += 1

        result = {
            "prompt": prompt,
            "model_output": response,
            "ground_truth": ground_truth,
            "metrics": {
                "model": model_name,
                "latency_ms": metrics["latency_ms"],
                "input_tokens": metrics["input_tokens"],
                "output_tokens": metrics["output_tokens"],
                "total_tokens": metrics["total_tokens"],
                "tokens_per_second": metrics["tokens_per_second"],
                "retry_count": retry_count,
            },
        }

        async with file_lock:
            output_file.write(json.dumps(result) + "\n")
            output_file.flush()

        return result

async def main():
    generation_results_dir = "generation_results"
    os.makedirs(generation_results_dir, exist_ok=True)

    openai_models = [
        # "gpt-4o", # gpt-4o-2024-08-06
        # "gpt-4o-mini", # gpt-4o-mini-2024-07-18
        "gpt-4.1", # gpt-4.1-2025-04-14
        # "gpt-4.1-mini", # gpt-4.1-mini-2025-04-14
        # "gpt-4.1-nano", # gpt-4.1-nano-2025-04-14
    ]

    anthropic_models = [
        # "claude-opus-4-20250514",
        # "claude-sonnet-4-20250514",
        # "claude-3-7-sonnet-20250219",
        # "claude-3-5-sonnet-20241022",
        # "claude-3-5-haiku-20241022",
        "claude-haiku-4-5-20250514",  # Claude 4.5 Haiku
    ]

    gemini_models = [
        # "gemini-2.5-pro",
        # "gemini-2.5-flash",
        "gemini-3-flash-preview",  # Gemini 3 Flash Preview
    ]

    deepseek_models = [
        # "deepseek-chat", # DeepSeek-V3-0324
        # "deepseek-reasoner", # DeepSeek-R1-0528
    ]

    # models = openai_models + anthropic_models + gemini_models + deepseek_models
    models = gemini_models

    for model in models:
        print(f"Processing model: {model}")
        # Determine the client and model group based on the model name
        if model in openai_models:
            client, model_group = openai_client, "openai"
        elif model in anthropic_models:
            client, model_group = anthropic_client, "anthropic"
        elif model in gemini_models:
            client, model_group = gemini_client, "gemini"
        elif model in deepseek_models:
            client, model_group = deepseek_client, "deepseek"

        max_concurrent_requests = 10
        semaphore = asyncio.Semaphore(max_concurrent_requests)
        file_lock = asyncio.Lock()

        with open("../dataset/crawl/test.jsonl", "r") as f:
            lines = f.readlines()

        output_path = f"{generation_results_dir}/{model.split('/')[-1]}_generation_results.jsonl"
        with open(output_path, "w") as output_file:
            tasks = [process_line(client, line, model, model_group, semaphore, output_file, file_lock) for line in lines]
            await tqdm_asyncio.gather(*tasks, desc="Processing lines")

        print(f"Results saved to {output_path}")

if __name__ == "__main__":
    asyncio.run(main())
