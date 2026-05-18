import time
from openai import OpenAI
import os, json
import google.generativeai as genai

TOKENS_IN = dict()
TOKENS_OUT = dict()
TOKENS_CACHE_READ = dict()
TOKENS_CACHE_CREATE = dict()


def _research_lab_deepseek_base_url() -> str:
    """Match main CLI: default official DeepSeek OpenAPI base; ofox via DEEPSEEK_BASE_URL."""
    raw = (os.getenv("DEEPSEEK_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip().rstrip("/")
    if not raw:
        return "https://api.deepseek.com/v1"
    if raw in ("https://api.deepseek.com", "http://api.deepseek.com"):
        return "https://api.deepseek.com/v1"
    return raw


def _research_lab_deepseek_api_model(model_str: str) -> str:
    """Ofox often expects deepseek/…; official api.deepseek.com uses plain ids (e.g. deepseek-chat, deepseek-v4-flash)."""
    base = _research_lab_deepseek_base_url().lower()
    if "ofox.ai" not in base:
        return model_str
    if model_str == "deepseek-v3.2":
        return "deepseek/deepseek-v3.2"
    if model_str == "deepseek-chat":
        return "deepseek-chat"
    if model_str.startswith("deepseek-"):
        return f"deepseek/{model_str}"
    return model_str


def _research_lab_deepseek_api_key(openai_api_key=None) -> str | None:
    d = (os.getenv("DEEPSEEK_API_KEY") or "").strip()
    if d:
        return d
    if openai_api_key:
        s = str(openai_api_key).strip()
        return s or None
    o = (os.getenv("OPENAI_API_KEY") or "").strip()
    return o or None


def extract_usage(response, model_str, api_type="openai", version="1.5"):
    global TOKENS_IN, TOKENS_OUT, TOKENS_CACHE_READ, TOKENS_CACHE_CREATE
    
    in_tokens = 0
    out_tokens = 0
    cache_read = 0
    cache_create = 0
    
    try:
        if api_type == "openai":
            if version == "0.28":
                usage = response.get("usage", {})
                in_tokens = usage.get("prompt_tokens", 0)
                out_tokens = usage.get("completion_tokens", 0)
            else:
                usage = getattr(response, "usage", None)
                if usage:
                    in_tokens = getattr(usage, "prompt_tokens", 0)
                    out_tokens = getattr(usage, "completion_tokens", 0)
                    details = getattr(usage, "prompt_tokens_details", None)
                    if details:
                        cache_read = getattr(details, "cached_tokens", 0)
                        
        elif api_type == "anthropic":
            usage = getattr(response, "usage", None)
            if usage:
                in_tokens = getattr(usage, "input_tokens", 0)
                out_tokens = getattr(usage, "output_tokens", 0)
                cache_create = getattr(usage, "cache_creation_input_tokens", 0)
                cache_read = getattr(usage, "cache_read_input_tokens", 0)
                
        elif api_type == "gemini":
            usage = getattr(response, "usage_metadata", None)
            if usage:
                in_tokens = getattr(usage, "prompt_token_count", 0)
                out_tokens = getattr(usage, "candidates_token_count", 0)
                cache_read = getattr(usage, "cached_content_token_count", 0)
                
        # Initialize if not exists
        if model_str not in TOKENS_IN:
            TOKENS_IN[model_str] = 0
            TOKENS_OUT[model_str] = 0
            TOKENS_CACHE_READ[model_str] = 0
            TOKENS_CACHE_CREATE[model_str] = 0
            
        # Update globals
        TOKENS_IN[model_str] += in_tokens
        TOKENS_OUT[model_str] += out_tokens
        TOKENS_CACHE_READ[model_str] += cache_read
        TOKENS_CACHE_CREATE[model_str] += cache_create
        
    except Exception as e:
        print(f"Error extracting token usage: {e}")

def curr_cost_est():
    costmap_in = {
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "claude-4-6-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "deepseek-v3.2": 1.00 / 1000000,
        "deepseek-v4-flash": 0.27 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
    }
    costmap_out = {
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 15.00 / 1000000,
        "claude-4-6-sonnet": 15.00 / 1000000,
        "deepseek-chat": 2.00 / 1000000,
        "deepseek-v3.2": 2.00 / 1000000,
        "deepseek-v4-flash": 1.10 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
        "claude-4-6-sonnet": 15.00 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,  anthropic_api_key=None, tries=12, timeout=8.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None and os.getenv('DEEPSEEK_API_KEY') is None and gemini_api_key is None:
        raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    for _ in range(tries):
        try:
            if model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI(base_url="https://api.ofox.ai/v1")
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)

            elif model_str == "gemini-2.0-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                response = model.generate_content(prompt)
                answer = response.text
                extract_usage(response, model_str, "gemini")
            elif model_str == "gemini-1.5-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                response = model.generate_content(prompt)
                answer = response.text
                extract_usage(response, model_str, "gemini")
            elif model_str.startswith("gemini-") and model_str not in ("gemini-2.0-pro", "gemini-1.5-pro"):
                gk = gemini_api_key or os.getenv("GEMINI_API_KEY")
                if not gk:
                    raise ValueError("GEMINI_API_KEY required for Gemini model backends in research_lab inference")
                genai.configure(api_key=gk)
                gname = model_str.replace("models/", "").strip()
                model = genai.GenerativeModel(model_name=gname, system_instruction=system_prompt)
                response = model.generate_content(prompt)
                answer = response.text
                extract_usage(response, model_str, "gemini")
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI(base_url="https://api.ofox.ai/v1")
                    completion = client.chat.completions.create(
                        model="o3-mini", messages=messages)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)

            elif model_str == "claude-3.5-sonnet" or model_str == "claude-3-5-sonnet":
                model_str = "claude-3-5-sonnet"
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    max_tokens=8192,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
                extract_usage(message, model_str, "anthropic")
                
            elif model_str == "claude-4.6-sonnet" or model_str == "claude-4-6-sonnet" or model_str == "claude-sonnet-4.6":
                model_str = "claude-4-6-sonnet"
                client = OpenAI(
                    api_key=os.environ["ANTHROPIC_API_KEY"],
                    base_url="https://api.ofox.ai/v1"
                )
                completion = client.chat.completions.create(
                    model="anthropic/claude-sonnet-4.6",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}],
                    max_tokens=8192
                )
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI(base_url="https://api.ofox.ai/v1")
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)
            elif model_str == "deepseek-chat" or model_str == "deepseek-v3.2" or model_str.startswith("deepseek-"):
                api_model_name = _research_lab_deepseek_api_model(model_str)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    dkey = _research_lab_deepseek_api_key(openai_api_key)
                    if not dkey:
                        raise ValueError(
                            "DEEPSEEK_API_KEY or OPENAI_API_KEY is required for DeepSeek models in research_lab inference"
                        )
                    deepseek_client = OpenAI(
                        api_key=dkey,
                        base_url=_research_lab_deepseek_base_url(),
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model=api_model_name,
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model=api_model_name,
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI(base_url="https://api.ofox.ai/v1")
                    completion = client.chat.completions.create(
                        model="o1-mini", messages=messages)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI(base_url="https://api.ofox.ai/v1")
                    completion = client.chat.completions.create(
                        model="o1", messages=messages)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI(base_url="https://api.ofox.ai/v1")
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content
                extract_usage(completion, model_str, "openai", version)
            else:
                raise ValueError(f"Unsupported model backend: {model_str}")

            try:
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost: print(f"Cost approximation has an error? {e}")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))