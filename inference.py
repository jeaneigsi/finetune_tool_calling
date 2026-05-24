#!/usr/bin/env python3
"""Interactive inference for fine-tuned Qwen 3.5 tool-calling model."""
import json, os, sys, re, argparse
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer

SYSTEM = "You are JBUJB assistant, a food ordering and restaurant discovery agent. Use only available tools. Never invent IDs — always resolve them through search or resolution tools. Ask for clarification when required information is missing (location, restaurant name, ambiguous results). Mutating order actions (create, add, remove, update, clear) require explicit user confirmation. Respond in the same language as the user (French, English, or Moroccan Arabic). Be concise, friendly, and helpful."

def load_model(checkpoint_dir: str, max_seq_length: int = 4096):
    """Load LoRA adapter from checkpoint."""
    print(f"Loading LoRA adapter from {checkpoint_dir}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=max_seq_length,
        dtype=torch.bfloat16,
        load_in_4bit=False,
        attn_implementation="sdpa",
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=32, lora_alpha=32, lora_dropout=0,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    # Load adapter weights
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def load_tools(registry_path: str = None) -> list:
    """Load tool definitions from registry."""
    if registry_path and os.path.exists(registry_path):
        with open(registry_path) as f:
            return json.load(f)
    # Default JBUJB tools
    return [
        {"type":"function","function":{"name":"search_food","description":"Search for dishes/food items","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Search term"},"language":{"type":"string","enum":["fr","en","ar"]},"limit":{"type":"integer","default":5},"page":{"type":"integer","default":1}},"required":["query"]}}},
        {"type":"function","function":{"name":"search_restaurants","description":"Search for restaurants","parameters":{"type":"object","properties":{"query":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]},"lat":{"type":"number"},"lon":{"type":"number"},"limit":{"type":"integer","default":5},"open_now":{"type":"boolean"},"is_verified":{"type":"boolean"},"is_delivery":{"type":"boolean"},"min_price":{"type":"number"},"max_price":{"type":"number"}},"required":["query"]}}},
        {"type":"function","function":{"name":"resolve_restaurant","description":"Resolve restaurant name to business ID","parameters":{"type":"object","properties":{"name":{"type":"string","description":"Restaurant name"},"language":{"type":"string","enum":["fr","en","ar"]},"city":{"type":"string"},"lat":{"type":"number"},"lon":{"type":"number"}},"required":["name"]}}},
        {"type":"function","function":{"name":"get_restaurant_menu","description":"Get a restaurant's menu","parameters":{"type":"object","properties":{"merchant_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]},"limit":{"type":"integer","default":20},"search":{"type":"string"},"product_type":{"type":"string"}},"required":["merchant_id"]}}},
        {"type":"function","function":{"name":"get_restaurant_details","description":"Get restaurant details","parameters":{"type":"object","properties":{"merchant_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["merchant_id"]}}},
        {"type":"function","function":{"name":"check_restaurant_open","description":"Check if restaurant is open","parameters":{"type":"object","properties":{"merchant_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["merchant_id"]}}},
        {"type":"function","function":{"name":"check_food_available","description":"Check food availability","parameters":{"type":"object","properties":{"product_id":{"type":"string"},"merchant_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["product_id"]}}},
        {"type":"function","function":{"name":"get_delivery_info","description":"Get delivery information","parameters":{"type":"object","properties":{"merchant_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["merchant_id"]}}},
        {"type":"function","function":{"name":"create_order","description":"Create a new order","parameters":{"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{"product_id":{"type":"string"},"quantity":{"type":"integer"}}},"description":"Order items"},"restaurant":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["items"]}}},
        {"type":"function","function":{"name":"add_to_order","description":"Add items to existing order","parameters":{"type":"object","properties":{"draft_id":{"type":"string"},"items":{"type":"array","items":{"type":"object","properties":{"product_id":{"type":"string"},"quantity":{"type":"integer"}}}},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["draft_id","items"]}}},
        {"type":"function","function":{"name":"remove_from_order","description":"Remove items from order","parameters":{"type":"object","properties":{"draft_id":{"type":"string"},"product_ids":{"type":"array","items":{"type":"string"}},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["draft_id","product_ids"]}}},
        {"type":"function","function":{"name":"get_order","description":"Get order details","parameters":{"type":"object","properties":{"draft_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["draft_id"]}}},
        {"type":"function","function":{"name":"find_nearby","description":"Find nearby restaurants","parameters":{"type":"object","properties":{"lat":{"type":"number"},"lon":{"type":"number"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["lat","lon"]}}},
        {"type":"function","function":{"name":"get_promotions","description":"Get active promotions","parameters":{"type":"object","properties":{"language":{"type":"string","enum":["fr","en","ar"]},"city":{"type":"string"},"search":{"type":"string"}}}}},
        {"type":"function","function":{"name":"get_food_details","description":"Get detailed food information","parameters":{"type":"object","properties":{"product_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["product_id"]}}},
        {"type":"function","function":{"name":"validate_products","description":"Validate product availability","parameters":{"type":"object","properties":{"product_ids":{"type":"array","items":{"type":"string"}},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["product_ids"]}}},
        {"type":"function","function":{"name":"get_user_profile","description":"Get user profile","parameters":{"type":"object","properties":{"user_id":{"type":"string"}}}}},
        {"type":"function","function":{"name":"search_all","description":"Search both dishes and restaurants","parameters":{"type":"object","properties":{"query":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]},"lat":{"type":"number"},"lon":{"type":"number"},"limit":{"type":"integer","default":5}},"required":["query"]}}},
        {"type":"function","function":{"name":"search_context","description":"Context-aware search","parameters":{"type":"object","properties":{"query":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]},"lat":{"type":"number"},"lon":{"type":"number"}},"required":["query"]}}},
        {"type":"function","function":{"name":"get_cities","description":"List available cities","parameters":{"type":"object","properties":{"language":{"type":"string","enum":["fr","en","ar"]}}}}},
        {"type":"function","function":{"name":"get_districts","description":"List districts in a city","parameters":{"type":"object","properties":{"city_id":{"type":"string"},"language":{"type":"string","enum":["fr","en","ar"]}},"required":["city_id"]}}},
    ]


def parse_prediction(raw: str) -> dict | None:
    """Parse model output into structured prediction."""
    # Strip thinking
    raw = re.sub(r'<think>.*?</think>', '', raw, flags=re.S | re.I)
    raw = re.sub(r'⟨think⟩.*?⟨/think⟩', '', raw, flags=re.S | re.I)
    
    # Qwen XML format
    tc_match = re.search(r'<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>', raw, re.S | re.I)
    if tc_match:
        name = tc_match.group(1)
        args_str = tc_match.group(2)
        args = {}
        for pm in re.finditer(r'<parameter=(\w+)>(.*?)</parameter>', args_str, re.S | re.I):
            args[pm.group(1)] = pm.group(2)
        return {"type": "tool_call", "name": name, "arguments": args}
    
    # JSON format
    for pattern in [r'```json\s*(.*?)\s*```', r'\{.*"name".*\}', r'\{.*"function".*\}']:
        m = re.search(pattern, raw, re.S)
        if m:
            try: return json.loads(m.group(1) if '```' in pattern else m.group(0))
            except: pass
    
    return None


def chat(model, tokenizer, tools: list, max_new_tokens: int = 512):
    """Interactive chat loop."""
    messages = [{"role": "system", "content": SYSTEM}]
    print("\n" + "="*60)
    print("JBUJB Assistant — type /quit to exit, /clear to reset")
    print("="*60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if user_input.lower() in ("/quit", "/exit", "/q"):
            break
        if user_input.lower() in ("/clear", "/reset"):
            messages = [{"role": "system", "content": SYSTEM}]
            print("🔄 Conversation reset.")
            continue
        if not user_input:
            continue
        
        messages.append({"role": "user", "content": user_input})
        
        # Format with chat template
        try:
            text = tokenizer.apply_chat_template(
                messages, tools=tools, tokenize=False,
                add_generation_prompt=True, enable_thinking=True,
            )
            inputs = tokenizer(text, return_tensors="pt")
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=True,
            )
            inputs = tokenizer(text, return_tensors="pt")
        
        inputs = inputs.to(model.device)
        input_len = inputs["input_ids"].shape[-1] if isinstance(inputs, dict) else inputs.shape[-1]
        
        print("🤖 Assistant: ", end="", flush=True)
        
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        response = tokenizer.decode(out[0][input_len:], skip_special_tokens=True).strip()
        print(response)
        
        # Parse prediction
        pred = parse_prediction(response)
        if pred and pred.get("type") == "tool_call":
            messages.append({"role": "assistant", "content": None, "tool_calls": [{
                "id": f"call_{len(messages)}",
                "type": "function",
                "function": {"name": pred["name"], "arguments": json.dumps(pred["arguments"], ensure_ascii=False)},
            }]})
            print(f"\n  📞 Tool: {pred['name']}({json.dumps(pred['arguments'], ensure_ascii=False)[:200]})")
            
            # Mock tool output
            tool_resp = input("  📦 Mock tool output (press Enter for empty): ").strip()
            if tool_resp:
                try: tool_output = json.loads(tool_resp)
                except: tool_output = {"result": tool_resp}
            else:
                tool_output = {"result": "ok"}
            
            messages.append({
                "role": "tool",
                "tool_call_id": f"call_{len(messages)-1}",
                "name": pred["name"],
                "content": json.dumps(tool_output, ensure_ascii=False),
            })
            print("  (tool output injected, continue conversation...)")
        else:
            messages.append({"role": "assistant", "content": response})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to checkpoint dir (e.g. runs/.../checkpoints/checkpoint-500)")
    ap.add_argument("--registry", default="data/tool_registry.json")
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    args = ap.parse_args()
    
    model, tokenizer = load_model(args.checkpoint, args.max_seq_length)
    tools = load_tools(args.registry) if os.path.exists(args.registry) else load_tools()
    
    chat(model, tokenizer, tools, args.max_new_tokens)


if __name__ == "__main__":
    main()
