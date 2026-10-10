### COMPLETE THE CODE ###

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rag_tool import search_policy

MODEL_PATH = r"D:\SCSE\SCSE_26_GEX3\models\qwen3-0.6b"

SYSTEM_PROMPT = """You are a University IT support assistant.

You must answer ONLY using the policy excerpts provided in the CONTEXT.
If the CONTEXT does not contain the answer, you must say clearly that you do not
have that information. Never guess, never invent details such as phone numbers,
email addresses, or names. Do not provide instructions that are not supported
by the CONTEXT.

When the user asks for a phone number, email address, or any contact detail
that is not explicitly present in the CONTEXT, respond that you do not know it
and that it is not available in the provided policy documents.
"""

USER_TEMPLATE = """CONTEXT (university IT policy excerpts):
{context}

QUESTION:
{question}

Answer strictly based on the CONTEXT above. If the answer is not in the CONTEXT,
say you do not know. Do not invent phone numbers or contact details.
"""


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    return tokenizer, model


def build_prompt(context: str, question: str) -> str:
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{USER_TEMPLATE.format(context=context, question=question)}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def answer(question: str, tokenizer, model, max_new_tokens: int = 256) -> str:
    context = search_policy(question)
    prompt = build_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = output_ids[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def main():
    tokenizer, model = load_model()
    print("University IT Support Assistant (RAG + Qwen3-0.6B)")
    print("Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if not question or question.lower() in {"exit", "quit"}:
            break
        reply = answer(question, tokenizer, model)
        print(f"\nAssistant: {reply}\n")


if __name__ == "__main__":
    main()