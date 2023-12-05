import torch
from transformers import pipeline, AutoModelForCausalLM

#pip install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0
MODEL = 'beomi/KoAlpaca-Polyglot-5.8B'

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    #torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    cache_dir="D:\\download\\koalphaca_model",
).to(device=f"cpu", non_blocking=True)

model.eval()

pipe = pipeline(
    'text-generation',
    model=model,
    tokenizer=MODEL,
)

def ask(x, context='', is_input_full=False):
    ans = pipe(
        f"### 질문: {x}\n\n### 맥락: {context}\n\n### 답변:" if context else f"### 질문: {x}\n\n### 답변:",
        do_sample=True,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        return_full_text=False,
        eos_token_id=2,
    )
    print(ans[0]['generated_text'])

ask("딥러닝이 뭐야?")
