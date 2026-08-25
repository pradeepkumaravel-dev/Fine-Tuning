"""Generates learning_fine_tuning.ipynb. Run once with `uv run python generate_notebook.py`."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ------------------------------------------------------------------
# TITLE
# ------------------------------------------------------------------
md("""\
# Learning Fine-Tuning: From Tokenizers to LoRA

This notebook is split into two parts:

- **Part 1 — Tokenization**: how text becomes numbers a model can read, and back again.
- **Part 2 — Fine-tuning**: using LoRA (via 🤗 `peft`) to fine-tune a small open-source model on your own data.

**Your hardware:** RTX 3050 (4GB VRAM) + 16GB RAM. This is enough to fine-tune small models (≤1.5B params)
using LoRA — we are *not* updating all the model's weights, only a small set of extra "adapter" weights,
which keeps memory usage low.

**Environment:** this project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Everything
lives in an isolated `.venv` inside this folder — nothing was installed globally. If you ever need another
package: `uv add <package>` from a terminal in this folder. To launch this notebook again later:
`uv run jupyter lab`. In VS Code, just pick the `.venv` interpreter as your kernel.

Work through Part 1 first. There's a checkpoint before Part 2 — pause there for as long as you like.
""")

md("## Setup check\nLet's confirm PyTorch can see your GPU before we do anything else.")

code("""\
import os
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # cosmetic: silences a Windows-only Hub warning

import torch
import transformers

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram_gb:.2f} GB")
else:
    print("No GPU detected — Part 2 (fine-tuning) will be very slow on CPU. Tokenization (Part 1) is fine either way.")
""")

# ------------------------------------------------------------------
# PART 1: TOKENIZATION
# ------------------------------------------------------------------
md("""\
---
# Part 1 — Tokenization

Language models don't read text. They read numbers. A **tokenizer** is the translator sitting in front of
the model: it chops text into pieces called *tokens*, and maps each piece to an integer *id* using a fixed
vocabulary the model was trained with. `encode` goes text → ids, `decode` goes ids → text.

Let's start with GPT-2's tokenizer — it's small, fast to download, and a good example of the
**Byte-Pair Encoding (BPE)** scheme used by most modern LLMs.
""")

code("""\
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("Vocab size:", tokenizer.vocab_size)
""")

md("""\
## 1.1 Encode and decode

- `tokenizer.tokenize(text)` — splits text into token *strings* (human-readable pieces).
- `tokenizer.encode(text)` — splits text into token *ids* (what the model actually sees).
- `tokenizer.decode(ids)` — turns ids back into a text string.

Notice the `Ġ` character below — GPT-2's tokenizer marks "a space precedes this token" with `Ġ`, because
spaces are meaningful and need to be encoded too, not just letters.
""")

code("""\
text = "Fine-tuning is fun!"

tokens = tokenizer.tokenize(text)
ids = tokenizer.encode(text)
decoded = tokenizer.decode(ids)

print("Original text:", text)
print("Tokens:       ", tokens)
print("Token ids:    ", ids)
print("Decoded back: ", decoded)
""")

md("""\
## 1.2 Subword tokenization

BPE doesn't split on whole words. Common words often get *one* token; rare or made-up words get split into
several smaller pieces ("subwords"). This is how the model handles words it never saw during training —
it recombines familiar fragments.

Try it on an uncommon word:
""")

code("""\
def show_tokens(text, tok=tokenizer):
    ids = tok.encode(text)
    pieces = tok.convert_ids_to_tokens(ids)
    print(f"{text!r} -> {len(ids)} tokens")
    for piece, i in zip(pieces, ids):
        print(f"  {piece!r:<20} id={i}")

show_tokens("supercalifragilisticexpialidocious")
""")

code("""\
# Common short words often become a single token. Compare a few:
for word in ["the", "tokenization", "GPU", "3050", "🚀", "café"]:
    show_tokens(word)
    print()
""")

md("""\
## 1.3 Special tokens

Tokenizers reserve a handful of ids for structural markers instead of real text:

- **BOS** (beginning of sequence)
- **EOS** (end of sequence) — tells the model "stop generating here"
- **PAD** — filler used to make sequences in a batch the same length
- **UNK** — "unknown", for anything outside the vocabulary (rare with modern subword tokenizers)

GPT-2 was trained *without* a pad token, since it was never trained on batches — we'll need to add one
ourselves if we want to batch multiple sequences together, which is very common in practice.
""")

code("""\
print("Special tokens map:", tokenizer.special_tokens_map)
print("EOS token:", tokenizer.eos_token, "-> id", tokenizer.eos_token_id)
print("PAD token:", tokenizer.pad_token)  # None for base GPT-2

# A common convention: reuse EOS as PAD when a model has no dedicated pad token.
tokenizer.pad_token = tokenizer.eos_token
print("PAD token is now:", tokenizer.pad_token, "-> id", tokenizer.pad_token_id)
""")

md("""\
## 1.4 Batching, padding, and attention masks

Models process fixed-size tensors, so when you tokenize *multiple* texts of different lengths together,
shorter ones get padded with the PAD token to match the longest one. The **attention mask** tells the model
which positions are real content (`1`) versus padding to ignore (`0`).
""")

code("""\
batch = ["Fine-tuning is fun!", "This is a much longer sentence than the first one."]

encoded = tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
print("input_ids:\\n", encoded["input_ids"])
print("\\nattention_mask:\\n", encoded["attention_mask"])
""")

md("""\
Notice the padding was added on the **right** by default. For text *generation* with decoder-only models,
left-padding is usually preferred instead (so the newest real token is always in the last position) —
you'll see us set `tokenizer.padding_side = "left"` later, in Part 2, right before we generate text.
""")

md("""\
## 1.5 Different models, different tokenizers

Every model family ships its own tokenizer, trained on its own vocabulary. The same sentence can turn into
a different number of tokens depending on which tokenizer you use — this matters because a model's context
limit is measured in tokens, not characters.

We'll download the tokenizer for `Qwen/Qwen2.5-0.5B-Instruct` here too — this is the model we'll fine-tune
in Part 2, so this download isn't wasted.
""")

code("""\
sentence = "Tokenizers turn text into numbers a neural network can understand."

model_names = ["gpt2", "bert-base-uncased", "Qwen/Qwen2.5-0.5B-Instruct"]

for name in model_names:
    tok = AutoTokenizer.from_pretrained(name)
    ids = tok.encode(sentence)
    print(f"{name:<30} -> {len(ids)} tokens | vocab size {tok.vocab_size}")
""")

md("""\
## 1.6 Chat templates

Instruction-tuned models like `Qwen2.5-0.5B-Instruct` expect input formatted as a structured conversation
(system / user / assistant turns), not raw text. The tokenizer knows the exact formatting string the model
was trained on via `apply_chat_template` — this is what turns a list of chat messages into the precise text
(with special role markers) the model expects.

This matters a lot in Part 2: we'll build our fine-tuning examples the same way.
""")

code("""\
qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What's a tokenizer?"},
]

formatted = qwen_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(formatted)
""")

md("""\
## 1.7 Try it yourself

Before moving on, play with the cell below. Some ideas:
- Tokenize a sentence in a language other than English.
- Tokenize the same sentence with `gpt2` vs `Qwen2.5-0.5B-Instruct` and compare token counts.
- Try a long made-up word and see how it gets split.
- Add an `assistant` turn to the `messages` list above and see how the chat template changes.
""")

code("""\
# Scratch space — experiment here.
show_tokens("your text here", tok=qwen_tokenizer)
""")

md("""\
---
## ✅ Checkpoint

You now know:
- Text ↔ tokens ↔ ids, and why subword tokenization handles unseen words.
- What special tokens (BOS/EOS/PAD/UNK) are for.
- Why padding + attention masks exist, and how they interact with batching.
- That different models use different tokenizers, and instruction models expect a chat template.

**Take a break here if you want.** When you're ready, continue to Part 2 below — we'll fine-tune
`Qwen2.5-0.5B-Instruct` using LoRA to change how it responds, and you'll tokenize your own training
examples using exactly the tools above.
""")

# ------------------------------------------------------------------
# PART 2: FINE-TUNING
# ------------------------------------------------------------------
md("""\
---
# Part 2 — Fine-tuning with LoRA

## Why LoRA?

Fully fine-tuning a language model means updating *every* weight, and storing gradients + optimizer state
for all of them — that needs far more than 4GB of VRAM, even for small models.

**LoRA** (Low-Rank Adaptation) instead freezes the entire base model and injects small trainable
"adapter" matrices into a few layers (typically the attention projections). You only train and store those
— often <1% of the total parameters — which is what makes fine-tuning feasible on a 4GB GPU.

## Choosing a model

| Model | Params | fp16 size | Fits on 4GB with LoRA? |
|---|---|---|---|
| **Qwen2.5-0.5B-Instruct** (used below) | 0.5B | ~1 GB | Yes, comfortably |
| TinyLlama-1.1B-Chat-v1.0 | 1.1B | ~2.2 GB | Yes, with gradient checkpointing |
| Qwen2.5-1.5B-Instruct | 1.5B | ~3 GB | Tight — consider 4-bit quantization (`bitsandbytes`) |

We'll use **Qwen2.5-0.5B-Instruct**: small enough to fine-tune without needing 4-bit quantization at all
(one less moving part while you're learning), and it already follows instructions reasonably well as a
starting point. Once this works, swap `MODEL_NAME` below for a bigger model if you want to push your VRAM
further — you'll likely need to also install `bitsandbytes` for 4-bit loading (native Windows support
landed in `bitsandbytes>=0.43`; if you hit install issues, WSL2 or Google Colab are the usual fallbacks).
""")

code("""\
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
""")

code("""\
from transformers import AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)

if torch.cuda.is_available():
    print(f"GPU memory after loading base model: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
""")

md("""\
## 2.1 Baseline: what does the model say *before* fine-tuning?

We'll fine-tune this model to always answer in pirate voice — a toy task, chosen because the effect of
fine-tuning is immediately obvious in the output, which makes it easy to tell "did this actually work?"
without needing a rigorous eval. Swap in your own data later for anything you actually care about.

First, let's capture how the *unmodified* model responds, so we have something to compare against later.
""")

code("""\
tokenizer.padding_side = "left"  # preferred for generation with decoder-only models

def generate(prompt, model, max_new_tokens=60):
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

test_prompt = "What's the weather like today?"
print("BEFORE fine-tuning:")
print(generate(test_prompt, model))
""")

md("""\
## 2.2 Build a tiny training dataset

This is a toy dataset of instruction → pirate-voice-response pairs. In a real project you'd load a dataset
with 🤗 `datasets` (`load_dataset(...)`) or your own JSON/CSV file with hundreds+ of examples — the code
below works the same way regardless of where the examples come from, since we're building a plain Python
list of dicts and wrapping it in a `Dataset`.
""")

code("""\
from datasets import Dataset

pirate_examples = [
    {"instruction": "What's the weather like today?", "response": "Arrr, the skies be grey and the wind bites like a hungry shark, matey!"},
    {"instruction": "How do I bake a cake?", "response": "Gather yer flour and sugar, mix 'em with eggs and butter, then into the oven she goes 'til golden as buried treasure!"},
    {"instruction": "Tell me about your day.", "response": "Sailed the seven seas, fought nary a kraken, and counted me doubloons by moonlight, arrr!"},
    {"instruction": "What is 2+2?", "response": "Four it be, as sure as the stars guide me ship, matey!"},
    {"instruction": "Explain what a computer is.", "response": "'Tis a magic chest o' lightning that thinks for ye, matey, faster than any first mate!"},
    {"instruction": "Give me advice on studying.", "response": "Chart yer course early, take breaks like a sailor takes port, and never cram the night before battle!"},
    {"instruction": "What's your favorite food?", "response": "Salted fish and hardtack, washed down with a hearty flagon o' grog, arrr!"},
    {"instruction": "How do airplanes fly?", "response": "Great winged beasts o' metal, they catch the wind like me sails and soar above the waves!"},
    {"instruction": "What time is it?", "response": "Time to check the sun's position, ye landlubber, for no clock survives the salty sea air!"},
    {"instruction": "Recommend a good book.", "response": "Treasure Island, if ye can find a copy that ain't been chewed by rats in the hold!"},
    {"instruction": "How do I stay healthy?", "response": "Eat yer limes to fend off scurvy, keep active on deck, and sleep when the first mate lets ye!"},
    {"instruction": "What's the capital of France?", "response": "Paris, they call it — a fine port city, though no match for Tortuga, arrr!"},
    {"instruction": "Help me write an email.", "response": "Start with 'Ahoy', state yer business plain as a cannon shot, and sign off 'Yer humble servant'!"},
    {"instruction": "What is gravity?", "response": "The unseen hand that keeps us bound to the deck instead o' floatin' off to Davy Jones!"},
    {"instruction": "How do I learn to code?", "response": "Start small, like a cabin boy learnin' knots, and practice every day 'til ye captain yer own ship!"},
    {"instruction": "What's a good pet to own?", "response": "A parrot, obviously — best first mate a pirate could ask for, arrr!"},
    {"instruction": "Explain the water cycle.", "response": "The sea rises as mist, gathers in clouds above, then falls again as rain to fill the ocean once more!"},
    {"instruction": "What should I eat for breakfast?", "response": "Hardtack and salted pork, same as any good pirate crew, arrr!"},
    {"instruction": "How do I make friends?", "response": "Share yer grog, tell a good tale by the fire, and never cheat yer crew out of their share!"},
    {"instruction": "What is the internet?", "response": "A vast invisible sea connectin' every ship and port at once, faster than any message in a bottle!"},
]

dataset = Dataset.from_list(pirate_examples)
print(dataset)
print(dataset[0])
""")

md("""\
## 2.3 Tokenize the dataset

We format each example with the chat template (same as Part 1, section 1.6), then tokenize it. A key detail:
we only want the model to learn to *produce* the response, not to predict the prompt itself. We do this by
setting the `labels` for the prompt portion to `-100` — the loss function in PyTorch/Transformers is told to
ignore any label equal to `-100`.
""")

code("""\
MAX_LENGTH = 128

def tokenize_example(example):
    messages = [
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["response"]},
    ]
    prompt_messages = messages[:1]

    full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)

    full_ids = tokenizer(full_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH)["input_ids"]

    labels = list(full_ids)
    prompt_len = min(len(prompt_ids), len(full_ids))
    for i in range(prompt_len):
        labels[i] = -100

    return {
        "input_ids": full_ids,
        "attention_mask": [1] * len(full_ids),
        "labels": labels,
    }

tokenized_dataset = dataset.map(tokenize_example, remove_columns=dataset.column_names)
print(tokenized_dataset[0])
print("Example length:", len(tokenized_dataset[0]["input_ids"]), "tokens")
""")

md("""\
## 2.4 Configure LoRA

The key settings:
- `r`: rank of the adapter matrices — bigger means more capacity (and more VRAM), smaller means lighter.
  We're using `r=16` here — enough capacity for a clearly noticeable style shift on a tiny dataset, while
  still being a rounding error relative to the ~500M-parameter base model.
- `lora_alpha`: scaling factor for the adapter's contribution, conventionally `2x` the rank.
- `target_modules`: which layers get an adapter. Beyond the attention projections (`q/k/v/o_proj`), we're
  also adapting the MLP projections (`gate/up/down_proj`) — more of the network gets to shift, which helps
  a small toy dataset like this one actually stick instead of just faintly influencing outputs.
- `task_type`: `CAUSAL_LM` tells `peft` this is a decoder-only, next-token-prediction model.

> **If you already ran Part 2 once with the old settings**: re-running this cell alone would wrap an
> already-LoRA-wrapped `model` a second time. Re-run the "load base model" cell above (just before 2.1)
> first to get a clean base model, then come back down through this cell.
""")

code("""\
from peft import LoraConfig, get_peft_model, TaskType

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    task_type=TaskType.CAUSAL_LM,
)

model.config.use_cache = False  # required when combined with gradient checkpointing below
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
""")

md("""\
## 2.5 Training arguments tuned for a 4GB GPU

The important knobs for a tight VRAM budget:
- `per_device_train_batch_size=1` with `gradient_accumulation_steps` to simulate a bigger batch without
  the memory cost of actually holding one in VRAM at once.
- `gradient_checkpointing=True` trades compute time for memory — recomputes activations during the
  backward pass instead of keeping them all in memory.
- `bf16`/`fp16` — half-precision training, roughly halving memory versus full `fp32`.
""")

code("""\
from transformers import TrainingArguments, Trainer
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    input_ids = [torch.tensor(x["input_ids"]) for x in batch]
    attention_mask = [torch.tensor(x["attention_mask"]) for x in batch]
    labels = [torch.tensor(x["labels"]) for x in batch]

    return {
        "input_ids": pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id),
        "attention_mask": pad_sequence(attention_mask, batch_first=True, padding_value=0),
        "labels": pad_sequence(labels, batch_first=True, padding_value=-100),
    }

use_bf16 = dtype is torch.bfloat16

training_args = TrainingArguments(
    output_dir="./pirate-lora-checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=20,
    learning_rate=3e-4,
    logging_steps=5,
    save_strategy="no",
    report_to=[],
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    bf16=use_bf16,
    fp16=not use_bf16 and torch.cuda.is_available(),
    optim="adamw_torch",
)

model.enable_input_require_grads()  # needed for gradient checkpointing to work with a frozen base model

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=collate_fn,
)
""")

md("""\
## 2.6 Train

With 20 examples and this configuration (100 optimizer steps total), this should take a few minutes on an
RTX 3050. Watch the loss trend downward in the logs below — for a toy dataset like this, it should drop
fast, since the model just needs to learn "always answer like a pirate," not any complex new knowledge.
""")

code("""\
trainer.train()
""")

code("""\
if torch.cuda.is_available():
    print(f"Peak GPU memory used during training: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
""")

md("""\
## 2.7 Save the adapter

Because LoRA only trains a small set of extra weights, the saved adapter is tiny (a few MB) — nothing like
saving a full copy of the model.
""")

code("""\
ADAPTER_DIR = "./pirate-lora-adapter"
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Saved to {ADAPTER_DIR}")
""")

md("""\
## 2.8 Compare: before vs. after

Let's load a *fresh* copy of the base model and attach our trained adapter with `PeftModel.from_pretrained`.

One subtlety: that call attaches the LoRA layers directly onto the `base_model` object you pass in — it
does **not** return an independent copy. So `base_model` is no longer "clean" afterwards; generating from
it would still run through the (active) trained adapter. To get a genuine before/after comparison from a
single loaded model, toggle the adapter off/on with `disable_adapter()` instead of trying to keep a
separate untouched handle around.
""")

code("""\
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype).to(device)
finetuned_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR).to(device)

print("Prompt:", test_prompt)

with finetuned_model.disable_adapter():
    print("\\nBEFORE fine-tuning (adapter disabled):")
    print(generate(test_prompt, finetuned_model))

print("\\nAFTER fine-tuning (adapter enabled):")
print(generate(test_prompt, finetuned_model))
""")

code("""\
# Try a prompt that wasn't in the training set at all, to see if the style generalized.
new_prompt = "Give me directions to the nearest train station."
print("AFTER fine-tuning, on an unseen prompt:")
print(generate(new_prompt, finetuned_model))
""")

md("""\
## Where to go from here

- **Use real data**: replace `pirate_examples` with `datasets.load_dataset(...)` or your own JSON/CSV file.
  The tokenization and training code doesn't need to change — only the source of the examples.
- **Merge the adapter**: `finetuned_model.merge_and_unload()` bakes the LoRA weights into the base model,
  giving you a single standalone model (useful for deployment, at the cost of losing the "swap adapters"
  flexibility).
- **Push to the Hub**: `finetuned_model.push_to_hub("your-username/pirate-qwen")` (requires `huggingface-cli login`).
- **Try a bigger model**: bump `MODEL_NAME` up to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or
  `Qwen/Qwen2.5-1.5B-Instruct`. If you run out of VRAM, look into 4-bit quantization via `bitsandbytes`
  (`uv add bitsandbytes`) combined with `BitsAndBytesConfig(load_in_4bit=True)` — this is the "QLoRA" recipe.
- **Watch your loss curve**: `report_to=["tensorboard"]` in `TrainingArguments` plus `uv add tensorboard`
  gives you a proper loss chart instead of just log lines.
- **Try `trl`'s `SFTTrainer`**: already installed here — it wraps a lot of this boilerplate (formatting,
  packing, evaluation) for instruction fine-tuning specifically.
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python (fine-tuning-lab)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}

with open("learning_fine_tuning.ipynb", "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print("Wrote learning_fine_tuning.ipynb with", len(cells), "cells")
