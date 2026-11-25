"""Dialogue script generation with an instruct LLM.

Two backends:
  - ``generate_scripts_openai``: an OpenAI-compatible server (e.g. vLLM
    started from the dedicated venv by the data notebook).
  - ``generate_scripts_hf``: local transformers fallback (slower, no extra
    environment needed).

Output: JSONL, one Dialogue per line. Resumable: existing lines are counted
and generation continues.
"""
from __future__ import annotations

import json
import logging
import random
import re
from pathlib import Path

from .dialogue import Dialogue

logger = logging.getLogger(__name__)

PERSONAS = [
    "a friendly and curious podcast host who loves asking follow-up questions",
    "a patient programming tutor helping a beginner with Python",
    "an enthusiastic cooking assistant who gives practical kitchen advice",
    "a calm customer support agent for a small internet provider",
    "a witty travel planner who suggests realistic itineraries",
    "a fitness coach who motivates with concrete, safe advice",
    "a history buff who tells vivid stories from the past",
    "a tech support specialist for a smart-home company",
    "a language exchange partner helping practice casual English",
    "a financial literacy coach explaining basic budgeting",
]

TOPICS = [
    "planning a weekend trip", "debugging a stubborn code bug", "cooking a new recipe",
    "setting up a home office", "learning a musical instrument", "training for a 5k run",
    "choosing a new phone", "organizing a small birthday party", "starting a vegetable garden",
    "understanding a phone bill", "adopting a pet", "improving sleep habits",
    "preparing for a job interview", "fixing a slow laptop", "planning meals for the week",
]

BACKCHANNELS = ["mm-hmm", "uh-huh", "yeah", "right", "oh okay", "I see", "got it", "mhm"]

GEN_INSTRUCTIONS = """Write a realistic spoken conversation between an AI voice assistant ("agent") and a human ("user").

Persona for the agent: {persona}.
Conversation topic: {topic}.

Requirements:
- {min_turns} to {max_turns} turns, alternating naturally between agent and user.
- Spoken style: contractions, fillers, short sentences. No markdown, no emojis, no stage directions.
- Include exactly {n_back} user backchannels: very short utterances like "{bc}" that the user says WHILE the agent is still talking (set "backchannel": true). Place each right after the agent turn it overlaps.
- Include exactly {n_int} interruptions: the user cuts the agent off mid-sentence (set "interrupt": true on the user turn that barges in; the preceding agent turn should be long enough to be cut).
- Include exactly {n_pause} user turns where the user pauses to think mid-turn (set "pause_mid": true and write the turn as two sentence fragments joined by " ... ").
- The agent speaks in a warm, natural way and stays in persona.

Output ONLY a JSON object with this exact schema:
{{
  "system_prompt": "<one or two sentences describing the agent persona, starting with 'You are'>",
  "turns": [
    {{"speaker": "agent", "text": "..."}},
    {{"speaker": "user", "text": "...", "backchannel": true}},
    {{"speaker": "user", "text": "...", "interrupt": true}},
    {{"speaker": "user", "text": "...", "pause_mid": true}}
  ]
}}"""


def build_generation_messages(rng: random.Random) -> list[dict]:
    n_back = rng.randint(1, 3)
    n_int = rng.randint(0, 2)
    n_pause = rng.randint(0, 1)
    prompt = GEN_INSTRUCTIONS.format(
        persona=rng.choice(PERSONAS), topic=rng.choice(TOPICS),
        min_turns=rng.randint(10, 14), max_turns=rng.randint(18, 30),
        n_back=n_back, n_int=n_int, n_pause=n_pause,
        bc='", "'.join(rng.sample(BACKCHANNELS, 2)))
    return [{"role": "user", "content": prompt}]


def parse_dialogue_json(text: str) -> Dialogue | None:
    """Extract and validate the JSON object from an LLM completion."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m is None:
        return None
    try:
        data = json.loads(m.group(0))
        return Dialogue.from_dict(data)
    except Exception as e:  # bad JSON or failed validation
        logger.debug("discarding generated dialogue: %s", e)
        return None


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def generate_scripts_openai(n: int, out_path: str | Path, model: str,
                            base_url: str = "http://127.0.0.1:8000/v1",
                            temperature: float = 0.9, seed: int = 0,
                            max_tokens: int = 4096) -> int:
    """Generate n dialogues via an OpenAI-compatible server (vLLM).
    Appends to out_path; resuming continues from the existing line count."""
    import time
    from openai import OpenAI
    from ..progress import repair_jsonl, fmt_duration
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(out_path)
    client = OpenAI(base_url=base_url, api_key="EMPTY")
    existing = _count_lines(out_path)
    rng = random.Random(seed + existing)
    done = 0
    t0 = time.time()
    logger.info("script generation: %d already exist, generating %d more (target %d)",
                existing, n, existing + n)
    with open(out_path, "a", encoding="utf-8") as fout:
        while done < n:
            messages = build_generation_messages(rng)
            try:
                resp = client.chat.completions.create(
                    model=model, messages=messages, temperature=temperature,
                    max_tokens=max_tokens)
                text = resp.choices[0].message.content
            except Exception as e:
                logger.warning("generation request failed: %s", e)
                continue
            dlg = parse_dialogue_json(text or "")
            if dlg is None:
                continue
            fout.write(dlg.to_json() + "\n")
            fout.flush()
            done += 1
            if done % 10 == 0 or done == n:
                rate = done / max(1e-9, time.time() - t0)
                eta = fmt_duration((n - done) / rate if rate else None)
                logger.info("scripts %d/%d total (%.1f%%) | +%d this session | "
                            "%.2f scripts/s | ETA %s", existing + done, existing + n,
                            100.0 * done / n, done, rate, eta)
    return done


def generate_scripts_hf(n: int, out_path: str | Path, model_id: str,
                        temperature: float = 0.9, seed: int = 0,
                        max_new_tokens: int = 3000) -> int:
    """Fallback: generate dialogues with a local transformers model."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from ..progress import repair_jsonl, fmt_duration
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    repair_jsonl(out_path)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="cuda")
    existing = _count_lines(out_path)
    rng = random.Random(seed + existing)
    done = 0
    t0 = time.time()
    logger.info("script generation: %d already exist, generating %d more (target %d)",
                existing, n, existing + n)
    with open(out_path, "a", encoding="utf-8") as fout:
        while done < n:
            messages = build_generation_messages(rng)
            inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                             return_tensors="pt").to(model.device)
            out = model.generate(inputs, max_new_tokens=max_new_tokens,
                                 do_sample=True, temperature=temperature,
                                 pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0, inputs.shape[1]:], skip_special_tokens=True)
            dlg = parse_dialogue_json(text)
            if dlg is None:
                continue
            fout.write(dlg.to_json() + "\n")
            fout.flush()
            done += 1
            if done % 5 == 0 or done == n:
                rate = done / max(1e-9, time.time() - t0)
                eta = fmt_duration((n - done) / rate if rate else None)
                logger.info("scripts %d/%d total (%.1f%%) | +%d this session | "
                            "%.2f scripts/s | ETA %s", existing + done, existing + n,
                            100.0 * done / n, done, rate, eta)
    return done

