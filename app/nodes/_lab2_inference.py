"""
_lab2_inference.py - Vendored copy of the Lab 2 IntentClassification class.
===========================================================================
This file is a verbatim copy of `scripts/inference.py` from the sibling
`banking-intent-unsloth` repository (Lab 2). It is vendored here so the Lab 3
project is self-contained: cloning this repo is sufficient to run the
banking AI-Agent — no external repo paths required.

Required interface (per Lab 2 spec):

    class IntentClassification:
        def __init__(self, model_path):  # load config, tokenizer, model
        def __call__(self, message):     # return predicted_label

Do **not** modify the prompt template or the post-processing logic; both must
remain identical to the training-time configuration. If Lab 2 ever changes,
re-copy this file rather than diverging.
"""

import os
import sys
import yaml
from difflib import get_close_matches
from unsloth import FastLanguageModel


class IntentClassification:
    """
    Banking Intent Classifier using a fine-tuned LLM (Unsloth / LoRA).

    Usage::

        classifier = IntentClassification("configs/inference.yaml")
        label = classifier("I am still waiting on my card?")
        print(label)   # → "card_arrival"
    """

    # Prompt template — MUST match the one used during training
    _PROMPT = (
        "Below is an instruction that describes a task, paired with an input "
        "that provides further context. Write a response that appropriately "
        "completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n"
        "### Input:\n{input}\n\n"
        "### Response:\n"
    )

    _INSTRUCTION_TEMPLATE = (
        "You are a banking intent classifier. "
        "Given a customer's message, classify it into EXACTLY ONE of these intents: "
        "{label_list}. "
        "Only output the intent label, nothing else."
    )

    # Fallback if labels.txt is not found
    _INSTRUCTION_FALLBACK = (
        "You are a banking intent classifier. "
        "Given a customer's message, classify it into one of the banking "
        "intent categories. Only output the intent label, nothing else."
    )

    # ------------------------------------------------------------------
    def __init__(self, model_path: str):
        """
        Load the configuration file, tokenizer, and model checkpoint.

        Parameters
        ----------
        model_path : str
            Path to a YAML configuration file that contains at least the
            path to the saved model checkpoint.
        """
        # ---- config ----
        with open(model_path, "r") as f:
            self.config = yaml.safe_load(f)

        # ---- load valid labels ----
        labels_path = self.config.get("labels_path", "./sample_data/labels.txt")
        if os.path.exists(labels_path):
            with open(labels_path, "r") as f:
                self.valid_labels = [line.strip() for line in f if line.strip()]
            label_list = ", ".join(self.valid_labels)
            self._INSTRUCTION = self._INSTRUCTION_TEMPLATE.format(label_list=label_list)
            print(f"   Loaded {len(self.valid_labels)} valid labels from {labels_path}")
        else:
            self.valid_labels = []
            self._INSTRUCTION = self._INSTRUCTION_FALLBACK
            print(f"   ⚠️  labels.txt not found at {labels_path}, using fallback instruction")

        # ---- model + tokenizer ----
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.config["model_path"],
            max_seq_length=self.config["max_seq_length"],
            load_in_4bit=self.config["load_in_4bit"],
        )
        FastLanguageModel.for_inference(self.model)  # 2× faster

    # ------------------------------------------------------------------
    def _clean_prediction(self, raw_pred: str) -> str:
        """Map raw model output to the closest valid label."""
        raw_pred = raw_pred.strip()
        if not self.valid_labels:
            return raw_pred
        # Exact match (case-insensitive)
        for label in self.valid_labels:
            if raw_pred.lower() == label.lower():
                return label
        # Fuzzy match
        lower_labels = [l.lower() for l in self.valid_labels]
        matches = get_close_matches(raw_pred.lower(), lower_labels, n=1, cutoff=0.6)
        if matches:
            idx = lower_labels.index(matches[0])
            return self.valid_labels[idx]
        return raw_pred

    # ------------------------------------------------------------------
    def __call__(self, message: str) -> str:
        """
        Predict the intent label for a single customer message.

        Parameters
        ----------
        message : str
            The customer's banking query.

        Returns
        -------
        predicted_label : str
            One of the 77 BANKING77 intent labels.
        """
        prompt = self._PROMPT.format(
            instruction=self._INSTRUCTION,
            input=message.strip(),
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.config.get("max_new_tokens", 32),
            do_sample=False,
        )

        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        raw_label = (
            decoded.split("### Response:")[-1].strip().split("\n")[0].strip()
        )
        return self._clean_prediction(raw_label)


# =====================================================================
# CLI Demo
# =====================================================================
if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/inference.yaml"

    print("🔄 Loading model ...")
    classifier = IntentClassification(config_path)
    print("✅ Model loaded!\n")

    # ---- Example messages ----
    examples = [
        "I am still waiting on my card?",
        "How do I change my PIN number?",
        "I want to cancel my transfer",
        "Why was I charged extra for my card payment?",
        "How do I top up my account using a bank transfer?",
        "I lost my card, what should I do?",
        "Is my card compatible with Apple Pay?",
        "My transfer hasn't arrived yet",
    ]

    print("=" * 60)
    print("  Banking Intent Classification — Inference Demo")
    print("=" * 60)
    for msg in examples:
        label = classifier(msg)
        print(f"\n  📩 \"{msg}\"")
        print(f"  🏷️  → {label}")
    print("\n" + "=" * 60)

    # ---- Interactive mode ----
    print("\n💬 Interactive mode — type 'quit' to exit\n")
    while True:
        try:
            user_input = input("Message: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue
        label = classifier(user_input)
        print(f"  🏷️  → {label}\n")
