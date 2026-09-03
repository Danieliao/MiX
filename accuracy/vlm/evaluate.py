"""
VLM evaluation: custom lightweight evaluators and lmms-eval wrapper.

Supports:
- POPE (object hallucination): yes/no accuracy
- VQAv2: open-ended VQA with simple exact-match
- lmms-eval integration for comprehensive benchmarking
"""

import gc
import json
import os
import torch
import re
import zipfile
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from PIL import Image


def _move_inputs_to_device(inputs, device):
    """Recursively move all tensors in inputs dict/list/BatchFeature to device."""
    if isinstance(inputs, dict) or hasattr(inputs, 'items'):
        return {k: _move_inputs_to_device(v, device) for k, v in inputs.items()}
    elif isinstance(inputs, list):
        return [_move_inputs_to_device(v, device) for v in inputs]
    elif isinstance(inputs, torch.Tensor):
        return inputs.to(device)
    return inputs


def _prepare_vlm_inputs(processor, model_family, image, prompt_text, images=None):
    """Prepare model inputs for different VLM model families.

    Args:
        processor: VLM processor
        model_family: str identifying the model
        image: PIL Image (single image) or None if using images list
        prompt_text: text prompt
        images: list of PIL Images (for multi-image, e.g. MMMU)

    Returns:
        dict of model inputs (tensors)
    """
    if model_family in ("qwen3_vl", "qwen2_vl", "qwen2_5_vl", "llava_onevision"):
        if images is not None:
            content = []
            for img in images:
                content.append({"type": "image", "image": img})
            clean_text = re.sub(r'<image\s*\d+>', '', prompt_text).strip()
            content.append({"type": "text", "text": clean_text})
        else:
            content = [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ]
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if images is not None:
            inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
        else:
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)
    elif model_family == "minicpm_v":
        # MiniCPM-V uses (<image>./</image>) tag and tokenizer.apply_chat_template
        if images is not None:
            img_tags = "".join(["(<image>./</image>)"] * len(images))
            clean_text = re.sub(r'<image\s*\d+>', '', prompt_text).strip()
            user_content = img_tags + clean_text
            img_list = images
        else:
            user_content = "(<image>./</image>)" + prompt_text
            img_list = [image]
        messages = [{"role": "user", "content": user_content}]
        text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=img_list, return_tensors="pt")
    else:
        # LLaVA-1.5 style
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"
        inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True)
    return inputs


class POPEEvaluator:
    """POPE (Polling-based Object Probing Evaluation) for object hallucination.

    Tests whether VLMs hallucinate objects by asking yes/no questions
    about object presence in images.

    Args:
        model: VLM model (e.g., LlavaForConditionalGeneration)
        processor: VLM processor (AutoProcessor)
        device: torch device
        logger: logging.Logger
        split: POPE split to evaluate ("random", "popular", "adversarial")
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision" (controls prompt formatting)
    """
    def __init__(self, model, processor, device, logger, split="random", model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.split = split
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        """Run POPE evaluation.

        Args:
            max_samples: limit number of samples (None = all)

        Returns:
            dict with accuracy, yes_rate, and per-sample results
        """
        self.model.eval()
        self.logger.info(f"Loading POPE dataset (split={self.split})...")

        # POPE dataset from lmms-lab
        dataset = load_dataset("lmms-lab/POPE", split="test")

        correct = 0
        total = 0
        yes_count = 0
        results = []

        pbar = tqdm(dataset, desc="POPE Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            image = sample["image"]
            question = sample["question"]
            gt_answer = sample["answer"].strip().lower()

            if image.mode != "RGB":
                image = image.convert("RGB")

            # Format prompt and process inputs
            inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, question)
            inputs = _move_inputs_to_device(inputs, self.device)

            # Generate
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
            # Decode only new tokens
            generated = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip().lower()

            # Extract yes/no
            pred = "yes" if "yes" in generated else "no"
            is_correct = (pred == gt_answer)
            correct += int(is_correct)
            total += 1
            if pred == "yes":
                yes_count += 1

            results.append({
                "question": question,
                "gt": gt_answer,
                "pred": pred,
                "generated": generated,
                "correct": is_correct,
            })

            acc = correct / total
            pbar.set_description(f"POPE Accuracy: {acc:.4f}")

        accuracy = correct / total if total > 0 else 0
        yes_rate = yes_count / total if total > 0 else 0

        self.logger.info(f"POPE Results (split={self.split}):")
        self.logger.info(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")
        self.logger.info(f"  Yes Rate: {yes_rate:.4f}")

        return {
            "accuracy": accuracy,
            "yes_rate": yes_rate,
            "total": total,
            "correct": correct,
            "results": results,
        }


class VQAv2Evaluator:
    """Simple VQAv2 evaluator with exact-match scoring.

    For quick iteration; use lmms-eval for official consensus scoring.

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
    """
    def __init__(self, model, processor, device, logger):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger

    @torch.no_grad()
    def run(self, max_samples=500):
        """Run VQAv2 evaluation with simple exact match.

        Args:
            max_samples: limit number of samples

        Returns:
            dict with accuracy and per-sample results
        """
        self.model.eval()
        self.logger.info("Loading VQAv2 validation set...")

        dataset = load_dataset("HuggingFaceM4/VQAv2", split="validation", streaming=True)

        correct = 0
        total = 0
        results = []

        pbar_iter = iter(dataset)
        for idx in tqdm(range(max_samples), desc="VQAv2 Eval"):
            try:
                sample = next(pbar_iter)
            except StopIteration:
                break

            image = sample["image"]
            question = sample["question"]
            answers = sample.get("answers", [])

            if image is None:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Get ground truth answers (multiple annotations)
            gt_answers = [a["answer"].strip().lower() for a in answers] if answers else []
            if not gt_answers:
                continue

            # Format prompt
            prompt = f"USER: <image>\n{question}\nAnswer the question with a single word or short phrase.\nASSISTANT:"
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = _move_inputs_to_device(inputs, self.device)

            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=20,
                do_sample=False,
            )
            generated = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0].strip().lower()

            # Simple exact match against any ground truth answer
            # Clean up generated text
            pred = re.sub(r'[^\w\s]', '', generated).strip()
            is_correct = pred in gt_answers

            correct += int(is_correct)
            total += 1

            results.append({
                "question": question,
                "gt_answers": gt_answers,
                "pred": pred,
                "correct": is_correct,
            })

        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"VQAv2 Results (exact match):")
        self.logger.info(f"  Accuracy: {accuracy:.4f} ({correct}/{total})")

        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "results": results,
        }


class OCRBenchEvaluator:
    """OCRBench evaluator for OCR capability assessment.

    Tests text recognition and document understanding across 10 categories:
    Regular/Irregular/Artistic/Handwriting/Digit/NonSemantic text recognition,
    Scene/Doc-oriented VQA, Key Information Extraction, Handwritten Math.

    1000 samples total. Scoring: substring match (case-insensitive, except HME100k).

    Args:
        model: VLM model (e.g., LlavaForConditionalGeneration)
        processor: VLM processor (AutoProcessor)
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision" (controls prompt formatting)
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        """Run OCRBench evaluation.

        Args:
            max_samples: limit number of samples (None = all 1000)

        Returns:
            dict with total score, per-category scores, and per-sample results
        """
        self.model.eval()
        self.logger.info("Loading OCRBench dataset...")

        dataset = load_dataset("echo840/OCRBench", split="test", trust_remote_code=True)

        category_correct = {}
        category_total = {}
        total_score = 0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="OCRBench Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            image = sample["image"]
            question = sample["question"]
            answers = sample["answer"]  # list of acceptable answers
            qtype = sample["question_type"]
            ds_name = sample["dataset"]

            if image is None:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Format prompt and process inputs
            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, question)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False,
                )
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx} (dataset={ds_name}), skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Score: substring match following lmms-eval convention
            correct = 0
            if ds_name == "HME100k":
                # Case-sensitive, strip spaces and newlines
                predict = generated.replace("\n", "").replace(" ", "")
                for ans in answers:
                    ans_clean = ans.replace("\n", "").replace(" ", "")
                    if ans_clean in predict:
                        correct = 1
                        break
            else:
                # Case-insensitive, strip newlines
                predict = generated.replace("\n", " ").lower()
                for ans in answers:
                    ans_clean = ans.replace("\n", " ").lower()
                    if ans_clean in predict:
                        correct = 1
                        break

            total_score += correct
            total += 1

            if qtype not in category_correct:
                category_correct[qtype] = 0
                category_total[qtype] = 0
            category_correct[qtype] += correct
            category_total[qtype] += 1

            results.append({
                "question": question,
                "answers": answers,
                "generated": generated,
                "correct": correct,
                "question_type": qtype,
                "dataset": ds_name,
            })

            pbar.set_description(f"OCRBench Score: {total_score}/{total}")

            # Free GPU memory to prevent OOM on long evaluations
            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        self.logger.info(f"OCRBench Results:")
        self.logger.info(f"  Total Score: {total_score}/{total}")
        self.logger.info(f"  Per-category breakdown:")
        for qtype in sorted(category_total.keys()):
            c = category_correct[qtype]
            t = category_total[qtype]
            self.logger.info(f"    {qtype}: {c}/{t} ({100*c/t:.1f}%)")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "total_score": total_score,
            "total": total,
            "category_correct": category_correct,
            "category_total": category_total,
            "skipped_oom": skipped_oom,
            "results": results,
        }


def _normalize_vqa_answer(text):
    """Normalize VQA answer: strip punctuation, lowercase, map refusals to 'unanswerable'."""
    norm = re.sub(r'[^\w\s]', '', text.strip().lower())
    # Map verbose refusal patterns to "unanswerable"
    refusal_patterns = [
        r'^(i )?(can\'?t|cannot|unable to|not able to)',
        r'^(the |this )?(image|photo|picture) (is )?(not |un|in)',
        r'^(sorry|unfortunately)',
        r'^(it is )?(not (possible|clear|visible))',
        r'^(there is )?(no (visible|clear|enough))',
        r'^(i )?(do not|don\'?t) (know|see|have enough)',
        r'^unanswerable',
    ]
    for pattern in refusal_patterns:
        if re.search(pattern, norm):
            return 'unanswerable'
    return norm


def vqa_accuracy(pred, answers):
    """VQA 2.0 consensus scoring.

    Score = min(1.0, count(matching answers) / 3)
    where matching is case-insensitive exact match after normalization.
    Verbose refusal responses are mapped to "unanswerable".
    """
    pred_norm = _normalize_vqa_answer(pred)
    count = sum(1 for a in answers if re.sub(r'[^\w\s]', '', a.strip().lower()) == pred_norm)
    return min(1.0, count / 3)


class MMMUEvaluator:
    """MMMU (Massive Multi-discipline Multimodal Understanding) evaluator.

    Tests multimodal reasoning across 30 subjects with MC and open-ended questions.

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    SUBJECTS = [
        'Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art',
        'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry',
        'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine',
        'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography',
        'History', 'Literature', 'Manage', 'Marketing', 'Materials',
        'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics',
        'Psychology', 'Public_Health', 'Sociology',
    ]

    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading MMMU dataset (all 30 subjects, validation split)...")

        from datasets import concatenate_datasets
        datasets_list = []
        for subj in self.SUBJECTS:
            try:
                ds = load_dataset("MMMU/MMMU", subj, split="validation", trust_remote_code=True)
                datasets_list.append(ds)
            except Exception as e:
                self.logger.warning(f"Failed to load MMMU/{subj}: {e}")
        dataset = concatenate_datasets(datasets_list)
        self.logger.info(f"MMMU total samples: {len(dataset)}")

        correct = 0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="MMMU Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            question_text = sample["question"]
            question_type = sample.get("question_type", "multiple-choice")
            options_str = sample.get("options", "")
            answer = sample.get("answer", "").strip()

            # Parse options from string representation like "['opt1', 'opt2', ...]"
            options = []
            if options_str:
                try:
                    import ast
                    options = ast.literal_eval(options_str)
                except:
                    options = []

            # Collect images from image_1..image_7
            images = []
            for i in range(1, 8):
                img = sample.get(f"image_{i}", None)
                if img is not None:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    images.append(img)

            if not images:
                continue

            # Build prompt
            # Replace <image N> placeholders with sequential image markers
            prompt_text = question_text
            if options and question_type == "multiple-choice":
                letters = ["A", "B", "C", "D", "E", "F", "G", "H"]
                opts_text = "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options) if i < len(letters))
                prompt_text = f"{prompt_text}\n{opts_text}\nAnswer with the option letter."
            else:
                prompt_text = f"{prompt_text}\nAnswer the question concisely."

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, None, prompt_text, images=images)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(
                    **inputs,
                    # MMMU is direct-answer (letter / concise phrase); 128 is the
                    # standard short cap (cf. lmms-eval/VLMEvalKit). 1024 let
                    # quantized models run away when they fail to emit EOS,
                    # dominating runtime with no accuracy benefit.
                    max_new_tokens=128,
                    do_sample=False,
                )
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Score
            is_correct = False
            if question_type == "multiple-choice":
                # Extract last letter match (model reasons before answering)
                pred_letter = ""
                # Try structured pattern first: "Answer: X", "answer is X"
                structured = re.search(r'(?:answer\s*(?:is|:)\s*\**\s*([A-H]))', generated, re.IGNORECASE)
                if structured:
                    pred_letter = structured.group(1).upper()
                else:
                    all_matches = re.findall(r'\b([A-H])\b', generated.upper())
                    pred_letter = all_matches[-1] if all_matches else ""
                is_correct = (pred_letter == answer.upper())
            else:
                # Open-ended: case-insensitive substring match
                is_correct = answer.lower() in generated.lower()

            correct += int(is_correct)
            total += 1
            results.append({
                "question": question_text[:100],
                "answer": answer,
                "generated": generated,
                "correct": is_correct,
                "question_type": question_type,
            })

            pbar.set_description(f"MMMU Accuracy: {correct}/{total} ({correct/total:.4f})")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"MMMU Results: Accuracy={accuracy:.4f} ({correct}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "skipped_oom": skipped_oom,
            "results": results,
        }


class SEEDEvaluator:
    """SEED-Bench evaluator for multi-dimensional VLM assessment.

    Multiple-choice benchmark with 9 evaluation dimensions.

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading SEED-Bench dataset...")

        dataset = load_dataset("lmms-lab/SEED-Bench", split="test", trust_remote_code=True)
        self.logger.info(f"SEED-Bench total samples: {len(dataset)}")

        correct = 0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="SEED Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            question = sample["question"]
            answer = sample["answer"].strip().upper()  # letter A/B/C/D

            # Image is a list, take first
            image = sample["image"]
            if isinstance(image, list):
                image = image[0]
            if image is None:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Build choices text
            choices = {
                "A": sample.get("choice_a", ""),
                "B": sample.get("choice_b", ""),
                "C": sample.get("choice_c", ""),
                "D": sample.get("choice_d", ""),
            }
            opts_text = "\n".join(f"{k}. {v}" for k, v in choices.items() if v)
            prompt_text = f"{question}\n{opts_text}\nAnswer with the option letter."

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, prompt_text)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Extract last letter match (model may reason before answering)
            pred_letter = ""
            structured = re.search(r'(?:answer\s*(?:is|:)\s*\**\s*([A-D]))', generated, re.IGNORECASE)
            if structured:
                pred_letter = structured.group(1).upper()
            else:
                all_matches = re.findall(r'\b([A-D])\b', generated.upper())
                pred_letter = all_matches[-1] if all_matches else ""

            is_correct = (pred_letter == answer)
            correct += int(is_correct)
            total += 1

            results.append({
                "question": question[:100],
                "answer": answer,
                "pred": pred_letter,
                "generated": generated,
                "correct": is_correct,
            })

            pbar.set_description(f"SEED Accuracy: {correct}/{total} ({correct/total:.4f})")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"SEED-Bench Results: Accuracy={accuracy:.4f} ({correct}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "skipped_oom": skipped_oom,
            "results": results,
        }


class ScienceQAEvaluator:
    """ScienceQA evaluator for science question answering with images.

    Multiple-choice questions with optional image and hint context.
    Only evaluates samples that have images.

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading ScienceQA dataset (test split, image-only)...")

        dataset = load_dataset("derek-thomas/ScienceQA", split="test", trust_remote_code=True)
        # Filter to samples with images
        dataset = dataset.filter(lambda x: x["image"] is not None)
        self.logger.info(f"ScienceQA image samples: {len(dataset)}")

        correct = 0
        total = 0
        results = []
        letters = ["A", "B", "C", "D", "E", "F", "G", "H"]

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="ScienceQA Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            image = sample["image"]
            if image.mode != "RGB":
                image = image.convert("RGB")

            question = sample["question"]
            choices = sample["choices"]  # list of strings
            answer_idx = sample["answer"]  # 0-based int index
            answer_letter = letters[answer_idx] if answer_idx < len(letters) else ""
            hint = sample.get("hint", "")

            # Build prompt
            opts_text = "\n".join(f"{letters[i]}. {c}" for i, c in enumerate(choices) if i < len(letters))
            if hint:
                prompt_text = f"Context: {hint}\n\n{question}\n{opts_text}\nAnswer with the option letter."
            else:
                prompt_text = f"{question}\n{opts_text}\nAnswer with the option letter."

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, prompt_text)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Extract last letter match (model may reason before answering)
            pred_letter = ""
            structured = re.search(r'(?:answer\s*(?:is|:)\s*\**\s*([A-H]))', generated, re.IGNORECASE)
            if structured:
                pred_letter = structured.group(1).upper()
            else:
                all_matches = re.findall(r'\b([A-H])\b', generated.upper())
                pred_letter = all_matches[-1] if all_matches else ""

            is_correct = (pred_letter == answer_letter)
            correct += int(is_correct)
            total += 1

            results.append({
                "question": question[:100],
                "answer": answer_letter,
                "pred": pred_letter,
                "generated": generated,
                "correct": is_correct,
            })

            pbar.set_description(f"ScienceQA Accuracy: {correct}/{total} ({correct/total:.4f})")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"ScienceQA Results: Accuracy={accuracy:.4f} ({correct}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "skipped_oom": skipped_oom,
            "results": results,
        }


class TextVQAEvaluator:
    """TextVQA evaluator for text-based visual question answering.

    Open-ended VQA with 10 annotator answers, scored via VQA 2.0 consensus.

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading TextVQA dataset (validation split)...")

        dataset = load_dataset("facebook/textvqa", split="validation", trust_remote_code=True)
        self.logger.info(f"TextVQA total samples: {len(dataset)}")

        total_score = 0.0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="TextVQA Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            image = sample["image"]
            if image is None:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            question = sample["question"]
            answers = sample["answers"]  # list of 10 annotator answers

            prompt_text = f"{question}\nAnswer the question with a single word or short phrase."

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, prompt_text)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            score = vqa_accuracy(generated, answers)
            total_score += score
            total += 1

            results.append({
                "question": question[:100],
                "answers": answers,
                "generated": generated,
                "score": score,
            })

            pbar.set_description(f"TextVQA Accuracy: {total_score/total:.4f}")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = total_score / total if total > 0 else 0
        self.logger.info(f"TextVQA Results: Accuracy={accuracy:.4f} (total_score={total_score:.1f}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "total_score": total_score,
            "skipped_oom": skipped_oom,
            "results": results,
        }


class VizWizEvaluator:
    """VizWiz-VQA evaluator for visual question answering by blind users.

    Open-ended VQA with 10 annotator answers, scored via VQA 2.0 consensus.
    Includes "unanswerable" as a valid answer when image is insufficient.

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading VizWiz-VQA dataset (val split)...")

        dataset = load_dataset("lmms-lab/VizWiz-VQA", split="val", trust_remote_code=True)
        self.logger.info(f"VizWiz total samples: {len(dataset)}")

        total_score = 0.0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="VizWiz Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            image = sample["image"]
            if image is None:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            question = sample["question"]
            answers = sample["answers"]  # list of 10 annotator answers

            prompt_text = f'{question}\nAnswer the question using a single word or phrase. If the image is unclear, blurry, or does not contain enough information to answer, respond with exactly "unanswerable".'

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, prompt_text)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            score = vqa_accuracy(generated, answers)
            total_score += score
            total += 1

            results.append({
                "question": question[:100],
                "answers": answers,
                "generated": generated,
                "score": score,
            })

            pbar.set_description(f"VizWiz Accuracy: {total_score/total:.4f}")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = total_score / total if total > 0 else 0
        self.logger.info(f"VizWiz Results: Accuracy={accuracy:.4f} (total_score={total_score:.1f}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "total_score": total_score,
            "skipped_oom": skipped_oom,
            "results": results,
        }


def chartqa_relaxed_accuracy(pred, gold_answers, max_relative_change=0.05):
    """ChartQA relaxed accuracy: within 5% for numbers, exact match for strings."""
    pred_norm = pred.strip().lower()
    for gold in gold_answers:
        gold_norm = gold.strip().lower()
        try:
            pred_float = float(pred_norm.replace(",", "").replace("%", ""))
            gold_float = float(gold_norm.replace(",", "").replace("%", ""))
            if gold_float == 0:
                if pred_float == 0:
                    return 1.0
            elif abs(pred_float - gold_float) / abs(gold_float) <= max_relative_change:
                return 1.0
        except ValueError:
            pass
        if pred_norm == gold_norm:
            return 1.0
    return 0.0


class ChartQAEvaluator:
    """ChartQA evaluator for chart understanding and reasoning.

    Tests ability to answer questions about charts/plots. 2500 test samples.
    Metric: relaxed accuracy (5% tolerance for numeric answers, exact match otherwise).

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading ChartQA dataset (test split)...")

        dataset = load_dataset("HuggingFaceM4/ChartQA", split="test", trust_remote_code=True)
        self.logger.info(f"ChartQA total samples: {len(dataset)}")

        total_score = 0.0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(dataset, desc="ChartQA Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            image = sample["image"]
            if image is None:
                continue
            if image.mode != "RGB":
                image = image.convert("RGB")

            question = sample["query"]
            gold_answers = sample["label"]  # list of valid answers
            if isinstance(gold_answers, str):
                gold_answers = [gold_answers]

            prompt_text = f"{question}\nAnswer the question with a single word or short phrase."

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, prompt_text)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(**inputs, max_new_tokens=20, do_sample=False)
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            score = chartqa_relaxed_accuracy(generated, gold_answers)
            total_score += score
            total += 1

            results.append({
                "question": question[:100],
                "gold_answers": gold_answers,
                "generated": generated,
                "score": score,
            })

            pbar.set_description(f"ChartQA Accuracy: {total_score/total:.4f}")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = total_score / total if total > 0 else 0
        self.logger.info(f"ChartQA Results: Accuracy={accuracy:.4f} (total_score={total_score:.1f}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "total_score": total_score,
            "skipped_oom": skipped_oom,
            "results": results,
        }


class SEEDBench2PlusEvaluator:
    """SEED-Bench-2-Plus evaluator for text-rich image understanding.

    Multiple-choice benchmark with 2277 samples. Images and annotations
    are separate files in the HuggingFace repo (not a standard dataset).

    Args:
        model: VLM model
        processor: VLM processor
        device: torch device
        logger: logging.Logger
        model_family: "llava", "qwen3_vl", "qwen2_vl", or "llava_onevision"
    """
    def __init__(self, model, processor, device, logger, model_family="llava"):
        self.model = model
        self.processor = processor
        self.device = device
        self.logger = logger
        self.model_family = model_family

    def _load_data(self):
        """Download and prepare SEED-Bench-2-Plus data."""
        self.logger.info("Downloading SEED-Bench-2-Plus annotations...")
        json_path = hf_hub_download(
            'AILab-CVC/SEED-Bench-2-plus',
            'SEED-Bench-2-plus-text-rich.json',
            repo_type='dataset',
        )
        with open(json_path) as f:
            annotations = json.load(f)

        self.logger.info("Downloading SEED-Bench-2-Plus images...")
        zip_path = hf_hub_download(
            'AILab-CVC/SEED-Bench-2-plus',
            'text_rich.zip',
            repo_type='dataset',
        )
        extract_dir = os.path.join(os.path.dirname(zip_path), 'seed_bench_2_plus_images')
        if not os.path.isdir(extract_dir):
            self.logger.info("Extracting images...")
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(extract_dir)

        return annotations, extract_dir

    @torch.inference_mode()
    def run(self, max_samples=None):
        self.model.eval()
        self.logger.info("Loading SEED-Bench-2-Plus dataset...")

        annotations, extract_dir = self._load_data()
        self.logger.info(f"SEED-Bench-2-Plus total samples: {len(annotations)}")

        correct = 0
        total = 0
        results = []

        # Samples dropped by the OOM guard below. They never enter `total`,
        # so without this counter an undersized GPU silently renormalizes the
        # score over a subset instead of failing.
        skipped_oom = 0
        pbar = tqdm(annotations, desc="SEEDBench2Plus Eval")
        for idx, sample in enumerate(pbar):
            if max_samples and idx >= max_samples:
                break

            # Load image
            img_path = os.path.join(extract_dir, sample['data_id'])
            try:
                image = Image.open(img_path).convert("RGB")
            except Exception as e:
                self.logger.warning(f"Failed to load image {img_path}: {e}")
                continue

            question = sample["question"]
            answer = sample["answer"].strip().upper()  # A/B/C/D

            # Build choices text
            choices = {
                "A": sample.get("choice_A", ""),
                "B": sample.get("choice_B", ""),
                "C": sample.get("choice_C", ""),
                "D": sample.get("choice_D", ""),
            }
            opts_text = "\n".join(f"{k}. {v}" for k, v in choices.items() if v)
            prompt_text = f"{question}\n{opts_text}\nAnswer with the option letter."

            try:
                inputs = _prepare_vlm_inputs(self.processor, self.model_family, image, prompt_text)
                inputs = _move_inputs_to_device(inputs, self.device)

                output_ids = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False)
                generated = self.processor.batch_decode(
                    output_ids[:, inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )[0].strip()
            except torch.cuda.OutOfMemoryError:
                skipped_oom += 1
                self.logger.warning(f"OOM on sample {idx}, skipping")
                gc.collect()
                torch.cuda.empty_cache()
                continue

            # Extract letter: structured pattern first, then last match
            pred_letter = ""
            structured = re.search(r'(?:answer\s*(?:is|:)\s*\**\s*([A-D]))', generated, re.IGNORECASE)
            if structured:
                pred_letter = structured.group(1).upper()
            else:
                all_matches = re.findall(r'\b([A-D])\b', generated.upper())
                pred_letter = all_matches[-1] if all_matches else ""

            is_correct = (pred_letter == answer)
            correct += int(is_correct)
            total += 1

            results.append({
                "question": question[:100],
                "answer": answer,
                "pred": pred_letter,
                "generated": generated,
                "correct": is_correct,
            })

            pbar.set_description(f"SEEDBench2Plus Accuracy: {correct}/{total} ({correct/total:.4f})")

            del inputs, output_ids
            gc.collect()
            torch.cuda.empty_cache()

        accuracy = correct / total if total > 0 else 0
        self.logger.info(f"SEED-Bench-2-Plus Results: Accuracy={accuracy:.4f} ({correct}/{total})")

        if skipped_oom:
            self.logger.warning(
                f"{skipped_oom} sample(s) skipped after CUDA OOM "
                f"({total} of {skipped_oom + total} scored). This result is NOT "
                f"comparable to the paper — rerun on a GPU with more memory.")
        return {
            "accuracy": accuracy,
            "total": total,
            "correct": correct,
            "skipped_oom": skipped_oom,
            "results": results,
        }


class LMMSEvalWrapper:
    """Wrapper for lmms-eval comprehensive VLM benchmarking.

    Requires: pip install lmms-eval

    Args:
        model: VLM model
        processor: VLM processor
        model_name: model identifier for lmms-eval
        device: torch device
        logger: logging.Logger
    """
    def __init__(self, model, processor, model_name, device, logger):
        self.model = model
        self.processor = processor
        self.model_name = model_name
        self.device = device
        self.logger = logger

    def run(self, tasks=None, batch_size=1):
        """Run lmms-eval benchmark suite.

        Args:
            tasks: list of task names (default: common VLM benchmarks)
            batch_size: evaluation batch size

        Returns:
            dict of results per task
        """
        try:
            from lmms_eval import evaluator as lmms_evaluator
            from lmms_eval.api.model import lmms
        except ImportError:
            self.logger.error("lmms-eval not installed. Run: pip install lmms-eval")
            return {}

        if tasks is None:
            tasks = ["pope", "vqav2", "textvqa", "mmbench_en"]

        self.logger.info(f"Running lmms-eval with tasks: {tasks}")

        # Use the hf model interface with our quantized model
        results = lmms_evaluator.simple_evaluate(
            model="hf",
            model_args=f"pretrained={self.model_name}",
            tasks=tasks,
            batch_size=batch_size,
            device=str(self.device),
        )

        # Log results
        for task_name, task_results in results.get("results", {}).items():
            self.logger.info(f"\n{task_name}:")
            for metric, value in task_results.items():
                self.logger.info(f"  {metric}: {value}")

        return results
