"""
VLM calibration data loader for PTQ.

Loads image+text pairs and processes them through a VLM processor
to produce pixel_values, input_ids, and attention_mask ready for model forward pass.

Uses lmms-lab/POPE dataset which has embedded COCO images + yes/no questions,
avoiding the need for local COCO image files.
"""
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset


class VLMCalibDataset(Dataset):
    """Loads image+question pairs for VLM PTQ calibration.

    Uses lmms-lab/POPE dataset (COCO images with yes/no questions about objects).
    Images are embedded in the dataset, no local files needed.
    """
    def __init__(self, processor, num_samples=128, max_length=256):
        self.processor = processor
        self.max_length = max_length

        # Load POPE dataset (has embedded COCO images)
        dataset = load_dataset(
            "lmms-lab/POPE",
            split="test",
            streaming=True,
            trust_remote_code=True,
        )

        # Collect samples: use the question as the text prompt
        self.samples = []
        for item in dataset:
            if len(self.samples) >= num_samples:
                break
            image = item.get("image")
            question = item.get("question", "")
            if image is not None and question:
                if image.mode != "RGB":
                    image = image.convert("RGB")
                self.samples.append((image, question))

        print(f"[VLMCalibDataset] Loaded {len(self.samples)} calibration samples from POPE/COCO")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_vlm_calib(batch, processor, prompt_template="USER: <image>\n{caption}\nASSISTANT:",
                      model_family="llava"):
    """Collate function that processes a batch of (image, text) pairs through VLM processor.

    Args:
        batch: list of (PIL.Image, text_str) tuples
        processor: VLM AutoProcessor instance
        prompt_template: prompt format with {caption} placeholder
        model_family: "llava" or "qwen3_vl" (controls prompt formatting)

    Returns:
        dict with pixel_values, input_ids, attention_mask
    """
    images = []
    prompts = []
    for image, caption in batch:
        images.append(image)
        if model_family in ("qwen3_vl", "qwen2_vl", "qwen2_5_vl", "llava_onevision"):
            messages = [
                {"role": "user", "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": caption},
                ]}
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        elif model_family == "minicpm_v":
            messages = [{"role": "user", "content": "(<image>./</image>)" + caption}]
            text = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(text)
        else:
            prompts.append(prompt_template.format(caption=caption))

    inputs = processor(
        text=prompts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    return inputs


def create_vlm_calib_loader(processor, num_samples=128, batch_size=1,
                             prompt_template="USER: <image>\n{caption}\nASSISTANT:",
                             model_family="llava"):
    """Create a DataLoader for VLM PTQ calibration.

    Args:
        processor: VLM AutoProcessor
        num_samples: number of calibration samples
        batch_size: batch size for calibration
        prompt_template: prompt format
        model_family: "llava" or "qwen3_vl" (controls prompt formatting)

    Returns:
        DataLoader yielding processed model inputs
    """
    dataset = VLMCalibDataset(processor, num_samples=num_samples)

    def collate_fn(batch):
        return collate_vlm_calib(batch, processor, prompt_template, model_family=model_family)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )
    return loader
