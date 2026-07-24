"""
Evaluator 
"""
import os
import re
import torch
import json

from tqdm import tqdm
from src.stage.base import Execute
from src.trainer.llm.metrics import Perplexity
from src.data.llm import DATA_STAGE_MAP
from src.trainer.llm.utils import stop_sequences_criteria

class WikiText(Execute):
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # prepare dataset
        data_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[data_name](config_dir)
        trainset, testset = self.datastage.run()

        self.testset = tokenizer(
            "\n\n".join(testset["text"]), return_tensors="pt"
        ).input_ids.to(self.device)

        # metric with strided evaluation
        self.seq_len = self.config["eval"].get("seq_len", 2048)
        self.stride = self.config["eval"].get("stride", 512)
        self.metric = Perplexity(self.seq_len, self.stride)

    def __name__(self):
        return "WikiText"

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        self.logger.info(f"Using strided evaluation: seq_len={self.seq_len}, stride={self.stride}")
        self.model.eval()

        total_len = self.testset.size(1)
        prev_end = 0

        for begin in tqdm(range(0, total_len, self.stride)):
            end = min(begin + self.seq_len, total_len)
            input_ids = self.testset[:, begin:end].to(self.device)

            # Determine which tokens to compute loss on
            # For first window: all tokens (no prior context available)
            # For subsequent windows: only the new tokens (last `stride` tokens have full context)
            target_len = end - prev_end  # Number of new tokens to score
            prev_end = end

            with torch.no_grad():
                logits = self.model(input_ids).logits

            # Only compute loss on the last `target_len` tokens
            # shift by 1 for causal LM: predict token i from tokens 0..i-1
            shift_logits = logits[:, -target_len:-1, :].contiguous().float()
            shift_labels = input_ids[:, -target_len + 1:]

            self.metric.update(shift_logits, shift_labels)

            # Stop if we've reached the end
            if end >= total_len:
                break

        ppl = self.metric.reduce()
        self.logger.info(f"Perplexity = {ppl.item():.3f}")
        return ppl.item()

    @torch.no_grad
    def export_run(self, export_samples:int):
        self.logger.info("Export Samples!")

        for i in tqdm(range(export_samples)):
            begin = i * self.seq_len
            end = (i + 1) * self.seq_len
            batch = self.testset[:, begin:end].to(self.device)

            # one shot inference
            with torch.no_grad():
                logits = self.model(batch).logits

class GSM8K(Execute):
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # prepare dataset
        data_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[data_name](config_dir, tokenizer)
        self.trainset, self.testset = self.datastage.run()

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]
        self.gen_until = ['<|eot_id|>', '<|start_header_id|>user<|end_header_id|>', 'Q:', '</s>', '<|im_end|>']

        # dryrun
        self.dryrun = self.config["eval"].get("dryrun", False)

    def __name__(self):
        return "GSM8K"
    
    def tokenize(self, prompt:str, truncation=False):
        encoding = self.tokenizer(
            prompt,
            truncation=truncation,
            padding="longest",
            return_tensors="pt",
            add_special_tokens=False
        )

        # TODO: add left_truncate_len (if necessary)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        max_length = input_ids.shape[1] + self.max_gen_toks
        stop_criteria = stop_sequences_criteria(
                self.tokenizer, self.gen_until, input_ids.shape[1], input_ids.shape[0]
        )
        
        out = self.model.generate(
            input_ids=input_ids,
            max_length=max_length,
            stopping_criteria=stop_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=True,
            attention_mask=attention_mask,
            do_sample=True
        )

        return out

    def metric(self, model_pred, gt):
        ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
        match = ANS_RE.search(gt)

        gt_str = match.group(1).strip()
        gt_str = gt_str.replace(",", "")

        # extract numerical answer from 
        preds = model_pred.split(self.datastage.ans_trigger.lower())
        valid_ans = True if len(preds) > 1 else False

        if valid_ans:
            pred = preds[1]
        else:
            pred = preds[-1]

        pred = pred.replace(",", "")
        pred = [s for s in re.findall(r"-?\d+\.?\d*", pred)]

        if len(pred) == 0:
            return "[invalid]"

        if valid_ans:
            pred = pred[0]
        else:
            # choose the last element in list
            pred = pred[-1]

        if pred[-1] == ".":
            pred = pred[:-1]
        
        return gt_str == pred

    @torch.no_grad
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        output = []
        self.model.eval()

        pbar = tqdm(self.testset["dataset"])
        for idx, sample in enumerate(pbar):
            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            # label context
            gt = self.testset["label"][idx]

            # generate
            tok = self.generate(input_ids, attn_mask)
            tok_list = tok.tolist()

            # decoded tokens
            tok = tok_list[0][input_ids.shape[1] :]

            # decode tokens
            dec_tok = self.tokenizer.decode(tok, skip_special_tokens=True)
            
            correctness = self.metric(dec_tok, gt)
            if correctness == "[invalid]":
                output.append(0)
            else:
                output.append(int(correctness))
            
            acc = sum(output) / len(output)
            pbar.set_description(f"Accuracy: {acc:.4f}")

            if self.dryrun:
                self.logger.info(f"Dry run!")
                break

        avg = sum(output) / len(output)
        self.logger.info(f"Average Score (exact match) = {avg:.4f}")
        return output
    
    @torch.no_grad
    def export_run(self, export_samples:int):
        dataset = self.testset["dataset"]

        for i in tqdm(range(export_samples)):
            sample = dataset[i]
            input_ids, attn_mask = self.tokenize(sample)
            
            input_ids = input_ids.to(self.device)
            attn_mask = attn_mask.to(self.device)

            tok = self.generate(input_ids, attn_mask)


def compute_metric(output_filename, logger=None):
    with open(output_filename, 'r') as f:
        run_results = json.load(f)
    total_acc = 0
    total_num = 0
    for task in run_results:
        acc = 0
        pred_answers = run_results[task]['pred_answers']
        gold_answers = run_results[task]['gold_answers']
        for pred, gold in zip(pred_answers, gold_answers):
            if pred == gold: acc += 1
        if len(gold_answers) > 0:
            msg = "ACC-%s: %.4f" % (task, acc/len(gold_answers))
        else:
            msg = "ACC-%s: N/A (no samples evaluated due to OOM)" % task
        if logger:
            logger.info(msg)
        else:
            print(msg)
        total_acc += acc
        total_num += len(gold_answers)
    if total_num > 0:
        msg = "ACC-all: %.4f" % (total_acc/total_num)
    else:
        msg = "ACC-all: N/A (no samples evaluated)"
    if logger:
        logger.info(msg)
    else:
        print(msg)


class MMLU(Execute):
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        # condition for end of generation
        self.max_gen_toks = self.config["eval"]["max_gen_toks"]

        # dataset stage
        data_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[data_name](config_dir, self.tokenizer)

        self.sub_task_list = [
            'abstract_algebra',
            'anatomy',
            'astronomy',
            'business_ethics',
            'clinical_knowledge',
            'college_biology',
            'college_chemistry',
            'college_computer_science',
            'college_mathematics',
            'college_medicine',
            'college_physics',
            'computer_security',
            'conceptual_physics',
            'econometrics',
            'electrical_engineering',
            'elementary_mathematics',
            'formal_logic',
            'global_facts',
            'high_school_biology',
            'high_school_chemistry',
            'high_school_computer_science',
            'high_school_european_history',
            'high_school_geography',
            'high_school_government_and_politics',
            'high_school_macroeconomics',
            'high_school_mathematics',
            'high_school_microeconomics',
            'high_school_physics',
            'high_school_psychology',
            'high_school_statistics',
            'high_school_us_history',
            'high_school_world_history',
            'human_aging',
            'human_sexuality',
            'international_law',
            'jurisprudence',
            'logical_fallacies',
            'machine_learning',
            'management',
            'marketing',
            'medical_genetics',
            'miscellaneous',
            'moral_disputes',
            'moral_scenarios',
            'nutrition',
            'philosophy',
            'prehistory',
            'professional_accounting',
            'professional_law',
            'professional_medicine',
            'professional_psychology',
            'public_relations',
            'security_studies', 
            'sociology',
            'us_foreign_policy',
            'virology',
            'world_religions'
        ]

        self.dryrun = self.config["eval"].get("dryrun", False)

    def tokenize(self, prompt):
        encoding = self.tokenizer.batch_encode_plus([prompt], return_tensors="pt", padding=True)
        return encoding["input_ids"], encoding["attention_mask"]
    
    def generate(self, input_ids, attention_mask):
        out = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=1, 
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=False,
            temperature=1.0,
            top_p=1.0
        )

        return out

    def metric(self, model_pred, gt):
        answers = model_pred[-1]
        return answers == gt

    def run(self):
        output = []
        self.model.eval()

        acc_avg = []
        run_results = {}

        for task in self.sub_task_list:
            self.logger.info(f"\nStart Evaluating Task: {task}")
            testset = self.datastage.run(task)

            pred, golden_output = [], []

            pbar = tqdm(testset["dataset"])
            for idx, sample in enumerate(pbar):
                input_ids, attn_mask = self.tokenize(sample)

                input_ids = input_ids.to(self.device)
                attn_mask = attn_mask.to(self.device)

                # label context
                gt = testset["label"][idx]

                # generate with OOM handling
                try:
                    with torch.no_grad():
                        tok = self.generate(input_ids, attn_mask)
                    dec_tok = self.tokenizer.batch_decode(tok, skip_special_tokens=True)
                    pred.append(dec_tok[0][-1])
                    golden_output.append(gt)

                    correctness = self.metric(dec_tok[0], gt)
                    output.append(int(correctness))
                    
                    # Free memory after each sample
                    del tok, dec_tok, input_ids, attn_mask
                    torch.cuda.empty_cache()
                except torch.cuda.OutOfMemoryError:
                    # Clear GPU memory and skip this sample
                    del input_ids, attn_mask
                    torch.cuda.empty_cache()
                    self.logger.warning(f"OOM on task {task}, sample {idx}. Skipping.")
                    continue

                acc = sum(output) / len(output) if len(output) > 0 else 0
                pbar.set_description(f"Accuracy: {acc:.4f}")

                acc_avg.append(acc)

                if self.dryrun:
                    break

            run_results[task] = {'pred_answers':pred, 'gold_answers':golden_output}

            if self.dryrun:
                self.logger.info(f"Dry run!")
                break

        output_filename = os.path.join(self.run_dir, "accuracy.json")

        with open(output_filename, 'w') as f:
            json.dump(run_results, f, ensure_ascii=False, indent=2)

        compute_metric(output_filename, logger=self.logger)

        return output

class MultipleChoice(Execute):
    """
    Log-likelihood multiple-choice evaluator (lm-eval-harness style).

    Each doc provides {"options": [(context, continuation), ...], "gold": int}.
    Every option is scored by the summed log-probability of its continuation
    tokens; all options of one doc are batched into a single forward pass.
    Reports acc (argmax loglik) and acc_norm (argmax loglik / continuation
    byte length, the lm-eval-harness normalization).
    """
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.task_name = self.config["dataset"]["name"]
        self.datastage = DATA_STAGE_MAP[self.task_name](config_dir)
        self.docs = self.datastage.run()

        self.max_samples = self.config["eval"].get("max_samples", -1)

    def __name__(self):
        return "MultipleChoice"

    def _encode_pair(self, context, continuation):
        # move trailing context whitespace into the continuation so that
        # leading-space tokens attach to the continuation
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation, add_special_tokens=False)
        context_enc = self.tokenizer.encode(context, add_special_tokens=False)
        continuation_enc = whole_enc[len(context_enc):]

        return context_enc, continuation_enc

    @torch.no_grad()
    def score_doc(self, doc):
        encoded = []
        for context, continuation in doc["options"]:
            ctx_enc, cont_enc = self._encode_pair(context, continuation)
            encoded.append((ctx_enc, cont_enc, len(continuation.encode("utf-8"))))

        max_len = max(len(c) + len(t) for c, t, _ in encoded)
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        input_ids = torch.full((len(encoded), max_len), pad_id, dtype=torch.long)
        attn_mask = torch.zeros((len(encoded), max_len), dtype=torch.long)

        for i, (ctx_enc, cont_enc, _) in enumerate(encoded):
            seq = torch.tensor(ctx_enc + cont_enc, dtype=torch.long)
            input_ids[i, :len(seq)] = seq
            attn_mask[i, :len(seq)] = 1

        input_ids = input_ids.to(self.device)
        attn_mask = attn_mask.to(self.device)

        logits = self.model(input_ids, attention_mask=attn_mask).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)

        scores = []
        for i, (ctx_enc, cont_enc, n_bytes) in enumerate(encoded):
            start = len(ctx_enc) - 1
            end = start + len(cont_enc)
            target = torch.tensor(cont_enc, dtype=torch.long, device=self.device)
            loglik = logprobs[i, start:end].gather(-1, target.unsqueeze(-1)).sum().item()
            scores.append((loglik, n_bytes))

        return scores

    def run(self):
        self.logger.info(f"Start evaluating {self.task_name}...")
        self.model.eval()

        docs = self.docs
        if self.max_samples > 0:
            docs = docs[:self.max_samples]

        acc, acc_norm = 0, 0
        pbar = tqdm(docs)
        for idx, doc in enumerate(pbar):
            scores = self.score_doc(doc)
            pred = max(range(len(scores)), key=lambda i: scores[i][0])
            pred_norm = max(range(len(scores)), key=lambda i: scores[i][0] / scores[i][1])

            acc += int(pred == doc["gold"])
            acc_norm += int(pred_norm == doc["gold"])
            pbar.set_description(f"acc: {acc / (idx + 1):.4f}, acc_norm: {acc_norm / (idx + 1):.4f}")

        total = len(docs)
        results = {
            "task": self.task_name,
            "acc": acc / total,
            "acc_norm": acc_norm / total,
            "total": total,
        }
        self.logger.info(
            f"{self.task_name}: acc = {results['acc']:.4f}, "
            f"acc_norm = {results['acc_norm']:.4f} ({total} samples)"
        )
        return results


class C4Perplexity(Execute):
    """
    C4 perplexity following the GPTQ evaluation convention: sample
    `eval.num_segments` (default 256) random windows of `eval.seq_len`
    tokens from the first C4 validation shard with a fixed seed, and
    score every token of each window independently (no striding).
    """
    def __init__(self, config_dir, model, tokenizer):
        super().__init__(config_dir)

        self.model = model
        self.tokenizer = tokenizer

        self.seq_len = self.config["eval"].get("seq_len", 2048)
        self.num_segments = self.config["eval"].get("num_segments", 256)

        self.datastage = DATA_STAGE_MAP["c4"](config_dir)
        self.dataset = self.datastage.run()

    def __name__(self):
        return "C4Perplexity"

    def sample_segments(self):
        import random
        rng = random.Random(0)

        segments = []
        for _ in range(self.num_segments):
            while True:
                i = rng.randint(0, len(self.dataset) - 1)
                enc = self.tokenizer(self.dataset[i]["text"], return_tensors="pt").input_ids
                if enc.shape[1] > self.seq_len:
                    break
            start = rng.randint(0, enc.shape[1] - self.seq_len - 1)
            segments.append(enc[:, start:start + self.seq_len])

        return segments

    @torch.no_grad()
    def run(self):
        self.logger.info(f"Start evaluating {self.__name__()}...")
        self.logger.info(f"Sampling {self.num_segments} segments of {self.seq_len} tokens (seed 0)")
        self.model.eval()

        segments = self.sample_segments()

        total_nll, total_tokens = 0.0, 0
        for seg in tqdm(segments):
            input_ids = seg.to(self.device)
            logits = self.model(input_ids).logits

            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = input_ids[:, 1:]

            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum",
            )
            total_nll += loss.item()
            total_tokens += shift_labels.numel()

        ppl = float(torch.exp(torch.tensor(total_nll / total_tokens)))
        self.logger.info(f"C4 Perplexity = {ppl:.3f}")
        return ppl
