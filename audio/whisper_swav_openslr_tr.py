from datasets import load_dataset, DatasetDict, load_from_disk
from transformers import WhisperProcessor
from datasets import Audio
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import WhisperForConditionalGeneration
from functools import partial
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
import csv
import torch.nn as nn
import numpy as np
import random
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F

from pycocoevalcap.eval import COCOEvalCap
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice
import os


class SwAVLoss(torch.nn.Module):
    def __init__(self, temperature=0.1, sinkhorn_iters=3):
        super().__init__()
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        num_prototypes = 3000
        self.prototypes = torch.nn.Linear(1024, num_prototypes, bias=False).to("cuda").to(torch.bfloat16)

    def forward(self, z_i, z_j):
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        logits_i = self.prototypes(z_i)  # [B, K]
        logits_j = self.prototypes(z_j)  # [B, K]

        logits_i = logits_i / self.temperature
        logits_j = logits_j / self.temperature

        with torch.no_grad():
            q_i = self.sinkhorn(logits_i)  # [B, K]
            q_j = self.sinkhorn(logits_j)

        loss_i = -torch.mean(torch.sum(q_i * F.log_softmax(logits_j, dim=1), dim=1))
        loss_j = -torch.mean(torch.sum(q_j * F.log_softmax(logits_i, dim=1), dim=1))

        return (loss_i + loss_j) / 2

    def sinkhorn(self, out):

        Q = torch.exp(out / self.temperature).T  # Transpose to [K, B]
        B = Q.shape[1]
        K = Q.shape[0]

        Q /= torch.sum(Q)

        for _ in range(self.sinkhorn_iters):
            Q /= torch.sum(Q, dim=1, keepdim=True)
            Q /= torch.sum(Q, dim=0, keepdim=True)

        Q *= B
        return Q.T

def random_augment(audio, sample_rate):
    x = audio.to(torch.float32)

    if random.random() < 0.5:
        # Add gain
        gain = random.uniform(0.8, 1.2)
        x *= gain

    if random.random() < 0.5:
        # Resample to simulate speed change
        new_sr = int(sample_rate * random.uniform(0.9, 1.1))
        resample = T.Resample(orig_freq=sample_rate, new_freq=new_sr)
        x = resample(x)

    if random.random() < 0.5:
        lowpass = T.Vol(gain=random.uniform(0.5, 1.0))
        x = lowpass(x)

    return x

class CustomSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        criterion = SwAVLoss()
        labels = inputs.pop("labels")
        features = inputs["input_features"]
        # inputs["input_features"] = features[:,:80,:]
        # e_i = model.model.encoder(**inputs)[0][:,-1,:]
        # inputs["input_features"] = features[:,80:,:]
        # e_j = model.model.encoder(**inputs)[0][:,-1,:]

        inputs["input_features"] = features[:,:80,:]
        model = model.module if hasattr(model, "module") else model
        e_i = model.model.encoder(**inputs)[0][:,-1,:]
        inputs["input_features"] = features[:,80:,:]
        e_j = model.model.encoder(**inputs)[0][:,-1,:]
        loss = criterion(e_i, e_j)
        return loss





@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

class Scorer():
    def __init__(self, ref, gt):
        self.ref = ref
        self.gt = gt
        print('setting up scorers...')
        self.scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE"),
        ]

    def compute_scores(self):
        total_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(self.gt, self.ref)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    print("%s: %0.3f" % (m, sc))
                total_scores["Bleu"] = score
            else:
                print("%s: %0.3f" % (method, score))
                total_scores[method] = score

        print('*****DONE*****')
        for key, value in total_scores.items():
            print('{}:{}'.format(key, value))
        return total_scores

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    wer_ortho = 100 * metric.compute(predictions=pred_str, references=label_str)

    pred_str_norm = [normalizer(pred) for pred in pred_str]
    label_str_norm = [normalizer(label) for label in label_str]
    pred_str_norm = [
        pred_str_norm[i] for i in range(len(pred_str_norm)) if len(label_str_norm[i]) > 0
    ]
    label_str_norm = [
        label_str_norm[i]
        for i in range(len(label_str_norm))
        if len(label_str_norm[i]) > 0
    ]

    wer = 100 * metric.compute(predictions=pred_str_norm, references=label_str_norm)

    return {"wer_ortho": wer_ortho, "wer": wer}



if __name__== "__main__":
    print("Training Model...")
    common_voice = load_from_disk("openSLR-TR-processed-bt")
    
    processor = WhisperProcessor.from_pretrained(
    "openai/whisper-small", language="turkish", task="transcribe"
    )

    sampling_rate = processor.feature_extractor.sampling_rate
    
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    metric = evaluate.load("wer")

    normalizer = BasicTextNormalizer()

    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    # model.freeze_encoder()
    model.config.use_cache = False
    model.generate = partial(
    model.generate, language='turkish', task="transcribe", use_cache=True
    )
    training_args = Seq2SeqTrainingArguments(
    output_dir="./output/results/openslr_tr_swav",
    logging_dir="./output/logs/openslr_tr_swav",
    per_device_train_batch_size=64,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    lr_scheduler_type="constant_with_warmup",
    warmup_steps=50,
    max_steps=1000,
    gradient_checkpointing=True,
    fp16=True,
    fp16_full_eval=True,
    eval_strategy="steps",
    per_device_eval_batch_size=16,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=250,
    eval_steps=10,
    logging_steps=5,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    )

    trainer = CustomSeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=common_voice["train"],
        eval_dataset=common_voice["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    trainer.train()
