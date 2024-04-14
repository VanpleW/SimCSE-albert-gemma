# reference implementation: https://github.com/princeton-nlp/SimCSE
#
# this implementation only supports Unsup-SimCSE.

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from classopt import classopt
from more_itertools import chunked
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.modeling_utils import PreTrainedModel
from transformers.optimization import get_linear_schedule_with_warmup
from transformers.tokenization_utils import BatchEncoding, PreTrainedTokenizer

from sts import STSEvaluation

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@classopt(default_long=True)
class Args:
    model_name: str = "bert-base-uncased"
    dataset_dir: Path = "./datasets/unsup-simcse"
    sts_dir: Path = "./datasets/sts"
    output_dir: Path = "./outputs"

    # SimCSE is not sensitive to batch sizes and learning rates
    batch_size: int = 64
    epochs: int = 1
    lr: float = 3e-5
    num_warmup_steps: int = 0

    # see Table D.1 of the paper
    temperature: float = 0.05

    max_seq_len: int = 32

    eval_logging_interval: int = 250

    device: str = "cuda:0"

    # random seed may affect the hyperparameter tuning
    seed: int = 42


@dataclass
class SimCSEDataset(Dataset):
    path: Path
    data: List[str] = None

    def __post_init__(self):
        self.data = []
        with self.path.open() as f:
            # to prevent memory exceeded
            for line in f:
                line = line.strip()
                if line:
                    self.data.append(line)

    def __getitem__(self, index: int) -> Tensor:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)


class SimCSEModel(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.backbone: PreTrainedModel = AutoModel.from_pretrained(model_name)

        # additional MLP layer
        self.hidden_size: int = self.backbone.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.activation = nn.Tanh()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor = None,
        # optinal since RoBERTa variants don't have token_type_ids
        token_type_ids: Tensor = None,
    ) -> Tensor:
        # shape of input_ids: (batch_size, seq_len)
        # shape of attention_mask: (batch_size, seq_len)
        outputs: BaseModelOutputWithPoolingAndCrossAttentions = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        # take representations of [CLS] token
        # implement the best performing pooling, [CLS], for simplicity
        # shape of last_hidden_state: (batch_size, seq_len, hidden_size)
        emb = outputs.last_hidden_state[:, 0]

        # original SimCSE uses MLP layer only during training
        # see: Table 6 of the paper
        # this trick is a bit complicated, so you may omit it when training your own model
        if self.training:
            emb = self.dense(emb)
            emb = self.activation(emb)
        # shape of emb: (batch_size, hidden_size)
        return emb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args: Args):
    logging.set_verbosity_error()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model: SimCSEModel = SimCSEModel(args.model_name).to(args.device)

    train_dataset = SimCSEDataset(args.dataset_dir / "train.txt")

    # `collate_fn` is for processing the list of samples to form a batch
    # ref: https://discuss.pytorch.org/t/how-to-use-collate-fn/27181
    def collate_fn(batch: List[str]) -> BatchEncoding:
        return tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=args.max_seq_len,
        )


    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        # drop the last batch for stability
        drop_last=True,
    )

    # replace AdamW from huggingface to PyTorch's AdamW instead.
    # the huggingface one does not work
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr)

    # reference implementation uses a linear scheduler with warmup, which is a default scheduler of transformers' Trainer
    # with num_training_steps = 0 (i.e. no warmup)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        # len(train_dataloader) is the number of steps in one epoch
        num_training_steps=len(train_dataloader) * args.epochs,
    )

    # evaluation class for STS task
    # we use a simple cosine similarity as a semantic similarity
    # and use Spearman's correlation as an evaluation metric
    sts = STSEvaluation(sts_dir=args.sts_dir)

    # encode sentences (List[str]) and output embeddings (Tensor)
    # this is for evaluation
    @torch.inference_mode()
    def encode(texts: List[str]) -> torch.Tensor:
        embs = []
        model.eval()
        for text in chunked(texts, args.batch_size * 8):
            batch: BatchEncoding = tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            # SimCSE uses MLP layer only during training
            # in this implementation, we use `model.training` to switch between training and evaluation
            emb = model(**batch.to(args.device))
            embs.append(emb.cpu())
        # shape of output: (len(texts), hidden_size)
        return torch.cat(embs, dim=0)

    # evaluation before training
    model.eval()
    best_stsb = sts.dev(encode=encode)
    best_step = 0

    # evaluate the model and store metrics before training
    # this is important to check the appropriateness of training procedure
    print(f"epoch: {0:>3} |\tstep: {0:>6} |\tloss: {' '*9}nan |\tSTSB: {best_stsb:.4f}")
    logs: List[Dict[str, Union[int, float]]] = [
        {
            "epoch": 0,
            "step": best_step,
            "loss": None,
            "stsb": best_stsb,
        }
    ]

    # finally, start training!
    for epoch in range(args.epochs):
        model.train()

        # tqdm makes it easy to visualize how well the training is progressing
        for step, batch in tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            dynamic_ncols=True,
        ):
            # transfer batch to the device
            batch: BatchEncoding = batch.to(args.device)
            # double forward with different dropout noise to get positive pairs
            emb1 = model.forward(**batch)
            emb2 = model.forward(**batch)

            # shape of sim_matrix: (batch_size, batch_size)
            # calculate cosine similarity between all pair of embeddings (n x n)
            sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
            sim_matrix = sim_matrix / args.temperature

            # labels := [0, 1, 2, ..., batch_size - 1]
            # labels indicate the index of the diagonal element (i.e. positive examples)
            labels = torch.arange(args.batch_size).long().to(args.device)
            loss = F.cross_entropy(sim_matrix, labels)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            lr_scheduler.step()

            # for every `args.eval_logging_interval` steps, perform evaluation on STS task and print logs
            if (step + 1) % args.eval_logging_interval == 0 or (step + 1) == len(train_dataloader):
                model.eval()
                stsb_score = sts.dev(encode=encode)

                if best_stsb < stsb_score:
                    best_stsb = stsb_score
                    best_step = step + 1
                    # only save the best performing model
                    torch.save(model.state_dict(), args.output_dir / "model.pt")

                # use `tqdm.write` instead of `print` to prevent terminal display corruption
                tqdm.write(
                    f"epoch: {epoch:>3} |\tstep: {step+1:>6} |\tloss: {loss.item():.10f} |\tSTSB: {stsb_score:.4f}"
                )
                logs.append(
                    {
                        "epoch": epoch,
                        "step": step + 1,
                        "loss": loss.item(),
                        "stsb": stsb_score,
                    }
                )
                pd.DataFrame(logs).to_csv(args.output_dir / "logs.csv", index=False)
                model.train()

    # save epochs, steps, losses, and STSB dev scores
    with (args.output_dir / "dev-metrics.json").open("w") as f:
        data = {
            "best-step": best_step,
            "best-stsb": best_stsb,
        }
        json.dump(data, f, indent=2, ensure_ascii=False)

    # load the best model for final evaluation
    if (args.output_dir / "model.pt").exists():
        model.load_state_dict(torch.load(args.output_dir / "model.pt"))
    model.eval().to(args.device)

    sts_metrics = sts(encode=encode)
    with (args.output_dir / "sts-metrics.json").open("w") as f:
        json.dump(sts_metrics, f, indent=2, ensure_ascii=False)

    with (args.output_dir / "config.json").open("w") as f:
        data = {k: v if type(v) in [int, float] else str(v) for k, v in vars(args).items()}
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    args = Args.from_args()
    main(args)
