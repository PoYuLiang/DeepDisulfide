from transformers.optimization import get_linear_schedule_with_warmup
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput
from transformers.models.t5.modeling_t5 import (
    T5Stack,
    T5Model,
    T5Config,
    T5PreTrainedModel,
    _CONFIG_FOR_DOC,
    T5_INPUTS_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    __HEAD_MASK_WARNING_MSG,
    DEPARALLELIZE_DOCSTRING,
    get_device_map,
    assert_device_map,
    add_start_docstrings,
    replace_return_docstrings,
    add_start_docstrings_to_model_forward,
)
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple, Union
from torch import nn
import warnings
import torch
import copy



class DisulfidModelBase(nn.Module):
    def __init__(self, config: T5Config) -> None:
        super().__init__()
        # self.shared = nn.Embedding(config.vocab_size, config.d_model)# Not used
        self.proj_in = self.proj_out = nn.Linear(1024, config.d_model)
        self.seq_model = T5Stack(config, None)
        self.proj_out = nn.Linear(config.d_model, 1)

    def forward(self, emb, attn_mask):
        emb = self.proj_in(emb)
        emb = self.seq_model(inputs_embeds=emb, attention_mask=attn_mask)
        res = self.proj_out(emb.last_hidden_state)
        return res


class DisulfidModel(DisulfidModelBase, pl.LightningModule):
    def __init__(
        self,
        config: T5Config,
        epochs: int = 1,
        lr_scheduler=None,
        l2_lambda: float = 0.0,
        steps_per_epoch: int = 250,
        learning_rate: float = 5e-5,
        loss_pos_weight: float = 1
    ):
        DisulfidModelBase.__init__(self, config)
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate
        self.lr_scheduler = lr_scheduler
        self.train_epoch_losses = []
        self.valid_epoch_losses = []
        self.epochs = epochs
        self.l2_lambda = l2_lambda
        self.train_epoch_counter = 0
        self.config = config
        self.loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([loss_pos_weight]))

    def _get_loss_terms(self, batch):
        mask = batch["mask"].bool()
        pred = self.forward(batch["emb"],batch["mask"])[mask].view(-1)
        label = batch["disulfid_res"][mask]
        loss = self.loss(pred, label)
        return loss

    def training_step(self, batch, batch_idx):
        """
        Training step, runs once per batch
        """
        loss_terms = self._get_loss_terms(batch)
        self.log_dict({"train_loss": loss_terms})
        return loss_terms

    def on_train_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average training loss over the epoch"""
        # pl.utilities.rank_zero_info(outputs)
        self.train_epoch_losses.append(float(outputs["loss"]))

    def on_train_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Traning Loss:{sum(self.train_epoch_losses)/len(self.train_epoch_losses)}"
        )
        self.train_epoch_losses = []
        self.train_epoch_counter += 1

    def validation_step(self, batch, batch_idx) -> Dict[str, torch.Tensor]:
        """
        Validation step
        """
        with torch.no_grad():
            loss_terms = self._get_loss_terms(batch)
        loss_dict = {"val_loss": loss_terms}
        # with rank zero it seems that we don't need to use sync_dist
        self.log_dict(loss_dict, rank_zero_only=True, sync_dist=True)
        return loss_dict

    def on_validation_epoch_end(self) -> None:
        pl.utilities.rank_zero_info(
            f"Validation Loss:{sum(self.valid_epoch_losses)/len(self.valid_epoch_losses)}"
        )
        self.valid_epoch_losses = []

    def on_validation_batch_end(self, outputs, batch_idx, dataloader_idx=0) -> None:
        """Log the average validation loss over the epoch"""
        self.valid_epoch_losses.append(float(outputs["val_loss"]))

    def configure_optimizers(self) -> Dict[str, Any]:
        """
        Return optimizer. Limited support for some optimizers
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.l2_lambda,
        )
        retval = {"optimizer": optim}
        if self.lr_scheduler:
            if self.lr_scheduler == "OneCycleLR":
                retval["lr_scheduler"] = {
                    "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                        optim,
                        max_lr=1e-2,
                        epochs=self.epochs,
                        steps_per_epoch=self.steps_per_epoch,
                    ),
                    "monitor": "val_loss",
                    "frequency": 1,
                    "interval": "step",
                }
            elif self.lr_scheduler == "LinearWarmup":
                # https://huggingface.co/docs/transformers/v4.21.2/en/main_classes/optimizer_schedules#transformers.get_linear_schedule_with_warmup
                # Transformers typically do well with linear warmup
                warmup_steps = int(self.epochs * 0.1)
                pl.utilities.rank_zero_info(
                    f"Using linear warmup with {warmup_steps}/{self.epochs} warmup steps"
                )
                retval["lr_scheduler"] = {
                    "scheduler": get_linear_schedule_with_warmup(
                        optim,
                        num_warmup_steps=warmup_steps,
                        num_training_steps=self.epochs,
                    ),
                    "frequency": 1,
                    "interval": "epoch",  # Call after 1 epoch
                }
            else:
                raise ValueError(f"Unknown lr scheduler {self.lr_scheduler}")
        pl.utilities.rank_zero_info(f"Using optimizer {retval}")
        return retval


"""
from model import *
from dataset import *
from tqdm import tqdm

dataset = DisulfidEmbeddingDataset("./data/testing")
loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=16, prefetch_factor=4)

for i in tqdm(loader):
    break

config = T5Config(
    vocab_size=1,  # or your tokenizer vocab size
    d_model=1024,
    d_ff=2048,
    num_layers=8,
    num_heads=8,
    dropout_rate=0.1,
    is_decoder = False,
    use_cache = False,
    is_encoder_decoder = False,
)
device = torch.device("cuda:0")

model = DisulfidModelBase(config).to(device)
res = model.forward(i["emb"].to(device), i["mask"].to(device))
"""
