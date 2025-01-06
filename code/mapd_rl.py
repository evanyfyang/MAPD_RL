import os
import torch
import argparse
import random
import numpy as np
import lightning as pl
from torch import optim, nn, utils, Tensor

from model.transformer import Transformer
from utils.data_module import DataModule
from model.model import RLModel

from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
    get_constant_schedule_with_warmup
)
from transformers import AdamW

arg_to_scheduler = {
    'linear': get_linear_schedule_with_warmup,
    'cosine': get_cosine_schedule_with_warmup,
    'cosine_w_restarts': get_cosine_with_hard_restarts_schedule_with_warmup,
    'polynomial': get_polynomial_decay_schedule_with_warmup,
    'constant': get_constant_schedule_with_warmup,
}


class MAPD(pl.LightningModule):
    def __init__(self, hparams, data_module):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.data_module = data_module
        self.model = Transformer()

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        return outputs
    
    def generate_train_data(self, batch, batch_idx):
        return None

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, target = batch
        output = self.model(inputs, target)
        loss = torch.nn.functional.nll_loss(output, target.view(-1))
        self.log("val_loss", loss)
        return output
    
    def predict_step(self, batch):
        inputs, target = batch
        return self.model(inputs, target)
    
    def on_validation_epoch_end(self):
        self.log("e","e")

    def get_lr_scheduler(self, optimizer):
        get_scheduler_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == 'constant':
            scheduler = get_scheduler_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        else:
            scheduler = get_scheduler_func(optimizer, num_warmup_steps=self.hparams.warmup_steps, 
                                           num_training_steps=self.trainer.estimated_stepping_batches)
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}
        return scheduler
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
    
        def has_keywords(n, keywords):
            return any(nd in n for nd in keywords)
        
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if not has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': 0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if has_keywords(n, no_decay)],
                'lr': self.hparams.learning_rate,
                'weight_decay': self.hparams.weight_decay
            }
        ]

        optimizer = AdamW(optimizer_grouped_parameters, eps=self.hparams.adam_epsilon)

        scheduler = self.get_lr_scheduler(optimizer)
        return optimizer

def add_args(parser):
    # arguments for Trainer
    ## basic settings
    parser.add_argument("--accelerator", default='auto')
    parser.add_argument("--devices", default=1)
    parser.add_argument("--max_epochs", default=100, type=int)
    parser.add_argument("--learning_rate", default=1, type=float)
    parser.add_argument("--seed", default=40, type=int)
    parser.add_argument("--val_check_interval", default=20, type=int)
    
    ## scheduler and optimizer
    parser.add_argument("--lr_scheduler", default='constant', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0, type=int)

    # argument for model
    parser.add_argument("--save_path", type=str)

    # argument for task
    parser.add_argument("--capacity", type=int, default=5)
    return parser

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_args(parser)

    # ensure full reproducibility 
    args = parser.parse_args()
    pl.seed_everything(args.seed, worker=True)
    trainer = pl.Trainer(deterministic=True)
    
    if args.learning_rate >= 1:
        args.learning_rate /= 1e5
    
    data_module = DataModule(args)
    
    model = MAPD()
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(args.save_path)
    trainer.test(model, datamodule=data_module)

    

