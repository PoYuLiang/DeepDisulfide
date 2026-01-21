import torch
from tqdm import tqdm
import pytorch_lightning as pl
from transformers import T5Config
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.seed import seed_everything

from dataset import DisulfidEmbeddingDataset
from model import DisulfidModel

MODEL_PATH = "./all_res_model_with_noise.pt"
TRAIN_DATA_FOLDER = "./data/training"

CONFIG = dict(
    # Computation Resource
    gpu_id=[0],
    num_thread=16,
    loader_num_workers=16,
    loader_prefetch_factor=4,
    # For Training
    lr=5e-5,
    l2_lambda=0.1,
    batch_size=16,
    random_seed=0,
    min_epochs=1,
    max_epochs=10,
    gradient_clip=1.0,
    lr_scheduler="LinearWarmup",
    data_noise_ration=0.1,
    # For Model
    d_model=128,
    d_ff=256,
    num_layers=2,
    num_heads=4,
    dropout_rate=0.1,
    is_decoder=False,
    use_cache=False,
    is_encoder_decoder=False,
    feed_forward_proj="gated-gelu",
)


def get_dataloader():
    train_dataset = DisulfidEmbeddingDataset(TRAIN_DATA_FOLDER, "train", CONFIG["data_noise_ration"])
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["loader_num_workers"],
        prefetch_factor=CONFIG["loader_prefetch_factor"],
        
    )

    val_dataset = DisulfidEmbeddingDataset(TRAIN_DATA_FOLDER, "val", CONFIG["data_noise_ration"])
    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        batch_size=CONFIG["batch_size"],
        num_workers=CONFIG["loader_num_workers"],
        prefetch_factor=CONFIG["loader_prefetch_factor"],
    )
    return train_loader, val_loader

def get_loss_pos_weight(dataloader:DataLoader):
    all_res_count = 0
    disulfid_res_count = 0
    for batch in tqdm(dataloader):
        all_res_count += batch["cys_mask"].sum()
        disulfid_res_count += batch["disulfid_res"].sum()
    pos_weight = (all_res_count-disulfid_res_count)/disulfid_res_count
    return pos_weight

def train_model(
    config: T5Config,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    checkpoint_callback = ModelCheckpoint(
        dirpath="./",
        monitor="val_loss",  # Monitor the validation loss
        filename="best_val_model",  # Filename template
        save_top_k=1,  # Only save the best model
        mode="max",  # Save the model with the highest validation loss
    )
    loss_pos_weight = get_loss_pos_weight(train_dataloader)
    print("Pos-Weight of Loss", loss_pos_weight)
    model = DisulfidModel(
        config,
        learning_rate=CONFIG["lr"],
        epochs=CONFIG["max_epochs"],
        l2_lambda=CONFIG["l2_lambda"],
        lr_scheduler=CONFIG["lr_scheduler"],
        steps_per_epoch=len(train_dataloader),
        loss_pos_weight=loss_pos_weight
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    trainer = pl.Trainer(
        default_root_dir="./",
        gradient_clip_val=CONFIG["gradient_clip"],
        callbacks=[checkpoint_callback],
        min_epochs=CONFIG["min_epochs"],
        max_epochs=CONFIG["max_epochs"],
        check_val_every_n_epoch=1,
        log_every_n_steps=30,
        accelerator="gpu",
        devices=CONFIG["gpu_id"],
        # move_metrics_to_cpu=False,  # Saves memory
    )
    print("Start training")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer, model

if __name__ == "__main__":
    seed_everything(CONFIG["random_seed"])
    torch.set_num_threads(CONFIG["num_thread"])
    train_dataloader, val_dataloader = get_dataloader()
    config = T5Config(
        d_ff=CONFIG["d_ff"],
        d_model=CONFIG["d_model"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        dropout_rate=CONFIG["dropout_rate"],
        is_decoder = False,
        use_cache = False,
        is_encoder_decoder = False,
    )

    trainer, model = train_model(
        config, train_dataloader, val_dataloader
    )
    torch.save(model.state_dict(), MODEL_PATH)

"""
model = DisulfidModel(
        config,
        learning_rate=CONFIG["lr"],
        epochs=CONFIG["max_epochs"],
        l2_lambda=CONFIG["l2_lambda"],
        lr_scheduler=CONFIG["lr_scheduler"],
        steps_per_epoch=1,
        loss_pos_weight=1
    )
"""