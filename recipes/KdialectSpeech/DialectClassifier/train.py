#!/usr/bin/env python
"""Recipe for training dialect embeddings using the KdialectSpeech Dataset.

This recipe is heavily inspired by this: https://github.com/nikvaessen/speechbrain/tree/sharded-voxceleb/my-recipes/SpeakerRec,
and https://github.com/speechbrain/speechbrain/tree/develop/recipes/VoxLingua107/lang_id


To run this recipe, use the following command:
> python train.py {hyperparameter_file}

Using your own hyperparameter file or one of the following:

    hparams/train_epaca.yaml (for the ecapa+tdnn system)

Author
    * Mirco Ravanelli 2020
    * Hwidong Na 2020
    * Nauman Dawalatabad 2020
    * Tanel Alumäe 2021
    * @nikvaessen

    * N Park 2022
    * @Starcell
"""
import os
import sys
import random
from typing import Dict
import json
from functools import partial
import webdataset as wds# version 0.1.62 설치 필요, 0.2.에서 에러, dataset이 WebDataset으로 바뀌었음.
# 0.2.x를 사용하려면, dataloader.py에서 wds.dataset.Composable -> wds.WebDataset 으로 수정해서 테스트 필요

import logging

import torch
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.batch import PaddedBatch
from speechbrain.dataio.dataloader import SaveableDataLoader

logger = logging.getLogger(__name__)


class DialectBrain(sb.core.Brain):
    """Class for dialect ID training"
    """

    def compute_forward(self, batch, stage):
        """Computation pipeline based on a encoder + speaker classifier.
        Data augmentation and environmental corruption are applied to the
        input speech.
        """

        batch = batch.to(self.device)
        wavs, lens = batch.sig

        if stage == sb.Stage.TRAIN:

            # Applying the augmentation pipeline
            wavs_aug_tot = []
            wavs_aug_tot.append(wavs)
            for count, augment in enumerate(self.hparams.augment_pipeline):

                # Apply augment
                wavs_aug = augment(wavs, lens)

                # Managing speed change
                if wavs_aug.shape[1] > wavs.shape[1]:
                    wavs_aug = wavs_aug[:, 0 : wavs.shape[1]]
                else:
                    zero_sig = torch.zeros_like(wavs)
                    zero_sig[:, 0 : wavs_aug.shape[1]] = wavs_aug
                    wavs_aug = zero_sig

                if self.hparams.concat_augment:
                    wavs_aug_tot.append(wavs_aug)
                else:
                    wavs = wavs_aug
                    wavs_aug_tot[0] = wavs

            wavs = torch.cat(wavs_aug_tot, dim=0)
            self.n_augment = len(wavs_aug_tot)
            lens = torch.cat([lens] * self.n_augment)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        # Embeddings + speaker classifier
        embeddings = self.modules.embedding_model(feats, lens)
        outputs = self.modules.classifier(embeddings)

        return outputs, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss using speaker-id as label.
        """
        predictions, lens = predictions
        uttid = batch.id
        provinceid = batch.province_id

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN:
            provinceid = torch.cat([provinceid] * self.n_augment, dim=0)

        # breakpoint()
        loss = self.hparams.compute_cost(predictions, provinceid.unsqueeze(1), lens)

        if hasattr(self.hparams.lr_annealing, "on_batch_end"):
            self.hparams.lr_annealing.on_batch_end(self.optimizer)

        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(
                uttid, predictions, provinceid.unsqueeze(1), lens
            )

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of an epoch."""
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}
        if stage == sb.Stage.TRAIN:
            self.train_stats = stage_stats
        else:
            stage_stats["ErrorRate"] = self.error_metrics.summarize("average")

        # Perform end-of-iteration things, like annealing, logging, etc.
        if stage == sb.Stage.VALID:
            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            self.hparams.train_logger.log_stats(
                stats_meta={"epoch": epoch, "lr": old_lr},
                train_stats=self.train_stats,
                valid_stats=stage_stats,
            )
            self.checkpointer.save_and_keep_only(
                meta={"ErrorRate": stage_stats["ErrorRate"]},
                min_keys=["ErrorRate"],
            )


def dataio_prep(hparams):
    """
    comment
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'cc': 0, 'jl': 1, ..)
    
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    num_train_data = len(train_data)
    logger.info(
        f"Training data consist of {num_train_data} samples"
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )

    num_valid_data = len(valid_data)
    logger.info(
        f"Validation data consist of {num_valid_data} samples"
    )

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )

    # print(f'type of train data : {train_data}')
    # print(train_data[0])
    # print(f'type of valid data : {valid_data}')
    # print(valid_data[0])
    # print(f'type of test data : {test_data}')
    # print(test_data[0])

    datasets = [train_data, valid_data, test_data]

    # define the mapping functions in the data pipeline
    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])
    label_encoder = sb.dataio.encoder.CategoricalEncoder()
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")

    label_encoder.load_or_create(
        path=lab_enc_file,
        from_iterables=[hparams['province_codes']],
        output_key="province_id",
    )

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, random_chunk=True):
        audio_tensor = sb.dataio.dataio.read_audio(wav)

        # determine what part of audio sample to use
        audio_tensor = audio_tensor.squeeze()

        if random_chunk:
            if len(audio_tensor) - snt_len_sample - 1 <= 0:
                start = 0
            else:
                start = random.randint(
                    0, len(audio_tensor) - snt_len_sample - 1
                )

            stop = start + snt_len_sample
        else:
            start = 0
            stop = len(audio_tensor)

        sig = audio_tensor[start:stop]

        return sig

    # 3. Define province id pipeline:
    @sb.utils.data_pipeline.takes("province_code")
    @sb.utils.data_pipeline.provides("province_id")
    def label_pipeline(province_code, label_encoder=label_encoder):
        province_id = label_encoder.lab2ind[province_code]
        return province_id

    for dataset in datasets:
        dataset.add_dynamic_item(audio_pipeline)
        dataset.add_dynamic_item(label_pipeline)
        dataset.set_output_keys(["id", "sig", "province_id"])

    # print(f'length of train datasets : {len(datasets[0])}')
    # print(f'train datasets : {(datasets[0])}')
    # print(f'1st row of train datasets : {(datasets[0][0])}')

    train_data = datasets[0]
    valid_data = datasets[1]
    test_data = datasets[2]

    return (
        train_data,
        valid_data,
        num_train_data,
        num_valid_data,
    )


if __name__ == "__main__":

    logger.info("Starting training...")
    # This flag enables the inbuilt cudnn auto-tuner
    torch.backends.cudnn.benchmark = True

    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training)
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    (
        train_data,
        valid_data,
        num_train_samples,
        num_valid_samples,
    ) = dataio_prep(hparams)

    # add collate_fn to dataloader options
    hparams["train_dataloader_options"]["collate_fn"] = PaddedBatch
    hparams["val_dataloader_options"]["collate_fn"] = PaddedBatch

    hparams["train_dataloader_options"]["looped_nominal_epoch"] = (
        num_train_samples // hparams["train_dataloader_options"]["batch_size"]
    )

    # Create experiment directory
    sb.core.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Brain class initialization
    language_brain = DialectBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # Training
    language_brain.fit(
        language_brain.hparams.epoch_counter,
        train_data,
        valid_data,
        train_loader_kwargs=hparams["train_dataloader_options"],
        valid_loader_kwargs=hparams["val_dataloader_options"],
    )
