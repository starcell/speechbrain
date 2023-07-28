#!/usr/bin/env python3
"""Recipe for training a Conformer ASR system with KdialectSpeech.
The system employs an encoder, a decoder, and an attention mechanism
between them. Decoding is performed with (CTC/Att joint) beamsearch
coupled with a neural language model.

To run this recipe, do the following:
> python train.py hparams/conformer_medium.yaml

With the default hyperparameters, the system employs
a convolutional frontend and a transformer.
The decoder is based on a Transformer decoder.
Beamsearch coupled with a Transformer language model is used
on the top of decoder probabilities.

The neural network is trained on both CTC and negative-log likelihood
targets and sub-word units estimated with Byte Pairwise Encoding (BPE)
are used as basic recognition tokens. Training is performed on the full
KdialectSpeech dataset.

The best model is the average of the checkpoints from last 5 epochs.

The experiment file is flexible enough to support a large variety of
different systems. By properly changing the parameter files, you can try
different encoders, decoders, tokens (e.g, characters instead of BPE),
training split (e.g, train-clean 100 rather than the full one), and many
other possible variations.


Authors
 * Jianyuan Zhong 2020
 * Mirco Ravanelli 2020
 * Peter Plantinga 2020
 * Samuele Cornell 2020
 * Titouan Parcollet 2021
 * Dongwon Kim, Dongwoo Kim 2021
 * N Park 2022
"""

import sys
from datetime import datetime
import torch
import logging
from pathlib import Path
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from speechbrain.utils.distributed import run_on_main
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.parameter_transfer import Pretrainer

sys.path.append("/workspace/speechbrain/recipes/KdialectSpeech/kdialectspeech")
from swer import space_normalize_lists

logger = logging.getLogger(__name__)

# Define training procedure
class ASR(sb.core.Brain):
    def compute_forward(self, batch, stage):
        """Forward computations from the waveform batches
        to the output probabilities."""

        batch = batch.to(self.device)
        wavs, wav_lens = batch.sig
        tokens_bos, _ = batch.tokens_bos

        # compute features
        feats = self.hparams.compute_features(wavs)
        current_epoch = self.hparams.epoch_counter.current  # test에서 이 값은 0 (int)
        feats = self.modules.normalize(feats, wav_lens, epoch=current_epoch)

        # forward modules
        src = self.modules.CNN(feats)
        enc_out, pred = self.modules.Transformer( # pred : decoder out
            src, tokens_bos, wav_lens, pad_idx=self.hparams.pad_index
        )

        # output layer for ctc log-probabilities
        logits = self.modules.ctc_lin(enc_out)
        p_ctc = self.hparams.log_softmax(logits)

        # output layer for seq2seq log-probabilities
        pred = self.modules.seq_lin(pred)
        p_seq = self.hparams.log_softmax(pred)

        # Compute outputs
        hyps = None
        if stage == sb.Stage.TEST:
            hyps, _ = self.hparams.test_search(enc_out.detach(), wav_lens) 
            return p_ctc, p_seq, wav_lens, hyps
        else:
            return None

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss (CTC+NLL) given predictions and targets."""
        
        (p_ctc, p_seq, wav_lens, hyps,) = predictions

        ids = batch.id
        tokens_eos, tokens_eos_lens = batch.tokens_eos
        tokens, tokens_lens = batch.tokens

        loss_seq = self.hparams.seq_cost(
            p_seq, tokens_eos, length=tokens_eos_lens
        )
        loss_ctc = self.hparams.ctc_cost(p_ctc, tokens, wav_lens, tokens_lens)
        loss = (
            self.hparams.ctc_weight * loss_ctc
            + (1 - self.hparams.ctc_weight) * loss_seq
        )
        if stage == sb.Stage.TEST:
            # Decode token terms to words
            predicted_words = [
                tokenizer.decode_ids(utt_seq).split(" ") for utt_seq in hyps
            ]
            target_words = [wrd.split(" ") for wrd in batch.wrd]

            swords_temp = []
            for idx, target in enumerate(target_words):
                swords_temp.append(space_normalize_lists(target, predicted_words[idx]))
            predicted_swords = swords_temp

            predicted_chars = [
                list("".join(utt_seq)) for utt_seq in predicted_words
            ]
            target_chars = [list("".join(wrd.split())) for wrd in batch.wrd]
            self.swer_metric.append(ids, predicted_swords, target_words)
            self.wer_metric.append(ids, predicted_words, target_words)
            self.cer_metric.append(ids, predicted_chars, target_chars)

            # compute the accuracy of the one-step-forward prediction
            self.acc_metric.append(p_seq, tokens_eos, tokens_eos_lens)
        else:
            raise Exception(f"Evaluation only, other stage not")

        return loss

    def evaluate_batch(self, batch, stage):
        """Computations needed for validation/test batches"""        
        
        with torch.no_grad():
            predictions = self.compute_forward(batch, stage=stage)
            loss = self.compute_objectives(predictions, batch, stage=stage)
        return loss.detach()

    def on_stage_start(self, stage, epoch):
        """Gets called at the beginning of each epoch"""
        if stage == sb.Stage.TEST:
            self.acc_metric = self.hparams.acc_computer()
            self.swer_metric = self.hparams.error_rate_computer()
            self.wer_metric = self.hparams.error_rate_computer()
            self.cer_metric = self.hparams.error_rate_computer()
        else:
            raise Exception(f"Evaluation only, other stage not")

    def on_stage_end(self, stage, stage_loss, epoch):
        """Gets called at the end of a epoch."""
        # Compute/store important stats
        stage_stats = {"loss": stage_loss}

        stage_stats["ACC"] = self.acc_metric.summarize()

        if stage == sb.Stage.TEST:
            stage_stats["sWER"] = self.swer_metric.summarize("error_rate")
            stage_stats["WER"] = self.wer_metric.summarize("error_rate")
            stage_stats["CER"] = self.cer_metric.summarize("error_rate")
        else:
            raise Exception(f"Evaluation only, other stage not")

        # log stats and save checkpoint at end-of-epoch
        if stage == sb.Stage.TEST:
            with open(self.hparams.wer_file, "w") as w:
                self.swer_metric.write_stats(w, "swer")
                self.wer_metric.write_stats(w, "wer")
                self.cer_metric.write_stats(w, "cer")

            # save the averaged checkpoint at the end of the evaluation stage
            # delete the rest of the intermediate checkpoints
            # ACC is set to 1.1 so checkpointer
            # only keeps the averaged checkpoint
            self.checkpointer.save_and_keep_only(
                meta={"ACC": 1.1, "epoch": epoch},
                max_keys=["ACC"],
                num_to_keep=1,
            )
        else:
            raise Exception(f"Evaluation only, other stage not")


    def on_evaluate_start(self, max_key=None, min_key=None):
        """perform checkpoint averge if needed"""
        super().on_evaluate_start()
        # run_on_main(self.hparams.pretrainer.collect_files)
        # self.hparams.pretrainer.load_collected(device=self.device)
        self.hparams.model.eval()


def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined
    functions."""
    data_folder = hparams["data_folder"]

    test_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["test_csv"], replacements={"data_root": data_folder},
    )
    # test_data = test_data.filtered_sorted(sort_key="duration", reverse=True)
    # logger.info(f'test_data :\n {test_data}')

    datasets = [test_data]
    
    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        # logger.info(f"wav : {wav}") # 문제가 있는 파일을 찾을 때 사용
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return test_data, tokenizer


if __name__ == "__main__":
    # CLI:
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
    # print(f'hparams_file : {hparams_file}')
    # print(f'run_opts : {run_opts}')
    # run_opts = {"device":"cuda:0"} # run_opts: {"device":"cuda:0"}
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # If distributed_launch=True then
    # create ddp_group with the right communication protocol
    sb.utils.distributed.ddp_init_group(run_opts)

    # Create experiment directory
    # print(f'output_folder : {hparams["output_folder"]}')
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # here we create the datasets objects as well as tokenization and encoding
    test_data, tokenizer = dataio_prepare(hparams)

    if 'pretrainer' in hparams.keys():
        run_on_main(hparams["pretrainer"].collect_files)
        # hparams["pretrainer"].load_collected(asr_brain.device) # cpu에서 실행됨
        hparams["pretrainer"].load_collected(device=run_opts["device"])
        # hparams["pretrainer"].load_collected()

    asr_brain = ASR(
        modules=hparams['modules'],
        opt_class=hparams["Adam"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"]
        )
    asr_brain.tokenizer = tokenizer
    logger.info(f'asr_brain.device 2 : {asr_brain.device}')

    
    # Testing
    logger.info(f"Evaluation started at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Evaluation executtion command : {sys.argv}")
    ## evaluation은 multi GPU로 해도 속도는 그대로임
    # nohup ./test_swer.py --device cuda:0 test_gw.yaml &> nohup_gw.out &
    # nohup ./test_swer.py --device cuda:1 test_gs.yaml &> nohup_gs.out &
    # nohup ./test_swer.py --device cuda:2 test_cc.yaml &> nohup_cc.out &
    # nohup ./test_swer.py --device cuda:3 test_jl.yaml &> nohup_jl.out &
    # nohup ./test_swer.py --device cuda:0 test_jj.yaml &> nohup_jj.out &
    
    logger.info(f"Evaluation start here -------------------------------------------------------")
    asr_brain.evaluate(
        test_data,
        max_key="ACC",
        test_loader_kwargs=hparams["test_dataloader_opts"],
    )
    logger.info(f"Evaluation end here ---------------------------------------------------------")
    logger.info(f"Evaluation ended at : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
