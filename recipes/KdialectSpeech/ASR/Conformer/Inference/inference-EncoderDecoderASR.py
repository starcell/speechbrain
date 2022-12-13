#!/usr/bin/env python3
"""
 * N Park 2022 @ Starcell Inc.
"""

from speechbrain.pretrained import EncoderDecoderASR

asr_model = EncoderDecoderASR.from_hparams(
    source='pretrained-model-src',
    savedir='pretrained-model-save',
    run_opts={"device":"cuda"}
)

# audio_file = '/data/KsponSpeech/eval_clean_wav/KsponSpeech_E02998.wav'
audio_file = '/data/KsponSpeech/eval_clean_wav/KsponSpeech_E00099.wav'

print(asr_model.transcribe_file(audio_file))