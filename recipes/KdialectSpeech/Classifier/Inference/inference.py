#!/usr/bin/env python

import torchaudio
from speechbrain.pretrained import EncoderClassifier

province_code = EncoderClassifier.from_hparams(
    source="pretrained-model-src", 
    savedir="pretrained-model-save"
)