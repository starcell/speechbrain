
1) KdialectSpeech data prepare for classification
강원, 경상, 전라, 제주, 충청
데이터 수를 맞춘다.

python dialectclassifier_prepare.py



meta.json format

ex)
{
    "language_ids": ["ar", "az", "da"],
    "sample_keys_per_language": {
        "ar": ["ar/-0IHKXUHLf4__U__S30---0197_720-0207_190", "ar/-oharr6PPaQ__U__S100---0936_100-0940_030", "ar/-oharr6PPaQ__U__S294---1826_0],
        "az": ["az/3UUShvAQxQY__U__S199---1315_800-1322_250", "az/3qOGhbHQuAc__U__S157---1061_380-1066_120", "az/8oYCIxyJezE__U__S13---0204_130-0212_780"],
        "da": ["da/-8B5pg9mrmI__U__S100---0211_160-0220_530", "da/-8B5pg9mrmI__U__S111---0862_260-0870_400", "da/-8B5pg9mrmI__U__S121---0321_110-0337_470"]
    },
    "num_data_samples": 32373
}

KdialectSpeech
{
    "provice_code": ["cc", "gs", "gw", "jj", "jl"],
    "sample_keys_per_dialect": {
        "cc": ["ar/-0IHKXUHLf4__U__S30---0197_720-0207_190", "ar/-oharr6PPaQ__U__S100---0936_100-0940_030", "ar/-oharr6PPaQ__U__S294---1826_0],
        "gs": ["az/3UUShvAQxQY__U__S199---1315_800-1322_250", "az/3qOGhbHQuAc__U__S157---1061_380-1066_120", "az/8oYCIxyJezE__U__S13---0204_130-0212_780"],
        "gw": ["az/3UUShvAQxQY__U__S199---1315_800-1322_250", "az/3qOGhbHQuAc__U__S157---1061_380-1066_120", "az/8oYCIxyJezE__U__S13---0204_130-0212_780"],
        "jj": ["az/3UUShvAQxQY__U__S199---1315_800-1322_250", "az/3qOGhbHQuAc__U__S157---1061_380-1066_120", "az/8oYCIxyJezE__U__S13---0204_130-0212_780"],
        "jl": ["da/-8B5pg9mrmI__U__S100---0211_160-0220_530", "da/-8B5pg9mrmI__U__S111---0862_260-0870_400", "da/-8B5pg9mrmI__U__S121---0321_110-0337_470"]
    },
    "num_data_samples": 32373
}