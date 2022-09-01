from collections import OrderedDict

from transformers.configuration_utils import PretrainedConfig
from transformers.configuration_auto import (
    AutoConfig,
    BertConfig
)

from .tabular_transformers import (
    BertWithTabular
)


MODEL_FOR_SEQUENCE_W_TABULAR_CLASSIFICATION_MAPPING = OrderedDict(
    [
        (BertConfig, BertWithTabular)
    ]
)


class AutoModelWithTabular:
    def __init__(self):
        raise EnvironmentError(
            "AutoModelWithTabular is designed to be instantiated "
            "using the `AutoModelWithTabular.from_pretrained(pretrained_model_name_or_path)` or "
            "`AutoModelWithTabular.from_config(config)` methods."
        )

    @classmethod
    def from_config(cls, config):
        for config_class, model_class in MODEL_FOR_SEQUENCE_W_TABULAR_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class(config)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_W_TABULAR_CLASSIFICATION_MAPPING.keys()),
            )
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):

        config = kwargs.pop("config", None)
        if not isinstance(config, PretrainedConfig):
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        for config_class, model_class in MODEL_FOR_SEQUENCE_W_TABULAR_CLASSIFICATION_MAPPING.items():
            if isinstance(config, config_class):
                return model_class.from_pretrained(pretrained_model_name_or_path, *model_args, config=config, **kwargs)
        raise ValueError(
            "Unrecognized configuration class {} for this kind of AutoModel: {}.\n"
            "Model type should be one of {}.".format(
                config.__class__,
                cls.__name__,
                ", ".join(c.__name__ for c in MODEL_FOR_SEQUENCE_W_TABULAR_CLASSIFICATION_MAPPING.keys()),
            )
        )
