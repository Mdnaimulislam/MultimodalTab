from torch import nn
from transformers import (
    BertForSequenceClassification
)
from transformers.modeling_bert import BERT_INPUTS_DOCSTRING
from transformers.file_utils import add_start_docstrings_to_callable
from MultimodalTab.tabular_combiner import TabularFeatCombiner
from MultimodalTab.layer_utils import MLP, calc_mlp_dims, hf_loss_func

class TabularConfig:
    def __init__(self,
                 num_labels,
                 mlp_division=4,
                 combine_feat_method='text_only',
                 mlp_dropout=0.1,
                 numerical_bn=True,
                 use_simple_classifier=True,
                 mlp_act='relu',
                 gating_beta=0.2,
                 numerical_feat_dim=0,
                 cat_feat_dim=0,
                 **kwargs
                 ):
        self.mlp_division = mlp_division
        self.combine_feat_method = combine_feat_method
        self.mlp_dropout = mlp_dropout
        self.numerical_bn = numerical_bn
        self.use_simple_classifier = use_simple_classifier
        self.mlp_act = mlp_act
        self.gating_beta = gating_beta
        self.numerical_feat_dim = numerical_feat_dim
        self.cat_feat_dim = cat_feat_dim
        self.num_labels = num_labels


class BertWithTabular(BertForSequenceClassification):

    def __init__(self, hf_model_config):
        super().__init__(hf_model_config)
        tabular_config = hf_model_config.tabular_config
        if type(tabular_config) is dict:  # when loading from saved model
            tabular_config = TabularConfig(**tabular_config)
        else:
            self.config.tabular_config = tabular_config.__dict__

        tabular_config.text_feat_dim = hf_model_config.hidden_size
        tabular_config.hidden_dropout_prob = hf_model_config.hidden_dropout_prob
        self.tabular_combiner = TabularFeatCombiner(tabular_config)
        self.num_labels = tabular_config.num_labels
        combined_feat_dim = self.tabular_combiner.final_out_dim
        if tabular_config.use_simple_classifier:
            self.tabular_classifier = nn.Linear(combined_feat_dim,
                                                tabular_config.num_labels)
        else:
            dims = calc_mlp_dims(combined_feat_dim,
                                 division=tabular_config.mlp_division,
                                 output_dim=tabular_config.num_labels)
            self.tabular_classifier = MLP(combined_feat_dim,
                                          tabular_config.num_labels,
                                          num_hidden_lyr=len(dims),
                                          dropout_prob=tabular_config.mlp_dropout,
                                          hidden_channels=dims,
                                          bn=True)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING.format("(batch_size, sequence_length)"))
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        class_weights=None,
        output_attentions=None,
        output_hidden_states=None,
        cat_feats=None,
        numerical_feats=None
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        combined_feats = self.tabular_combiner(pooled_output,
                                               cat_feats,
                                               numerical_feats)
        loss, logits, classifier_layer_outputs = hf_loss_func(combined_feats,
                                                              self.tabular_classifier,
                                                              labels,
                                                              self.num_labels,
                                                              class_weights)
        return loss, logits, classifier_layer_outputs