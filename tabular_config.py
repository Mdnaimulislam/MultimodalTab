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
