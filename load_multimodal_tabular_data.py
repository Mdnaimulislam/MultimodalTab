from functools import partial
import logging
from os.path import join, exists
import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import logging
import types

import numpy as np
from sklearn import preprocessing

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer


logger = logging.getLogger(__name__)

class TorchTabularTextDataset(TorchDataset):
    def __init__(self,
                 encodings,
                 categorical_feats,
                 numerical_feats,
                 labels=None,
                 df=None,
                 label_list=None,
                 class_weights=None
                 ):
        self.df = df
        self.encodings = encodings
        self.cat_feats = categorical_feats
        self.numerical_feats = numerical_feats
        self.labels = labels
        self.class_weights = class_weights
        self.label_list = label_list if label_list is not None else [i for i in range(len(np.unique(labels)))]

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]) if self.labels is not None  else None
        item['cat_feats'] = torch.tensor(self.cat_feats[idx]).float() \
            if self.cat_feats is not None else torch.zeros(0)
        item['numerical_feats'] = torch.tensor(self.numerical_feats[idx]).float()\
            if self.numerical_feats is not None else torch.zeros(0)
        return item

    def __len__(self):
        return len(self.labels)

    def get_labels(self):
        return self.label_list


class CategoricalFeatures:

    def __init__(self, df, categorical_cols, encoding_type, handle_na=False):
        self.df = df
        self.cat_feats = categorical_cols
        self.enc_type = encoding_type
        self.handle_na = handle_na
        self.label_encoders = dict()
        self.binary_encoders = dict()
        self.ohe = None

        if self.handle_na:
            for c in self.cat_feats:
                self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-9999999")
        self.output_df = self.df.copy(deep=True)

    def _label_encoding(self):
        for c in self.cat_feats:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(self.df[c].values)
            self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
            self.label_encoders[c] = lbl
        return self.output_df[self.cat_feats].values

    def _label_binarization(self):
        vals =[]
        self.feat_names = []

        def change_name_func(x):
            return x.lower().replace(', ', '_').replace(' ', '_')
        for c in self.cat_feats:
            self.df[c] = self.df[c].astype(str)
            classes_orig = self.df[c].unique()
            val = preprocessing.label_binarize(self.df[c].values, classes=classes_orig)
            vals.append(val)
            if len(classes_orig) == 2:
                classes = [c + '_binary']
            else:
                change_classes_func_vec = np.vectorize(lambda x: c + '_' + change_name_func(x))
                classes = change_classes_func_vec(classes_orig)
            self.feat_names.extend(classes)
        return np.concatenate(vals, axis=1)

    def _one_hot(self):
        ohe = preprocessing.OneHotEncoder(sparse=False)
        ohe.fit(self.df[self.cat_feats].values)
        self.feat_names = list(ohe.get_feature_names(self.cat_feats))
        return ohe.transform(self.df[self.cat_feats].values)

    def fit_transform(self):
        if self.enc_type == "label":
            return self._label_encoding()
        elif self.enc_type == "binary":
            return self._label_binarization()
        elif self.enc_type == "ohe":
            return self._one_hot()
        elif self.enc_type is None or self.enc_type == "none":
            return self.df[self.cat_feats].values
        else:
            raise Exception("Encoding type not understood")

    def transform(self, dataframe):
        if self.handle_na:
            for c in self.cat_feats:
                dataframe.loc[:, c] = dataframe.loc[:, c].astype(str).fillna("-9999999")

        if self.enc_type == "label":
            for c, lbl in self.label_encoders.items():
                dataframe.loc[:, c] = lbl.transform(dataframe[c].values)
            return dataframe

        elif self.enc_type == "binary":
            for c, lbl in self.binary_encoders.items():
                val = lbl.transform(dataframe[c].values)
                dataframe = dataframe.drop(c, axis=1)

                for j in range(val.shape[1]):
                    new_col_name = c + f"__bin_{j}"
                    dataframe[new_col_name] = val[:, j]
            return dataframe

        elif self.enc_type == "ohe":
            return self.ohe(dataframe[self.cat_feats].values)

        else:
            raise Exception("Encoding type not understood")


def normalize_numerical_feats(numerical_feats, transformer=None):
    if numerical_feats is None or transformer is None:
        return numerical_feats
    return transformer.transform(numerical_feats)


def convert_to_func(container_arg):
    """convert container_arg to function that returns True if an element is in container_arg"""
    if container_arg is None:
        return lambda df, x: False
    if not isinstance(container_arg, types.FunctionType):
        assert type(container_arg) is list or type(container_arg) is set
        return lambda df, x: x in container_arg
    else:
        return container_arg


def agg_text_columns_func(empty_row_values, replace_text, texts):
    """replace empty texts or remove empty text str from a list of text str"""
    processed_texts = []
    for text in texts.astype('str'):
        if text not in empty_row_values:
            processed_texts.append(text)
        else:
            if replace_text is not None:
                processed_texts.append(replace_text)
    return processed_texts


def load_cat_and_num_feats(df, cat_bool_func, num_bool_func, enocde_type=None):
    cat_feats = load_cat_feats(df, cat_bool_func, enocde_type)
    num_feats = load_num_feats(df, num_bool_func)
    return cat_feats, num_feats


def load_cat_feats(df, cat_bool_func, encode_type=None):
    """load categorical features from DataFrame and do encoding if specified"""
    cat_cols = get_matching_cols(df, cat_bool_func)
    logger.info(f'{len(cat_cols)} categorical columns')
    if len(cat_cols) == 0:
        return None
    cat_feat_processor = CategoricalFeatures(df, cat_cols, encode_type)
    return cat_feat_processor.fit_transform()


def load_num_feats(df, num_bool_func):
    num_cols = get_matching_cols(df, num_bool_func)
    logger.info(f'{len(num_cols)} numerical columns')
    df = df.copy()
    df[num_cols] = df[num_cols].astype(float)
    df[num_cols] = df[num_cols].fillna(dict(df[num_cols].median()), inplace=False)
    if len(num_cols) == 0:
        return None
    return df[num_cols].values


def get_matching_cols(df, col_match_func):
    return [c for c in df.columns if col_match_func(df, c)]


def load_data_into_folds(data_csv_path,
                         num_splits,
                         validation_ratio,
                         text_cols,
                         tokenizer,
                         label_col,
                         label_list=None,
                         categorical_cols=None,
                         numerical_cols=None,
                         sep_text_token_str=' ',
                         categorical_encode_type='ohe',
                         numerical_transformer_method='quantile_normal',
                         empty_text_values=None,
                         replace_empty_text=None,
                         max_token_length=None,
                         debug=False
                         ):
    assert 0 <= validation_ratio <= 1, 'validation ratio needs to be between 0 and 1'
    all_data_df = pd.read_csv(data_csv_path)
    folds_df, val_df = train_test_split(all_data_df, test_size=validation_ratio, shuffle=True,
                                        train_size=1-validation_ratio, random_state=5)
    kfold = KFold(num_splits, shuffle=True, random_state=5)

    train_splits, val_splits, test_splits = [], [], []

    for train_index, test_index in kfold.split(folds_df):
        train_df = folds_df.copy().iloc[train_index]
        test_df = folds_df.copy().iloc[test_index]

        train, val, test = load_train_val_test_helper(train_df, val_df.copy(),
                                                      test_df,
                                                      text_cols, tokenizer,
                                                      label_col,
                                                      label_list,
                                                      categorical_cols,
                                                      numerical_cols,
                                                      sep_text_token_str,
                                                      categorical_encode_type,
                                                      numerical_transformer_method,
                                                      empty_text_values,
                                                      replace_empty_text,
                                                      max_token_length,
                                                      debug)
        train_splits.append(train)
        val_splits.append(val)
        test_splits.append(test)

    return train_splits, val_splits, test_splits


def load_data_from_folder(folder_path,
                          text_cols,
                          tokenizer,
                          label_col,
                          label_list=None,
                          categorical_cols=None,
                          numerical_cols=None,
                          sep_text_token_str=' ',
                          categorical_encode_type='ohe',
                          numerical_transformer_method='quantile_normal',
                          empty_text_values=None,
                          replace_empty_text=None,
                          max_token_length=None,
                          debug=False,
                          ):
    train_df = pd.read_csv(join(folder_path, 'train.csv'), index_col=0)
    test_df = pd.read_csv(join(folder_path, 'test.csv'), index_col=0)
    if exists(join(folder_path, 'val.csv')):
        val_df = pd.read_csv(join(folder_path, 'val.csv'), index_col=0)
    else:
        val_df = None

    return load_train_val_test_helper(train_df, val_df, test_df,
                                      text_cols, tokenizer, label_col,
                                      label_list, categorical_cols, numerical_cols,
                                      sep_text_token_str,
                                      categorical_encode_type,
                                      numerical_transformer_method,
                                      empty_text_values,
                                      replace_empty_text,
                                      max_token_length,
                                      debug)


def load_train_val_test_helper(train_df,
                               val_df,
                               test_df,
                               text_cols,
                               tokenizer,
                               label_col,
                               label_list=None,
                               categorical_cols=None,
                               numerical_cols=None,
                               sep_text_token_str=' ',
                               categorical_encode_type='ohe',
                               numerical_transformer_method='quantile_normal',
                               empty_text_values=None,
                               replace_empty_text=None,
                               max_token_length=None,
                               debug=False):
    if categorical_encode_type == 'ohe' or categorical_encode_type == 'binary':
        dfs = [df for df in [train_df, val_df, test_df] if df is not None]
        data_df = pd.concat(dfs, axis=0)
        cat_feat_processor = CategoricalFeatures(data_df, categorical_cols, categorical_encode_type)
        vals = cat_feat_processor.fit_transform()
        cat_df = pd.DataFrame(vals, columns=cat_feat_processor.feat_names)
        data_df = pd.concat([data_df, cat_df], axis=1)
        categorical_cols = cat_feat_processor.feat_names

        len_train = len(train_df)
        len_val = len(val_df) if val_df is not None else 0

        train_df = data_df.iloc[:len_train]
        if val_df is not None:
            val_df = data_df.iloc[len_train: len_train + len_val]
            len_train = len_train + len_val
        test_df = data_df.iloc[len_train:]

        categorical_encode_type = None

    if numerical_transformer_method != 'none':
        if numerical_transformer_method == 'yeo_johnson':
            numerical_transformer = PowerTransformer(method='yeo-johnson')
        elif numerical_transformer_method == 'box_cox':
            numerical_transformer = PowerTransformer(method='box-cox')
        elif numerical_transformer_method == 'quantile_normal':
            numerical_transformer = QuantileTransformer(output_distribution='normal')
        else:
            raise ValueError(f'preprocessing transformer method '
                             f'{numerical_transformer_method} not implemented')
        num_feats = load_num_feats(train_df, convert_to_func(numerical_cols))
        numerical_transformer.fit(num_feats)
    else:
        numerical_transformer = None

    train_dataset = load_data(train_df,
                              text_cols,
                              tokenizer,
                              label_col,
                              label_list,
                              categorical_cols,
                              numerical_cols,
                              sep_text_token_str,
                              categorical_encode_type,
                              numerical_transformer,
                              empty_text_values,
                              replace_empty_text,
                              max_token_length,
                              debug
                              )
    test_dataset = load_data(test_df,
                             text_cols,
                             tokenizer,
                             label_col,
                             label_list,
                             categorical_cols,
                             numerical_cols,
                             sep_text_token_str,
                             categorical_encode_type,
                             numerical_transformer,
                             empty_text_values,
                             replace_empty_text,
                             max_token_length,
                             debug
                             )

    if val_df is not None:
        val_dataset = load_data(val_df,
                                text_cols,
                                tokenizer,
                                label_col,
                                label_list,
                                categorical_cols,
                                numerical_cols,
                                sep_text_token_str,
                                categorical_encode_type,
                                numerical_transformer,
                                empty_text_values,
                                replace_empty_text,
                                max_token_length,
                                debug
                                )
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset


def load_data(data_df,
              text_cols,
              tokenizer,
              label_col,
              label_list=None,
              categorical_cols=None,
              numerical_cols=None,
              sep_text_token_str=' ',
              categorical_encode_type='ohe',
              numerical_transformer=None,
              empty_text_values=None,
              replace_empty_text=None,
              max_token_length=None,
              debug=False,
              ):
    if debug:
        data_df = data_df[:500]
    if empty_text_values is None:
        empty_text_values = ['nan', 'None']

    text_cols_func = convert_to_func(text_cols)
    categorical_cols_func = convert_to_func(categorical_cols)
    numerical_cols_func = convert_to_func(numerical_cols)

    categorical_feats, numerical_feats = load_cat_and_num_feats(data_df,
                                                                categorical_cols_func,
                                                                numerical_cols_func,
                                                                categorical_encode_type)
    numerical_feats = normalize_numerical_feats(numerical_feats, numerical_transformer)
    agg_func = partial(agg_text_columns_func, empty_text_values, replace_empty_text)
    texts_cols = get_matching_cols(data_df, text_cols_func)
    logger.info(f'Text columns: {texts_cols}')
    texts_list = data_df[texts_cols].agg(agg_func, axis=1).tolist()
    for i, text in enumerate(texts_list):
        texts_list[i] = f' {sep_text_token_str} '.join(text)
    logger.info(f'Raw text example: {texts_list[0]}')
    hf_model_text_input = tokenizer(texts_list, padding=True, truncation=True,
                                    max_length=max_token_length)
    tokenized_text_ex = ' '.join(tokenizer.convert_ids_to_tokens(hf_model_text_input['input_ids'][0]))
    logger.debug(f'Tokenized text example: {tokenized_text_ex}')
    labels = data_df[label_col].values

    return TorchTabularTextDataset(hf_model_text_input, categorical_feats,
                                   numerical_feats, labels, data_df, label_list)
