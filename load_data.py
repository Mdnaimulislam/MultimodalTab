from functools import partial
import logging
from os.path import join, exists

import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

from .tabular_torch_dataset import TorchTabularTextDataset
from .data_utils import (
    CategoricalFeatures,
    agg_text_columns_func,
    convert_to_func,
    get_matching_cols,
    load_num_feats,
    load_cat_and_num_feats,
    normalize_numerical_feats,
)

logger = logging.getLogger(__name__)


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
