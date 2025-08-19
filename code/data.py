import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import transformers

def load_data(train_path, test_path, submission_path):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    submission = pd.read_csv(submission_path)
    return train, test, submission

def compute_class_weights(train_labels):
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    return {0: class_weights[0], 1: class_weights[1]}

def bert_encode(texts, tokenizer, max_len=160):
    all_tokens, all_masks = [], []
    for text in texts:
        text = str(text)
        tokens = tokenizer.tokenize(text)[:max_len-2]
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1]*len(input_ids)
        pad_len = max_len - len(input_ids)
        input_ids += [0]*pad_len
        attention_mask += [0]*pad_len
        all_tokens.append(input_ids[:max_len])
        all_masks.append(attention_mask[:max_len])
    return np.array(all_tokens), np.array(all_masks)
