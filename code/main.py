from data import load_data, bert_encode, compute_class_weights
from train import train_kfold
import transformers
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report


train, test, submission = load_data("/kaggle/input/nlp-getting-started/train.csv",
                                    "/kaggle/input/nlp-getting-started/test.csv",
                                    "/kaggle/input/nlp-getting-started/sample_submission.csv")


tokenizer = transformers.DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')


train_input, train_masks = bert_encode(train.text.values, tokenizer)
test_input, test_masks = bert_encode(test.text.values, tokenizer)
train_labels = train.target.values


class_weight_dict = compute_class_weights(train_labels)
print(f"Class weights: {class_weight_dict}")


oof_preds, test_preds = train_kfold(train_input, train_labels, test_input, transformer_layer, class_weight_dict)


final_preds_binary = (oof_preds>0.5).astype(int)
print(f"Final F1 Score: {f1_score(train_labels, final_preds_binary):.4f}")
print(classification_report(train_labels, final_preds_binary))

# Submission
optimal_threshold = 0.5
submission['target'] = (test_preds>optimal_threshold).astype(int)
submission.to_csv('submission.csv', index=False)
print("Submission saved to submission.csv")