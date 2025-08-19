import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

def train_kfold(train_input, train_labels, test_input, transformer_layer, class_weight_dict, n_splits=5, epochs=4, batch_size=16):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    test_preds = np.zeros(len(test_input))
    oof_preds = np.zeros(len(train_labels))
    oof_labels = np.zeros(len(train_labels))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=1e-7, verbose=1)
    ]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_input, train_labels)):
        print(f"\n{'='*50}\nFold {fold+1}/{n_splits}\n{'='*50}")
        tf.keras.backend.clear_session()
        from model import build_advanced_model
        model = build_advanced_model(transformer_layer)
        
        X_train, X_val = train_input[train_idx], train_input[val_idx]
        y_train, y_val = train_labels[train_idx], train_labels[val_idx]
        
        history = model.fit(X_train, y_train,
                            validation_data=(X_val, y_val),
                            epochs=epochs,
                            batch_size=batch_size,
                            class_weight=class_weight_dict,
                            callbacks=callbacks,
                            verbose=1)
        
        val_pred = model.predict(X_val, verbose=0).flatten()
        test_pred = model.predict(test_input, verbose=0).flatten()
        
        oof_preds[val_idx] = val_pred
        oof_labels[val_idx] = y_val
        test_preds += test_pred / n_splits
        
        val_pred_binary = (val_pred>0.5).astype(int)
        print(f"Fold {fold+1} F1 Score: {f1_score(y_val, val_pred_binary):.4f}")
    
    return oof_preds, test_preds
