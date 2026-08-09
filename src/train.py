"""Train and evaluate the SmartLeafNet hybrid classification pipeline."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import optuna
import tensorflow as tf
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.applications import EfficientNetB3, ResNet50
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from xgboost import XGBClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, default=Path("data/train"))
    parser.add_argument("--valid-dir", type=Path, default=Path("data/valid"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pca-components", type=int, default=100)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(args: argparse.Namespace):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.train_dir,
        labels="inferred",
        label_mode="int",
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    class_names = train_ds.class_names
    valid_ds = tf.keras.utils.image_dataset_from_directory(
        args.valid_dir,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    return train_ds, valid_ds, class_names


def build_extractor(backbone, image_size: int) -> Sequential:
    base = backbone(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False
    return Sequential([base, GlobalAveragePooling2D()], name=f"{base.name}_extractor")


def extract_features(dataset, extractor: Sequential, preprocess):
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for images, batch_labels in dataset:
        batch = preprocess(tf.cast(images, tf.float32))
        features.append(extractor(batch, training=False).numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(features), np.concatenate(labels)


def fused_features(dataset, image_size: int):
    efficientnet = build_extractor(EfficientNetB3, image_size)
    resnet = build_extractor(ResNet50, image_size)

    efficient_features, labels = extract_features(
        dataset, efficientnet, efficientnet_preprocess
    )
    resnet_features, resnet_labels = extract_features(
        dataset, resnet, resnet_preprocess
    )
    if not np.array_equal(labels, resnet_labels):
        raise RuntimeError("Feature and label ordering changed during extraction.")
    return np.concatenate([efficient_features, resnet_features], axis=1), labels


def xgb_parameters(trial: optuna.Trial, num_classes: int, seed: int) -> dict:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": -1,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, valid_ds, class_names = load_datasets(args)
    train_features, train_labels = fused_features(train_ds, args.image_size)
    valid_features, valid_labels = fused_features(valid_ds, args.image_size)

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    valid_scaled = scaler.transform(valid_features)

    component_count = min(
        args.pca_components, train_scaled.shape[0] - 1, train_scaled.shape[1]
    )
    pca = PCA(n_components=component_count, random_state=args.seed)
    train_reduced = pca.fit_transform(train_scaled)
    valid_reduced = pca.transform(valid_scaled)

    tune_train, tune_valid, y_tune_train, y_tune_valid = train_test_split(
        train_reduced,
        train_labels,
        test_size=0.2,
        random_state=args.seed,
        stratify=train_labels,
    )
    tune_train_balanced, y_tune_balanced = SMOTE(
        random_state=args.seed
    ).fit_resample(tune_train, y_tune_train)
    num_classes = len(class_names)

    def objective(trial: optuna.Trial) -> float:
        model = XGBClassifier(**xgb_parameters(trial, num_classes, args.seed))
        model.fit(tune_train_balanced, y_tune_balanced)
        predictions = model.predict(tune_valid)
        return f1_score(y_tune_valid, predictions, average="macro")

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    full_train_balanced, full_labels_balanced = SMOTE(
        random_state=args.seed
    ).fit_resample(train_reduced, train_labels)
    final_parameters = xgb_parameters(
        optuna.trial.FixedTrial(study.best_params), num_classes, args.seed
    )
    final_model = XGBClassifier(**final_parameters)
    final_model.fit(full_train_balanced, full_labels_balanced)

    predictions = final_model.predict(valid_reduced)
    metrics = {
        "accuracy": accuracy_score(valid_labels, predictions),
        "macro_f1": f1_score(valid_labels, predictions, average="macro"),
        "classes": class_names,
        "train_samples": int(len(train_labels)),
        "validation_samples": int(len(valid_labels)),
        "best_optuna_value": study.best_value,
        "best_parameters": study.best_params,
        "classification_report": classification_report(
            valid_labels,
            predictions,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    joblib.dump(scaler, args.output_dir / "scaler.joblib")
    joblib.dump(pca, args.output_dir / "pca.joblib")
    final_model.save_model(args.output_dir / "xgboost_model.json")
    (args.output_dir / "class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps({"accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()

