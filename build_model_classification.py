import math
import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
from keras.callbacks import ModelCheckpoint
from keras.losses import CategoricalCrossentropy
from keras.metrics import Precision, Recall
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow_addons.optimizers import AdamW

import base_model
import optuna_util
from custom_metrics import F1Score, MetricFromLogits
from optuna_util import TrialParameters


def prepare_for_quantization(model_or_path, train_ds, val_ds, best_weights_out):
    if isinstance(model_or_path, str) and os.path.exists(model_or_path):
        model = tf.keras.models.load_model(model_or_path)
        print(f"Loaded model from path: {model_or_path}")
    else:
        model = model_or_path
        print("Using the provided model directly.")

    epochs = 30
    batch_size = 32

    num_samples = tf.data.experimental.cardinality(train_ds).numpy()
    steps_per_epoch = math.ceil(num_samples / batch_size)
    total_steps = steps_per_epoch * epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=0.00005,
        decay_steps=total_steps,
        alpha=0.2
    )
    optimizer = tfa.optimizers.AdamW(learning_rate=lr_schedule, weight_decay=1e-6)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_weights_out,
            monitor='val_accuracy',
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True
        ),
    ]

    qat_model = tfmot.quantization.keras.quantize_model(model)
    for layer in qat_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    qat_model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()]
    )
    qat_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
    )
    return qat_model


def print_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    n = len(class_names)

    # Header
    header = " " * 12 + "".join([f"{name:<10}" for name in class_names])
    print(header)
    print("-" * (12 + 10 * n))

    # Rows
    for i, row in enumerate(cm):
        row_str = f"{class_names[i]:<12}" + "".join([f"{val:<10}" for val in row])
        print(row_str)


def evaluate_on_test(model_or_path, test_ds, class_names=None):
    print("\n=== Running evaluation on test dataset ===", flush=True)

    test_seg = (test_ds
                .batch(8, drop_remainder=False)
                .prefetch(tf.data.AUTOTUNE))

    if isinstance(model_or_path, str) and os.path.exists(model_or_path):
        model = tf.keras.models.load_model(model_or_path)
        print(f"Loaded model from path: {model_or_path}")
    else:
        model = model_or_path
        print("Using the provided model directly.")

    y_true = []
    y_pred = []

    for images, labels in test_seg:
        predictions = model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(predictions, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred) * 100
    print(f"Total samples: {len(y_true)}")
    print(f"Correct predictions: {(accuracy / 100) * len(y_true):.0f} ({accuracy:.4f}%) ({accuracy})\n")

    # Class names
    if class_names is None:
        class_names = [f"Class {i}" for i in range(model.output_shape[-1])]

    # Detailed per-class metrics
    print("Per-class metrics (Precision, Recall, F1-score):")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)

    print("Confusion matrix:")
    print_confusion_matrix(y_true, y_pred, class_names)
    print("\nTest evaluation complete.\n", flush=True)
    return accuracy


def train(input_shape, train_ds, val_ds, test_ds, class_list):
    do_fine_tune = False

    shape_w, shape_h, shape_c = input_shape
    if shape_w != shape_h:
        raise ValueError("Input shape should have square dimensions.")

    batch_size = 32
    lr = 0.0008
    epochs = 50
    alpha = 1

    train_seg = (train_ds
                 .batch(batch_size, drop_remainder=False)
                 .prefetch(tf.data.AUTOTUNE))

    val_seg = (val_ds
               .batch(batch_size, drop_remainder=False)
               .prefetch(tf.data.AUTOTUNE))


    model = base_model.build_model(input_shape, alpha, len(class_list), weights=None)

    model.compile(optimizer=AdamW(learning_rate=lr, weight_decay=1e-4),
                  loss=CategoricalCrossentropy(from_logits=False),
                  metrics=[
                      'accuracy',
                      MetricFromLogits(tf.keras.metrics.Precision, name="precision", from_logits=False),
                      MetricFromLogits(tf.keras.metrics.Recall, name="recall", from_logits=False),
                      F1Score(from_logits=False),
                  ])

    best_model_path = "best_model_classification.h5"
    callbacks = [
        ModelCheckpoint(best_model_path, monitor='val_f1_score', save_best_only=True, save_weights_only=False,
                        mode="max"),
    ]
    model.fit(train_seg,
              validation_data=val_seg,
              batch_size=batch_size,
              validation_batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              callbacks=callbacks)
    model.load_weights(best_model_path)

    if do_fine_tune:
        if test_ds:
            print("Testing before fine tuning")
            evaluate_on_test(model, test_ds, class_list)

        print("Fine-tuning.")
        model = prepare_for_quantization(model,
                          train_seg,
                          val_seg,
                          best_model_path)
        model.load_weights(best_model_path)

    if test_ds:
        print("Testing after fine tuning")
        evaluate_on_test(model, test_ds, class_list)

    print("Training done")
    return model


def get_trial_parameters(input_shape, num_classes, train_ds, val_ds, train_size, val_size, epochs, alphas):
    metrics = [
        "accuracy",
        Precision(),
        Recall(),
        F1Score(name="f1"),
    ]

    return TrialParameters(
        build_model_fn=base_model.build_model,
        input_shape=input_shape,
        num_classes=num_classes,
        metrics=metrics,
        monitored_metric="val_f1",
        monitored_metric_mode="max",
        monitored_metric_threshold=1,
        monitored_metric_threshold_save_amount=200,
        train_ds=train_ds,
        train_size=train_size,
        val_ds=val_ds,
        val_size=val_size,
        epochs=epochs,
        lossfn="categorical_crossentropy",
        alphas=alphas,
        is_object_detection=False,
    )


def find_best_parameters(tune_name, input_shape, num_classes, train_ds, val_ds, train_size, val_size, epochs, alphas):
    optuna_util.start_optuna_trial(
        get_trial_parameters(input_shape, num_classes, train_ds, val_ds, train_size, val_size, epochs, alphas),
        mode="maximize",
        nr_trials=10000,
        name=tune_name)


def recreate_model(trial_id, input_shape, classes, train_ds, val_ds, train_size, val_size):
    optuna_util.recreate_trial(get_trial_parameters(input_shape, classes, train_ds, val_ds, train_size, val_size, None),
                               trial_id)


def find_best_qat(tune_name, model_or_path, input_shape, classes, train_ds, val_ds, train_size, val_size):
    optuna_util.start_optuna_quantization_trial(
        get_trial_parameters(input_shape, classes, train_ds, val_ds, train_size, val_size, 0),
        model_or_path,
        mode="maximize",
        nr_trials=10000,
        name=tune_name)

# trial 1 best: [I 2025-12-13 17:11:59,822] Trial 49 finished with value: 0.9176470041275024 and parameters: {'learning_rate': 0.005558028761321086, 'batch_size': 32, 'weight_decay': 0.000534548635061232, 'alpha': 0.1, 'dropout_rate': 0.3228579616945298, 'optimizer': 'SGD'}. Best is trial 24 with value: 0.9651162028312683.
#
#
# [I 2025-12-13 18:22:30,112] Trial 136 finished with value: 0.9767441153526306 and parameters: {'learning_rate': 0.0034840596767749094, 'batch_size': 8, 'weight_decay': 0.0009568686500680142, 'alpha': 0.35, 'dropout_rate': 0.2627924013334022, 'optimizer': 'Nadam', 'nadam_beta_1': 0.9590880020818104, 'nadam_beta_2': 0.9560060290144828, 'nadam_epsilon': 1.240353883671581e-07}. Best is trial 124 with value: 0.9883720278739929.
