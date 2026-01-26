import math
import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Union, Tuple, Any, Literal, Optional

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_model_optimization as tfmot
from keras import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from optuna.integration import TFKerasPruningCallback
from tensorflow.python.data.ops.dataset_ops import DatasetV2

import optuna


def save_best_models(
        model,
        study,
        trial,
        current_metric,
        monitored_metric,
        metric_mode: Literal["max", "min"],
        save_dir,
        metric_threshold=None,
        metric_threshold_save_amount=None,
):
    if metric_mode not in {"max", "min"}:
        raise ValueError("metric_mode should be 'max' or 'min'")

    os.makedirs(save_dir, exist_ok=True)

    try:
        if metric_mode == "max":
            is_best_trial = current_metric > study.best_value
        else:
            is_best_trial = current_metric < study.best_value
    except ValueError:
        print("Best trial not set yet.")
        is_best_trial = True

    threshold_count = study.user_attrs.get("threshold_model_count", 0)
    metric_str = f"{current_metric:.6f}"

    if is_best_trial:
        print(f"New best trial found with {monitored_metric}: {current_metric}")

        model_path = os.path.join(
            save_dir,
            f"best_model_trial_{trial.number}_{monitored_metric}_{metric_str}.h5",
        )
        model.save(model_path)
        print(f"Best model saved to {model_path}")

    elif (
            metric_threshold is not None
            and (
                    (metric_mode == "max" and current_metric >= metric_threshold) or
                    (metric_mode == "min" and current_metric <= metric_threshold)
            )
    ):
        threshold_count += 1
        study.set_user_attr("threshold_model_count", threshold_count)

        print(
            f"Model with {monitored_metric} passing threshold "
            f"{metric_threshold}: {current_metric}"
        )

        model_path = os.path.join(
            save_dir,
            f"threshold_model_trial_{trial.number}_{monitored_metric}_{metric_str}.h5",
        )
        model.save(model_path)
        print(f"Threshold model saved to {model_path}")

        if metric_threshold_save_amount is not None and threshold_count >= metric_threshold_save_amount:
            print(
                f"Stopping study early: {threshold_count} models "
                f"passed threshold {metric_threshold}."
            )
            study.stop()


def get_trial_epochs(trial: optuna.Trial, epochs: Tuple[int, int], is_re_run=False) -> int:
    if is_re_run:
        return trial.params["epochs"]

    if epochs is None:
        raise ValueError("epochs cannot be None")

    if isinstance(epochs, tuple) and len(epochs) == 2:
        min_epochs, max_epochs = epochs
        if not (isinstance(min_epochs, int) and isinstance(max_epochs, int)):
            raise TypeError("tuple elements must be integers")
        if min_epochs <= 0 or max_epochs <= 0:
            raise ValueError("tuple elements must be positive integers")
        if min_epochs > max_epochs:
            raise ValueError("min_epochs cannot be greater than max_epochs")
        return trial.suggest_int("epochs", min_epochs, max_epochs)

    # Invalid type
    raise TypeError("epochs must be an int or a tuple of exactly two elements")


@dataclass
class TrialParameters:
    build_model_fn: Callable[[tuple[int, ...], float, int, Any, float], Model]
    input_shape: tuple[int, ...]
    num_classes: int
    metrics: list[Any]
    lossfn_gen: Callable[..., Union[Callable[..., Any], str]]
    monitored_metric: str
    monitored_metric_mode: Literal["max", "min"]
    monitored_metric_threshold: Optional[Any]
    monitored_metric_threshold_save_amount: Optional[int]
    train_ds: DatasetV2
    train_size: int
    val_ds: DatasetV2
    val_size: int
    epochs: Tuple[int, int]
    alphas: list[float]
    is_object_detection: bool
    enable_pruning: bool = False
    evaluate_on_test_fn: Callable[..., float] = None
    test_ds: DatasetV2 = None


def build_optimizer(trial, lr_schedule, weight_decay):
    opt = trial.suggest_categorical(
        "optimizer", ["AdamW", "SGD", "Adam", "RMSprop", "Nadam"]
    )

    if opt == "AdamW":
        return tfa.optimizers.AdamW(
            learning_rate=lr_schedule,
            weight_decay=weight_decay,
            amsgrad=True,
        )

    if opt == "SGD":
        return tf.keras.optimizers.SGD(
            learning_rate=lr_schedule,
            momentum=trial.suggest_float("sgd_momentum", 0.5, 0.95),
        )

    if opt == "Adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr_schedule,
            beta_1=trial.suggest_float("adam_beta_1", 0.85, 0.99),
            beta_2=trial.suggest_float("adam_beta_2", 0.9, 0.999),
            epsilon=trial.suggest_float("adam_epsilon", 1e-8, 1e-4, log=True),
        )

    if opt == "RMSprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr_schedule,
            momentum=trial.suggest_float("rmsprop_momentum", 0.0, 0.9),
            rho=trial.suggest_float("rmsprop_rho", 0.7, 0.99),
            epsilon=trial.suggest_float("rmsprop_epsilon", 1e-8, 1e-4, log=True),
        )

    return tf.keras.optimizers.Nadam(
        learning_rate=lr_schedule,
        beta_1=trial.suggest_float("nadam_beta_1", 0.85, 0.99),
        beta_2=trial.suggest_float("nadam_beta_2", 0.9, 0.999),
        epsilon=trial.suggest_float("nadam_epsilon", 1e-8, 1e-4, log=True),
    )


def objective(trial, study, best_model_path, params: TrialParameters, is_re_run=False):
    """Runs a trial objective with the passed parameters."""
    final_epochs = get_trial_epochs(trial, params.epochs, is_re_run)

    lr = trial.suggest_float("learning_rate", 0.00001, 0.001, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64])
    weight_decay = trial.suggest_float("weight_decay", 0.00001, 0.001)
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)

    if params.alphas is None:
        raise ValueError("Alpha values should be a list of floats")

    if params.train_size < batch_size:
        raise ValueError("Training dataset size must be larger than batch size.")

    alpha = trial.suggest_categorical("alpha_pretrained", params.alphas)

    steps_per_epoch = math.ceil(params.train_size / batch_size)
    total_steps = steps_per_epoch * final_epochs

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=lr,
        decay_steps=total_steps,
        alpha=0.1
    )

    train_seg = (
        params.train_ds
        .shuffle(1024, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_seg = (
        params.val_ds
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    model = params.build_model_fn(params.input_shape, alpha, params.num_classes, None, dropout_rate)

    if params.is_object_detection:
        class_weight_val = trial.suggest_float("class_weight_val", 50.0, 100.0, log=True)
        class_weights = [class_weight_val for _ in range(params.num_classes)]
        class_weights[0] = 1.0

        fp_weight = trial.suggest_float("fp_weight", 1.0, 100.0, log=True)
        fp_threshold = trial.suggest_float("fp_threshold", 0.1, 0.4)
        lossfn = params.lossfn_gen(class_weights, fp_weight, fp_threshold)

        print(f"class weights: {class_weights}")
        print(f"fp weight: {fp_weight}")
        print(f"fp threshold: {fp_threshold}")
    else:
        lossfn = params.lossfn_gen()

    model.compile(
        optimizer=build_optimizer(trial, lr_schedule, weight_decay),
        loss=lossfn,
        metrics=params.metrics,
    )

    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor=params.monitored_metric,
            save_best_only=True,
            save_weights_only=True,
            mode=params.monitored_metric_mode,
        ),
    ]

    if params.enable_pruning:
        callbacks.append(TFKerasPruningCallback(trial, monitor=params.monitored_metric))

    history = model.fit(
        train_seg,
        validation_data=val_seg,
        epochs=final_epochs,
        verbose=1,
        callbacks=callbacks
    )

    print("Loading best weights.")
    model.load_weights(best_model_path)

    if params.evaluate_on_test_fn is not None and params.test_ds is not None:
        best_metric = params.evaluate_on_test_fn(model,
                                                 params.test_ds.batch(1),
                                                 params.input_shape,
                                                 model.output.shape[1],
                                                 params.num_classes)
    else:
        if params.monitored_metric_mode == "max":
            best_metric = max(history.history[params.monitored_metric])
        else:
            best_metric = min(history.history[params.monitored_metric])

    if not is_re_run:
        trial.report(best_metric, step=final_epochs)

    optuna.logging.get_logger("optuna").info(
        f"Trial {trial.number}: {params.monitored_metric}={best_metric:.4f}"
    )

    if not is_re_run:
        save_best_models(model,
                         study,
                         trial,
                         best_metric,
                         params.monitored_metric,
                         params.monitored_metric_mode,
                         f"./optuna/study_{study.study_name}",
                         params.monitored_metric_threshold,
                         params.monitored_metric_threshold_save_amount)

    return best_metric


def quantization_objective(study, trial, model_or_path, params: TrialParameters):
    """Runs a trial for quantization-aware training."""

    if isinstance(model_or_path, str) and os.path.exists(model_or_path):
        model = tf.keras.models.load_model(model_or_path)
        print(f"Loaded model from path: {model_or_path}")
    else:
        model = model_or_path
        print("Using the provided model directly.")

    final_epochs = trial.suggest_int("epochs", 10, 120)
    lr = trial.suggest_float("learning_rate", 0.0000005, 0.001, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.000001, 0.0001)

    train_seg = (
        params.train_ds
        .shuffle(1024, reshuffle_each_iteration=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_seg = (
        params.val_ds
        .batch(batch_size, drop_remainder=False)
        .prefetch(tf.data.AUTOTUNE)
    )

    qat_model = tfmot.quantization.keras.quantize_model(model)
    for layer in qat_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    best_model_path = "optuna_quantization_current_best.weights.h5"
    callbacks = [
        ModelCheckpoint(
            filepath=best_model_path,
            monitor=params.monitored_metric,
            save_best_only=True,
            save_weights_only=True,
            mode=params.monitored_metric_mode,
        ),
        EarlyStopping(monitor=params.monitored_metric, patience=15, mode=params.monitored_metric_mode),
        ReduceLROnPlateau(monitor=params.monitored_metric, factor=0.5, patience=8, mode=params.monitored_metric_mode),
        TFKerasPruningCallback(trial, monitor=params.monitored_metric),
    ]

    if params.is_object_detection:
        class_weight_val = trial.suggest_float("class_weight_val", 50.0, 100.0, log=True)
        class_weights = [class_weight_val for _ in range(params.num_classes)]
        class_weights[0] = 1.0

        fp_weight = trial.suggest_float("fp_weight", 1.0, 100.0, log=True)
        fp_threshold = trial.suggest_float("fp_threshold", 0.1, 0.4)
        lossfn = params.lossfn_gen(class_weights, fp_weight, fp_threshold)

        print(f"class weights: {class_weights}")
        print(f"fp weight: {fp_weight}")
        print(f"fp threshold: {fp_threshold}")
    else:
        lossfn = params.lossfn_gen()

    qat_model.compile(
        optimizer=build_optimizer(trial, lr, weight_decay),
        loss=lossfn,
        metrics=params.metrics,
    )
    history = qat_model.fit(
        train_seg,
        validation_data=val_seg,
        epochs=final_epochs,
        callbacks=callbacks
    )
    qat_model.load_weights(best_model_path)

    if params.monitored_metric_mode == "max":
        best_metric = max(history.history[params.monitored_metric])
    else:
        best_metric = min(history.history[params.monitored_metric])

    trial.report(best_metric, step=final_epochs)
    optuna.logging.get_logger("optuna").info(
        f"Trial {trial.number}: {params.monitored_metric}={best_metric:.4f}"
    )
    save_best_models(model,
                     study,
                     trial,
                     best_metric,
                     params.monitored_metric,
                     params.monitored_metric_mode,
                     f"./optuna/study_quantization_{study.study_name}",
                     params.monitored_metric_threshold,
                     params.monitored_metric_threshold_save_amount)
    return best_metric


optuna_db_location = "sqlite:///optuna.db"


def start_optuna_trial(params: TrialParameters, nr_trials=100, name="optuna_tuning", parallel=1):
    """
        Starts an optuna study for the passed parameters.

        :param params: Model building and training parameters.
        :param nr_trials: Maximum number of trials to run.
        :param name: Name of the study, use a unique one to create a new study, use the same names to continue an already existing study.
    """

    study = optuna.create_study(
        direction="maximize" if params.monitored_metric_mode == "max" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=20),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        study_name=name,
        storage=optuna_db_location,
        load_if_exists=True,
    )

    def objective_fn(trial):
        tf.keras.backend.clear_session()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, f"best_model.h5")
            return objective(trial, study, path, params)

    study.optimize(
        objective_fn,
        n_trials=nr_trials,
        n_jobs=parallel,
    )

    print("Best trial:")
    print(study.best_trial.params)


def start_optuna_quantization_trial(params: TrialParameters, model_or_path, nr_trials=100, name="optuna_quantization_tuning"):
    """
        Starts an optuna quantization-aware study training for the passed parameters.

        :param params: Model building and training parameters.
        :param model_or_path: The model for QAT.
        :param nr_trials: Maximum number of trials to run.
        :param name: Name of the study, use a unique one to create a new study, use the same names to continue an already existing study.
    """

    print("Runnning quantization trial")
    study = optuna.create_study(
        direction="maximize" if params.monitored_metric_mode == "max" else "minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
        study_name=name,
        storage=optuna_db_location,
        load_if_exists=True,
    )

    study.optimize(
        lambda trial: quantization_objective(study, trial, model_or_path, params),
        n_trials=nr_trials,
    )

    print("Best trial:")
    print(study.best_trial.params)


def recreate_trial(params: TrialParameters, trial_id, study_name="optuna_tuning"):
    """
        Recreates the model generated by a specific trial from an study.

        :param params: Model building parameters.
        :param trial_id: Optuna trial number.
        :param study_name: Name of the study.
        :return:
    """

    study = optuna.load_study(study_name=study_name, storage=optuna_db_location)
    trial = study.trials[trial_id - 1]
    objective(trial, params, True)


def test_output_models(study_name: str, mode: Literal["max", "min"], test_fn: Callable[[Model], float]):
    """
        Runs a test function for all best models from a study and returns the better one.

        :param study_name: Name of the study.
        :param mode: Mode of rating for the best score. Should be "max" or "min".
        :param test_fn: Test function to be executed.
        :return:
    """

    model_folder = f"./optuna/study_{study_name}"

    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"Study folder '{model_folder}' not found.")

    model_files = [f for f in os.listdir(model_folder) if f.endswith('.h5')]
    best_score = -1  # Initialize to a very low value
    best_model = None

    for model_file in model_files:
        model_path = os.path.join(model_folder, model_file)
        print(f"Testing model: {model_path}")

        try:
            model = tf.keras.models.load_model(model_path)
            score = test_fn(model)

            if (mode == "max" and score > best_score) or (mode == "min" and score < best_score):
                best_score = score
                best_model = model_path

        except Exception as e:
            print(f"Error with model {model_file}: {e}")
            continue

    if best_model is not None:
        print(f"The best model is: {best_model} with a score of {best_score}%")
        return best_model
    else:
        print("No valid models found or evaluated.")
        return None
