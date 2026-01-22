import sys

from train import get_datasets, get_data_length
import build_model_classification as bmic
import tensorflow_model_optimization as tfmot
import util.data.tflite_util as tlu


COMMANDS = {}
def command(fn):
    COMMANDS[fn.__name__] = fn
    return fn


@command
def tuning():
    bmic.find_best_parameters("tuning_cropped_no_imagenet_128x3_notop",
                              input_shape,
                              label_list,
                              train_ds,
                              val_ds,
                              train_size,
                              val_size,
                              (30, 30),
                              alphas=[0.35])


@command
def train():
    print("\n"
          "---------------------------------------------------------------------------------\n"
          "-- TRAINING ---------------------------------------------------------------------\n"
          "---------------------------------------------------------------------------------\n")

    bmic.train(input_shape, train_ds, val_ds, test_ds, label_list)
    print("Labels: " + str(label_list))


@command
def test():
    bmic.evaluate_on_test(best_path, test_ds, label_list)
    bmic.find_best_qat("tuning_testing_quantization", best_path, input_shape, label_list, train_ds, val_ds, train_size, val_size)


@command
def export():
    with tfmot.quantization.keras.quantize_scope():
        tlu.quantize_model(model_or_path="best_model_classification.h5",
                           output_path="best_model_classification_quantized.tflite", representative_ds=test_ds)
        tlu.evaluate_on_test_tflite(tflite_model_path="best_model_classification_quantized.tflite",
                                    test_ds=test_ds.batch(1), class_names=label_list)


if __name__ == "__main__":
    root_folder = "./data/classification"
    input_shape = (64, 64, 3)
    train_ds, val_ds, test_ds, label_map, label_list = get_datasets(input_shape, "./data/classification", False)
    train_size, val_size = get_data_length(root_folder)
    best_path = "./optuna/study_tuning_testing_3/threshold_model_trial_701_val_f1_1.000000.h5"

    COMMANDS[sys.argv[1]]()

