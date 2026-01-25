import sys

from keras import Model

import plotting
from train import get_datasets, get_data_length
import build_model_detection as bmod
import tensorflow as tf
import util.data.tflite_util as tlu
from util.training_tool import collect_data


COMMANDS = {}
def command(fn):
    COMMANDS[fn.__name__] = fn
    return fn


@command
def tuning():
    bmod.find_best_parameters("tuning_detection_custom_arch2_4_a_1",
                              input_shape,
                              len(label_list),
                              train_ds,
                              val_ds,
                              train_size,
                              val_size,
                              test_ds,
                              (40, 150))


@command
def train():
    bmod.train(input_shape, train_ds, val_ds, test_ds, label_list, (75, 75, 0), 16, best_model_path)


@command
def test():
    test_model = tf.keras.models.load_model(best_model_path, compile=False)
    bmod.evaluate_on_test(test_model,
                          test_ds.batch(1),
                          input_shape,
                          test_model.output.shape[1],
                          len(label_list) + 1)

    outputs = test_model.output
    outputs = tf.nn.softmax(outputs, name="output_wrapper", axis=-1)
    final_model = Model(test_model.input, outputs)

    _, testing_data = collect_data(root_folder)
    test_images = []
    test_bboxes = []
    for img, gt_bboxes in test_ds:
        test_images.append(img)
        test_bboxes.append(gt_bboxes)

    combined = list(zip(test_images, test_bboxes, testing_data))
    grid_size = 12
    for img, gt_bboxes, img_name in combined[:20]:
        ytrue = bmod.y_true_mapper(gt_bboxes, input_shape, grid_size, len(label_list) + 1, bmod.get_n_cells(grid_size))
        plotting.overlay_ytrue_heatmap_on_tensor(img, ytrue)
        img = tf.expand_dims(img, axis=0)  # (1, H, W, C)
        out = final_model(img)
        plotting.plot_model_heatmap(out, img_name=img_name)


@command
def export():
    test_model = tf.keras.models.load_model(best_model_path, compile=False)
    outputs = test_model.output
    outputs = tf.nn.softmax(outputs, name="output_wrapper", axis=-1)
    # outputs = bmod.local_max_filter(outputs)
    final_model = Model(test_model.input, outputs)

    quantized_model_path = "detection_quantized.tflite"
    tlu.quantize_model(final_model, quantized_model_path, representative_ds=test_ds)


if __name__ == '__main__':
    cmd = len(sys.argv) > 1 and sys.argv[1] or ""
    if cmd not in COMMANDS:
        raise ValueError(f"Command \"{cmd}\" is not valid, valid commands are: {str([k for k in COMMANDS])}")

    input_shape = (96, 96, 3)
    root_folder = "./data/detection"
    train_ds, val_ds, test_ds, label_map, label_list = get_datasets(input_shape, root_folder, True)
    train_size, val_size = get_data_length(root_folder)

    # best_model_path = "best_model_detection.h5"
    best_model_path = "optuna/study_tuning_detection_custom_arch2_4_a_1/best_model_trial_1_val_f1_0.916201.h5"

    print("\n"
          "---------------------------------------------------------------------------------\n"
          f"-- {cmd} ---------------------------------------------------------------------\n"
          "---------------------------------------------------------------------------------\n")
    COMMANDS[cmd]()