import tensorflow_model_optimization as tfmot

import build_model_classification as bmic
import build_model_detection as bmod
import util.data.tflite_util as tlu
from util.training_tool import (process_images,
                                summarize_dataset,
                                collect_data,
                                generate_label_map,
                                split_training_data,
                                create_tf_dataset)


def get_datasets(input_shape, root_folder, is_detection):
    process_images(root_folder, input_shape[0], 'fit-shortest')

    # Collect training data
    training_data, testing_data = collect_data(root_folder)

    label_map, label_list = generate_label_map(training_data, testing_data)
    print(f"Generated label_map and label_list: {label_map}/{label_list}")

    # Split into training and validation datasets
    train_split, val_split = split_training_data(training_data, split_ratio=0.8)

    summarize_dataset(train_split, "TRAINING")
    summarize_dataset(val_split, "VALIDATION")
    summarize_dataset(testing_data, "TESTING")

    # Convert to TensorFlow datasets
    train_ds = create_tf_dataset(train_split, input_shape, label_map, is_detection)
    val_ds = create_tf_dataset(val_split, input_shape, label_map, is_detection, augmentation=False)
    test_ds = create_tf_dataset(testing_data, input_shape, label_map, is_detection, augmentation=False)

    return train_ds, val_ds, test_ds, label_map, label_list


def get_data_length(root_folder):
    training_data, _ = collect_data(root_folder)
    train, val = split_training_data(training_data, split_ratio=0.8)
    return len(train), len(val)


def train_detection():
    input_shape = (96, 96, 3)
    root_folder = "./data/detection"
    train_ds, val_ds, test_ds, label_map, label_list = get_datasets(input_shape, root_folder, True)
    train_size, val_size = get_data_length(root_folder)

    print("\n"
          "---------------------------------------------------------------------------------\n"
          "-- TRAINING ---------------------------------------------------------------------\n"
          "---------------------------------------------------------------------------------\n")

    # best_model_path = "best_model_detection.h5"
    # best_model_path = "optuna/study_tuning_detection_custom_arch2_a0_80/best_model_trial_7_val_f1_0.939227.h5"
    #
    # test_model = tf.keras.models.load_model(best_model_path, compile=False)
    # bmod.evaluate_on_test(test_model,
    #                       test_ds.batch(1),
    #                       input_shape,
    #                       test_model.output.shape[1],
    #                       len(label_list) + 1)
    #
    # outputs = test_model.output
    # outputs = tf.nn.softmax(outputs, name="output_wrapper", axis=-1)
    # outputs = bmod.local_max_filter(outputs)
    # final_model = Model(test_model.input, outputs)
    #
    # quantized_model_path = "detection_quantized.tflite"
    # tlu.quantize_model(final_model, quantized_model_path, representative_ds=test_ds)
    # # profile(quantized_model_path)
    #
    # _, testing_data = collect_data(root_folder)
    # test_images = []
    # test_bboxes = []
    # for img, gt_bboxes in test_ds:
    #     test_images.append(img)
    #     test_bboxes.append(gt_bboxes)
    #
    # combined = list(zip(test_images, test_bboxes, testing_data))
    # grid_size = 12
    # for img, gt_bboxes, img_name in combined[:20]:
    #     ytrue = bmod.y_true_mapper(gt_bboxes, input_shape, grid_size, len(label_list) + 1, bmod.get_n_cells(grid_size))
    #     plotting.overlay_ytrue_heatmap_on_tensor(img, ytrue)
    #     img = tf.expand_dims(img, axis=0)  # (1, H, W, C)
    #     out = final_model(img)
    #     plotting.plot_model_heatmap(out, img_name=img_name)

    bmod.find_best_parameters("tuning_detection_custom_arch2_a0_80",
                              input_shape,
                              len(label_list),
                              train_ds,
                              val_ds,
                              train_size,
                              val_size,
                              test_ds,
                              (40, 150))


def train_classification():
    input_shape = (64, 64, 3)
    train_ds, val_ds, test_ds, label_map, label_list = get_datasets(input_shape, "./data/classification", False)

    print("\n"
          "---------------------------------------------------------------------------------\n"
          "-- TRAINING ---------------------------------------------------------------------\n"
          "---------------------------------------------------------------------------------\n")

    bmic.train(input_shape, train_ds, val_ds, test_ds, label_list)
    print("Labels: " + str(label_list))

    with tfmot.quantization.keras.quantize_scope():
        tlu.quantize_model(model_or_path="best_model_classification.h5",
                           output_path="best_model_classification_quantized.tflite", representative_ds=test_ds)
        tlu.evaluate_on_test_tflite(tflite_model_path="best_model_classification_quantized.tflite",
                                    test_ds=test_ds.batch(1), class_names=label_list)

    # profile("best_model_classification_quantized.tflite")

    # bmic.find_best_parameters("tuning_cropped_no_imagenet_128x3_notop",
    #                           INPUT_SHAPE,
    #                           label_list,
    #                           train_ds,
    #                           val_ds,
    #                           len(train_split),
    #                           len(val_split),
    #                           (30, 30),
    #                           alphas=[0.35])

    # bmic.recreate_model(13, INPUT_SHAPE, label_list, train_ds, val_ds, len(train_split), len(val_split))

    # optuna_util.test_output_models("tuning_testing_3",
    #                                lambda model: bmic.evaluate_on_test(
    #                                        model,
    #                                        test_ds,
    #                                        label_list))
    #
    # best_path = "./optuna/study_tuning_testing_3/threshold_model_trial_701_val_f1_1.000000.h5"
    # bmic.evaluate_on_test(best_path, test_ds, label_list)
    # bmic.find_best_qat("tuning_testing_quantization", best_path, INPUT_SHAPE, label_list, train_ds, val_ds, len(train_split), len(val_split))

if __name__ == '__main__':
    # train_classification()
    train_detection()
    pass
