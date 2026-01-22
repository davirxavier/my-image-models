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