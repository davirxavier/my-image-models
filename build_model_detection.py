import os

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras import Model
from keras.callbacks import ModelCheckpoint
from keras.layers import Conv2D
from keras.optimizers import Adam

import base_model
import optuna_util
from base_model import DOWNSAMPLE_CONV, EXTRA_DEPTHWISE, EXTRA_FULL_CONV, EXTRA_CONV_1x1
from custom_metrics import DetectionMetrics, F1LossCombo
from optuna_util import TrialParameters

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model(input_shape, alpha, num_classes, weights, dropout_rate=0.1):
    backbone = base_model.build_model(input_shape,
                                      alpha,
                                      num_classes,
                                      weights,
                                      conv_blocks=(
                                          (16, DOWNSAMPLE_CONV, [EXTRA_DEPTHWISE]),
                                          (48, DOWNSAMPLE_CONV, [EXTRA_DEPTHWISE]),
                                          (96, None, [EXTRA_DEPTHWISE]),

                                          # (96, DOWNSAMPLE_CONV),
                                          # (96, None, [EXTRA_DEPTHWISE]),
                                          # (96, None, [EXTRA_CONV_1x1]),
                                          # (96, None, [EXTRA_DEPTHWISE])

                                          (96, DOWNSAMPLE_CONV, [EXTRA_FULL_CONV]),
                                          (96, None, [EXTRA_FULL_CONV]),
                                          (128, None, [EXTRA_FULL_CONV]),
                                          (128, None, [EXTRA_CONV_1x1]),
                                          (128, None)

                                          # (96, DOWNSAMPLE_CONV, [EXTRA_DEPTHWISE]),
                                          # (96, None, [EXTRA_DEPTHWISE]),
                                          # (48, None),
                                          # (24, None),
                                      ),
                                      include_head=False,
                                      dropout_rate=dropout_rate)

    x = backbone.output
    output = Conv2D(num_classes, 1, 1, name="output_1x1_conv", activation=None)(x)

    model = Model(inputs=backbone.input, outputs=output)
    for l in model.layers:
        print(l.name, l.input_shape)

    return model


@tf.function
def y_true_mapper(
        bboxes,
        input_shape,
        grid_size,
        num_classes,
        n_cells
):
    """
    Returns:
        y_true: (grid_size, grid_size, num_classes + 1)
            one-hot targets
            class 0 = background
    """

    min_intersection_pct = 0.3
    C = num_classes

    # Initialize all cells as background
    y_true = tf.zeros((grid_size, grid_size, C), dtype=tf.float32)
    y_true = tf.tensor_scatter_nd_update(
        y_true,
        tf.reshape(
            tf.stack(
                tf.meshgrid(
                    tf.range(grid_size),
                    tf.range(grid_size),
                    indexing="ij"
                ),
                axis=-1
            ),
            (-1, 2)
        ),
        tf.repeat(
            tf.one_hot(0, C)[None, :],
            grid_size * grid_size,
            axis=0
        )
    )

    img_h = tf.cast(input_shape[0], tf.float32)
    img_w = tf.cast(input_shape[1], tf.float32)

    stride_x = img_w / tf.cast(grid_size, tf.float32)
    stride_y = img_h / tf.cast(grid_size, tf.float32)

    num_boxes = tf.shape(bboxes)[0]

    def body(i, y_true):
        xmin, ymin, xmax, ymax, label = tf.unstack(bboxes[i])

        def place_object(y_true):
            cls = tf.cast(label, tf.int32) + 1  # background offset

            intersections = []

            for gy in range(grid_size):
                for gx in range(grid_size):
                    gx_min = gx * stride_x
                    gy_min = gy * stride_y
                    gx_max = (gx + 1) * stride_x
                    gy_max = (gy + 1) * stride_y

                    inter_xmin = tf.maximum(gx_min, xmin)
                    inter_ymin = tf.maximum(gy_min, ymin)
                    inter_xmax = tf.minimum(gx_max, xmax)
                    inter_ymax = tf.minimum(gy_max, ymax)

                    inter_area = tf.maximum(inter_xmax - inter_xmin, 0.0) * \
                                 tf.maximum(inter_ymax - inter_ymin, 0.0)

                    intersections.append([inter_area, gy, gx])

            intersections = tf.convert_to_tensor(intersections, tf.float32)
            # shape: [N, 3] -> [intersection_area, gy, gx]

            # --------------------------------------------------
            # Grid-like selection logic
            # --------------------------------------------------

            # Box center in grid coordinates
            box_cx = (xmin + xmax) * 0.5 / stride_x
            box_cy = (ymin + ymax) * 0.5 / stride_y

            gy = intersections[:, 1]
            gx = intersections[:, 2]

            # Euclidean distance in grid space
            dist = tf.sqrt(
                tf.square(gx - box_cx) +
                tf.square(gy - box_cy)
            )

            # Distance-weighted score
            alpha = 0.7  # higher = tighter, more square clusters
            scores = intersections[:, 0] * tf.exp(-alpha * dist)

            # Sort by combined score
            sorted_idx = tf.argsort(scores, direction="DESCENDING")
            sorted_cells = tf.gather(intersections, sorted_idx)

            # --- Always take best cell ---
            best_cell = sorted_cells[0:1]
            max_intersection = best_cell[0, 0]

            # --- Percentage-based threshold ---
            threshold = max_intersection * min_intersection_pct

            remaining = sorted_cells[1:]
            valid_mask = remaining[:, 0] >= threshold
            remaining = tf.boolean_mask(remaining, valid_mask)

            # Limit total number of cells
            remaining = remaining[: n_cells - 1]

            selected_cells = tf.concat([best_cell, remaining], axis=0)

            # Apply one-hot encoding
            for cell in selected_cells:
                _, gy, gx = tf.unstack(cell)
                gy = tf.cast(gy, tf.int32)
                gx = tf.cast(gx, tf.int32)

                update = tf.one_hot(cls, C, dtype=tf.float32)
                coords = tf.reshape([gy, gx], (1, 2))

                y_true = tf.tensor_scatter_nd_update(
                    y_true,
                    coords,
                    tf.reshape(update, (1, C))
                )

            return y_true

        y_true = tf.cond(label >= 0, lambda: place_object(y_true), lambda: y_true)
        return i + 1, y_true

    _, y_true = tf.while_loop(
        lambda i, _: i < num_boxes,
        body,
        [0, y_true]
    )

    return y_true


def loss(pos_weights, fp_weight=50.0, threshold=0.3):
    pos_weights = tf.constant(pos_weights, dtype=tf.float32)

    def loss(y_true, y_pred):
        ce = tf.nn.weighted_cross_entropy_with_logits(
            labels=y_true,
            logits=y_pred,
            pos_weight=pos_weights
        )

        probs = tf.nn.sigmoid(y_pred)

        false_positive_mask = tf.cast(
            tf.logical_and(probs > threshold, y_true == 0),
            tf.float32
        )

        # Confidence-scaled FP penalty
        fp_confidence = tf.nn.relu(probs - threshold)
        fp_penalty = false_positive_mask * fp_weight * fp_confidence

        total_loss = ce + fp_penalty
        return tf.reduce_mean(total_loss)

    return loss


def local_max_filter(probs, k=3):
    # probs: (B, H, W, C+1) softmaxed
    fg_probs = probs[..., 1:]  # exclude background
    maxpool = tf.nn.max_pool2d(
        fg_probs,
        ksize=k,
        strides=1,
        padding="SAME"
    )

    # keep only local maxima
    keep = tf.cast(fg_probs >= maxpool, fg_probs.dtype)

    fg_probs = fg_probs * keep

    # recompute background prob so sum = 1
    bg = 1.0 - tf.reduce_sum(fg_probs, axis=-1, keepdims=True)
    bg = tf.clip_by_value(bg, 0.0, 1.0)

    return tf.concat([bg, fg_probs], axis=-1)


def get_n_cells(grid_size):
    return int(max(grid_size / 3.0, 1.0))


def evaluate_on_test(
        model,
        test_dataset,
        input_shape,
        grid_size,
        num_classes_with_background,
        radius=1,
        conf_thresh=0.5,
):

    img_h, img_w = input_shape[:2]
    stride_x = img_w / grid_size
    stride_y = img_h / grid_size

    TP = {}
    FP = {}
    FN = {}

    final_model = Model(model.input, local_max_filter(model.output))
    for images, bboxes in test_dataset:
        y_pred = final_model(images, training=False)

        y_pred_cls = tf.argmax(y_pred, axis=-1).numpy()  # (B, G, G)
        y_pred_score = tf.reduce_max(y_pred, axis=-1).numpy()  # (B, G, G)

        batch_size = images.shape[0]

        for b in range(batch_size):

            # ----------------------
            # Build GT centroids
            # ----------------------
            gt_by_class = {}

            for box in bboxes[b]:
                xmin, ymin, xmax, ymax, label = box.numpy()

                if label < 0:
                    continue

                cls = int(label) + 1  # background offset

                cx = (xmin + xmax) * 0.5
                cy = (ymin + ymax) * 0.5

                gx = int(cx // stride_x)
                gy = int(cy // stride_y)

                gx = max(0, min(grid_size - 1, gx))
                gy = max(0, min(grid_size - 1, gy))

                gt_by_class.setdefault(cls, []).append((gy, gx))

            # ----------------------
            # Build predictions
            # ----------------------
            pred_by_class = {}

            for gy in range(grid_size):
                for gx in range(grid_size):
                    cls = int(y_pred_cls[b, gy, gx])
                    score = y_pred_score[b, gy, gx]

                    if cls > 0 and score >= conf_thresh:
                        pred_by_class.setdefault(cls, []).append((gy, gx))

            # ----------------------
            # Match predictions ↔ GT
            # ----------------------
            for cls in range(1, num_classes_with_background):
                gt_centroids = gt_by_class.get(cls, [])
                pred_cells = pred_by_class.get(cls, [])

                TP.setdefault(cls, 0)
                FP.setdefault(cls, 0)
                FN.setdefault(cls, 0)

                matched_gt = set()
                matched_pred = set()

                for pi, (py, px) in enumerate(pred_cells):
                    for gi, (gy, gx) in enumerate(gt_centroids):
                        if gi in matched_gt:
                            continue
                        if max(abs(py - gy), abs(px - gx)) <= radius:
                            matched_gt.add(gi)
                            matched_pred.add(pi)
                            TP[cls] += 1
                            break

                FP[cls] += len(pred_cells) - len(matched_pred)
                FN[cls] += len(gt_centroids) - len(matched_gt)

    # ===================== PRINT RESULTS =====================

    print("\n" + "=" * 72)
    print("CENTROID DETECTION EVALUATION (OBJECT-LEVEL)")
    print(f"Grid tolerance radius: ±{radius} cell(s)")
    print(f"Confidence threshold : {conf_thresh}")
    print("Background class (0) ignored")
    print("=" * 72)

    def safe_div(a, b):
        return a / b if b > 0 else 0.0

    total_tp = sum(TP.values())
    total_fp = sum(FP.values())
    total_fn = sum(FN.values())

    overall_precision = safe_div(total_tp, total_tp + total_fp)
    overall_recall = safe_div(total_tp, total_tp + total_fn)
    overall_f1 = safe_div(2 * overall_precision * overall_recall, overall_precision + overall_recall)

    print(f"\nOverall Precision : {overall_precision:.4f}")
    print(f"Overall Recall    : {overall_recall:.4f}")
    print(f"Overall F1 Score  : {overall_f1:.4f}")

    print("\nPer-class metrics:")
    print("-" * 60)

    for cls in sorted(set(TP) | set(FP) | set(FN)):
        tp, fp, fn = TP.get(cls, 0), FP.get(cls, 0), FN.get(cls, 0)
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        print(
            f"Class {cls:>3} | "
            f"TP: {tp:>4}  FP: {fp:>4}  FN: {fn:>4} | "
            f"P: {precision:.4f}  R: {recall:.4f}  F1: {f1:.4f}"
        )

    print("\n" + "=" * 72)
    return overall_f1


def train(input_shape, train_ds, val_ds, test_ds, class_list, class_weights, batch_size=16, best_model_path="best_model_detection.h5"):
    num_classes = len(class_list) + 1
    epochs = 100
    alpha = 1
    model = build_model(input_shape, alpha, num_classes, None)
    grid_size = model.output.shape[1]

    mapper = lambda img, boxes: (img, y_true_mapper(boxes, input_shape, grid_size, num_classes, get_n_cells(grid_size)))

    train_seg = (train_ds
                 .map(mapper)
                 .batch(batch_size, drop_remainder=False)
                 .prefetch(tf.data.AUTOTUNE))

    val_seg = (val_ds
               .map(mapper)
               .batch(batch_size, drop_remainder=False)
               .prefetch(tf.data.AUTOTUNE))

    lossfn = loss(class_weights)
    model.compile(optimizer=Adam(learning_rate=0.0007),
                  loss=lossfn,
                  metrics=[
                      DetectionMetrics(grid_size),
                      F1LossCombo(grid_size, lossfn, alpha=0.3),
                  ])

    model.fit(train_seg,
              epochs=epochs,
              validation_data=val_seg,
              verbose=2,
              callbacks=[
                  ModelCheckpoint(filepath=best_model_path,
                                  monitor="val_f1",
                                  mode="max",
                                  verbose=1,
                                  save_weights_only=False,
                                  save_best_only=True)
              ])

    model.load_weights(best_model_path)
    print("Training done.")


def prune_model(model, input_shape, num_classes_without_bg, train_ds, val_ds, train_size, val_size, class_weights):
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    EXCLUDE_LAYERS = (
        tf.keras.layers.Lambda,
        tf.keras.layers.Reshape,
        tf.keras.layers.Flatten,
        tf.keras.layers.MaxPooling2D,
        tf.keras.layers.GlobalAveragePooling2D,
        tf.keras.layers.GlobalMaxPooling2D,
    )

    def prune_if_allowed(layer):
        # Skip layers with no trainable weights
        if not layer.trainable_weights:
            return layer

        # Skip explicitly excluded layer types
        if isinstance(layer, EXCLUDE_LAYERS):
            return layer

        # Everything else gets pruned
        return prune_low_magnitude(
            layer,
            pruning_schedule=tfmot.sparsity.keras.ConstantSparsity(
                target_sparsity=0.5,
                begin_step=0
            )
        )

    grid_size = model.output.shape[1]
    num_classes = num_classes_without_bg + 1
    batch_size = 16

    mapper = lambda img, boxes: (img, y_true_mapper(boxes, input_shape, grid_size, num_classes, get_n_cells(grid_size)))

    train_seg = (train_ds
                 .map(mapper)
                 .batch(batch_size, drop_remainder=False)
                 .prefetch(tf.data.AUTOTUNE))

    val_seg = (val_ds
               .map(mapper)
               .batch(batch_size, drop_remainder=False)
               .prefetch(tf.data.AUTOTUNE))

    pruned_model = tf.keras.models.clone_model(
        model,
        clone_function=prune_if_allowed
    )

    pruned_model.compile(
        optimizer="adam",
        loss=loss(class_weights),
        metrics=[
            DetectionMetrics(grid_size),
        ]
    )

    pruned_model.fit(
        train_seg,
        validation_data=val_seg,
        epochs=1,
        callbacks=[tfmot.sparsity.keras.UpdatePruningStep()]
    )

    final_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
    # final_model.save("best_model_detection_pruned.h5")
    return final_model


def get_trial_parameters(input_shape, num_classes_without_bg, train_ds, val_ds, train_size, val_size, test_ds, epochs, alphas):
    num_classes = num_classes_without_bg + 1

    model = build_model(input_shape, alphas[0], num_classes, None)
    grid_size = model.output.shape[1]
    mapper = lambda img, boxes: (img, y_true_mapper(boxes, input_shape, grid_size, num_classes, get_n_cells(grid_size)))

    metrics = [
        DetectionMetrics(grid_size),
    ]

    return TrialParameters(
        build_model_fn=build_model,
        input_shape=input_shape,
        num_classes=num_classes,
        metrics=metrics,
        monitored_metric="val_f1",
        monitored_metric_mode="max",
        monitored_metric_threshold=0.95,
        monitored_metric_threshold_save_amount=200,
        train_ds=train_ds.map(mapper),
        train_size=train_size,
        val_ds=val_ds.map(mapper),
        val_size=val_size,
        epochs=epochs,
        lossfn_gen=lambda class_weights, fp_weight, fp_t: loss(class_weights, fp_weight, fp_t),
        alphas=alphas,
        is_object_detection=True,
        enable_pruning=False,
        evaluate_on_test_fn=evaluate_on_test,
        test_ds=test_ds
    )


def find_best_parameters(tune_name, input_shape, num_classes, train_ds, val_ds, train_size, val_size, test_ds, epochs, alphas, parallel, nr_trials=10000):
    optuna_util.start_optuna_trial(
        get_trial_parameters(input_shape, num_classes, train_ds, val_ds, train_size, val_size, test_ds, epochs, alphas),
        nr_trials=nr_trials,
        name=tune_name,
        parallel=parallel)
