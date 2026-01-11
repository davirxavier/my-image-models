import tensorflow as tf
from keras.saving.object_registration import get_custom_objects


class MetricFromLogits(tf.keras.metrics.Metric):
    def __init__(self, metric_class, name=None, from_logits=False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.from_logits = from_logits
        self._metric = metric_class(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.from_logits:
            y_pred_labels = tf.argmax(y_pred, axis=-1)
            # Support both one-hot and integer y_true
            if y_true.shape[-1] > 1:
                y_true_labels = tf.argmax(y_true, axis=-1)
            else:
                y_true_labels = tf.squeeze(y_true, axis=-1)
        else:
            y_pred_labels = y_pred
            y_true_labels = y_true

        self._metric.update_state(y_true_labels, y_pred_labels, sample_weight)

    def result(self):
        return self._metric.result()

    def reset_state(self):
        self._metric.reset_state()


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', from_logits=False, **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        self.from_logits = from_logits

        self._metrics = [self.precision, self.recall]

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert logits to predicted class indices
        if self.from_logits:
            y_pred_labels = tf.argmax(y_pred, axis=-1)
            y_true_labels = tf.argmax(y_true, axis=-1)
        else:
            y_pred_labels = y_pred
            y_true_labels = y_true

        self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
        self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)

    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()


get_custom_objects().update({"F1Score": F1Score})


class DetectionMetrics(tf.keras.metrics.Metric):
    def __init__(self, grid_size, radius=1, name="f1", **kwargs):
        super().__init__(name=name, **kwargs)
        self.grid_size = grid_size
        self.radius = radius

        self.tp = self.add_weight(name="tp", initializer="zeros")
        self.fp = self.add_weight(name="fp", initializer="zeros")
        self.fn = self.add_weight(name="fn", initializer="zeros")

    def _local_max_suppression(self, heatmap, ksize=3):
        """Apply local maximum suppression to a 2D heatmap."""
        # Add batch and channel dimensions
        heatmap = tf.expand_dims(tf.expand_dims(heatmap, axis=0), axis=-1)
        pooled = tf.nn.max_pool2d(heatmap, ksize=ksize, strides=1, padding='SAME')
        # Keep only local maxima
        maxima = tf.cast(tf.equal(heatmap, pooled), tf.float32)
        # Multiply by heatmap to preserve original values (or just keep 1s)
        suppressed = heatmap * maxima
        # Remove batch and channel dimensions
        return tf.squeeze(suppressed, axis=[0, -1])

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_cls = tf.argmax(y_true, axis=-1)
        y_pred_cls = tf.argmax(y_pred, axis=-1)

        batch_size = tf.shape(y_true_cls)[0]

        for b in tf.range(batch_size):
            # Apply local maximum suppression to predictions
            pred_heatmap = tf.cast(y_pred_cls[b], tf.float32)
            pred_heatmap = self._local_max_suppression(pred_heatmap, ksize=3)
            pred_cells = tf.where(pred_heatmap > 0)

            gt_cells = tf.where(y_true_cls[b] > 0)

            n_gt = tf.shape(gt_cells)[0]
            n_pred = tf.shape(pred_cells)[0]

            matched_gt = tf.zeros(n_gt, dtype=tf.bool)
            matched_pred = tf.zeros(n_pred, dtype=tf.bool)

            for pi in tf.range(n_pred):
                pred_cell = pred_cells[pi]
                py = pred_cell[0]
                px = pred_cell[1]

                for gi in tf.range(n_gt):
                    if matched_gt[gi]:
                        continue

                    gt_cell = gt_cells[gi]
                    gy = gt_cell[0]
                    gx = gt_cell[1]

                    if tf.maximum(tf.abs(py - gy), tf.abs(px - gx)) <= self.radius:
                        matched_gt = tf.tensor_scatter_nd_update(matched_gt, [[gi]], [True])
                        matched_pred = tf.tensor_scatter_nd_update(matched_pred, [[pi]], [True])
                        self.tp.assign_add(1.0)
                        break

            self.fp.assign_add(tf.reduce_sum(tf.cast(tf.logical_not(matched_pred), tf.float32)))
            self.fn.assign_add(tf.reduce_sum(tf.cast(tf.logical_not(matched_gt), tf.float32)))

    def result(self):
        precision = tf.math.divide_no_nan(self.tp, self.tp + self.fp)
        recall = tf.math.divide_no_nan(self.tp, self.tp + self.fn)
        return tf.math.divide_no_nan(2.0 * precision * recall, precision + recall)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)


class F1LossCombo(tf.keras.metrics.Metric):
    def __init__(self, grid_size, loss_fn, alpha=0.1, name="f1_loss_combo", **kwargs):
        super().__init__(name=name, **kwargs)

        self.alpha = alpha

        self.f1 = DetectionMetrics(grid_size)
        self.loss_fn = loss_fn

        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        f1_value = self.f1(y_true, y_pred)
        loss_value = self.loss_fn(y_true, y_pred)

        combo = f1_value - self.alpha * loss_value

        self.total.assign_add(combo)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / self.count

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)
        self.f1.reset_state()