import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import resize


def plot_model_heatmap(output, batch_idx=0, class_idx=None, background_class=0,
                       img_name="img", threshold=0.0, show_values=True):
    """
    Plots a heatmap from a model output and optionally overlays probability values on each cell.
    Text color automatically adjusts for readability.
    """
    if hasattr(output, "numpy"):
        output = output.numpy()

    # Determine heatmap
    if class_idx is not None:
        heatmap = output[batch_idx, :, :, class_idx]
        title = f'{img_name} - Class {class_idx}'
        color_label = f'Class {class_idx} Probability'
    else:
        fg_probs = output[batch_idx, :, :, :].copy()
        fg_probs[..., background_class] = -1
        heatmap = np.max(fg_probs, axis=-1)
        title = f'{img_name} - Max Foreground Probability'
        color_label = 'Max Foreground Probability'

    # Apply threshold
    heatmap_masked = np.where(heatmap >= threshold, heatmap, np.nan)

    # Plot heatmap
    plt.figure(figsize=(8, 8))
    im = plt.imshow(heatmap_masked, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    plt.colorbar(im, label=color_label)
    plt.title(title + f" (threshold={threshold})")

    if show_values:
        nrows, ncols = heatmap_masked.shape
        for i in range(nrows):
            for j in range(ncols):
                val = heatmap_masked[i, j]
                if not np.isnan(val):
                    # Use white text on dark cells, black text on bright cells
                    text_color = 'white' if val < 0.5 else 'black'
                    plt.text(j, i, f'{val:.2f}', ha='center', va='center', color=text_color, fontsize=8)

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()


def overlay_ytrue_heatmap_on_tensor(img_tensor, y_true, class_idx=None, background_class=0,
                                    alpha=0.5, threshold=0.1, title="Heatmap Overlay"):
    """
    Overlays a heatmap from y_true on top of a processed image tensor.

    Parameters:
    - img_tensor: np.array or TensorFlow tensor, shape (H, W, 3) or (H, W), values assumed in [0,1]
    - y_true: one-hot tensor or array of shape (grid_H, grid_W, num_classes)
    - class_idx: int or None, which class to plot. If None, plot max foreground class
    - background_class: index of background class (default 0)
    - alpha: transparency of overlay
    - threshold: min probability to show
    - title: plot title
    """
    # Convert tensors to numpy if needed
    if hasattr(img_tensor, "numpy"):
        img = img_tensor.numpy()
    else:
        img = img_tensor
    if hasattr(y_true, "numpy"):
        y_true = y_true.numpy()

    # Compute heatmap
    if class_idx is not None:
        heatmap = y_true[..., class_idx]
    else:
        fg_probs = y_true.copy()
        fg_probs[..., background_class] = -1  # ignore background
        heatmap = np.max(fg_probs, axis=-1)

    # Threshold
    heatmap_masked = np.where(heatmap >= threshold, heatmap, 0)

    # Resize heatmap to image size
    heatmap_resized = resize(heatmap_masked, (img.shape[0], img.shape[1]),
                             order=0, preserve_range=True, anti_aliasing=False)

    # Plot
    plt.figure(figsize=(8, 8))
    if img.ndim == 2:  # grayscale
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    plt.imshow(heatmap_resized, cmap='hot', alpha=alpha)
    plt.colorbar(label="Foreground probability")
    plt.title(title)
    plt.axis('off')

    mng = plt.get_current_fig_manager()
    mng.resize(*mng.window.maxsize())
    plt.show()
