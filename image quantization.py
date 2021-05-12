import numpy as np
from skimage.io import imread
from matplotlib import plt

PRINT_TEMPLATE = "{name:<30} {compression:<15%} 

def quantize_image(image, bin_count):
    # Create centers of bins, and for each pixel
    # pick the bin that is the closest.
    bin_centroids = np.linspace(0, 255, bin_count, dtype=np.int)
    # Bit of broadcasting tricks: compute distance
    # from all centroids to all pixels
    distances = np.abs(image[..., None] - bin_centroids[None, None])
    # Pick the closest match
    binned_image = np.argmin(distances, axis=2)

    return binned_image, bin_centroids


def unquantize_image(binned_image, bin_centroids):
    return bin_centroids[binned_image]
  
  
original_image = imread("building.png")
grayscale_image = original_image[:, :, 0]

fig, axs = plt.subplots(ncols=4)

axs[0].imshow(grayscale_image, cmap="gray")
axs[0].set_title("Original")

bins_list = [2, 8, 16]

for i, bins in enumerate(bins_list):
    ax = axs[i + 1]

    binned_image, bin_centroids = quantize_image(grayscale_image, bins)
        # Map back to original image
    restored_image = bin_centroids[binned_image]

    ax.imshow(restored_image, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Bins {}".format(bins))

        # We only need log2(bins) bits as compared to full 8
    compression = np.log2(bins) / 8

    print(PRINT_TEMPLATE.format(name="Quantization (bins={})".format(bins), compression=compression))
plt.show()
