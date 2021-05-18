#run length encoding

from skimage.io import imread  
import numpy as np

def run_length_encode(data):
    """
    Apply RLE on an array of data, return [values, repetitions].
    """
    values = []
    repetitions = []
    last_item = data[0]
    item_count = 1
    for i in range(1, len(data)):
        d = data[i]
        if last_item != d:
            values.append(last_item)
            repetitions.append(item_count)
            last_item = d
            item_count = 1
        else:
            item_count += 1
    values.append(last_item)
    repetitions.append(item_count)

    return values, repetitions


def run_length_decode(values, repetitions):
    original_data = []
    for value, repetition in zip(values, repetitions):
        for _ in range(repetition):
            original_data.append(value)
    return original_data
  
  
  original_bytes = np.product(grayscale_image.shape)


original_image = imread("building.png")
grayscale_image = original_image[:, :, 0]

original_bytes = np.product(grayscale_image.shape)

flat_grayscale_image = grayscale_image.ravel().tolist()
grayscale_values, grayscale_repetitions = run_length_encode(flat_grayscale_image)
reconstructed_grayscale_image = run_length_decode(grayscale_values, grayscale_repetitions)
reconstructed_grayscale_image = np.array(reconstructed_grayscale_image).reshape(grayscale_image.shape)

# Both values and repetitions are stored as 8-bit integers
compression = (len(grayscale_values) + (len(grayscale_repetitions))) / original_bytes
