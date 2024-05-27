import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

class Settings:
    @classmethod
    def show_img(cls, img, figsize=(8, 4), subplot=(1, 2, 1)):
        plt.figure(figsize=figsize)
        plt.subplot(*subplot)
        plt.imshow(img, cmap="gray", interpolation='nearest')
        plt.show()

    @classmethod
    def randomize_images(cls, x_datas, y_datas=None):
        new_x_datas = np.zeros_like(x_datas)
        new_y_datas = np.zeros_like(y_datas)

        for i, data in enumerate(tqdm(x_datas)):
            _data = x_datas[i].reshape(28, 28) 

            # Random scale
            scale_factor = np.random.uniform(0.9, 1.1)
            scaled_image = cv2.resize(_data, None, fx=scale_factor, fy=scale_factor)

            # Random rotation
            angle = np.random.uniform(-45, 45)
            rows, cols = _data.shape
            rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
            rotated_image = cv2.warpAffine(scaled_image, rotation_matrix, (cols, rows))

            # Random translation (offset)
            dx, dy = np.random.randint(-2, 2, size=2)  # Random integers between -2 and 2 for x and y translation
            translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])  # Translation matrix
            translated_image = cv2.warpAffine(rotated_image, translation_matrix, (cols, rows))

            # Add a little bit of noise
            probability = random.uniform(0, 0.3)
            noise_mask = np.random.random(translated_image.shape) <= probability
            noise_strength = np.random.random(translated_image.shape) - 0.5
            noise_value = translated_image * noise_strength
            noisy_image = np.clip(np.where(noise_mask, 1 * noise_strength, translated_image), 0, 1)

            if y_datas is not None:
                new_x_datas[i] = noisy_image.reshape(1, -1)
                new_y_datas[i] = np.array([y_datas[i]])
            else:
                x_datas[i] = noisy_image.reshape(784)

        if y_datas is not None:
            x_datas = np.concatenate((x_datas, new_x_datas), axis=0)
            y_datas = np.concatenate((y_datas, new_y_datas), axis=0)

        return x_datas, y_datas
