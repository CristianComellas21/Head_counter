from matplotlib import pyplot as plt
from pathlib import Path
import cv2

def plot_results(images, all_results, all_labels, save_images=False):
    
    fig, axs = plt.subplots(len(images), 2, figsize=(15, 5*len(images)))
    axs = axs.ravel()
    
    for i, (image, results, labels) in enumerate(zip(images, all_results, all_labels)):
        __plot_result(image, results, labels, axs, i*2, save_images=save_images)
    
def __plot_result(image, results, labels, ax, i, save_images=False, save_path=Path('results')):
    ax[0].imshow(image)
    ax[0].scatter(labels['x'], labels['y'], color='red', marker='o', s=10)
    ax[0].axis('off')
    ax[0].set_title('Ground truth ({} people)'.format(len(labels)))

    if save_images:
        # Put circles in a copy of the image
        image_copy = image.copy()
        for x, y in zip(labels['x'], labels['y']):
            cv2.circle(image_copy, (int(x), int(y)), 5, (255, 0, 0), -1)
        cv2.imwrite(str(save_path / 'ground_truth_{}.png'.format(i)), image_copy)

    
    ax[1].imshow(image)
    ax[1].scatter(results['x'], results['y'], color='blue', marker='o', s=10)
    ax[1].axis('off')
    ax[1].set_title('Result ({} people)'.format(len(results)))

    if save_images:
        # Put circles in a copy of the image
        image_copy = image.copy()
        for x, y in zip(results['x'], results['y']):
            cv2.circle(image_copy, (int(x), int(y)), 5, (0, 0, 255), -1)
        cv2.imwrite(str(save_path / 'final_result{}.png'.format(i)), image_copy)

    
def plot_results_different_figures(images, all_results, all_labels, save_images=False):
    
    for i, (image, results, labels) in enumerate(zip(images, all_results, all_labels)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        __plot_result(image, results, labels, axs, i, save_images=save_images)
        plt.show()
    
    