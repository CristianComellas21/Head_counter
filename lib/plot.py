from matplotlib import pyplot as plt

def plot_results(images, all_results, all_labels):
    
    fig, axs = plt.subplots(len(images), 2, figsize=(15, 5*len(images)))
    axs = axs.ravel()
    
    for i, (image, results, labels) in enumerate(zip(images, all_results, all_labels)):
        __plot_result(image, results, labels, axs, i*2)
    
def __plot_result(image, results, labels, ax, i):
    ax[i].imshow(image)
    ax[i].scatter(labels['x'], labels['y'], color='red', marker='o', s=10)
    ax[i].axis('off')
    ax[i].set_title('Ground truth ({} people)'.format(len(labels)))
    
    ax[i+1].imshow(image)
    ax[i+1].scatter(results['x'], results['y'], color='blue', marker='o', s=10)
    ax[i+1].axis('off')
    ax[i+1].set_title('Result ({} people)'.format(len(results)))
    
def plot_results_different_figures(images, all_results, all_labels):
    
    for i, (image, results, labels) in enumerate(zip(images, all_results, all_labels)):
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        __plot_result(image, results, labels, axs, 0)
        plt.show()
    
    