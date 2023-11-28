import numpy as np

def evaluate_method(all_results, all_labels, width, height, max_distance=20):
    
    image_level_performance = np.zeros(len(all_results))
    person_level_performance = np.zeros(len(all_results), dtype=object)


    for idx, (results, labels) in enumerate(zip(all_results, all_labels)):

        image_level, person_level = image_level_evaluation(results, labels), person_level_evaluation(results, labels, width, height, max_distance=max_distance)

        image_level_performance[idx] = image_level
        person_level_performance[idx] = person_level


    image_level_result = np.mean(image_level_performance)

    person_level_result = {
        'accuracy': np.mean([person_level['accuracy'] for person_level in person_level_performance]),
        'recall': np.mean([person_level['recall'] for person_level in person_level_performance]),
        'precision': np.mean([person_level['precision'] for person_level in person_level_performance]),
        'f1': np.mean([person_level['f1'] for person_level in person_level_performance])
    }

    return {
        'image_level': image_level_result,
        'person_level': person_level_result
    
    }
        


def image_level_evaluation(results, labels):
    n_results = len(results)
    n_labels = len(labels)

    return (n_results - n_labels)**2


def person_level_evaluation(results, labels, width, height, max_distance=20):

    # Variables of interest for evaluation
    tp, fp, fn = 0, 0, 0

    # Define Euclidean distance
    def distance(x1, y1, x2, y2):
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)

    results_sorted_indices = []
    results_sorted_distances = []

    # Loop over all results
    for _, result in results.iterrows():
        x = result['x']
        y = result['y']

        distances = np.zeros(len(labels))

        # For each result, calculate the distance to all labels
        for idx_label, (_, label) in enumerate(labels.iterrows()):
            x_label = label['x']
            y_label = label['y']

            dist = distance(x, y, x_label, y_label)
            dist = dist if dist < max_distance else np.inf

            distances[idx_label] = dist


        # If all distances are inf, then it is a false positive and we can continue to the next result    
        if np.all(distances == np.inf):
            fp += 1
            continue


        # Sort the distances keeping track of the indices
        sorted_indices = np.argsort(distances)
        sorted_distances = distances[sorted_indices]

        # Save the sorted indices and distances
        results_sorted_indices.append(sorted_indices)
        results_sorted_distances.append(sorted_distances)

    # Solve collisions by assigning the label only to the closest result
    collisions = True
    while collisions:
        collisions = False

        # Compare each index with all the others to check if there are two results with the same label
        for i in range(len(results_sorted_indices)):
            for j in range(i + 1, len(results_sorted_indices)):

                # Take the first index of each result (assigned label)
                idx_i0 = results_sorted_indices[i][0]
                idx_j0 = results_sorted_indices[j][0]

                # If the indices are the same, then we have a collision
                if idx_i0 == idx_j0:

                    # Take the first distance of each result
                    dist_i0 = results_sorted_distances[i][0]
                    dist_j0 = results_sorted_distances[j][0]

                    # If one of the distances is inf we can skip this collision
                    if dist_i0 == np.inf or dist_j0 == np.inf:
                        continue

                    # We take the furthest result and remove it from the list
                    if dist_i0 < dist_j0:
                        results_sorted_indices[j] = results_sorted_indices[j][1:]
                        results_sorted_distances[j] = results_sorted_distances[j][1:]

                    else:
                        results_sorted_indices[i] = results_sorted_indices[i][1:]
                        results_sorted_distances[i] = results_sorted_distances[i][1:]

                    # If a collision was found, we start again from the beginning
                    collisions = True
                    break
                
            if collisions:
                break



    for sorted_distances in results_sorted_distances:

        if np.all(sorted_distances == np.inf):
            fp += 1
            continue

        tp += 1

    fn = len(labels) - tp


    accuracy = float(tp) / (tp + fp + fn)
    recall = float(tp) / (tp + fn)
    precision = float(tp) / (tp + fp)
    f1 = 2 * precision * recall / (precision + recall)

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1
    }