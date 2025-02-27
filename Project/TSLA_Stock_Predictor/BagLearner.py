import numpy as np

class BagLearner:
    def __init__(bag_model, learner, kwargs={}, bags=50, boost=False, verbose=False):
        """
        Initialize Bag Learner with multiple weak learners.

        :param learner: The base learner (e.g., DTLearner, RTLearner)
        :param kwargs: Arguments for the base learner
        :param bags: Number of learners (default: 20)
        :param boost: Boosting flag (not required)
        :param verbose: Debugging mode
        """
        bag_model.bags = bags
        bag_model.learners = [learner(**kwargs) for _ in range(bags)]
        bag_model.verbose = verbose

    def add_evidence(bag_model, data_x, data_y):
        """
        Train the ensemble learners using bootstrap aggregation.

        :param data_x: Training feature data (NumPy array)
        :param data_y: Training target values (NumPy array)
        """
        n_samples = data_x.shape[0]
        for model in bag_model.learners:
            sample_indices = np.random.choice(n_samples, n_samples, replace=True)
            model.add_evidence(data_x[sample_indices], data_y[sample_indices])

    def query(bag_model, points):
        """
        Make predictions by averaging over all base learners.

        :param points: Test feature data
        :return: Averaged predictions from all learners
        """
        predictions = np.array([model.query(points) for model in bag_model.learners])
        return np.mean(predictions, axis=0)  # Averaging for regression

    def author(bag_model):
        return "your_gt_username"

    def study_group(bag_model):
        return "your_study_group"
