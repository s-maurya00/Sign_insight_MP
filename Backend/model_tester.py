import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

# My functions/constants/variables
from data_preprocessing import X_test, y_test

def test_model(model):
    """
    The test_model function tests the accuracy of a given model on a test dataset.

    Args:
    model: A trained Keras model.

    Returns:
    None. The function prints the multilabel confusion matrix and accuracy score for the model's performance on the test dataset.
    """

    yhat = model.predict(X_test)

    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()

    print(multilabel_confusion_matrix(ytrue, yhat))

    print(accuracy_score(ytrue, yhat))