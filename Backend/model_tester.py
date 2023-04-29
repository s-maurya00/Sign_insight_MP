import numpy as np
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score


def test_model(model, X_test, y_test):
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

    print("\nmultilabel_confusion_matrix of the trained model is: \n", multilabel_confusion_matrix(ytrue, yhat))

    print("\naccuracy_score of the trained model is: ", accuracy_score(ytrue, yhat))