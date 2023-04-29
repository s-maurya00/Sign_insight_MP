# My functions/constants/variables
from utility import *
from model_tester import test_model


def begin_training(X_train, y_train, X_test, y_test):
    """
    Trains a machine learning model for action recognition using LSTM neural network.

    This function first creates an LSTM model for action recognition by calling the `create_model()` function from the `utility` module. It then compiles the model with the `compile_model()` function and trains the model using the `train_model()` function with the preprocessed training data `X_train` and `y_train` from the `data_preprocessing` module. 

    After the model is trained, this function saves the model's architecture as a JSON file named 'action_recog.json' and saves the trained weights as an HDF5 file named 'action_recog.h5'. Finally, it tests the accuracy of the trained model using the `test_model()` function from the `model_tester` module.

    Returns:
    None
    """

    # create a model 
    model = create_model()


    # Check if the model has been created
    if (model):
        print("Model has been created")
    else:
        print("model is not created successfully")


    # compile the model
    compile_model(model)


    # Check if the model has been compiled
    if (model.optimizer is not None):
        print('Model has been compiled')
    else:
        print("Model has not been compiled")


    # train the model
    train_model(model, X_train, y_train)


    # Check if the model has been trained
    if ((model.history is not None) and (model.history.history is not None)):
        print('Model has been trained')
    else:
        print('Model has not been trained')

    
    # Save the model's architecture to directory in json format
    model_json = model.to_json()
    with open("./Model/action_recog.json", "w") as json_file:
        json_file.write(model_json)

    # Save the model to directory in .h5 format
    model.save('./Model/action_recog.h5')

    test_model(model, X_test, y_test)

if __name__ == '__main__':

    # Import only if file is run independently
    from data_preprocessing import begin_data_preprocessing

    print("Data Preprocessing Started")
    X_train, y_train, X_test, y_test = begin_data_preprocessing()
    print("Data Preprocessing Completed")

    begin_training(X_train, y_train, X_test, y_test)