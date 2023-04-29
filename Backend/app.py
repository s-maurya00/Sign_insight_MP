from utility import create_model, compile_model
from data_collection import start_data_collection
from data_preprocessing import begin_data_preprocessing
from model_trainer import begin_training


choice = input("\nChoose:\n1. Collect data and train the model\n2. Use collected data for training the model\n3. Use the trained model for making predictions\n--> ")


# Collect data and train model
if(choice == '1'):
    
    print("Data Collection Started")
    start_data_collection()
    print("Data Collection Completed")

    print("Data Preprocessing Started")
    X_train, y_train, X_test, y_test = begin_data_preprocessing()
    print("Data Preprocessing Completed")

    begin_training(X_train, y_train, X_test, y_test)
    print("Training of model is completed successfully")


# Use collected data for training the model
elif(choice == '2'):

    print("Data Preprocessing Started")
    X_train, y_train, X_test, y_test = begin_data_preprocessing()
    print("Data Preprocessing Completed")

    begin_training(X_train, y_train, X_test, y_test)


# Use trained model for making predictions
elif(choice == '3'):

    model = create_model()

    compile_model(model)

    model.load_weights('./Model/action_recog.h5')
    # incomplete ==> create a file for doing predictions on using this model


else:
    print("\nNo such option exists!!!")