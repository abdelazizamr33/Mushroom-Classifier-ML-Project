# importing libararies will be used
import pandas as pd
import pickle
import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder


# 1: Preprocess the data 
def preprocess(df,label_encoder):
    # using label encoder to change values of dtaset from str to int
    
    encoded_df=df.iloc[1:,1:].apply(label_encoder.fit_transform)
    
    # make the features and outputs extraction
    x=np.array(encoded_df)
    y=np.array(df['class'].iloc[1:])
    return x,y

# 2:Train the model
def train(x,y):
    #split train test samples
    x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.2)

    # The used models
    modelRF=RandomForestClassifier()
    modelKNN=KNeighborsClassifier()
    modelLR=LogisticRegression()
    modelRF.fit(x_train,y_train)
    modelKNN.fit(x_train,y_train)
    modelLR.fit(x_train,y_train)

    #making list of models that will be used in the app
    Models_list=[modelRF,modelKNN,modelLR]
    accuracy_score=[]
    model_name=[]

    for i in Models_list:
        accuracy=i.score(x_test,y_test)
        accuracy_score.append(accuracy)
        model_name.append(i)
    # make dictinary with each model and its accuracy value
    model_dictionary={model_name[0]:accuracy_score[0],model_name[1]:accuracy_score[1],model_name[2]:accuracy_score[2]}
    max_models_score=max(accuracy_score)
    # Now check which is the highest model accuracy that will be used in model
    for key, val in model_dictionary.items():
        if val == max_models_score:
              used_model=key
    
    return used_model


def prediction_encoder(predictions,label_encoder):
    encoded_predictions=label_encoder.fit_transform(np.array(predictions).reshape(-1,1))
    return encoded_predictions.flatten()


# 3: Save the trained model using pickle
def save_data(used_model):
    with open("mymodel3.pkl", "wb") as file:
        pickle.dump(used_model, file)
    
    st.session_state.save = True



# 4: call back functions that will be used down for buttons to be clicked in nested buttons
def call_back1():
    st.session_state.preprocess_clicked=True
def call_back2():
    st.session_state.train_clicked=True
def call_back3():
    st.session_state.predict_clicked=True
def call_back4():
    st.session_state.savebut_clicked=True
def call_back5():
    st.session_state.resetbut_clicked=True

# reset function
def reset():
    st.session_state.preprocess_clicked=False
    st.session_state.train_clicked=False
    st.session_state.predict_clicked=False
    st.session_state.savebut_clicked=False
    st.session_state.resetbut_clicked=False




# 5: main function of app
def main():
    # import image and title to webapp
    st.image('https://snipboard.io/CaVyOt.jpg')
    st.title("Mushroom Classification Project BY Abdelaziz amr")

    #enter the csv file
    csv_file = st.file_uploader("Upload CSV file", type="csv")
    st.session_state.save = False

    # import the label encoder to turn the values from string to int to start training it
    label_encoder=LabelEncoder()

    # check if there is a csv file to start or not
    if csv_file is not None:
        df = pd.read_csv(csv_file)
       
       #displaying the first five samples of data frame
        st.write("First five samples of the data frame:")
        st.dataframe(df.head())

        # making buttons ready to be clicked without any glitches
        if 'preprocess_clicked' not in st.session_state:
            st.session_state.preprocess_clicked = False
        if 'train_clicked' not in st.session_state:
            st.session_state.train_clicked = False
        if 'predict_clicked' not in st.session_state:
            st.session_state.predict_clicked = False
        if 'savebut_clicked' not in st.session_state:
            st.session_state.savebut_clicked = False
        if 'resetbut_clicked' not in st.session_state:
            st.session_state.resetbut_clicked = False


        #processing the data button
        if st.button("Preprocess",on_click=call_back1) or st.session_state.preprocess_clicked  :
            st.session_state.button1_clicked=True
            x, y = preprocess(df,label_encoder)
            st.write("Data preprocessed")



            # Check if the model is saved before or not
            if st.session_state.save==False:
              #start the train button
              if st.button("Train",on_click=call_back2) or st.session_state.train_clicked and st.session_state.preprocess_clicked==True :
                st.session_state.button2_clicked=True
                used_model=train(x, y)
                st.write("Please enter it as this: 'x', 's',...")
                predictions = st.text_input("Enter sample values:")
                # start predicting the values
                if st.button('Predict',on_click=call_back3) or st.session_state.predict_clicked and st.session_state.train_clicked==True :
                    if ' ' in predictions:
                        predictions.replace(' ','')
                    if predictions:
                        predictions=predictions.split(',')
                        prediction_list=[]
                        # here the program can predict by the numerical data also that is typed in the input textbox
                        # here we check if the predictions is a digit 
                        if predictions[0].isdigit():

                            # making a loop that will take the values from the textbox which is string
                            # and turn it into int values
                            for i in predictions:
                                value=int(i)
                                prediction_list.append(value)
                            
                            #passing the values to predict
                            y=used_model.predict([prediction_list])
                            if y=='e':
                                y='Edible'
                            elif y=='p':
                                y='Poisonous'
                            st.write('predicted value: ',y)

                        # here if we have a str values that you enter in textbox
                        else:

                            #change this str values into numerical alues that the odel can train with
                            predictions=prediction_encoder(predictions,label_encoder)
                            prediction_list=[]
                            for i in predictions:
                                if i==',':
                                    continue
                                else:
                                    value=int(i)
                                    prediction_list.append(value)
                            y=used_model.predict([prediction_list])
                            if y=='e':
                                y='Edible'
                            elif y=='p':
                                y='Poisonous'
                            st.write('predicted values: ',y)
                    else:
                        st.write('Please enter a value')
                    if st.button('Save',on_click=call_back4) or st.session_state.savebut_clicked and st.session_state.predict_clicked==True :
                        save_data(used_model)   
                        st.write('the model save is ',st.session_state.save)
                        st.write('Please Twice Tab reset to full reset of page')
                        if st.button('reset',on_click=call_back5) or st.session_state.resetbut_clicked and st.session_state.savebut_clicked==True :
                            reset()
                    

           # here if we have the model already saved before so we will run it without the train function used
           # and the same steps up that we made 
            elif st.session_state.save == True:
                with open("mymodel3.pkl", "rb") as file:
                    used_model = pickle.load(file)
                predictions = st.text_input("Enter prediction values:")
                if st.button('Predict',on_click=call_back3) or st.session_state.predict_clicked and st.session_state.train_clicked==True :
                    if ' ' in predictions:
                        predictions.replace(' ','')
                    if predictions:
                            predictions=predictions.split(',')
                            prediction_list=[]
                            # here the program can predict by the numerical data also that is typed in the input textbox
                            # here we check if the predictions is a digit 
                            if predictions[0].isdigit():

                                # making a loop that will take the values from the textbox which is string
                                # and turn it into int values
                                for i in predictions:
                                    value=int(i)
                                    prediction_list.append(value)
                                
                                #passing the values to predict
                                y=used_model.predict([prediction_list])
                                if y=='e':
                                    y='Edible'
                                elif y=='p':
                                    y='Poisonous'
                                st.write('predicted value: ',y)

                            # here if we have a str values that you enter in textbox
                            else:

                                #change this str values into numerical alues that the odel can train with
                                predictions=prediction_encoder(predictions,label_encoder)
                                prediction_list=[]
                                for i in predictions:
                                    if i==',':
                                        continue
                                    else:
                                        value=int(i)
                                        prediction_list.append(value)
                                y=used_model.predict([prediction_list])
                                if y=='e':
                                    y='Edible'
                                elif y=='p':
                                    y='Poisonous'
                                st.write('predicted values: ',y)
                    else:
                            st.write('Please enter a value')
                    if st.button('Save',on_click=call_back4) or st.session_state.savebut_clicked and st.session_state.predict_clicked==True :
                        save_data(used_model)   
                        st.write('the model save is ',st.session_state.save)
                        st.write('Please Twice Tab reset to full reset of page')
                        if st.button('reset',on_click=call_back5) or st.session_state.resetbut_clicked and st.session_state.savebut_clicked==True :
                            reset()

# values you can take to try the prediction: 'x', 's', 'n', 't', 'p', 'f', 'c', 'n', 'k', 'e', 'e', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'k', 's', 'u'
#                                            'x', 's', 'y', 't', 'a', 'f', 'c', 'b', 'k', 'e', 'c', 's', 's', 'w', 'w', 'p', 'w', 'o', 'p', 'n', 'n', 'g'

# Hope this project is perfect as you want :)

# Finally Calling the main function
main()
