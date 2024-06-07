import pandas as pd
import requests
import streamlit as st

@st.cache_data(show_spinner=False)
def load_customer_data() :
    # Load customers features as saved in .csv file thanks to a previous notebook
    clients = pd.read_csv("data/clients.csv", sep=";")
    
    # Use the line number as an ID for each customer
    #clients['ID'] = clients.index
    
    # Move the ID column at 1st position in dataframe
    #clients = clients[clients.columns.insert(0, 'ID')[:-1]]
    
    return clients

@st.cache_data(show_spinner=False)
def request_prediction(json_input):
    response = requests.request(
        method='POST',
        headers={"Content-Type": "application/json"},
        
        #url='http://localhost:5000/predict', # when using a local flask server
        url='https://oc-datascientist-p8-api.azurewebsites.net/predict/', # when using an Azure clouded server
        
        json=json_input)
    
    if response.status_code != 200 :
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()

def is_correct(input):
    # Vérifie si l'identifiant de client saisi par l'utilisateur est correct
    
    clients = pd.read_csv("data/clients.csv", sep=";")
    
    if input.isdigit() :
        input_num = int(input)
        if  (input_num >= 0) and (input_num < clients.shape[0]) :
            return True
            
    return False
