import streamlit as st
import sys
import numpy as np
import matplotlib.pyplot as plt
import shap
#st.set_option('deprecation.showPyplotGlobalUse', False)

sys.path.append('./functions')
from dash_functions import load_customer_data
from dash_functions import request_prediction
from dash_plots import my_waterfall

# -----------------------------------------------------

def display_score():
    st.session_state.display_score = True

def display_analysis():
    st.session_state.display_analysis = True
    
def reset_widgets():
    st.session_state.display_score = False
    st.session_state.display_analysis = False

def input_id():
    st.session_state.global_user_input = st.session_state.user_input
    st.session_state.id = int(st.session_state.user_input) if st.session_state.user_input.isdigit() else None
    st.session_state.print_msg = True 

    st.session_state.display_score = False
    st.session_state.display_analysis = False
# -----------------------------------------------------
# Page display
# -----------------------------------------------------

# Page
st.set_page_config(page_title="Prêt à dépenser", page_icon="💰")

# Sidebar
with st.sidebar :
    st.caption("_Menu_")

# Main content
st.image('static/images/logo.png')
st.write("## Analyse d'une décision d'octroi de crédit")
st.text("\n")
st.text("\n")

# -----------------------------------------------------
# Init variables
# -----------------------------------------------------

clients = load_customer_data()

id = None
pred = None
   
if 'id' not in st.session_state: st.session_state.id = None
if 'display_score' not in st.session_state: st.session_state.display_score = False
if 'print_msg' not in st.session_state: st.session_state.print_msg = False
if 'display_analysis' not in st.session_state: st.session_state.display_analysis = False
if 'global_user_input' not in st.session_state: st.session_state.global_user_input = None

# -----------------------------------------------------
# Get the client ID
# -----------------------------------------------------
st.text_input(label="Identifiant client",
              value = st.session_state.id if st.session_state.id is not None else st.session_state.global_user_input,
              key='user_input',
              help=f"Nombre entier entre 0 et {clients.shape[0]-1} correspondant au n° de ligne dans application_test.csv",
              placeholder="Saisir un identifiant",
              on_change=input_id
)

if st.session_state.print_msg :
    if st.session_state.global_user_input.isdigit() : 
        st.markdown(f"Identifiant du client sélectionné : :grey-background[{st.session_state.id}]")
    else :
        st.error("Cet identifiant ne correspond à aucun client.")

st.divider()

# -----------------------------------------------------
# Launch and display the model's prediction
# -----------------------------------------------------
predict_btn = st.button('Etudier la demande de crédit', disabled = (st.session_state['id'] is None), on_click=display_score)

if st.session_state['display_score'] :
    # Remove NaN value and convert to json format
    client = clients.fillna('').loc[st.session_state['id'], :].to_dict()
    client_values = list(clients.loc[st.session_state['id'], :])
    json_input = {'data': client, 'id' : st.session_state['id']}
    
    # Get results from the API
    pred = request_prediction(json_input)
    
    if pred['classe'] == 'accepté' :
        status_color = 'lightgreen'
        reponse = 'crédit accordé  '
        st.write(f'Réponse : :grey-background[:green[{reponse}]]')
    else :
        status_color = 'lightcoral'
        reponse = 'crédit refusé  '
        st.write(f'Réponse : :grey-background[:red[{reponse}]]')


    fig, ax = plt.subplots(figsize=(5, 0.2))
    bars = ax.barh(y=['Score'], width=round(100*(1-pred['proba_echec']), 1), height=0.2, color=status_color)
    
    ax.bar_label(bars, fontsize = 6)
    ax.yaxis.set_visible(False)
    ax.set_xticks([0, 50, 100], labels=['0', "seuil d'octroi", '100'], fontsize = 6, style='italic', color = 'gray')
    ax.tick_params(width=0.5)
    
    plt.axvline(x=50, linewidth=0.5, color='black')
    plt.title('Score client : probabilité de remboursement (%)', fontsize=6)
    st.pyplot(fig)
    
    st.caption("_Le crédit est accordé lorsque le modèle prédictif retourne un score supérieur à 50._")

st.divider()

# -----------------------------------------------------
# Launch and display SHAP analysis
# -----------------------------------------------------

analyze_btn = st.button('Voir les détails de la réponse', disabled = (pred is None), on_click=display_analysis)

if st.session_state['display_analysis'] :
    
    # we want to print a score value : a 0 to 100 value with 100 being the score of the perfect customer 
    customer_shap = shap.Explanation(
        values = 100 * (-1) * np.array(pred['shap_values']), 
        base_values = 100 * (1 - pred['shap_base_values']),
        data = client_values,
        #data=list(client.values()),
        feature_names=list(client.keys())
    )
    
    st.write("Principales contributions des caractéristiques du client à la décision :")

    fig, ax = plt.subplots()
    #st.pyplot(shap.waterfall_plot(customer_shap))
    st.pyplot(my_waterfall(customer_shap, max_display=21))

st.divider()

