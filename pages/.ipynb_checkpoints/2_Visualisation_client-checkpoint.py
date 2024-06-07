import streamlit as st
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.append('./functions')
from dash_functions import load_customer_data
from dash_functions import request_prediction
from dash_functions import is_correct
from dash_plots import my_waterfall
from dash_plots import my_histo
from dash_plots import my_scatter

# -----------------------------------------------------

def input_id():
    st.session_state.global_user_input = st.session_state.user_input
    st.session_state.id = int(st.session_state.user_input) if is_correct(st.session_state.user_input) else None
    st.session_state.print_msg = True 

    st.session_state.display_score = False
    st.session_state.display_analysis = False

def df_select():
    if len(st.session_state.df_event.selection.rows) == 0 : st.session_state.global_feat1 = None
    if len(st.session_state.df_event.selection.rows) == 1 : st.session_state.global_feat1 = features[st.session_state.df_event.selection.rows[-1]]
        
def feat1_change():
    st.session_state.global_feat1 = None if st.session_state.feat1 == '(effacer la sélection)' else st.session_state.feat1

def feat2_change():
    st.session_state.global_feat2 = None if st.session_state.feat2 == '(effacer la sélection)' else st.session_state.feat2
    

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
st.write("## Affichage des caractéristiques d'un client")
st.text("\n")
st.text("\n")

# -----------------------------------------------------
# Init variables
# -----------------------------------------------------

clients = load_customer_data()
features = clients.columns.sort_values().to_list()

if 'id' not in st.session_state: st.session_state.id = None
if 'global_feat1' not in st.session_state : st.session_state.global_feat1 = None
if 'global_feat2' not in st.session_state: st.session_state.global_feat2 = None
if 'global_user_input' not in st.session_state: st.session_state.global_user_input = None
if 'print_msg' not in st.session_state: st.session_state.print_msg = False

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
    if is_correct(st.session_state.global_user_input) : 
        st.markdown(f"Identifiant du client sélectionné : :grey-background[{st.session_state.id}]")
    else :
        st.error(f"Cet identifiant ne correspond à aucun client : veuillez saisir un nombre entre 0 et {clients.shape[0]-1}")

st.divider()
st.write("### Caractéristique du client")
if st.session_state['id'] is not None : 
    df = pd.DataFrame(clients.loc[st.session_state['id']])
    df.reset_index(inplace=True)
    df.columns = ['Caractéristique', 'Valeur']
    df.sort_values('Caractéristique', inplace=True)
    a=st.dataframe(df,
                 hide_index=True,
                 column_config={
                     'Caractéristique' : st.column_config.Column(width = 'large'),
                     'Valeur' : st.column_config.NumberColumn(width = 'medium', format="%.2f")},
                 key='df_event',
                 on_select=df_select,
                 selection_mode="single-row"
                )

st.caption("___Aide___ _: Cocher une ligne dans ce tableau pour la représenter visuellement dans la section suivante._")

# -----------------------------------------------------
# Launch and display the model's prediction
# -----------------------------------------------------
st.divider()
st.write("### Comparaisons avec la moyenne des clients")
st.markdown('''
Sélectionner 1 caractéristique afin d'afficher l'histogramme associé.  
Sélectionner 2 caractéristiques pour afficher un nuage de points où (x, y) = (attribut 1, attribut 2)  
''')
st.text("\n")
col1, col2 = st.columns(2)


with col1:
    feat1_list = features.copy()
    feat1_list.insert(0, '(effacer la sélection)')
    st.selectbox(
        "Caractéristique n°1",
        feat1_list,
        index = None if st.session_state['global_feat1'] is None else features.index(st.session_state['global_feat1']) + 1,
        key = 'feat1',
        placeholder= 'Liste' if st.session_state['global_feat1'] is None else st.session_state['global_feat1'],
        on_change = feat1_change,
    )

with col2:
    feat2_list = features.copy()
    feat2_list.insert(0, '(effacer la sélection)')
    st.selectbox(
        "Caractéristique n°2",
        feat2_list,
        index = None if st.session_state['global_feat2'] is None else features.index(st.session_state['global_feat2']) + 1,
        key = 'feat2',
        placeholder = "Liste" if st.session_state['global_feat2'] is None else st.session_state['global_feat2'],
        on_change = feat2_change
    )

# Aide aux utilisations
st.caption("___Aide___ _: Saisir quelques caractères permet d'appliquer un filtre sur la liste._")

# Affichage d'un histogramme si 1 variable sélectionnée
if (st.session_state['global_feat1'] is not None) and (st.session_state['global_feat2'] is None) and (st.session_state['id'] is not None):
    serie = clients[st.session_state['global_feat1']]
    client_value = clients.loc[st.session_state['id'], st.session_state['global_feat1']]
    st.text("\n")
    st.text("\n")
    st.markdown(f'''
        Valeurs de {st.session_state.global_feat1} :    
         - pour le client : :grey-background[{client_value:.2f}]
         - moyenne autres clients : :grey-background[{serie.mean():.2f}]
        ''')  
    st.text("\n")
    st.text("\n")
    st.markdown(f"Histogramme de {st.session_state.global_feat1} :")
    fig, ax = my_histo(serie=serie, client_value=client_value)
    st.pyplot(fig)
    st.caption("___Note___ _: L'histogramme est borné à la plage de valeurs correspondant aux pourcentiles 5% - 95% (plage étendue si nécessaire pour inclure la valeur du client)._")

# Affichage d'un nuage de points si 2 variables sélectionnées
if (st.session_state['global_feat1'] is not None) and (st.session_state['global_feat2'] is not None) and (st.session_state['id'] is not None):
    df = clients[[st.session_state.global_feat1, st.session_state.global_feat2]]
    client_values = (clients.loc[st.session_state.id, st.session_state.global_feat1], clients.loc[st.session_state.id, st.session_state.global_feat2])
    st.text("\n")
    st.text("\n")
    st.markdown(f'''
        Valeurs pour le client :  
         - {st.session_state.global_feat1} : :grey-background[{client_values[0]:.2f}]  
         - {st.session_state.global_feat2} : :grey-background[{client_values[1]:.2f}]
    ''')
    st.markdown(f'''
        Moyenne autres clients :  
         - {st.session_state.global_feat1} : :grey-background[{df.iloc[:, 0].mean():.2f}]
         - {st.session_state.global_feat2} : :grey-background[{df.iloc[:, 1].mean():.2f}]
    ''')  
    st.text("\n")
    st.text("\n")
    st.markdown(f"Nuage de points {st.session_state.global_feat1} vs. {st.session_state.global_feat2} :")
    fig, ax = my_scatter(df=df, client_values=client_values)
    st.pyplot(fig)
    st.caption("___Note___ _: Le nuage de points est borné aux plages de valeurs correspondant aux pourcentiles 5% - 95% sur chaque axe (plage étendue si nécessaire pour inclure la valeur du client)._")

    
   