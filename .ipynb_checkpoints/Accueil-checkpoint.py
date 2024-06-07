import streamlit as st

# Page
st.set_page_config(page_title="Prêt à dépenser", page_icon="💰")

# Sidebar
with st.sidebar :
    st.caption("_Menu_")

# Main content
st.image('static/images/logo.png')
st.write("## Tableau de bord intéractif de l'outil de scoring")
#st.text("\n")
st.markdown("Utilisez le menu de gauche pour naviguer entre les pages du dashboard :  ")
st.text("\n")
st.markdown('''
-  **Demande de crédit** : Calcul et analyse du score retourné par le modèle de prédiction suite à une demande de crédit.

    
-  **Visualisation client** : Affichage des caractéristiques d'un client ainsi que son positionnement par rapport à la moyenne de la clientèle.


-  **Définitions** : Liste et signification des caractéristiques client utilisés dans le dashboard.
''')


