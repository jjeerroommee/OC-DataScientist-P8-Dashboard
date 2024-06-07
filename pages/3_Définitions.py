import streamlit as st

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
st.write("## Définitions des caractéristiques d'un client")
st.text("\n")
st.text("\n")

url = "https://www.kaggle.com/c/home-credit-default-risk/data"

st.markdown('''
Les données utilisées dans ce tableau de bord sont issues de la compétition Kaggle _Home Credit Default Risk_ parue en 2018.  
  
La description des variables fournies dans ce dataset est disponible [à cette adresse](%s).
''' % url)