import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
st.title('Quelle est l\'espèce du gros penguin ?')
st.write(
  """Cette application utilise 6 entrées via le formulaire ci-dessous pour prédire l'espèce de penguin en utilisant un modèle de classification random forest construit à 
   partir d'un dataset de Kaggle."""
)

mdp = st.text_input('entrer le mdp')
if mdp != st.secrets['password']:
   st.stop()

penguin_file = st.file_uploader('Charger votre propre dataset sur les penguins')

if penguin_file is None:
 rf_pickle = open('random_forest_penguin.pickle', 'rb')
 map_pickle = open('output_penguin.pickle', 'rb')
 rfc = pickle.load(rf_pickle)
 unique_penguin_mapping = pickle.load(map_pickle)
 rf_pickle.close()
 map_pickle.close()

else:
 penguin_df = pd.read_csv(penguin_file)
 penguin_df = penguin_df.dropna()
 output = penguin_df['species']
 features = penguin_df[['island', 'bill_length_mm', 'bill_depth_mm',
 'flipper_length_mm', 'body_mass_g', 'sex']]
 features = pd.get_dummies(features)
 output, unique_penguin_mapping = pd.factorize(output)
 x_train, x_test, y_train, y_test = train_test_split(
 features, output, test_size=.8)
 rfc = RandomForestClassifier(random_state=15)
 rfc.fit(x_train.values, y_train)
 y_pred = rfc.predict(x_test.values)
 score = round(accuracy_score(y_pred, y_test), 2)
 st.write(
 f"""Score du Random Forest : {score}!"""
 )

 with st.form('user_inputs'):
    island = st.selectbox('Penguin Island', options= ['Biscoe', 'Dream',
'Torgerson'])
    sex = st.selectbox('Sex', options=['Female', 'Male'])
    bill_length = st.number_input('Bill Length (mm)', min_value=0)
    bill_depth = st.number_input('Bill Depth (mm)', min_value=0)
    flipper_length = st.number_input('Flipper Length (mm)', min_value=0)
    body_mass = st.number_input('Body Mass (g)', min_value=0)
    st.form_submit_button()
    island_biscoe, island_dream, island_torgerson = 0, 0, 0
    if island == 'Biscoe':
        island_biscoe = 1
    elif island == 'Dream':
        island_dream = 1
    elif island == 'Torgerson':
        island_torgerson = 1
    sex_female, sex_male = 0, 0
    if sex == 'Female':
        sex_female = 1
    elif sex == 'Male':
        sex_male = 1

    new_prediction = rfc.predict(
    [
        [
            bill_length,
            bill_depth,
            flipper_length,
            body_mass,
            island_biscoe,
            island_dream,
            island_torgerson,
            sex_female,
            sex_male,
        ]
    ]
    )
    prediction_species = unique_penguin_mapping[new_prediction][0]
    st.subheader("Prédiction de l\'espèce du penguin :")
    st.write(f"Votre penguin est de l\'espèce {prediction_species}")
    st.write(
    """Voici ci-dessous les features utilisés par le modèle classées par importance"""
    )
    st.image('feature_importance.png')
    st.write(
    """Voici ci-dessous les histogrammes de chaque variable continue du formulaire, chaque trait représente la valeur entrée par l\'utilisateur."""
    )
    fig, ax = plt.subplots()
    ax = sns.displot(x=penguin_df['bill_length_mm'],
    hue=penguin_df['species'])
    plt.axvline(bill_length)
    plt.title('Bill Length by Species')
    st.pyplot(ax)
    fig, ax = plt.subplots()
    ax = sns.displot(x=penguin_df['bill_depth_mm'],
    hue=penguin_df['species'])
    plt.axvline(bill_depth)
    plt.title('Bill Depth by Species')
    st.pyplot(ax)
    fig, ax = plt.subplots()
    ax = sns.displot(x=penguin_df['flipper_length_mm'],
    hue=penguin_df['species'])
    plt.axvline(flipper_length)
    plt.title('Flipper Length by Species')
    st.pyplot(ax)
