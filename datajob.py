import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("kaggle_survey_2020_responses.csv")
st.title("Projet Data Job")
st.sidebar.title("Sommaire")
pages=["Le projet Data Job","Exploration des données bruttes","Pré-traitement et nettoyage", "Analyse démographique", "Analyse Compétences techniques"]
page=st.sidebar.radio("Aller vers", pages)

#introduction et problématique
if page == pages[0] : 
  st.write("### Intruduction ")
  st.markdown("""
Le sondage « Kaggle 2020 » fournit une base de données riche pour comprendre les métiers de la science des données et 
leur évolution. Réalisée par la plateforme Kaggle, l’enquête recueille les réponses de plus de 20 000 professionnels 
du monde entier, abordant leurs rôles, outils, langages, plateformes, formations et pratiques. Cette diversité offre 
un aperçu représentatif du secteur à l’échelle internationale.
""")

  st.write("### Problématique & Objectifs")
  st.markdown("""La problémaatique retenue est la suivante :
              
Comment les réponses à l’enquête ou au sondage Kaggle 2020 permettent-elles de comprendre et de caractériser les différents profils techniques au sein de l'industrie de la Data, en se concentrant sur les tâches effectuées et les outils utilisés ?

L’analyse vise à:

- Identifier les outils et technologies les plus utilisés (langages, frameworks, IDE).

- Comprendre le profil démographique des répondants (âge, genre, pays, éducation).

- Explorer les pratiques professionnelles (fonctions occupées, usage du machine learning).

- Détecter les tendances futures à travers les outils que les répondants souhaitent apprendre d’ici deux ans.
""")
  
#Explotation

if page == pages[1] : 
  
  df = pd.read_csv('kaggle_survey_2020_responses.csv')
  st.write("##### Apercu du jeu de donnée")
  st.dataframe(df.head(10))
  st.write("Dimenssion du DataFrame:",df.shape)
  st.write("##### Description du DataFrame")
  st.dataframe(df.describe())
  if st.checkbox("Afficher le type des colonne"):
    st.dataframe(df.dtypes)
  if st.checkbox("Afficher le pourcentage de valeurs manquantes  (NA)") :
     st.dataframe(df.isna().mean() * 100)
     #st.dataframe(df.isna().sum())
  if st.checkbox("Afficher le nombre de doublon") :
     st.write(df.duplicated().sum())


else:
  #recharger le dataframe en ignorant la premiere ligne
  @st.cache_data
  def load_data():
    # Charger le fichier CSV
     df = pd.read_csv('kaggle_survey_2020_responses.csv',skiprows=[1])
     return df
  
  df=load_data()
  #fonction Regroupement des colonnes
  def regrouper_Q(df, prefixe_question, nom_colonne_finale):
    """
    Regrouper les colonnes d'une question à choix multiples
    en une seule colonne contenant une liste, puis supprime les colonnes originales.

    Args:
        df : DataFrame
        prefixe_question : str, le préfixe commun (ex: "Q7")
        nom_colonne_finale : str, nom de la nouvelle colonne à créer (ex: "Languages")

    Returns:
        df mis à jour avec la nouvelle colonne et sans les colonnes originales
    """

    colonnes = [col for col in df.columns if col.startswith(prefixe_question + "_")]
    colonnes_parts = [col for col in colonnes if "Part" or "OTHER" in col]
    #df[nom_colonne_finale] = df[colonnes_parts].apply(lambda x: [i for i in x if pd.notna(i)] if any(pd.notna(x)) else np.nan,axis=1)
    df[nom_colonne_finale] = df[colonnes_parts].apply(lambda x: ",".join(str(i) for i in x if pd.notna(i)) if any(pd.notna(x)) else np.nan,axis=1)
    df.drop(columns=colonnes, inplace=True)

    return df
  
  #application de la fonction Regroupement des questions
  
  questions = [
    "Q7", "Q9", "Q10", "Q12", "Q14", "Q16", "Q17", "Q18", "Q19", "Q23",
    "Q26_A", "Q27_A", "Q28_A", "Q29_A", "Q31_A", "Q33_A", "Q34_A", "Q35_A",
    "Q36", "Q37", "Q39",
    "Q26_B", "Q27_B", "Q28_B", "Q29_B", "Q31_B", "Q33_B", "Q34_B", "Q35_B"]
 

  for q in questions:
    df = regrouper_Q(df, q, q)

   #Les colonnes ont été regroupé dans le désordre, donc cette ligne sert à les remettre dans l'ordre croissant des questions
  from natsort import natsorted #importer la librairie natsorted sur python pour pouvoir ordonner les colonnes grace a 1 ligne de code

  df = df[natsorted(df.columns)]

  df=df.drop('Time from Start to Finish (seconds)',axis=1)
  

  #df.to_csv('kaggle_survey_2020_group_Q.csv', index=False)
  #df.head()

  #Suppression des doublons en gardant seulement la 1ère ligne du doublon
  df.drop_duplicates(keep = 'first', inplace=True)

  ## Suppresion des colonnes avec plus de 80% de valeurs manquantes
  # Seuil de 80%
  seuil = 0.80

  # Calcul du pourcentage de valeurs manquantes par colonne
  pourcent_nan = df.isna().mean()

  # Filtrer les colonnes à supprimer
  colonnes_a_supprimer = pourcent_nan[pourcent_nan > seuil].index

  # Suppression des colonnes avec un taux de valeur manquante >= 80% 
  df.drop(columns=colonnes_a_supprimer, inplace=True)

  # Optionnel : afficher les colonnes supprimées
  #print(f"{len(colonnes_a_supprimer)} colonnes supprimées (plus de 80% de NaN) :")
  #print(list(colonnes_a_supprimer))
  #suppression des colonnes inutiles
  df.drop(columns=['Q8', 'Q12', 'Q13', 'Q17', 'Q18', 'Q21', 'Q25', 'Q26_B', 'Q27_B', 'Q28_B', 'Q29_B', 'Q31_B', 'Q33_B', 'Q34_B', 'Q35_B', 'Q33_A', 'Q35_A'], inplace=True)
  #Suppression des espaces vides début et fin de cellule
  df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)
  #6. Renommer les colonnes
  new_column_names = {
    'Q1': 'Age',
    'Q2': 'Gender',
    'Q3': 'Country_of_residence',
    'Q4': 'Education',
    'Q5': 'job_title',
    'Q6': 'Coding_experience',
    'Q7': 'programming_languages',
    'Q9': 'IDE',
    'Q10':'Hosted_notebook',
    'Q11':'Computing_platform',
    'Q14':'Data_viz_libraries',
    'Q15':'Ml_experience',
    'Q16':'Ml_frameworks',
    'Q19':'NLP',
    'Q20':'Company_size',
    'Q22':'Ml_incorporation',
    'Q23':'Role/Activities',
    'Q24':'Compensation',
    'Q26_A':'Cloud_platforms',
    'Q27_A':'Cloud_products',
    'Q28_A':'ML_products',
    'Q29_A':'Bigdata_products',
    'Q30':'Bigdata_products-twins',
    'Q31_A':'BI_tools',
    'Q32':'BI_tools-twins',
    'Q34_A':'Auto_ML_tools',
    'Q36':'Sharing_platforms',
    'Q37':'Courses_Platforms',
    'Q38':'Work/School_primaryTools',
    'Q39':'Media_sources'

   }
  df.rename(columns=new_column_names, inplace=True)

   # Appliquer le renommage
  if page == pages[2] :
    st.markdown("""Un nettoyage des données a été effectué pour réduire la taille du DataFrame de 355 à 25 colonnes,
   en regroupant les questions à choix multiples. Un traitement des valeurs manquantes (suppression des colonnes avec 
   plus de 80 %\ de valeurs manquantes) et des doublons a également été réalisé. Enfin, les colonnes ont été renommées avec des noms explicites pour faciliter la lecture et l’analyse
   """)
    col1, col2 = st.columns([3, 1])
    with col1:
     
     st.write("##### Apercu du jeu de donnée")
     st.dataframe(df.head(10))
     st.write("Dimenssion du DataFrame:",df.shape)
     st.write("##### Description du DataFrame")
     st.dataframe(df.describe())
    
     with col2:
      st.write("##### Statistiques descriptives")
      st.write(f"Nombre total de répondants: {len(df)}")
        
      st.metric("Pourcentage d'hommes",f"{round(df[df['Gender']=='Man'].shape[0]/len(df)*100, 1)}%")
      st.metric("Pourcentage de femmes", f"{round(df[df['Gender']=='Woman'].shape[0]/len(df)*100, 1)}%")

    #DATAVIZ
  if page == pages[3] :
   col1, col2 = st.columns([3, 1])  # 3:1 ratio pour le contenu vs filtres
   with col2:
        top_10 = df['Country_of_residence'].value_counts().head(10).index.tolist()
        
        st.subheader("Filtres")
        use_top_10 = st.checkbox("Sélectionner automatiquement le Top 10 des pays", value=True)
        if use_top_10:
         default_selection = top_10
        else:
         default_selection = []
        selected_countries = st.multiselect(
        "Ou Choisissez les pays :",
        options=sorted(df['Country_of_residence'].unique()),
        default=default_selection)


        selected_ages = st.multiselect(
       "Sélectionnez des tranches d'âge", 
       options=df['Age'].unique(),
       default=df['Age'].unique())


        selected_education = st.multiselect(
       "Sélectionnez des niveaux d'éducation", 
       options=df['Education'].unique(),
       default=df['Education'].unique())


# Appliquer les filtres
   filtered_df = df[
    (df['Country_of_residence'].isin(selected_countries)) & 
    (df['Age'].isin(selected_ages)) & 
    (df['Education'].isin(selected_education))]


   with col1:
        st.header("👥 Analyse Démographique")
        
        # Visualisation 1 - Répartition par pays
        st.subheader("Top 10 Répartition par pays")
        country_counts = filtered_df['Country_of_residence'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(y=country_counts.index, x=country_counts.values, palette="viridis")
        plt.xlabel("Nombre de répondants")
        st.pyplot(fig)
        
        # Visualisation 2 - Répartition par âge
        st.subheader("Répartition par âge")
        age_counts = filtered_df['Age'].value_counts()
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(x=age_counts.index, y=age_counts.values, palette="viridis")
        plt.xticks(rotation=45)
        plt.xlabel("Tranche d'âge")
        plt.ylabel("Nombre de répondants")
        st.pyplot(fig)

        # Visualisation 3 - Répartition par genre
        st.subheader("Répartition par genre")
        gender_counts = filtered_df['Gender'].value_counts()
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.pie(gender_counts, labels=gender_counts.index,textprops={'fontsize': 8},autopct='%1.1f%%',pctdistance=0.8)
        plt.title("Répartition par genre")
        st.pyplot(fig)

        # Visualisation 3 - Répartition niveau D'éducation
        st.subheader("Top 10 niveau D'éducation")
        Education_counts = filtered_df['Education'].value_counts().head(10)
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.barplot(y=Education_counts.index, x=Education_counts.values, palette="viridis")
        plt.xlabel("Nombre de répondants")
        st.pyplot(fig)
    

  
   
    












 
 
 
 
 
 
 
 
 
 
# Pied de page
st.markdown("---")
st.markdown("""
**Source des données:** [Kaggle ML & DS Survey 2020](https://www.kaggle.com/c/kaggle-survey-2020)
""")