import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots  # <-- Import manquant


#import plotly.graph_objects as go

#------------------------------------------------------------------------------------------------------------------------
#fonction regroupement des colonnes
def regrouper_Q(df, prefixe_question, nom_colonne_finale):

    colonnes = [col for col in df.columns if col.startswith(prefixe_question + "_")]
    colonnes_parts = [col for col in colonnes if "Part" or "OTHER" in col]
    df[nom_colonne_finale] = df[colonnes_parts].apply(lambda x: ",".join(str(i) for i in x if pd.notna(i)) if any(pd.notna(x)) else np.nan,axis=1)
    df.drop(columns=colonnes, inplace=True)

    return df

#------------------------------------------------------------------------------------------------------------------------
#definir une fonction d'exploration et de nettoyage du dataframe

def clean_data(df) :
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
  return df
#------------------------------------------------------------------------------------------------------------------------
#Mise en page onglets Sidebar

st.title("Projet Data Job")
st.sidebar.title("Sommaire")
pages=["Le projet Data Job","Exploration des données bruttes","Pré-traitement et nettoyage","Profil des Répondants","Analyse Metiers","Méthodologie ou démarche de travail","Machine learning","conclusion"]
page=st.sidebar.radio("Aller vers", pages)

#------------------------------------------------------------------------------------------------------------------------
#configuration première page introduction et problématique

#------------------------------------------------------------------------------------------------------------------------
#configuration 2eme page Exploration des données bruttes" 

#------------------------------------------------------------------------------------------------------------------------
#recharger le dataframe en ignorant la premiere ligne et l'enregistrer dans le cache
@st.cache_data
def load_data():
     df=pd.read_csv('kaggle_survey_2020_responses.csv',skiprows=[1])
     df = clean_data(df)
     return df
df=load_data()
#----------------------------------------------------------------------------------------------------------------------------------
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


#------------------------------------------------------------------------------------------------------------------------
#configuration 2eme page Exploration des données bruttes" 

if page == pages[1] : 
  @st.cache_data
  def load_data():
   return pd.read_csv('kaggle_survey_2020_responses.csv')
  d=load_data()
  st.write("##### Apercu du jeu de donnée")
  st.dataframe(d.head(10))
  st.write("Dimenssion du DataFrame:",d.shape)
  st.write("##### Description du DataFrame")
  st.dataframe(d.describe())
  if st.checkbox("Afficher le type des colonne"):
    st.dataframe(d.dtypes)
  if st.checkbox("Afficher le pourcentage de valeurs manquantes  (NA)") :
     st.dataframe(d.isna().mean() * 100)
     #st.dataframe(df.isna().sum())
  if st.checkbox("Afficher le nombre de doublon") :
     st.write(d.duplicated().sum())
  

#------------------------------------------------------------------------------------------------------------------------
#recharger le dataframe en ignorant la premiere ligne et l'enregistrer dans le cache
else:
 
  @st.cache_data
  def load_data():
     df=pd.read_csv('kaggle_survey_2020_responses.csv',skiprows=[1])
     df = clean_data(df)
     return df
  df=load_data()

#------------------------------------------------------------------------------------------------------------------------
 #configuration 3eme page Pré-traitement et nettoyage df
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




#------------------------------------------------------------------------------------------------------------------------
 #configuration 3eme page Pré-traitement et nettoyage df

#------------------------------------------------------------------------------------------------------------------------
 
 #DATAVIZ

 #configuration 4eme page Analyse démographique"

if page == pages[3]:
    # =============================================
    # SECTION FILTRES - Dans la sidebar
    # =============================================
    st.sidebar.header("🔍 Filtres")
    
    with st.sidebar:
       top_10_label = "🏆TOP 10 Pays"
       all_label = "🌍Tous les pays"
       country_options = [top_10_label , all_label] + sorted(df['Country_of_residence'].dropna().unique()) 
       

# Widget
       selected_options = st.selectbox("Sélectionnez un pays: :", options=country_options,index=0 )

# Gérer la sélection
       
       if selected_options ==top_10_label:
          selected_countries = df['Country_of_residence'].value_counts().head(10).index.tolist()
       elif selected_options ==all_label :
          selected_countries = country_options
       else :
        selected_countries = [selected_options]


    # =============================================
    # SECTION PRINCIPALE - Visualisations
    # =============================================
    
    # Appliquer les filtres
    filtered_df = df[(df['Country_of_residence'].isin(selected_countries)) & df['Age'] & df['Education']& df['job_title']].copy()
        
        
        
    # KPI en haut de pageAge_str
    st.markdown(""" Cette visualisation interactive explore les caractéristiques clés des participants à l'enquête Kaggle 2020\:
                leurs provenance géographique,
                 âge, genre,
                Niveaux d'éducation, et
                Professions exercées. """)
    
    # Avertissement si aucun résultat
    if filtered_df.empty:
        st.warning("Aucun résultat ne correspond à vos critères de filtrage. Veuillez élargir votre sélection.")
    else:
        # Metrics row
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Nombre de répondants", filtered_df.shape[0])
        with col2:
            st.metric("Nombre de pays", filtered_df['Country_of_residence'].nunique())
        with col3:
            st.metric("Top Pays", filtered_df['Country_of_residence'].mode()[0] if not filtered_df.empty else "N/A")
        
        # Bouton de téléchargement
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger les données filtrées",
            data=csv,
            file_name='profils_repondants_filtrees.csv',
            mime='text/csv'
        )
        
        # Onglets pour organiser les visualisations
        tab1, tab2, tab3, tab4,tab5= st.tabs([
            "🌍 pays", 
            "👶🏽👴🏽 Tranche d'âge", 
            "🚻 Genre", 
            "🎓 Niveau d'éducation",
            "💻 Metier"
        ])
        
        with tab1:
            #st.subheader("Répartition géographique")

            country_counts = filtered_df['Country_of_residence'].value_counts().sort_values(ascending=False)
            fig = px.bar(
                    country_counts, 
                    orientation='h',
                    color=country_counts.values,
                    color_continuous_scale='Viridis',
                    labels={'value': 'Nombre Repondant', 'index': 'Pays'},
                    text=country_counts.values,
                    title="Top 10 des pays"
                )
            fig.update_layout(
            yaxis={'categoryorder':'total ascending'},  # Ordonner les catégories par valeur
            uniformtext_minsize=8,  # Taille minimale du texte
            coloraxis_showscale=False ) # Masquer l'échelle de couleur
    
            st.plotly_chart(fig, use_container_width=True)
            
        
        with tab2:
            #st.subheader("Analyse par tranche d'âges")
            age_counts = filtered_df['Age'].value_counts().sort_values(ascending=False)
            fig = px.bar(
                    age_counts,
                    color= age_counts.values,
                    color_continuous_scale='Viridis',
                    labels={'value': 'Nombre Repondants', 'index': 'Âge'},
                    text=age_counts.values,
                    title="Répartition par âge"
                )
            fig.update_layout(
            yaxis={'categoryorder':'total ascending'},  # Ordonner les catégories par valeur
            uniformtext_minsize=8,  # Taille minimale du texte
            coloraxis_showscale=False ) # Masquer l'échelle de couleur
    
            st.plotly_chart(fig, use_container_width=True)
        

        with tab3:
            
            if 'Gender' in filtered_df.columns:

                    gender_counts = filtered_df['Gender'].value_counts()
                    fig = px.pie(
                        gender_counts,
                        names=gender_counts.index,
                        values=gender_counts.values,
                        hole=0.4,
                        title="Répartition par genre"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.warning("La colonne 'Gender' n'est pas disponible dans les données.")
        

        with tab4:
            
            col1, col2 = st.columns([2, 1])
            with col1:
               
                edu_counts = filtered_df['Education'].value_counts().reset_index()
                edu_counts.columns = ['Education', 'count']  # Renommer les colonnes correctement
                
                fig = px.treemap(
                    edu_counts,
                    path=['Education'],  # Utiliser le nom de colonne correct
                    values='count',
                    color='count',
                    color_continuous_scale='Viridis',
                    title="Répartition par niveau d'éducation"
                )

                fig.update_layout(
                coloraxis_colorbar=dict(
                title="",  # Nouveau titre de la légende
                )) # Position du titre
    

                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.dataframe(
                    edu_counts.rename(columns={'Education': 'classement Niveau détude', 'count': 'Nombre'}),
                    hide_index=True
                )

        with tab5:
            
            if 'job_title' in filtered_df.columns:
                job_counts = filtered_df['job_title'].value_counts().sort_values(ascending=False)
                fig = px.bar(
                job_counts,
                orientation='h',
                text=job_counts.values,
                color= job_counts.values,
                color_continuous_scale='Viridis',
                labels={'value': 'Nombre de Repondants', 'index': 'Metier'},
                title="Répartition par Metier")
                fig.update_layout(
                yaxis={'categoryorder':'total ascending'},  # Ordonner les catégories par valeur
                uniformtext_minsize=14,  # Taille minimale du texte
                coloraxis_showscale=False ) # Masquer l'échelle de couleur
            st.plotly_chart(fig, use_container_width=True)
            
#------------------------------------------------------------------------------------------------------------------------  
if page == pages[4]:
    # =============================================
    # SECTION FILTRES - Dans la sidebar
    # =============================================
    
    
    
    #Filtre Pays

 # Liste des métiers liés à la data (extraite de votre fichier CSV)
  data_jobs = [
    "Data Scientist", "Data Analyst", 
    "Machine Learning Engineer", "Data Engineer",
    "DBA/Database Engineer"
 ]
  all_data=df['job_title'].dropna().unique().tolist()
 # Filtre Pays
  st.sidebar.header("🔍 Filtres")

  with st.sidebar:
     # Options pour les pays
     country_options = df['Country_of_residence'].dropna().unique().tolist()
     all_label = "🌍 Tous les pays"
     top_10_label = "🏆 Top 10 pays"

    # Widget de sélection des pays
     selected_country = st.selectbox(
     "Sélectionnez un pays:",
     options=[all_label, top_10_label] + sorted(country_options),
     index=1)  # Sélectionne 'all_label' par défaut


     # Gérer la sélection des pays
     if selected_country == top_10_label:
      selected_countries = df['Country_of_residence'].value_counts().head(10).index.tolist()
     elif selected_country == all_label:
        selected_countries = country_options
     else:
      selected_countries = [selected_country]  # Convertir en liste pour maintenir la compatibilité

     # Filtre Métier (version selectbox)
     metiers_options = ["🌟 Top métiers Data"] + sorted(all_data)
     selected_metier = st.selectbox("Sélectionnez un métier :", options=metiers_options,index=0 )
      
     
      # Sélectionne 'Top  métiers Data' par défaut


      # Gérer la sélection des métiers
     if selected_metier == "🌟 Top métiers Data":
      metiers_selectionnes = df[df["job_title"].isin(data_jobs)]["job_title"].value_counts().head(6).index.tolist()
     else:
      metiers_selectionnes = [selected_metier]

# Application des filtres
     if not selected_countries and not metiers_selectionnes:
    # Cas par défaut (normalement impossible avec nos sélections par défaut)
      df_filtre = df[df["job_title"].isin(data_jobs)]
      selected_countries = country_options
      st.info("ℹ️ Affichage des données globales pour les métiers Data.")
     else:
    # Cas général
      df_filtre = df[
        df["Country_of_residence"].isin(selected_countries) & 
        df["job_title"].isin(metiers_selectionnes)
    ]


# Le reste de votre code (tabs, visualisations) reste inchangé
   # Tabs
  tab1, tab2 = st.tabs(["Compétences techniques", "Salaire et emploi"])

  with tab1:

      st.markdown("""
     **🌐 Dashboard des Compétences techniques**  :Visualisations interactives.""")
     
      pages=["Les **langages de programmations** plébiscités ","Les **outils** de visualisation préférés",
             "Les **IDE** les plus utilisés","Les **frameworks ML** en vogue",
             "**Expérience** en programmation","Les **tâches clés** du métier"]
      page = st.radio("*Cliquez, filtrez et découvrez les insights!*", pages)
      
      if page == pages[0]:
         #st.subheader("Langages de programmation")
         if len(df_filtre) == 0:
          st.warning("Aucune donnée disponible avec les filtres sélectionnés.")
         else:
          lang_cols = ["Python", "R", "SQL", "Java", "C++", "JavaScript", "MATLAB", "Bash","Julia","Swift"]
        
        # Préparer les données avec le métier inclus
         lang_data = []
         for lang in lang_cols:
            for metier in df_filtre['job_title'].unique():
                count = df_filtre[(df_filtre['job_title'] == metier) & 
                                 (df_filtre['programming_languages'].str.contains(lang, na=False))].shape[0]
                lang_data.append({'Langage': lang, 'Métier': metier, 'Count': count})
        
         lang_df = pd.DataFrame(lang_data)
         # Créer le graphique
         fig = px.bar(lang_df,
                     x='Count',
                     y='Métier',
                     color='Langage',
                     title="Popularité des langages de programmation par métier",
                     labels={'Count': 'Nombre Repondants'},
                     barmode='group')  # 'group' pour des barres côte à côte
        
         st.plotly_chart(fig, use_container_width=True)


      elif page == pages[1]:
       #Fonction de nettoyage des données de visualisation"""
         def clean_data_viz(data):
          cleaned_data = (
          data.dropna()                  # Supprime les valeurs manquantes
          .str.split(',')                # Sépare par virgules
          .explode()                     # Crée une ligne par outil
          .str.strip()                   # Supprime les espaces
         .loc[lambda x: x != '']        # Filtre les chaînes vides
             )
          return cleaned_data.value_counts().reset_index()
    
       
         dfnew = df_filtre[["job_title", "Data_viz_libraries"]].copy()
         dfnew['Data_viz_libraries'] = dfnew['Data_viz_libraries'].str.split(',')
         dfnew = dfnew.explode('Data_viz_libraries')
         dfnew['Data_viz_libraries'] = dfnew['Data_viz_libraries'].str.strip()

         # 3. Compter les occurrences
         dfnew = dfnew.groupby(['job_title', 'Data_viz_libraries']).size().reset_index(name='counts')

         # 4. Visualisation
         fig=px.bar(
         dfnew,
         y='counts',
         x='job_title',
         color='Data_viz_libraries',
         title="Popularité des outils de Dataviz par métier",
         labels={"counts": 'Nombre Repondants','job_title':'Metier'},
         barmode='group'
        )
         st.plotly_chart(fig, use_container_width=True)
        
      elif page == pages[2]:
        
           dfnew = df_filtre[["job_title", "IDE"]].copy()
           dfnew['IDE'] = dfnew['IDE'].str.replace(r'\(.*?\)', '', regex=True).str.split(',')
           dfnew = dfnew.explode('IDE')
           dfnew['IDE'] = dfnew['IDE'].str.strip()
       
          # 3. Compter les occurrences
           dfnew = dfnew.groupby(['job_title', 'IDE']).size().reset_index(name='counts')
            # 4. Visualisation
           fig=px.bar(
           dfnew,
           y='counts',
           x='job_title',
           color='IDE',
           title="Popularité des outils IDE par métier",
            labels={"counts": 'Nombre Repondants','job_title':'Metier'},
           
           barmode='group'
        )
           st.plotly_chart(fig, use_container_width=True)
        
      elif page == pages[3]:
         dfnew = df_filtre[["job_title", "Ml_frameworks"]].copy()
         dfnew['Ml_frameworks'] = dfnew['Ml_frameworks'].str.replace(r'\(.*?\)', '', regex=True).str.split(',')
         dfnew = dfnew.explode('Ml_frameworks')
         dfnew['Ml_frameworks'] = dfnew['Ml_frameworks'].str.strip()
          # 3. Compter les occurrences
         dfnew = dfnew.groupby(['job_title', 'Ml_frameworks']).size().reset_index(name='counts')

         heatmap_data = dfnew.pivot(index="Ml_frameworks", 
                                 columns="job_title", 
                                 values="counts").fillna(0)
    
         fig = px.imshow(
         heatmap_data,
         labels=dict(x="Métier", y="Framework", color="Utilisateurs"),
         color_continuous_scale='Viridis',
         aspect="auto")
    
         fig.update_xaxes(side="top")
         st.plotly_chart(fig, use_container_width=True)

      elif page == pages[4]:
         dfnew= df_filtre.dropna(subset=['Coding_experience', 'Age','job_title']) 
         dfnew = dfnew[dfnew['Coding_experience'] != ''] 
         experience_order = ['I have never written code', '< 1 years', '1-2 years', '3-5 years', 
                    '5-10 years', '10-20 years', '20+ years']
         def sort_age(age_str):
           if isinstance(age_str, str):
            if '+' in age_str:
                return int(age_str.replace('+', ''))
            if '-' in age_str:
                return int(age_str.split('-')[0])
           return 0  # Valeur par défaut si format inattendu
    
    # Créer une colonne temporaire pour le tri
         dfnew['age_sort'] = dfnew['Age'].apply(sort_age)
         dfnew = dfnew.sort_values('age_sort')

         fig = px.histogram(dfnew, 
                  x="Age", 
                  color="Coding_experience",
                  facet_col="job_title",
                  facet_col_wrap=3,
                  category_orders={"Coding_experience": experience_order},
                  color_discrete_sequence=px.colors.sequential.Viridis,
                  labels={"count": 'Nombre Repondants'},
                  title="Expérience en programation par âge et métier")

         fig.update_layout(barmode='stack')
         fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
         st.plotly_chart(fig)
        
      elif page == pages[5]:
           dfnew = df_filtre[["job_title", "Role/Activities"]].copy()
           import re # Séparation uniquement sur les virgules sans espace
           dfnew['Role/Activities'] = dfnew['Role/Activities'].apply(lambda x: re.split(r',(?!\s)', str(x)) if pd.notna(x) else [])
    
    
           #dfnew['Role/Activities'] = dfnew['Role/Activities'].str.replace(r'\(.*?\)', '', regex=True).str.split(',')
           dfnew = dfnew.explode('Role/Activities')
           dfnew['Role/Activities'] = dfnew['Role/Activities'].str.strip()
       
          # 3. Compter les occurrences
           dfnew = dfnew.groupby(['job_title', 'Role/Activities']).size().reset_index(name='counts')
            # 4. Visualisation
           fig=px.bar(
           dfnew,
           y='Role/Activities',
           x='counts',
           color='job_title',
           orientation='h',
           title="Tâches/Activitées principales par métier",
           labels={'counts': 'Nombre Repondants','job_title':"Metier"},
           
           barmode='group')
           
           st.plotly_chart(fig, use_container_width=True)
          
  with tab2:  
     st.markdown("""
     **📊 Benchmark Salarial Intelligent**  
       Cette visualisation interactive révèle la distribution salariale au sein des professions data et autres.
       Explorez les dynamiques de rémunération à travers 2 dimensions (Par Métier et par Pays)
     """)

     pages=["Salaire par Metiers","salaire par expérience"]
     page = st.radio("Choisissez une visualisation", pages)
     if page == pages[0]: 
      dfnew = df_filtre.groupby(['job_title', 'Compensation']).size().reset_index(name='counts')
            # 4. Visualisation
      fig=px.bar(
           dfnew,
           y='Compensation',
           x='counts',
           color='job_title',
           title="remunération par métier",
           labels={
           'counts': 'Nombre de répondants',  
           'Compensation': 'Tranche de salaire',  
           'job_title': 'Métier'  
            },
           barmode='group')
 
      st.plotly_chart(fig, use_container_width=True)

     elif page == pages[1]:
         dfnew= df_filtre.dropna(subset=['Coding_experience', 'Compensation','job_title']) 
         dfnew = dfnew[dfnew['Coding_experience'] != ''] 
         experience_order = ['I have never written code', '< 1 years', '1-2 years', '3-5 years', 
                    '5-10 years', '10-20 years', '20+ years']

         fig = px.histogram(dfnew, 
                  x="Compensation", 
                  color="Coding_experience",
                  facet_col="job_title",
                  facet_col_wrap=3,
                  category_orders={"Coding_experience": experience_order},
                  color_discrete_sequence=px.colors.sequential.Viridis,
                  labels={
                  'counts': 'Nombre de répondants',  
                  'Compensation': 'Tranche de salaire',  
                 
                  },
                  title="salaire par Expérience et métier")

         fig.update_layout(barmode='stack')
         fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
         st.plotly_chart(fig)


#-----------------------------------------------------------------------------------





#-----------------------------------------------------------------------------------

st.sidebar.markdown("---")

st.sidebar.markdown("""
<div style="padding: 12px; border-left: 3px solid #FF4B4B; margin-bottom: 20px;">
    <div style="font-weight: 600; margin-bottom: 4px;">👩🏾💻 Olga TIWOA</div>
    <div style="color: #666; margin-bottom: 8px; font-size: 14px;"> Telecom&Networks Data Analyst Engineer</div>
    <a href="https://linkedin.com/in/olgatiwoa" style="color: #0077B5; text-decoration: none; font-size: 14px; display: block; margin-bottom: 8px;">LinkedIn Profile</a>
    <div style="font-size: 13px; color: #555;">🎓 DataScientest [mar25_bootcamp_da]</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("""
**Source des données:** [Kaggle ML & DS Survey 2020](https://www.kaggle.com/c/kaggle-survey-2020)
""")
st.sidebar.markdown("---")
