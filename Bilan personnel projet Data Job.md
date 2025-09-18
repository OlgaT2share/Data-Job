# Bilan Personnel - Projet Data Job

## 1. Méthodologie de Travail et Rôle dans l'Équipe

Notre équipe a adopté une méthodologie de travail collaborative et efficace. La démarche employée était la suivante :

- Pour chaque tâche, chaque membre réalisait une analyse de manière indépendante.
- Des points de synchronisation réguliers étaient organisés pour partager nos découvertes et nos approches.
- Nous sélectionnions collectivement la solution la plus simple comme base de travail commune.
- Nous l'enrichissions ensuite en intégrant des optimisations issues des meilleures idées et améliorations proposées par chacun.

Mon rôle dans cette équipe a été actif et contributeur. J'ai mené mes explorations de manière autonome tout en participant activement aux réunions d'équipe pour partager mes résultats, challenger ceux des autres, et converger vers une analyse cohérente et de qualité.

## 2. Réalisation Technique Personnelle : Application Streamlit

**Lien vers l'application Streamlit développée personnellement** :  
[https://data-job-khzdnzizvppgvfayrhm5mx.streamlit.app/](https://data-job-khzdnzizvppgvfayrhm5mx.streamlit.app/)

J'ai conçu et développé cette application Streamlit dans le but d'analyser et de visualiser les résultats de l'enquête Kaggle 2020 sur le Machine Learning et la Science des Données, en proposant une analyse personnelle et originale.

Le projet, entièrement fonctionnel, est versionné sur GitHub et déployé sur le cloud Streamlit, permettant des mises à jour en continu et un partage aisé. Je poursuis activement son optimisation, avec pour prochaine étape l'intégration de fonctionnalités de Machine Learning à l'issue de la formation dédiée.

L'application illustre déjà ma philosophie de développement :
- ✅ **Interface épurée**
- ✅ **Performances optimisées** (grâce à `@st.cache_data` pour une exécution rapide)
- ✅ **Visualisations interactives et pertinentes**

Elle intègre également certaines de mes propositions d'analyse qui, bien que non retenues en contexte de groupe, reflètent selon moi une vision pertinente des données.

Ce projet s'inscrit dans la construction active de mon portfolio data et démontre ma capacité à prendre en charge l'intégralité du cycle de développement d'une application data – de l'analyse à la visualisation et au déploiement – de manière autonome.

## 3. Exploration et Compréhension du Projet

### a. Mise en Place de l'Environnement

J'ai configuré mon environnement de travail technique en installant :
- **Anaconda** et **VS Code** avec **Python 3.12.7**
- **Bibliothèques nécessaires** : Pandas, NumPy, Matplotlib, Seaborn, Streamlit, Plotly

### b. Acquisition et Compréhension des Données

- Téléchargement direct du jeu de données du **Kaggle ML & DS Survey 2020**
- Analyse approfondie du contexte de cette enquête annuelle
- Compréhension des objectifs : identifier les tendances, les outils et le profil des professionnels de la data

### c. Objectifs Définis

Mon analyse visait à :
- 📊 Fournir une vue d'ensemble des pratiques et tendances en data en 2020
- 🛠️ Identifier les technologies et compétences les plus populaires
- 👤 Esquisser le profil type du répondant (formation, genre, pays, salaire)
- 💡 Produire des insights actionnables pour guider les choix de formation ou les stratégies d'entreprise

## 4. Réalisations Techniques et Défis Relevés

### a. Prétraitement et Nettoyage des Données

**Développement et optimisation d'une fonction d'agrégation Python** :
- Développement d'une fonction pour regrouper les réponses aux QCM en une seule colonne exploitable
- **Optimisation** : Passage d'une boucle `for` (quelques millisecondes de latence) à la fonction `apply()` (plus rapide et simple)

**Nettoyage et préparation des données** :
- Suppression des doublons
- Élimination des colonnes non pertinentes ou à majorité de valeurs manquantes
- Renommage des colonnes pour une meilleure lisibilité

### b. Analyse Exploratoire des Données et Visualisation

**Tendances principales identifiées** :

| Catégorie | Résultat |
|-----------|----------|
| **Langage de programmation** | Python dominant |
| **Environnements de développement** | Jupyter Notebook et VS Code les plus populaires |
| **Niveau de formation** | Master le plus répandu |
| **Genre** | Masculin majoritairement représenté (~79%) |
| **Pays** | Inde en tête, France hors du top 10, Chine hors du top 5 |
| **Professions** | Data Analyst en 6ème position (Étudiants et Data Scientist en top 1 et 2) |
| **Salaire** | Grande partie dans la tranche 0-999$ |
| **Analyse salariale** | Aucune valeur aberrante, aucun lien détecté avec les années d'expérience |

## 5. Contributions aux rendus d'équipe

**Premier rendu** :
- Collaboration à la rédaction interactive du rapport
- Contribution technique sur l'analyse des environnements de développement (IDE) et des outils de data visualisation
- Identification des tendances d'usage et contextualisation selon les profils de répondants

**Deuxième rendu** :
- Collaboration à la rédaction interactive du rapport
- Participation à la conception et développement de l'application Streamlit
- Implémentation de pages interactives et de filtres dynamiques
- Création de visualisations pertinentes
- Proposition d'idées d'interface et de dataviz alignées avec les objectifs d'analyse

## 6. Défis techniques et limitations

### a. Défis techniques surmontés

- 🖥️ **Configuration manuelle** sur MacOS avec résolution avancée de dépendances
- 🔄 **Gestion des conflits de versions** Python (local vs cloud)
- 🧹 **Nettoyage ciblé** des données selon les besoins spécifiques
- ⚙️ **Conception et optimisation** de fonction d'agrégation pour QCM
- 🎨 **Implémentation complète** de l'interface Streamlit
- ⚡ **Intégration du système de cache** via `@st.cache_data` pour l'optimisation des performances

### b. Limitations et Analyse Critique

**Limitations identifiées** :

1. **📅 Données datées (2020)** :
   - Écosystème data science en évolution rapide
   - Outils populaires en 2020 potentiellement supplantés en 2024
   - Formation basée uniquement sur ces résultats serait incomplète

2. **📊 Nature des Données** :
   - Variables uniquement catégorielles (QCM)
   - Limitation des analyses prédictives et calculs de corrélations statistiques avancées
   - Portée réduite à des descriptions et tendances

3. **🔍 Portée de l'Analyse** :
   - Un seul fichier CSV limite la simulation d'un cas pratique complet "de bout en bout"
   - Impossible de créer un modèle de données complexe, ETL avancé ou data warehouse
   - Manque de temps pour enrichir avec des sources externes (données économiques, tendances marché)

## 7. Conclusion Personnelle et Bilan

### Réalisations principales
- ✅ Contribution significative à la dimension technique et visuelle du projet
- ✅ Transformation de données brutes en visualisations éclairantes et application interactive performante
- ✅ Consolidation de l'expertise en data visualization, développement Streamlit et optimisation d'applications

### Développement des compétences
- 🚀 Opportunité majeure de développement des compétences techniques et analytiques
- 📈 Maîtrise autonome des bibliothèques de visualisation (Seaborn, Matplotlib, Streamlit, Plotly)
- 💪 Renforcement de la capacité à produire des scripts performants et à concevoir des analyses pertinentes

### Valeur de l'expérience
- 🧭 Les enquêtes de ce type constituent une boussole précieuse pour les aspirants data
- 💎 Insights précieux sur les compétences à développer et les tendances du marché
- 🏆 Référentiel précieux des compétences valorisées dans le domaine de la data
- 🔍 Base solide pour une analyse rigoureuse malgré les limitations

---

*Document généré à partir du bilan personnel du projet Data Job*
