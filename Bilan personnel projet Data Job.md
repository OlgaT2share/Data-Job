
1. Méthodologie de Travail et Rôle dans l'Équipe

Notre équipe a adopté une méthodologie de travail collaborative et efficace. La tactique employée était la suivante :
Pour chaque tâche, chaque membre réalisait une analyse de manière indépendante.
Des points de synchronisation réguliers étaient organisés pour partager nos découvertes et nos approches.
Nous sélectionnions collectivement la solution la plus simple comme base de travail commune.
Nous enrichissions ensuite en intégrant une optimisation à travers les meilleures idées et améliorations proposées par chacun.
Mon rôle dans cette équipe a été actif et contributeur. J'ai mené mes explorations de manière autonome tout en participant activement aux réunions d'équipe pour partager mes résultats, challenger ceux des autres, et converger vers une analyse cohérente et de qualité.

2. Réalisation Technique Personnelle : Application Streamlit

Lien vers l'application Streamlit développée personnellement : ;;;;;;;;;;

J'ai conçu et développé cette application Streamlit dans le but d'analyser et de visualiser les résultats de l'enquête Kaggle 2020 sur le Machine Learning et la Science des Données, en proposant une analyse personnelle et originale.

Le projet, entièrement fonctionnel, est versionné sur GitHub et déployé sur le cloud Streamlit, permettant des mises à jour en continu et un partage aisé. Je poursuis activement son optimisation, avec pour prochaine étape l'intégration de fonctionnalités de Machine Learning à l'issue de la formation dédiée.

L'application illustre déjà ma philosophie de développement : une interface épurée, des performances optimisées (grâce à @st.cache_data pour une exécution rapide) et des visualisations interactives et pertinentes. Elle intègre également certaines de mes propositions d'analyse qui, bien que non retenues en contexte de groupe, reflètent selon moi une vision pertinente des données.

Ce projet s'inscrit dans la construction active de mon portfolio data et démontre ma capacité à prendre en charge l'intégralité du cycle de développement d'une application data – de l'analyse à la visualisation et au déploiement – de manière autonome.

3. Exploration et Compréhension du Projet

   a. Mise en Place de l'Environnement :

J'ai configuré mon environnement de travail technique en installant Anaconda et VS Code avec Python 3.12.7, assurant ainsi la compatibilité avec les bibliothèques nécessaires au projet (Pandas, NumPy, Matplotlib, Seaborn, Streamlit).

   b. Acquisition et Compréhension des Données :

J'ai téléchargé le jeu de données du Kaggle ML & DS Survey 2020 sur la plateforme Kaggle. J'ai ensuite consacré du temps à comprendre le contexte de cette enquête annuelle, son objectif (identifier les tendances, les outils et le profil des professionnels de la data) et sa valeur pour la communauté.

   c. Objectifs Définis :

Mon analyse visait à :
- Fournir une vue d'ensemble des pratiques et tendances en data en 2020
- Identifier les technologies et compétences les plus populaires
- Esquisser le profil type du répondant (formation, genre, pays, salaire)
- Produire des insights actionnables pour guider les choix de formation ou les stratégies d'entreprise

4. Réalisations Techniques et Défis Relevés

   a. Prétraitement et Nettoyage des Données :

Développement et optimisation d'une fonction d'agrégation Python : J'ai développé et optimisé une fonction en Python pour regrouper les réponses aux QCM d'une même question (réparties sur plusieurs colonnes) en une seule colonne exploitable, simplifiant ainsi l'analyse.

La première version de ma fonction était basée sur une boucle for et, bien que fonctionnelle, présentait quelques millisecondes de latence. Je l'ai par la suite optimisée en utilisant la fonction apply() qui parcourt plus simplement et plus rapidement les lignes d'un DataFrame.

Nettoyage et préparation des données : J'ai procédé à un nettoyage fondamental incluant la suppression des doublons, l'élimination des colonnes non pertinentes pour nos objectifs ou contenant une majorité de valeurs manquantes, ainsi que le renommage des colonnes pour une meilleure lisibilité.

   b. Analyse Exploratoire des Données et Visualisation :

Mon analyse initiale a permis de dégager rapidement les tendances principales, confirmées par la suite avec l'équipe :
- Python est le langage de programmation dominant
- Les environnements de développement les plus populaires sont Jupyter Notebook et VS Code
- Le niveau de formation le plus répandu est le Master's Degree
- Le genre masculin est majoritairement représenté (~80%)
- L'Inde constitue le pays le plus représenté parmi les répondants, ce qui peut s'expliquer par la conduite du sondage en anglais ainsi que par la démographie importante et la forte orientation technologique de ce pays. À l'inverse, la France ne figure pas parmi les dix premiers pays représentés. Plus surprenant encore, la Chine n'apparaît pas dans le top 5 des pays participants
- Le titre de Data Analyst se classe en sixième position des professions les plus citées dans le sondage
- Une grande partie des répondants déclare un salaire dans la tranche 0-999$ (incluant potentiellement des étudiants ou des pays avec des écarts de coût de vie)
- Aucune valeur aberrante observée dans les salaires et aucun lien particulier détecté entre salaire et années d'expérience

5. Contributions aux rendus d'équipe

Dans le cadre du premier rendu, j'ai collaboré à la rédaction interactive du rapport. Ma contribution technique a porté sur l'analyse des environnements de développement (IDE) et des outils de data visualisation, présentés en section 5.visualisation point 2 et 3 du rapport. J'ai identifié les tendances d'usage et contextualisé leur adoption selon les profils de répondants.

Dans le cadre du deuxième rendu, j'ai également participé à la conception et au développement de l'application Streamlit, avec une implication particulière dans :
- L'implémentation de pages interactives et de filtres dynamiques permettant une exploration ciblée des données
- La création de visualisations pertinentes pour étayer notre problématique
- La proposition d'idées d'interface et de dataviz alignées avec les objectifs d'analyse

6. Défis techniques et limitations

   a. Défis techniques surmontés :

- Configuration et installation manuelle de bibliothèques spécifiques sur VS Code avec un environnement MacOS, nécessitant une résolution avancée de dépendances
- Gestion des conflits de versions Python entre l'environnement local et le cloud lors du déploiement sur GitHub, résolus par la containerisation et la configuration précise des requirements
- Nettoyage ciblé des données en fonction des besoins spécifiques de visualisation et d'analyse
- Conception et optimisation de la fonction d'agrégation pour regrouper les réponses à choix multiples (QCM) en une structure exploitable
- Sélection stratégique et implémentation complète de l'interface Streamlit pour une présentation interactive des résultats
- Intégration avancée du système de cache via @st.cache_data pour optimiser les performances et réduire les temps de chargement

   b. Limitations et analyse critique :

Mon analyse m'a conduit à identifier plusieurs limitations importantes :

- **Données datées (2020)** : L'écosystème de la data science évolue extrêmement vite. Les outils et langages populaires en 2020 (comme certains outils de dataviz) peuvent avoir été supplantés par d'autres en 2024 (e.g., Power BI, Tableau). Une formation basée uniquement sur ces résultats serait donc incomplète.

- **Nature des données** : Le jeu de données est constitué uniquement de variables catégorielles (réponses à des questions à choix multiples). Cela limite toute analyse prédictive ou tout calcul de corrélations statistiques avancées, limitant la portée de l'analyse à des descriptions et des tendances.

- **Portée de l'analyse** : Avec un seul fichier CSV, le projet ne permet pas de simuler un cas pratique complet d'analyse "de bout en bout", comme on le ferait en entreprise (e.g., création d'un modèle de données complexe, ETL avancé, construction d'un data warehouse). Le manque de temps n'a pas permis d'enrichir l'analyse avec des sources externes (données économiques, tendances marché, benchmarks sectoriels), qui auraient pourtant permis de contextualiser et d'approfondir significativement les insights obtenus.

7. Conclusion Personnelle et Bilan

J'ai grandement contribué à la dimension technique et visuelle de ce projet en transformant des données brutes en visualisations éclairantes et en une application interactive performante. Ce travail m'a permis de consolider mon expertise en data visualization, en développement web avec Streamlit et en optimisation d'applications. Je suis satisfaite d'avoir concrétisé une partie significative de la vision du projet à travers cette interface.

Cette expérience a été une opportunité majeure de développement de mes compétences techniques et analytiques, notamment dans le nettoyage, l'exploration et la visualisation de données sur un jeu de données réel et assez complexe. Je me suis challengée à chaque étape, en allant bien au-delà des contenus de formation pour réaliser des traitements avancés en Python et maîtriser les bibliothèques de visualisation (Seaborn, Matplotlib, Streamlit) en autonomie. Cette immersion a renforcé ma capacité à produire des scripts performants et ma confiance dans la conception d'analyses pertinentes.

J'ai pu constater à quel point les enquêtes de ce type constituent une boussole précieuse pour les aspirants au métier de la data, offrant des insights sur les compétences à développer et les tendances du marché. Bien que non récent, ce sondage reste un référentiel précieux des compétences valorisées dans le domaine de la data et a servi de base solide à une analyse rigoureuse.
