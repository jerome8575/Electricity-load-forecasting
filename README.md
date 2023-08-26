# Prédiction de la demande d'éléctricité au Québec, 24 heures à l'avance 

Code de mon stage d’été 2023 sous la supervision de Fabian Bastin
<br>
<br>
Étude et développement de modèle de prédiction de la demande d’électricité au Québec, 24 heures à l’avance, en utilisant les données ouvertes d’Hydro-Québec. 
<br>
<br>
La première partie du stage fut consacrée à une analyse des données de consommation de l’électricité avec une emphase sur l’effet de la température sur la demande. Suivant cette analyse, un premier modèle paramétrique classique a été développé.

Dans la deuxième partie, nous avons étudié des méthodes de combinaison de modèle dans l’optique d’améliorer la précision des prédictions ainsi que de réduire les erreurs. Une première approche par régression linéaire a été envisagée or des problèmes théoriques ont été observés dû à l’effet de colinéarité entre les modèles similaires. L’ajout d’un terme de régularisation (régression ridge) régla ses problèmes numériques en plus d’améliorer considérablement les résultats.

La présentation du modèle paramétrique et de la méthode d’ensemble ridge est disponible dans le dossier presentation_slides.
