"""
@author: AYA RAZZOUK
"""
# Librairie Utilisées :
import numpy as np
import random
import time
import os

# Initialize parameters
Nb_Allumettes = 21
ACTIONS = [1, 2, 3]
ALPHA = 0.1
GAMMA = 0.9
EPSILON = 0.1
Nb_tours = 1000000  # Ceci est Grand pour Assurer un table adéquate pour les décisions à faire !
Q_TABLE_FILE = 'q_table200.npy'

# Chargement Q_Table Fichier
if os.path.exists(Q_TABLE_FILE):
    Q_Table_Py = np.load(Q_TABLE_FILE)
else:
    Q_Table_Py = np.zeros((Nb_Allumettes + 1, len(ACTIONS)))

# Heuristic-based initialization : Optimisation de Strategie de la table-Q
for etat in range(Nb_Allumettes + 1):
    for action in ACTIONS:
        Prochain_etat = max(etat - action, 0)
        if Prochain_etat == 0:
            Q_Table_Py[etat, ACTIONS.index(action)] = 1  # Choix Gagnant
        elif (Prochain_etat) % 4 == 0:
            Q_Table_Py[etat, ACTIONS.index(action)] = 0.75  # Choix Optimal
        else:
            Q_Table_Py[etat, ACTIONS.index(action)] = 0.5  # Choix Neutre


def Choisir_Action(etat, Apprend=True):
    """
    Chooses an action based on the current state and the exploration-exploitation strategy.

    Args:
        etat (int): Current state (number of matches remaining).
        Apprend (bool): Flag indicating if the agent is in learning mode. Defaults to True.

    Returns:
        int: The chosen action (number of matches to take).
    """
    if Apprend and random.uniform(0.0, 1.0) < EPSILON:
        return random.choice(ACTIONS)
    return ACTIONS[np.argmax(Q_Table_Py[etat])]


def Pas(etat, action):
    """
    Executes the action and returns the next state and reward.

    Args:
        etat (int): Current state (number of matches remaining).
        action (int): Action to take (number of matches to remove).

    Returns:
        tuple: A tuple containing the next state and the reward.
    """
    Prochain_etat = max(etat - action, 0)
    if Prochain_etat == 0:
        Recompense = 2
    elif Prochain_etat % 4 == 0:
        Recompense = 1
    else:
        Recompense = 0

    return Prochain_etat, Recompense


def Mise_A_Jour_Table(Historique_tour):
    """
    Updates the Q-table based on the history of the game.

    Args:
        Historique_tour (list): A list of tuples containing the state, action, reward, and next state for each step in the game.
    """
    for etat, action, Recompense, Prochain_etat in Historique_tour:
        Meilleur_Action_AFaire = np.argmax(Q_Table_Py[Prochain_etat])
        Q_Table_Py[etat, ACTIONS.index(action)] += ALPHA * (
            Recompense + GAMMA * Q_Table_Py[Prochain_etat, Meilleur_Action_AFaire] - Q_Table_Py[etat, ACTIONS.index(action)]
        )


def Enregistre_Table():
    """
    Saves the Q-table to a file.
    """
    np.save(Q_TABLE_FILE, Q_Table_Py)


def Entraine_Table():
    """
    Trains the Q-table through a series of games.
    """
    for Tour in range(Nb_tours):
        etat = Nb_Allumettes
        Historique_tour = []
        while etat > 0:
            action = Choisir_Action(etat)
            Prochain_etat, Recompense = Pas(etat, action)
            Historique_tour.append((etat, action, Recompense, Prochain_etat))
            etat = Prochain_etat

        Mise_A_Jour_Table(Historique_tour)
    Enregistre_Table()


def play_nim():
    """
    Plays the Nim game against the user.
    """
    etat = Nb_Allumettes
    Historique_tour = []
    print('Demarrage Jeu de Nims!')
    print("Pile ou Face pour savoir qui commence en premier! (Je vous assure que si je commence en premier vous allez perdre!)")
    time.sleep(2)

    IA_Tour = True  # random.choice([True, False])
    print("L'IA Commence" if IA_Tour else "L'utilisateur Commence")

    while etat > 0:
        print(f'Allumettes Restantes: {etat}')
        print('| ' * etat)
        if IA_Tour:
            action = Choisir_Action(etat, Apprend=False)
            print(f"L'IA Prend: {action} Allumette(s)")
            Prochain_etat, Recompense = Pas(etat, action)
            Historique_tour.append((etat, action, Recompense, Prochain_etat))
            etat = Prochain_etat
            if etat == 0:
                print("L'IA Gagne!")
                break
        else:
            while True:
                try:
                    Action_Joueur = int(input('Ton Tour (Prenez 1, 2, or 3 Allumettes): '))
                    if Action_Joueur in ACTIONS and Action_Joueur <= etat:
                        break
                    else:
                        print("Choix Invalide. Réessayez. Je vous rappelle que c'est minimum 1 Allumette et Maximum 3!")
                except ValueError:
                    print('Entrée invalide. Choisissez un nombre valide.')

            Prochain_etat, Recompense = Pas(etat, Action_Joueur)
            Historique_tour.append((etat, Action_Joueur, Recompense, Prochain_etat))
            etat = Prochain_etat
            if etat == 0:
                print('Vous Gagnez!')
                Recompense = -1  # Recompense Negative pour IA s'il perd !
                break

        IA_Tour = not IA_Tour
        time.sleep(2)

    Mise_A_Jour_Table(Historique_tour)
    Enregistre_Table()


def IA_vs_IA(Nb_Matchs):
    """
    Plays a series of Nim games between two AIs to train the Q-table further.

    Args:
        Nb_Matchs (int): Number of matches to play.
    """
    IA_1_Victoires = 0
    IA_2_Victoires = 0
    IA_1_Debut = 0
    IA_2_Debut = 0
    print('Début de Jeu de Nim IA contre IA ! Pour Apprendre ! ')
    for i in range(Nb_Matchs):
        etat = Nb_Allumettes
        Historique_tour = []
        print(f'Début de Jeu de Nim IA contre IA ! Jeu Numero: {i+1}')
        print('- ' * 50)
        Tour_IA_1 = random.choice([True, False])

        if Tour_IA_1:
            IA_1_Debut += 1
            print("IA 1 Commence")
        else:
            IA_2_Debut += 1
            print("IA 2 Commence")

        while etat > 0:
            print(f'Allumettes Restantes: {etat}')
            print('| ' * etat)
            action = Choisir_Action(etat)
            Prochain_etat, Recompense = Pas(etat, action)
            Historique_tour.append((etat, action, Recompense, Prochain_etat))
            etat = Prochain_etat
            if Tour_IA_1:
                print(f'IA 1 Prend: {action} Allumettes')
                if etat == 0:
                    IA_1_Victoires += 1
                    print("IA 1 Gagne")
                    print('- ' * 50)
                    break
            else:
                print(f'IA 2 Prend: {action} Allumettes')
                if etat == 0:
                    IA_2_Victoires += 1
                    print("IA 2 Gagne")
                    print('- ' * 50)
                    break
            Tour_IA_1 = not Tour_IA_1
            time.sleep(0.1)

        Mise_A_Jour_Table(Historique_tour)

    print(f'IA 1 a Commencé: {IA_1_Debut} Jeux et a gagné : {IA_1_Victoires}')
    print(f'IA 2 a Commencé: {IA_2_Debut} Jeux et a gagné : {IA_2_Victoires}')

    Enregistre_Table()


# Entrainer IA en Lancement du Code
Entraine_Table()

# Jouer au Jeu Utilisateur contre IA
play_nim()

# IA contre IA matchs pour Apprendre 
# IA_vs_IA(100)
