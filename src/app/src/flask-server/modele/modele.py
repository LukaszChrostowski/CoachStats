from joblib import load
import pandas as pd
from math import sqrt
import math
import numpy as np
import xgboost
from sklearn.preprocessing import OrdinalEncoder

#xgBoost
def xgboost_predict_proba(position_name='Center Forward', shot_body_part_name='Right Foot', 
                          shot_technique_name='Normal', shot_type_name='Open Play', shot_first_time=False, 
                          shot_one_on_one=False, shot_aerial_won=False,
                          shot_open_goal=False, shot_follows_dribble=False, under_pressure = True,
                          number_of_players_opponents=0, number_of_players_teammates=0, 
                          angle=0.0, distance=0.0, x_player_opponent_Goalkeeper=np.nan, 
                          x_player_opponent_8=np.nan, x_player_opponent_1=np.nan, x_player_opponent_2=np.nan, 
                          x_player_opponent_3=np.nan, x_player_teammate_1=np.nan, x_player_opponent_4=np.nan, 
                          x_player_opponent_5=np.nan, x_player_opponent_6=np.nan, x_player_teammate_2=np.nan, 
                          x_player_opponent_9=np.nan, x_player_opponent_10=np.nan, x_player_opponent_11=np.nan, 
                          x_player_teammate_3=np.nan, x_player_teammate_4=np.nan, x_player_teammate_5=np.nan, 
                          x_player_teammate_6=np.nan, x_player_teammate_7=np.nan, x_player_teammate_8=np.nan, 
                          y_player_opponent_Goalkeeper=np.nan, y_player_opponent_8=np.nan, 
                          x_player_teammate_9=np.nan, x_player_teammate_10=np.nan,
                          y_player_opponent_1=np.nan, y_player_opponent_2=np.nan, y_player_opponent_3=np.nan, 
                          y_player_teammate_1=np.nan, y_player_opponent_4=np.nan, y_player_opponent_5=np.nan, 
                          y_player_opponent_6=np.nan, y_player_teammate_2=np.nan, y_player_opponent_9=np.nan, 
                          y_player_opponent_10=np.nan, y_player_opponent_11=np.nan, y_player_teammate_3=np.nan, 
                          y_player_teammate_4=np.nan, y_player_teammate_5=np.nan, y_player_teammate_6=np.nan, 
                          y_player_teammate_7=np.nan, y_player_teammate_8=np.nan, y_player_teammate_9=np.nan, 
                          y_player_teammate_10=np.nan, x_player_opponent_7=np.nan, y_player_opponent_7=np.nan, 
                          x_player_teammate_Goalkeeper=np.nan, y_player_teammate_Goalkeeper=np.nan):

   model = xgboost.XGBClassifier() 
   model.load_model('xgboost.json')

   enc = OrdinalEncoder() 
   enc = load('labelEncoder.joblib')

   X_new = pd.DataFrame(columns=['position_name', 'shot_body_part_name', 'shot_technique_name',
      'shot_type_name', 'shot_first_time', 'shot_one_on_one',
      'shot_aerial_won', 'shot_open_goal',
      'shot_follows_dribble', 'under_pressure',
      'number_of_players_opponents', 'number_of_players_teammates',
      'angle', 'distance', 'x_player_opponent_Goalkeeper',
      'x_player_opponent_8', 'x_player_opponent_1', 'x_player_opponent_2',
      'x_player_opponent_3', 'x_player_teammate_1', 'x_player_opponent_4',
      'x_player_opponent_5', 'x_player_opponent_6', 'x_player_teammate_2',
      'x_player_opponent_9', 'x_player_opponent_10', 'x_player_opponent_11',
      'x_player_teammate_3', 'x_player_teammate_4', 'x_player_teammate_5',
      'x_player_teammate_6', 'x_player_teammate_7', 'x_player_teammate_8',
      'x_player_teammate_9', 'x_player_teammate_10',
      'y_player_opponent_Goalkeeper', 'y_player_opponent_8',
      'y_player_opponent_1', 'y_player_opponent_2', 'y_player_opponent_3',
      'y_player_teammate_1', 'y_player_opponent_4', 'y_player_opponent_5',
      'y_player_opponent_6', 'y_player_teammate_2', 'y_player_opponent_9',
      'y_player_opponent_10', 'y_player_opponent_11', 'y_player_teammate_3',
      'y_player_teammate_4', 'y_player_teammate_5', 'y_player_teammate_6',
      'y_player_teammate_7', 'y_player_teammate_8', 'y_player_teammate_9',
      'y_player_teammate_10', 'x_player_opponent_7', 'y_player_opponent_7',
      'x_player_teammate_Goalkeeper', 'y_player_teammate_Goalkeeper'])
    
   X_new.loc[len(X_new.index)] = [position_name, shot_body_part_name, shot_technique_name,
      shot_type_name, shot_first_time, shot_one_on_one,
      shot_aerial_won, shot_open_goal,
      shot_follows_dribble, under_pressure,
      number_of_players_opponents, number_of_players_teammates,
      angle, distance, x_player_opponent_Goalkeeper,
      x_player_opponent_8, x_player_opponent_1, x_player_opponent_2,
      x_player_opponent_3, x_player_teammate_1, x_player_opponent_4,
      x_player_opponent_5, x_player_opponent_6, x_player_teammate_2,
      x_player_opponent_9, x_player_opponent_10, x_player_opponent_11,
      x_player_teammate_3, x_player_teammate_4, x_player_teammate_5,
      x_player_teammate_6, x_player_teammate_7, x_player_teammate_8,
      x_player_teammate_9, x_player_teammate_10,
      y_player_opponent_Goalkeeper, y_player_opponent_8,
      y_player_opponent_1, y_player_opponent_2, y_player_opponent_3,
      y_player_teammate_1, y_player_opponent_4, y_player_opponent_5,
      y_player_opponent_6, y_player_teammate_2, y_player_opponent_9,
      y_player_opponent_10, y_player_opponent_11, y_player_teammate_3,
      y_player_teammate_4, y_player_teammate_5, y_player_teammate_6,
      y_player_teammate_7, y_player_teammate_8, y_player_teammate_9,
      y_player_teammate_10, x_player_opponent_7, y_player_opponent_7,
      x_player_teammate_Goalkeeper, y_player_teammate_Goalkeeper]
   
   X_new[['position_name', 
    'shot_technique_name', 
    'shot_type_name', 
    'shot_body_part_name']] = enc.transform(X_new[['position_name', 'shot_technique_name', 'shot_type_name', 'shot_body_part_name']], )

   X_new[['position_name', 
           'shot_technique_name', 
           'shot_type_name', 
           'number_of_players_opponents', 
           'number_of_players_teammates', 
           'shot_body_part_name']] = X_new[['position_name', 
                                            'shot_technique_name', 
                                            'shot_type_name', 
                                            'number_of_players_opponents', 
                                            'number_of_players_teammates', 
                                            'shot_body_part_name']].astype(int)

#    X_new[['minute',
#            'position_name', 
#            'shot_technique_name', 
#            'shot_type_name', 
#            'number_of_players_opponents', 
#            'number_of_players_teammates', 
#            'shot_body_part_name']] = X_new[['minute', 
#                                             'position_name', 
#                                             'shot_technique_name', 
#                                             'shot_type_name', 
#                                             'number_of_players_opponents', 
#                                             'number_of_players_teammates', 
#                                             'shot_body_part_name']].astype('category')
    
   X_new[['shot_first_time', 
           'shot_one_on_one', 
           'shot_aerial_won', 
           'shot_open_goal', 
           'shot_follows_dribble']] = X_new[['shot_first_time', 
                                      'shot_one_on_one', 
                                      'shot_aerial_won', 
                                       'shot_open_goal', 
                                       'shot_follows_dribble']].astype(int)
   print(X_new)
   print(X_new.dtypes)


   return model.predict_proba(X_new)[0][1].round(3)

#XgBoost_2 

def xgboost_predict_proba_v2(shooter,goalkeeper,teamMatesList,opponentsList, minute,position_name,shot_body_part_name,
                             shot_technique_name,shot_type_name,shot_first_time,shot_aerial_won,shot_open_goal,
                             shot_follows_dribble,shot_redirect):
   model = load('xgboost.joblib')
   X_new = pd.DataFrame(columns=['minute', 'position_name', 'shot_body_part_name',
       'shot_technique_name','shot_type_name', 'shot_first_time',
       'shot_one_on_one','shot_aerial_won',
       'shot_open_goal','shot_follows_dribble', 'shot_redirect',
       'x1', 'y1','number_of_players_opponents',
       'number_of_players_teammates','angle', 'distance', 
       'x_player_opponent_Goalkeeper', 'x_player_opponent_8',
       'x_player_opponent_1', 'x_player_opponent_2','x_player_opponent_3',
       'x_player_opponent_4','x_player_opponent_5', 'x_player_opponent_6',
       'x_player_teammate_2','x_player_opponent_9', 'x_player_opponent_10',
       'x_player_opponent_11','x_player_teammate_1',
       'x_player_teammate_3', 'x_player_teammate_4','x_player_teammate_5',
       'x_player_teammate_6', 'x_player_teammate_7', 'x_player_teammate_8',
       'x_player_teammate_9', 'x_player_teammate_10',
       'y_player_opponent_Goalkeeper', 'y_player_opponent_8',
       'y_player_opponent_1', 'y_player_opponent_2', 'y_player_opponent_3',
       'y_player_teammate_1', 'y_player_opponent_4', 'y_player_opponent_5',
       'y_player_opponent_6', 'y_player_teammate_2', 'y_player_opponent_9',
       'y_player_opponent_10', 'y_player_opponent_11', 'y_player_teammate_3',
       'y_player_teammate_4', 'y_player_teammate_5', 'y_player_teammate_6',
       'y_player_teammate_7', 'y_player_teammate_8', 'y_player_teammate_9',
       'y_player_teammate_10', 'x_player_opponent_7', 'y_player_opponent_7',
       'x_player_teammate_Goalkeeper', 'y_player_teammate_Goalkeeper'])
   
  
   shooter = konwertujDoListy(shooter)
   teamMatesList = konwertujDoListy(teamMatesList)
   opponentsList = konwertujDoListy(opponentsList)
   goalkeeper = konwertujDoListy(goalkeeper)
  
   # obliczenie katu oraz odleglosci
   angle = loc2angle(x = shooter[0], y = shooter[1])
   print("kat: " + str(angle))
   distance = loc2distance(x = shooter[0], y = shooter[1])
   print("dystans: " + str(distance))
   #Sortowanie list zawodnikow
   teamMatesList = sortNearestToShooter(teamMatesList,shooter= shooter)
   opponentsList = sortNearestToShooter(opponentsList, shooter=shooter)
   # Obliczenie ilosci zawosników
   teamMatesList = zmienWMaciez(teamMatesList)
   opponentsList = zmienWMaciez(opponentsList)


   # DO ROZBUDOWY
   number_of_players_opponents = 3
   number_of_players_teammates = 3
   # Bramkarz 
   x_player_opponent_Goalkeeper = goalkeeper[0]
   y_player_opponent_Goalkeeper = goalkeeper[1]
   # Uproszczona funkcja trzeba rozbudowac
   shot_one_on_one = True if number_of_players_opponents == 1 else False
   # TeamMate goalkeppe

   x_player_teammate_Goalkeeper = np.nan
   y_player_teammate_Goalkeeper = np.nan
   #Reszta Zawodnikow 
   X_new.loc[len(X_new.index)] = [minute, position_name, shot_body_part_name,
       shot_technique_name,shot_type_name, shot_first_time,
       shot_one_on_one,shot_aerial_won,
       shot_open_goal,shot_follows_dribble, shot_redirect, 
       shooter[0], shooter[1],number_of_players_opponents, 
       number_of_players_teammates,angle, distance,
       x_player_opponent_Goalkeeper, opponentsList[8][0],
       opponentsList[0][0], opponentsList[1][0], opponentsList[2][0],
       opponentsList[3][0], opponentsList[4][0], opponentsList[5][0],
       opponentsList[6][0], opponentsList[7][0], teamMatesList[8][0],
       opponentsList[9][0], opponentsList[10][0],
       teamMatesList[0][0], teamMatesList[1][0], teamMatesList[2][0],
       teamMatesList[3][0], teamMatesList[4][0], teamMatesList[5][0],
       teamMatesList[6][0], teamMatesList[7][0],teamMatesList[8][0],
       teamMatesList[9][0],
       y_player_opponent_Goalkeeper,
       opponentsList[0][1], opponentsList[1][1], opponentsList[2][1],
       opponentsList[3][1], opponentsList[4][1],opponentsList[5][1],
       opponentsList[6][1], opponentsList[7][1], opponentsList[8][1],
       opponentsList[9][1], opponentsList[10][1],
       teamMatesList[0][1],teamMatesList[1][1],  teamMatesList[2][1],
       teamMatesList[4][1],teamMatesList[5][1], teamMatesList[6][1], teamMatesList[7][1],
       teamMatesList[8][1], teamMatesList[9][1],
       x_player_teammate_Goalkeeper, y_player_teammate_Goalkeeper]
   
   categorical_columns = ['position_name', 'shot_technique_name', 'shot_type_name', 'number_of_players_opponents', 'number_of_players_teammates', 'shot_body_part_name']  # list all your object columns here
   # X_new = pd.get_dummies(X_new, columns=categorical_columns)
   X_new[categorical_columns] = X_new[categorical_columns].astype('category')
   
   return model.predict_proba(X_new)[0][1].round(2)


# Pomocnicze Funkcje
# trzeba uzupelnic
def sortNearestToShooter(playerList, shooter):
    
    return playerList


def loc2distance(x, y):
    return math.sqrt(x**2 + (y - 34)**2)

def loc2locdistance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def loc2angle(x, y):
    rads = math.atan(7.32 * x / (x**2 + (y - 34)**2 - (7.32/2)**2))
    rads = math.pi + rads if rads < 0 else rads
    deg = math.degrees(rads)
    return deg
# zamiana stringa odzielonego ',' na liste
def konwertujDoListy(listaString):
    listaString = listaString.split(",")
    listaFloat = []
    for elem in listaString:
        listaFloat.append(float(elem))
    return listaFloat
#Zamiana listy w formie [1x,1y,2x,2y,3x,3y...] do postaci maciezy [xi, yi]
def zmienWMaciez(lista):
   maciez = []
   for i in range(0,len(lista),2):
       player = [lista[i],lista[i+1]]
       maciez.append(player)
   if(len(maciez) < 10):
       for i in range(len(maciez),11,1):
           maciez.append([np.nan,np.nan])
   return maciez
