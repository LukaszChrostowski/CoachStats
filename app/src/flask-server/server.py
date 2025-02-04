from flask import Flask, request, jsonify
from flask_cors import CORS
from modele.modele import LogisticRegression_predict_proba, xgboost_predict_proba, xgboost_predict_proba_v2
import numpy as np
import math

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

# zapytanie o regresje logistyczną na podstawie pozycji piłki

# @app.route("/LRegresion<x>&<y>")
@app.route("/get_model", methods = ['GET'])
def get_model():
    
    # Pozycja strzelca
    shooter = request.args.get('shooter')
    
    print("wprowadzone dane:")
    print("strzelec: " + shooter)

    if shooter[0] is None and shooter[1] is None:
        return jsonify({"error": "Brak wymaganych parametrów"}), 400
    
    # Elementy z list: czesc ciała,rodzaj uderzenia,pozycja strzelca oraz typ akcji

    body_part = request.args.get('bodyPart')
    technique = request.args.get('technique')
    acionType = request.args.get('actionType')
    minute = request.args.get('gameMinute')
    position_name = request.args.get('shooterPossition')
    gameMinute = request.args.get('gameMinute')

    shot_first_time = request.args.get('shot_first_time')
    shot_one_on_one = request.args.get('shot_one_on_one')
    shot_aerial_won = request.args.get('shot_aerial_won')
    shot_open_goal = request.args.get('shot_open_goal')
    shot_follows_dribble = request.args.get('shot_follows_dribble')
    shot_redirect = request.args.get('shot_redirect')


    shooter_x, shooter_y = shooter.split(',')

    shooter_x = float(shooter_x)
    shooter_y = float(shooter_y)

    print("minuta: ", minute)
    print("pozycja x1 strzelca: ", shooter_x)
    print("pozycja y1 strzelca: ", shooter_y)
    print("czesc ciala: " + body_part)
    print("technika: " + technique)
    print("typ akcji: " + acionType)

    # lista atakujacy oraz obroncy

    def konwertujDoListy(listaString):
        listaString = listaString.split(",")
        listaFloat = []
        for elem in listaString:
            listaFloat.append(float(elem))
        return listaFloat

    def zmienWMaciez(lista):
        maciez = []
        for i in range(0,len(lista),2):
            player = [float(lista[i]),float(lista[i+1])]
            maciez.append(player)
        if(len(maciez) < 10):
            for i in range(len(maciez),11,1):
                maciez.append([np.nan,np.nan])
        return maciez


    obroncy = request.args.get('defenders')
    atakujacy = request.args.get('strickers')

    print("wspolrzedne obroncow: ", obroncy)

    if obroncy == "":
        obroncy_macierz = [[np.nan, np.nan]]
    else:
        obroncy_lista = konwertujDoListy(obroncy)
        obroncy_macierz = zmienWMaciez(obroncy_lista)
    
    if atakujacy == "":
        atakujacy_macierz = [[np.nan, np.nan]]
    else:
        atakujacy_lista = konwertujDoListy(atakujacy)
        atakujacy_macierz = zmienWMaciez(atakujacy_lista)

    if gameMinute == "":
        gameMinute = 1.0 # to change

    print("wspolrzedne obroncow: ", obroncy)
    print("wspolrzedne atakujacych: ", atakujacy)

    def euclidean_distance(point1, point2):
        """Calculate the Euclidean distance between two points."""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

    def sort_coordinates_by_distance(coordinates, reference_point):
        """Sort a list of coordinates based on Euclidean distance from a reference point.
        Coordinates with NaN values are placed at the end of the list.
        """
        def distance_or_inf(coord):
            """Helper function to handle coordinates with NaN values."""
            if hasattr(coord, '__iter__') and len(coord) >= 2:
                if np.isnan(coord[0]) or np.isnan(coord[1]):
                    return float('inf')  # Set distance to infinity for NaN coordinates
                return euclidean_distance(coord, reference_point)
            return float('inf')  # Set distance to infinity for non-iterable coordinates
        
        sorted_coordinates = sorted(coordinates, key=distance_or_inf)

        if(len(sorted_coordinates) < 10):
            for i in range(len(sorted_coordinates),11,1):
                sorted_coordinates.append([np.nan,np.nan])
        return sorted_coordinates
    
    # POSORTOWANI OBRONCY I ATAKUJACY
    posortowani_obroncy = sort_coordinates_by_distance(obroncy_macierz, [shooter_x, shooter_y])
    posortowani_atakujacy = sort_coordinates_by_distance(atakujacy_macierz, [shooter_x, shooter_y])

    print("Posortowani obroncy: ", posortowani_obroncy)
    
    print("Posortowani atakujacy: ", posortowani_atakujacy)
    

    number_of_players_opponents = float(np.sum(~np.isnan(posortowani_obroncy).all(axis=1)))
    number_of_players_teammates = float(np.sum(~np.isnan(posortowani_atakujacy).all(axis=1)))
    
    print("Liczba obroncow:", number_of_players_opponents)
    print("Liczba atakujacych:", number_of_players_teammates)
    ## add angle, match minutes and number of players
  
    def loc2angle(x, y):
        rads = math.atan(7.32 * x / (x**2 + (y - 34)**2 - (7.32/2)**2))
        rads = math.pi + rads if rads < 0 else rads
        deg = math.degrees(rads)
        return deg
    
    def loc2distance(x, y):
        return math.sqrt(x**2 + (y - 34)**2)

    angle = loc2angle(x = shooter_x, y = shooter_y)
    dist = loc2distance(x = shooter_x, y = shooter_y)

    # BRAMKARZ
    goalkepper = request.args.get('goalkeeper')
    
    if goalkepper == "null":
        goalkepper_x, goalkepper_y = np.nan, np.nan
    else:
        goalkepper_x, goalkepper_y = goalkepper.split(',')
        goalkepper_x = float(goalkepper_x)
        goalkepper_y = float(goalkepper_y)

    print("bramkarz:", goalkepper_x, goalkepper_y)

    if shot_first_time == 'true':
        shot_first_time = True
    else:
        shot_first_time = False
    if shot_one_on_one == 'true':
        shot_one_on_one = True
    else:
        shot_one_on_one = False
    if shot_aerial_won == 'true':
        shot_aerial_won = True
    else:
        shot_aerial_won = False
    if shot_open_goal == 'true':
        shot_open_goal = True
    else:
        shot_open_goal = False
    if shot_follows_dribble == 'true':
        shot_follows_dribble = True
    else:
        shot_follows_dribble = False
    if shot_redirect == 'true':
        shot_redirect = True
    else: 
        shot_redirect = False

    #### CHECKLIST
    print('CHECKLISTA')
    print('shot_first_time', shot_first_time)
    print('shot_one_on_one', shot_one_on_one)
    print('shot_aerial_won', shot_aerial_won)
    print('shot_open_goal', shot_open_goal)
    print('shot_follows_dribble', shot_follows_dribble)
    print('shot_redirect', shot_redirect)

    # MODEL XGBOOST
    response = xgboost_predict_proba(minute=int(gameMinute), #minute=0,
                                     position_name=position_name, 
                                     shot_body_part_name=body_part, 
                                     shot_technique_name=technique, 
                                     shot_type_name=acionType, 
                                     shot_first_time=shot_first_time, 
                                     shot_one_on_one=shot_one_on_one, 
                                     shot_aerial_won=shot_aerial_won, 
                                     shot_open_goal=shot_open_goal, 
                                     shot_follows_dribble=shot_follows_dribble, 
                                     shot_redirect=shot_redirect,
                                     x1=shooter_x, y1=shooter_y,
                                     number_of_players_opponents=number_of_players_opponents, 
                                     number_of_players_teammates=number_of_players_teammates, 
                                     angle=angle, distance=dist, 
                                     x_player_opponent_Goalkeeper=goalkepper_x, y_player_opponent_Goalkeeper=goalkepper_y,
                                     x_player_opponent_1=posortowani_obroncy[0][0], y_player_opponent_1=posortowani_obroncy[0][1],
                                     x_player_opponent_2=posortowani_obroncy[1][0], y_player_opponent_2=posortowani_obroncy[1][1],
                                     x_player_opponent_3=posortowani_obroncy[2][0], y_player_opponent_3=posortowani_obroncy[2][1],
                                     x_player_opponent_4=posortowani_obroncy[3][0], y_player_opponent_4=posortowani_obroncy[3][1],
                                     x_player_opponent_5=posortowani_obroncy[4][0], y_player_opponent_5=posortowani_obroncy[4][1],
                                     x_player_opponent_6=posortowani_obroncy[5][0], y_player_opponent_6=posortowani_obroncy[5][1],
                                     x_player_opponent_7=posortowani_obroncy[6][0], y_player_opponent_7=posortowani_obroncy[6][1],
                                     x_player_opponent_8=posortowani_obroncy[7][0], y_player_opponent_8=posortowani_obroncy[7][1],
                                     x_player_opponent_9=posortowani_obroncy[8][0], y_player_opponent_9=posortowani_obroncy[8][1],
                                     x_player_opponent_10=posortowani_obroncy[9][0], y_player_opponent_10=posortowani_obroncy[9][1],
                                     x_player_teammate_Goalkeeper=np.nan, y_player_teammate_Goalkeeper=np.nan,
                                     x_player_teammate_1=posortowani_atakujacy[0][0], y_player_teammate_1=posortowani_atakujacy[0][1],
                                     x_player_teammate_2=posortowani_atakujacy[1][0], y_player_teammate_2=posortowani_atakujacy[1][1],
                                     x_player_teammate_3=posortowani_atakujacy[2][0], y_player_teammate_3=posortowani_atakujacy[2][1],
                                     x_player_teammate_4=posortowani_atakujacy[3][0], y_player_teammate_4=posortowani_atakujacy[3][1],
                                     x_player_teammate_5=posortowani_atakujacy[4][0], y_player_teammate_5=posortowani_atakujacy[4][1],
                                     x_player_teammate_6=posortowani_atakujacy[5][0], y_player_teammate_6=posortowani_atakujacy[5][1],
                                     x_player_teammate_7=posortowani_atakujacy[6][0], y_player_teammate_7=posortowani_atakujacy[6][1],
                                     x_player_teammate_8=posortowani_atakujacy[7][0], y_player_teammate_8=posortowani_atakujacy[7][1],
                                     x_player_teammate_9=posortowani_atakujacy[8][0], y_player_teammate_9=posortowani_atakujacy[8][1],
                                     x_player_teammate_10=posortowani_atakujacy[9][0], y_player_teammate_10=posortowani_atakujacy[9][1])

    # response = xgboost_predict_proba(shooter = shooter,
    #                                   opponentsList = opponentsList, teamMatesList = teamMatesList, minute = 20, position_name = position_name, shot_body_part_name = body_part, shot_technique_name = technique,
    #                                   shot_type_name = acionType, shot_first_time = False,
    #                                   shot_aerial_won = False, shot_deflected = False, shot_open_goal = False,
    #                                   shot_follows_dribble = False, shot_redirect = False, shot_kick_off = False, goalkeeper= goalkepper
    #                                   )
    
    #print(response)
    res = str(response)
    #return {"response":res}
    return jsonify({"response":res})

""""
def get_model():

    # x = int(x[0:2])
    # y = int(y[0:2])
    shooter = request.args.get('shooter')
    shooter = konwertujDoListy(shooter)
    
    print("wprowadzone dane:")
    print("x strzelca: " + str(shooter[0]))
    print("y strzelca: " + str(shooter[1]))
    
    ## change model on xgboost
    ## add angle, match minutes and number of players
  
    angle = loc2angle(x = shooter[0], y = shooter[1])
    dist = loc2distance(x = shooter[0], y = shooter[1])

    if shooter[0] is None and shooter[1] is None:
        return jsonify({"error": "Brak wymaganych parametrów"}), 400

    response = LogisticRegression_predict_proba(position_x=shooter[0],
                                                position_y=shooter[1],
                                                distance_to_goalM = dist,
                                                angle = angle,
                                                match_minute=13,
                                                Number_Intervening_Opponents=3,
                                                Number_Intervening_Teammates=0,
                                                isFoot=1,
                                                isHead=0)

    #print(response)
    res = str(response)
    #return {"response":res}
    return jsonify({"response":res})
# uruchomienie serwera
"""

if __name__ == "__main__":
    app.run(debug = True)