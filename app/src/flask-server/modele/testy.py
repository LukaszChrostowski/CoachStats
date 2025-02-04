from modele import zmienWMaciez,xgboost_predict_proba_v2,konwertujDoListy


#zmien maciez test
'''
lista = [2,3,43,12,323,12,23,43,54,90]
print(zmienWMaciez(lista))
lista = zmienWMaciez(lista)
print(lista[3][0])
print(lista[3][1])
print(lista[0][0])
'''

'''
string = "49.27,49.73,48.49,34.75,19.95,28.44"

string = konwertujDoListy(string)
print(string)
print(string[3])

string = zmienWMaciez(string)
print(string)
'''
'''
a = ['minute', 'position_name', 'shot_body_part_name', 'shot_technique_name',
       'shot_type_name', 'shot_first_time', 'shot_one_on_one',
       'shot_aerial_won', 'shot_deflected', 'shot_open_goal',
       'shot_follows_dribble', 'shot_redirect', 'x1', 'y1',
       'number_of_players_opponents', 'number_of_players_teammates',
       'angle', 'distance', 'x_player_opponent_Goalkeeper',
       'x_player_opponent_1', 'x_player_opponent_2','x_player_opponent_3', 'x_player_opponent_4',
       'x_player_opponent_5', 'x_player_opponent_6', 'x_player_teammate_2',
       'x_player_opponent_9', 'x_player_opponent_10', 'x_player_opponent_11','x_player_teammate_1'
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
       'x_player_teammate_Goalkeeper', 'y_player_teammate_Goalkeeper',
       'shot_kick_off']
print(len(a))
'''