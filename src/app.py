import dash
from dash import dcc, html, Input, Output, State, callback_context
import pandas as pd
import base64
import io
import multiprocessing
import math
import numpy as np
from joblib import load
import plotly.graph_objects as go
from PIL import Image

# Ustawienie metody multiprocessingu (np. dla macOS)
multiprocessing.set_start_method('spawn', force=True)

# Używamy backendu non-interactive dla matplotlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Import funkcji analitycznych – zakładamy, że zwracają obiekt Figure
from notebooks.data_analysis_funs import (
    plot_shot_minute_histogram, plot_shot_trajectory, 
    plot_distance_histogram, plot_angle_distance, 
    plot_pitch_heatmap, plot_shot_body_part,
    plot_shot_technique
)

# Inicjalizacja aplikacji Dash z wykorzystaniem zewnętrznego szablonu (Bootstrap)
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css']
app = app = dash.Dash(__name__, external_stylesheets=external_stylesheets, suppress_callback_exceptions=True)
server = app.server

# Załaduj dane analityczne
df_analysis = pd.read_csv('notebooks/data_to_ml.csv')
# Załaduj wyuczony model xG
xgb_model = load('notebooks/xgboost.joblib')
# Ścieżka do obrazu boiska
pitch_image_path = "assets/bojo.png"

# Przygotowanie list do dropdownów dla drużyn i zawodników
teams = ['all'] + sorted(df_analysis['team_name'].unique().tolist())
players = ['all'] + sorted(df_analysis['player_name'].unique().tolist())

# Helper: konwersja obiektu Figure z Matplotlib na obrazek zakodowany base64
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return "data:image/png;base64," + encoded


# Mapowania dla zmiennych kategorycznych – kolejność musi odpowiadać tej użytej podczas treningu
position_options = ['Right Center Forward', 'Center Midfield', 'Left Center Midfield', 'Right Center Back', 
                    'Right Center Midfield', 'Right Wing', 'Left Center Forward', 'Center Back', 'Left Center Back', 
                    'Left Midfield', 'Secondary Striker', 'Center Forward', 'Center Attacking Midfield', 'Left Wing', 
                    'Left Defensive Midfield', 'Right Back', 'Right Wing Back', 'Right Defensive Midfield', 
                    'Center Defensive Midfield', 'Left Back', 'Right Midfield', 'Left Wing Back', 
                    'Left Attacking Midfield', 'Right Attacking Midfield', 'Goalkeeper']
position_mapping = {val: i for i, val in enumerate(position_options)}

shot_body_part_options = ['Right Foot', 'Left Foot', 'Head', 'Other']
shot_body_part_mapping = {val: i for i, val in enumerate(shot_body_part_options)}

shot_technique_options = ['Normal', 'Half Volley', 'Volley', 'Lob', 'Diving Header', 'Overhead Kick', 'Backheel']
shot_technique_mapping = {val: i for i, val in enumerate(shot_technique_options)}

shot_type_options = ['Open Play', 'Free Kick', 'Penalty', 'Corner', 'Kick Off']
shot_type_mapping = {val: i for i, val in enumerate(shot_type_options)}

# Funkcje pomocnicze – obliczanie kąta i odległości na podstawie pozycji strzelca
def loc2angle(x, y):
    # Wzor: angle = rad2deg(atan(7.32 * x / (x^2 + (y - 34)^2 - (7.32/2)^2)))
    denominator = (x**2 + (y - 34)**2 - (7.32/2)**2)
    if denominator == 0:
        rads = math.pi/2
    else:
        rads = math.atan((7.32 * x) / denominator)
    if rads < 0:
        rads = math.pi + rads
    return math.degrees(rads)

def loc2distance(x, y):
    return math.sqrt(x**2 + (y - 34)**2)

# Konwersja obrazu na Base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

# Załaduj obraz w formacie Base64
encoded_pitch_image = encode_image(pitch_image_path)

def create_pitch_figure(points_data=None):
    fig = go.Figure()

    # Ustawienie obrazu boiska – oficjalne wymiary 105 m x 68 m
    fig.update_layout(
        images=[dict(
            source=encoded_pitch_image,
            xref="x",
            yref="y",
            x=0,
            y=68,         # Górna krawędź obrazu
            sizex=105,
            sizey=68,
            xanchor="left",
            yanchor="top",
            layer="below"
        )]
    )

    # Dodajemy zapisane punkty (np. shooter, goalkeeper, teammate, opponent)
    if points_data:
        colors = {
            'shooter': 'lightgreen',
            'goalkeeper': 'orange',
            'teammate': 'blue',
            'opponent': 'red'
        }
        for key, pts in points_data.items():
            if pts:
                xs = [pt[0] for pt in pts]
                ys = [pt[1] for pt in pts]
                fig.add_trace(go.Scatter(
                    x=xs,
                    y=ys,
                    mode='markers',
                    marker=dict(color=colors.get(key, 'black'), size=12),
                    name=key.capitalize(),
                    hoverinfo='none'
                ))

    # Tworzymy gęstą siatkę punktów z dokładnością co 0,5 metra
    grid_x = np.linspace(0, 105, num=211)  # 211 punktów na osi x (co ok. 0.5 m)
    grid_y = np.linspace(0, 68, num=137)    # 137 punktów na osi y (co ok. 0.5 m)
    xx, yy = np.meshgrid(grid_x, grid_y)
    fig.add_trace(go.Scatter(
        x=xx.flatten(),
        y=yy.flatten(),
        mode='markers',
        marker=dict(opacity=0, size=1),
        hoverinfo='none',
        name='clickLayer',
        showlegend=False
    ))

    # Ustawienia układu – zakresy osi zgodne z wymiarami boiska
    fig.update_layout(
        xaxis=dict(
            range=[0, 105],
            showgrid=False,
            zeroline=False,
            visible=False,
            fixedrange=True
        ),
        yaxis=dict(
            range=[0, 68],
            showgrid=False,
            zeroline=False,
            visible=False,
            scaleanchor="x",
            fixedrange=True
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        clickmode='event',
        dragmode=False,
        hoverdistance=-1,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        uirevision='constant'
    )

    return fig

# Układ główny – wielostronicowy (używamy dcc.Location)
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Nav([
        dcc.Link("Data Insights", href="/analysis", style={'marginRight': '20px'}),
        dcc.Link("Goal Predictor", href="/xg")
    ], style={'padding': '20px', 'backgroundColor': '#e9ecef'}),
    html.Div(id='page-content')
])

# Układ strony z analizą (dotychczasowy)
analysis_layout = html.Div([
    html.H1("CoachStats Powered Football Insights", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Filtry: dropdowny dla drużyny i zawodnika
    html.Div([
        html.Div([
            html.Label("Team:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='team-dropdown',
                options=[{'label': t, 'value': t} for t in teams],
                value='all',
                clearable=False,
                style={'marginBottom': '10px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top', 'padding': '10px'}),
        html.Div([
            html.Label("Player:", style={'fontWeight': 'bold'}),
            dcc.Dropdown(
                id='player-dropdown',
                options=[{'label': p, 'value': p} for p in players],
                value='all',
                clearable=False,
                style={'marginBottom': '10px'}
            )
        ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '20px', 'verticalAlign': 'top', 'padding': '10px'}),
    ], style={'backgroundColor': '#f8f9fa', 'padding': '20px', 'borderRadius': '5px', 'marginBottom': '20px'}),
    
    # Przycisk generujący wykresy analityczne
    html.Div([
        html.Button("Generate Visualisations", id='generate-analysis-button', n_clicks=0,
                    style={'width': '100%', 'padding': '10px', 'fontWeight': 'bold'})
    ], style={'marginBottom': '20px'}),
    
    # Kontener dla wykresów analitycznych
    html.Div(id='analysis-output'),
    
    # Przycisk i wykres gęstości strzałów
    html.Div([
        html.Button("Draw Shot Density", id='draw-density-button', n_clicks=0,
                    style={'width': '100%', 'padding': '10px', 'fontWeight': 'bold', 'marginTop': '40px'})
    ], style={'marginBottom': '20px'}),
    html.Div([
        html.H2("Shots Density", style={'textAlign': 'center'}),
        html.Img(id='density-output', style={'width': '100%', 'padding': '10px'})
    ], style={'margin': '20px'})
])

# Układ drugiej strony – xG Predictor
xg_layout = html.Div([
    html.H1("xG Prediction", style={'textAlign': 'center', 'marginBottom': '20px'}),
    
    # Sekcja interaktywnej makiety boiska
    html.Div([
        html.H3("Interactive Pitch - Click to place markers"),
        dcc.RadioItems(
            id='marker-type',
            options=[
                {'label': 'Scorer', 'value': 'shooter'},
                {'label': 'Goalkeeper', 'value': 'goalkeeper'},
                {'label': 'Teammate', 'value': 'teammate'},
                {'label': 'Opponent', 'value': 'opponent'}
            ],
            value='shooter',
            labelStyle={'display': 'inline-block', 'marginRight': '10px'}
        ),
        html.Button("Clear Markers", id='clear-markers', n_clicks=0, style={'marginLeft': '20px'}),
        dcc.Graph(
            id='pitch-graph',
            figure=create_pitch_figure(),
            config={
                'displayModeBar': False,
                'scrollZoom': False,
                'doubleClick': False,
                'showTips': False,
                'responsive': True,
                'staticPlot': False,
                'modeBarButtonsToRemove': ['zoom', 'pan', 'select', 'lasso2d']
            },
            style={'width': '100%', 'height': '700px'}
        ),
        # Przechowywanie punktów – każdy typ jako lista współrzędnych
        dcc.Store(id='points-store', data={'shooter': [], 'goalkeeper': [], 'teammate': [], 'opponent': []})
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f8f9fa', 'borderRadius': '5px'}),
    
    # Dodatkowe dane wejściowe dla modelu xG
    html.Div([
        html.H3("Additional Shot Information"),
        html.Div([
            html.Label("Position Name:"),
            dcc.Dropdown(
                id='position-dropdown',
                options=[{'label': pos, 'value': pos} for pos in position_options],
                value=position_options[0]
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot Body Part:"),
            dcc.Dropdown(
                id='shot-body-dropdown',
                options=[{'label': part, 'value': part} for part in shot_body_part_options],
                value=shot_body_part_options[0]
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot Technique:"),
            dcc.Dropdown(
                id='shot-technique-dropdown',
                options=[{'label': tech, 'value': tech} for tech in shot_technique_options],
                value=shot_technique_options[0]
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot Type:"),
            dcc.Dropdown(
                id='shot-type-dropdown',
                options=[{'label': stype, 'value': stype} for stype in shot_type_options],
                value=shot_type_options[0]
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot First Time:"),
            dcc.RadioItems(
                id='shot-first-time',
                options=[{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}],
                value=False,
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot Aerial Won:"),
            dcc.RadioItems(
                id='shot-aerial-won',
                options=[{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}],
                value=False,
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot Follows Dribble:"),
            dcc.RadioItems(
                id='shot-follows-dribble',
                options=[{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}],
                value=False,
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'marginBottom': '10px'}),
        html.Div([
            html.Label("Shot Kick Off:"),
            dcc.RadioItems(
                id='shot-kick-off',
                options=[{'label': 'Yes', 'value': True}, {'label': 'No', 'value': False}],
                value=False,
                labelStyle={'display': 'inline-block', 'marginRight': '10px'}
            )
        ], style={'marginBottom': '10px'}),
    ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#e9ecef', 'borderRadius': '5px'}),
    
    # Przycisk obliczający xG i wyświetlający wynik
    html.Div([
        html.Button("Calculate xG", id='calculate-xg', n_clicks=0, 
                    style={'width': '100%', 'padding': '10px', 'fontWeight': 'bold'}),
        html.H2("Predicted xG:", style={'marginTop': '20px'}),
        html.Div(id='xg-output', style={'fontSize': '24px', 'fontWeight': 'bold'})
    ], style={'margin': '20px'})
])

# Callback aktualizujący zawartość strony w zależności od adresu URL
@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/xg':
        return xg_layout
    else:
        return analysis_layout

# Callback-e dla strony analitycznej (bez zmian)
@app.callback(
    Output('analysis-output', 'children'),
    Input('generate-analysis-button', 'n_clicks'),
    State('team-dropdown', 'value'),
    State('player-dropdown', 'value')
)
def update_analysis(n_clicks, team, player):
    if n_clicks == 0:
        return ""
    
    df_filtered = df_analysis.copy()
    if team != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == team]
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    
    fig_trajectory = plot_shot_trajectory(df_filtered, club=team, player=player, n=200)
    fig_distance = plot_distance_histogram(df_filtered, club=team, player=player)
    fig_angle = plot_angle_distance(df_filtered, club=team, player=player)
    fig_mins_hist = plot_shot_minute_histogram(df_filtered, club=team, player=player)
    fig_shot_by_body = plot_shot_body_part(df_filtered, club=team, player=player)
    fig_shot_by_technique = plot_shot_technique(df_filtered, club=team, player=player)
    
    img_trajectory = fig_to_base64(fig_trajectory)
    img_distance = fig_to_base64(fig_distance)
    img_angle = fig_to_base64(fig_angle)
    img_mins_hist = fig_to_base64(fig_mins_hist)
    img_shot_by_body = fig_to_base64(fig_shot_by_body)
    img_shot_by_technique = fig_to_base64(fig_shot_by_technique)
    
    layout = html.Div([
        html.H2("General Shot Analysis", style={'textAlign': 'center', 'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("Shots by Match Minute"),
                html.Img(src=img_mins_hist, style={'width': '100%', 'padding': '10px'})
            ], className="col-md-6"),
            html.Div([
                html.H4("Shot Trajectory"),
                html.Img(src=img_trajectory, style={'width': '100%', 'padding': '10px'})
            ], className="col-md-6")
        ], className="row", style={'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("Shot Distance Histogram"),
                html.Img(src=img_distance, style={'width': '100%', 'padding': '10px'})
            ], className="col-md-6"),
            html.Div([
                html.H4("Angle vs. Distance"),
                html.Img(src=img_angle, style={'width': '100%', 'padding': '10px'})
            ], className="col-md-6")
        ], className="row", style={'marginBottom': '20px'}),
        html.Div([
            html.Div([
                html.H4("Shots by Body Part"),
                html.Img(src=img_shot_by_body, style={'width': '100%', 'padding': '10px'})
            ], className="col-md-6"),
            html.Div([
                html.H4("Shots by Shot Technique"),
                html.Img(src=img_shot_by_technique, style={'width': '100%', 'padding': '10px'})
            ], className="col-md-6")
        ], className="row", style={'marginBottom': '20px'})
    ], style={'margin': '20px'})
    
    return layout

@app.callback(
    Output('density-output', 'src'),
    Input('draw-density-button', 'n_clicks'),
    State('team-dropdown', 'value'),
    State('player-dropdown', 'value')
)
def update_density(n_clicks, team, player):
    if n_clicks == 0:
        return ""
    
    df_filtered = df_analysis.copy()
    if team != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == team]
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    
    fig_pitch = plot_pitch_heatmap(df_filtered, x='x1', y='y1', club=team, player=player, cmap='Greens')
    
    return fig_to_base64(fig_pitch)

@app.callback(
    Output('points-store', 'data'),
    Input('pitch-graph', 'clickData'),
    Input('clear-markers', 'n_clicks'),
    State('points-store', 'data'),
    State('marker-type', 'value'),
    prevent_initial_call=True
)
def update_points(clickData, clear_clicks, current_data, marker_type):
    ctx = callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate

    trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # Inicjalizacja, jeśli current_data jest None
    if current_data is None:
        current_data = {'shooter': [], 'goalkeeper': [], 'teammate': [], 'opponent': []}

    if trigger_id == 'clear-markers':
        return {'shooter': [], 'goalkeeper': [], 'teammate': [], 'opponent': []}

    if trigger_id == 'pitch-graph' and clickData:
        point = clickData['points'][0]
        # Pobieramy współrzędne
        x = float(point['x'])
        y = float(point['y'])
        # Przycinamy współrzędne do zakresu 0-105 (dla x) oraz 0-68 (dla y)
        x = max(min(x, 105), 0)
        y = max(min(y, 68), 0)

        # Wprowadzamy ograniczenia dla poszczególnych typów punktów:
        if marker_type in ['shooter', 'goalkeeper']:
            # Strzelec i bramkarz – tylko jeden punkt (zastępujemy poprzedni)
            current_data[marker_type] = [[x, y]]
        elif marker_type == 'teammate':
            # Maksymalnie 9 punktów
            if len(current_data['teammate']) < 9:
                current_data['teammate'].append([x, y])
        elif marker_type == 'opponent':
            # Maksymalnie 10 punktów
            if len(current_data['opponent']) < 10:
                current_data['opponent'].append([x, y])
    return current_data

# Callback aktualizujący wykres boiska w oparciu o zapisane punkty
@app.callback(
    Output('pitch-graph', 'figure'),
    Input('points-store', 'data')
)
def refresh_pitch(points_data):
    return create_pitch_figure(points_data)

# Callback, który po kliknięciu "Calculate xG" zbiera wszystkie dane, oblicza zmienne i wywołuje model
@app.callback(
    Output('xg-output', 'children'),
    Input('calculate-xg', 'n_clicks'),
    State('points-store', 'data'),
    State('position-dropdown', 'value'),
    State('shot-body-dropdown', 'value'),
    State('shot-technique-dropdown', 'value'),
    State('shot-type-dropdown', 'value'),
    State('shot-first-time', 'value'),
    State('shot-aerial-won', 'value'),
    State('shot-follows-dribble', 'value'),
    State('shot-kick-off', 'value')
)
def calculate_xg(n_clicks, points_data, position, shot_body, shot_technique, shot_type,
                 shot_first_time, shot_aerial_won, shot_follows_dribble, shot_kick_off):
    if n_clicks == 0:
        return ""
    # Weryfikacja – musi być dokładnie jeden strzelec
    if len(points_data['shooter']) != 1:
        return "Please place exactly one shooter on the pitch."
    shooter = points_data['shooter'][0]
    # Sprawdzenie obecności bramkarza
    goalkeeper_present = len(points_data['goalkeeper']) >= 1
    num_teammates = len(points_data['teammate'])
    num_opponents = len(points_data['opponent'])
    
    # Obliczenie kąta i odległości od bramki (przyjmujemy, że bramka znajduje się w punkcie (0,34))
    angle = loc2angle(shooter[0], shooter[1])
    distance = loc2distance(shooter[0], shooter[1])
    open_goal = 1 if not goalkeeper_present else 0
    under_pressure = 0
    for opp in points_data['opponent']:
        if math.sqrt((opp[0]-shooter[0])**2 + (opp[1]-shooter[1])**2) <= 2:
            under_pressure = 1
            break
    shot_one_on_one = 1 if (goalkeeper_present and num_teammates == 0 and num_opponents == 0) else 0

    # Kodowanie zmiennych kategorycznych
    encoded_position = position_mapping.get(position, 0)
    encoded_shot_body = shot_body_part_mapping.get(shot_body, 0)
    encoded_shot_technique = shot_technique_mapping.get(shot_technique, 0)
    encoded_shot_type = shot_type_mapping.get(shot_type, 0)

    # Przygotowanie wektora cech – kolejność musi odpowiadać treningowi modelu
    features = np.array([[num_opponents, num_teammates, angle, distance, open_goal, 
                           under_pressure, shot_one_on_one, 
                           encoded_position, encoded_shot_body, encoded_shot_technique, encoded_shot_type, 
                           int(shot_first_time), int(shot_aerial_won), int(shot_follows_dribble), int(shot_kick_off)]])
    
    # Predykcja xG
    xg_prob = xgb_model.predict_proba(features)[0, 1]
    # Przekształcamy wynik na procenty
    xg_percent = xg_prob * 100
    return f"{xg_percent:.2f}%"

if __name__ == '__main__':
    app.run_server(debug=True)