import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer.pitch import Pitch
from scipy.spatial import Voronoi
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch
import seaborn as sns
import numpy as np

def plot_heatmap(data, club='all', player='all'):
   """
   Plots shot map (scatter) on a football pitch.
   
   Args:
       data: DataFrame containing cols: 'x1', 'y1', 'is_goal', 'team_name', 'player_name'
       club: team name to filter shots (default 'all' - no filter)
       player: player name to filter shots (default 'all' - no filter)
   """
   # Filter data if specific values provided
   df_filtered = data.copy()
   if club != 'all':
       df_filtered = df_filtered[df_filtered['team_name'] == club]
   if player != 'all':
       df_filtered = df_filtered[df_filtered['player_name'] == player]
   
   # Make sure 'is_goal' is boolean
   if df_filtered['is_goal'].dtype != 'bool':
       df_filtered['is_goal'] = df_filtered['is_goal'].astype(bool)
   
   # Define colors: goals - green, other shots - red
   colors = {True: 'green', False: 'red'}
   
   # Initialize pitch
   pitch = Pitch(pitch_color='#aabb97', line_color='white', stripe=True)
   fig, ax = pitch.draw(figsize=(12, 8))
   
   # Plot points - each point is shot location (x1, y1)
   ax.scatter(df_filtered['x1'], df_filtered['y1'],
              c = df_filtered['is_goal'].map(colors),
              s = 50,         # marker size
              alpha = 0.8, 
              edgecolors = 'black',
              zorder = 5)
   
   # Set plot title with club/player info if filtering was applied
   title = "Shot Map"
   if club != 'all':
       title += " - " + club
   if player != 'all':
       title += " - " + player
   ax.set_title(title)
   return fig

def plot_shot_trajectory(data, club='all', player='all', n=None):
   """
   Plots shot map with trajectories on a football pitch.
   
   Args:
       data: DataFrame with columns:
           'x1', 'y1' - starting shot coordinates
           'x1_end', 'y1_end' - end trajectory coordinates  
           'is_goal' - shot outcome (True if goal, False otherwise)
           'team_name' - club name
           'player_name' - player name
       club: Team name to filter data. Default 'all' (no filter)
       player: Player name to filter. Default 'all' (no filter)
       n: Optional, number of random shots to plot. If None, uses all data
   """
   # Filter data
   df_filtered = data.copy()
   if club != 'all':
       df_filtered = df_filtered[df_filtered['team_name'] == club]
   if player != 'all':
       df_filtered = df_filtered[df_filtered['player_name'] == player]
   
   # If n provided and we have more rows than n, get sample
   if n is not None and len(df_filtered) > n:
       df_plot = df_filtered.sample(n=n, random_state=42)
   else:
       df_plot = df_filtered

   # Make sure 'is_goal' is boolean
   if df_plot['is_goal'].dtype != 'bool':
       df_plot['is_goal'] = df_plot['is_goal'].astype(bool)
   
   # Define colors - goals in green, other shots in red
   colors = {True: 'green', False: 'red'}
   
   # Initialize pitch
   pitch = Pitch(pitch_color='#aabb97', line_color='white', stripe=True)
   fig, ax = pitch.draw(figsize=(12, 8))
   
   # Draw arrows - shot trajectories
   for _, row in df_plot.iterrows():
       ax.arrow(row['x1'], row['y1'],
                row['x1_end'] - row['x1'], row['y1_end'] - row['y1'],
                head_width=0.8, head_length=1.0,
                fc='gray', ec='gray', alpha=0.5, zorder=4)
   
   # Draw points - starting shot locations
   ax.scatter(df_plot['x1'], df_plot['y1'],
              c=df_plot['is_goal'].map(colors),
              s=50, alpha=0.8, edgecolors='black', zorder=5)
   
   # Set plot title
   title = "Shot Map with Trajectories"
   if club != 'all':
       title += " - " + club
   if player != 'all':
       title += " - " + player
   ax.set_title(title)
   return fig


def plot_distance_histogram(data, club='all', player='all', bins=30):
    """
    Plots histogram of shot distances with KDE curve overlay.
    
    Args:
        data: DataFrame with 'distance' column and optional 'team_name'/'player_name'
        club: Team name to filter shots. Default 'all' (no filter)
        player: Player name to filter shots. Default 'all' (no filter) 
        bins: Number of histogram bins (default 30)
    """
    # Copy data to avoid modifying original DataFrame
    df_filtered = data.copy()
    
    # Filter by club if specified
    if club != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == club]
    
    # Filter by player if specified 
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    
    # Create figure and axis using subplots
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot shot distance histogram with KDE overlay
    sns.histplot(df_filtered['distance'], bins=bins, kde=True, color='skyblue', ax=ax)
    ax.set_xlabel("Shot Distance")
    ax.set_ylabel("Number of Shots")
    
    # Set title with club/player info if filtered
    title = "Shot Distance Histogram"
    if club != 'all':
        title += " - " + club
    if player != 'all':
        title += " - " + player
    ax.set_title(title)
    
    return fig



def plot_angle_distance(data, club='all', player='all'):
    """
    Plots scatter plot showing relationship between shot distance and angle.
    
    Args:
        data: DataFrame with cols: 'distance', 'angle', 'is_goal', 'team_name', 'player_name'
        club: Team name (e.g. "Barcelona"). Default 'all' - no filter.
        player: Player name. Default 'all' - no filter.
    
    Returns:
        fig: Matplotlib Figure object with the generated scatter plot.
    """
    # Copy data to avoid modifying original
    df_filtered = data.copy()
    
    # Filter by club if specified
    if club != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == club]
    
    # Filter by player if specified
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    
    # Make sure 'is_goal' is boolean
    if df_filtered['is_goal'].dtype != 'bool':
        df_filtered['is_goal'] = df_filtered['is_goal'].astype(bool)
    
    # Define colors - goals green, other shots red
    colors = {True: 'green', False: 'red'}
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot scatter using the provided axis
    sns.scatterplot(data=df_filtered, x='distance', y='angle', 
                    hue='is_goal', palette=colors, alpha=0.7, ax=ax)
    ax.set_xlabel("Shot Distance")
    ax.set_ylabel("Shot Angle")
    
    # Set title with filter info
    title = "Angle vs. Distance of Shots"
    if club != 'all':
        title += " - " + club
    if player != 'all':
        title += " - " + player
    ax.set_title(title)
    
    return fig

def plot_shot_minute_histogram(data, club='all', player='all'):
    """
    Generates a histogram of shots and goals by match minute.
    
    Args:
        data: DataFrame with columns 'minute', 'is_goal', 'team_name', 'player_name'
        club: Team name to filter data (default 'all' means no filter)
        player: Player name to filter data (default 'all' means no filter)
        
    Returns:
        fig: Matplotlib Figure object containing the histogram plot.
    """
    # Filter data
    df_filtered = data.copy()
    if club != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == club]
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    
    # Ensure 'is_goal' is boolean
    if df_filtered['is_goal'].dtype != 'bool':
        df_filtered['is_goal'] = df_filtered['is_goal'].astype(bool)
    
    max_minute = df_filtered['minute'].max()
    bins = np.arange(0, max_minute + 1, 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    # Plot histogram for all shots (skyblue) with KDE overlay
    sns.histplot(data=df_filtered, x='minute', bins=bins, color='skyblue',
                 kde=True, stat='count', alpha=0.5, ax=ax, label='Shots')
    # Plot histogram for goals (green)
    sns.histplot(data=df_filtered[df_filtered['is_goal'] == True], x='minute', bins=bins, color='green',
                 kde=True, stat='count', alpha=0.5, ax=ax, label='Goals')
    ax.set_xlabel("Match Minute")
    ax.set_ylabel("Count")
    ax.set_title("Shots and Goals by Match Minute")
    ax.set_xticks(np.arange(0, max_minute + 1, 10))
    ax.legend()
    
    plt.tight_layout()
    return fig

def plot_shot_body_part(data, club='all', player='all'):
    """
    Generates a bar plot of shots and goals by shot body part.
    
    Args:
        data: DataFrame with columns 'shot_body_part_name', 'is_goal', 'team_name', 'player_name'
        club: Team name to filter data (default 'all')
        player: Player name to filter data (default 'all')
        
    Returns:
        fig: Matplotlib Figure object containing the bar plot.
    """
    # Filter data
    df_filtered = data.copy()
    if club != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == club]
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    if df_filtered['is_goal'].dtype != 'bool':
        df_filtered['is_goal'] = df_filtered['is_goal'].astype(bool)
    
    # Group data by shot_body_part_name
    df_body = df_filtered.groupby('shot_body_part_name', as_index=False).agg(
        total_shots=('is_goal', 'count'),
        goals=('is_goal', 'sum')
    ).sort_values(by='total_shots', ascending=False)
    
    x = np.arange(len(df_body))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    # Plot background bars for all shots (lightblue)
    bar_total = ax.bar(x, df_body['total_shots'], color='lightblue', label='Shots')
    # Plot overlay bars for goals (green)
    bar_goals = ax.bar(x, df_body['goals'], color='green', label='Goals')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_body['shot_body_part_name'], rotation=0)
    ax.set_xlabel("Body Part")
    ax.set_ylabel("Count")
    ax.set_title("Shots and Goals by Body Part")
    ax.legend()
    
    # Add annotations: goal percentage above each green bar
    for rect, (_, row) in zip(bar_goals, df_body.iterrows()):
        height = rect.get_height()
        x_center = rect.get_x() + rect.get_width() / 2
        percent = (row['goals'] / row['total_shots']) * 100 if row['total_shots'] > 0 else 0
        ax.text(x_center, height + 0.5, f"{percent:.1f}%", ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    return fig

def plot_shot_technique(data, club='all', player='all'):
    """
    Generates a bar plot of shots and goals by shot technique.
    
    Args:
        data: DataFrame with columns 'shot_technique_name', 'is_goal', 'team_name', 'player_name'
        club: Team name to filter data (default 'all')
        player: Player name to filter data (default 'all')
        
    Returns:
        fig: Matplotlib Figure object containing the bar plot.
    """
    # Filter data
    df_filtered = data.copy()
    if club != 'all':
        df_filtered = df_filtered[df_filtered['team_name'] == club]
    if player != 'all':
        df_filtered = df_filtered[df_filtered['player_name'] == player]
    if df_filtered['is_goal'].dtype != 'bool':
        df_filtered['is_goal'] = df_filtered['is_goal'].astype(bool)
    
    # Group data by shot_technique_name
    df_tech = df_filtered.groupby('shot_technique_name', as_index=False).agg(
        total_shots=('is_goal', 'count'),
        goals=('is_goal', 'sum')
    ).sort_values(by='total_shots', ascending=False)
    
    x = np.arange(len(df_tech))
    
    fig, ax = plt.subplots(figsize=(8, 4))
    # Background bars: all shots (lightblue)
    bar_total = ax.bar(x, df_tech['total_shots'], color='lightblue', label='Shots')
    # Overlay bars: goals (green)
    bar_goals = ax.bar(x, df_tech['goals'], color='green', label='Goals')
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_tech['shot_technique_name'], rotation=0)
    ax.set_xlabel("Shot Technique")
    ax.set_ylabel("Count")
    ax.set_title("Shots and Goals by Shot Technique")
    ax.legend()
    
    # Add annotations: goal percentage above each green bar
    for rect, (_, row) in zip(bar_goals, df_tech.iterrows()):
        height = rect.get_height()
        x_center = rect.get_x() + rect.get_width() / 2
        percent = (row['goals'] / row['total_shots']) * 100 if row['total_shots'] > 0 else 0
        ax.text(x_center, height + 0.5, f"{percent:.1f}%", ha='center', va='bottom', color='black')
    
    plt.tight_layout()
    return fig



def plot_pitch_heatmap(data, x, y, club='all', player='all', 
                       cmap='Greens', thresh=0.05, alpha=0.7, figsize=(15, 10), title=None):
    """
    Draws a heatmap overlay on a full pitch for given coordinates.
    
    Args:
        data: DataFrame with coordinate columns (e.g. 'x1', 'y1')
        x_col: name of column containing x coordinates
        y_col: name of column containing y coordinates  
        club: (optional) team name to filter data (default 'all' - no filter)
        player: (optional) player name to filter data (default 'all')
        cmap: colormap for heatmap (default 'Reds')
        thresh: threshold param passed to sns.kdeplot (default 0.05)
        alpha: heatmap transparency (default 0.7)
        figsize: figure size (default (12,8))
        title: plot title (default auto-generated)
       
    Filters data by club/player if specified and draws full pitch,
    then overlays heatmap for coordinates in x_col and y_col.
    
    Returns:
        fig: Matplotlib Figure object with the generated plot.
    """
    # Filter data if club or player specified
    df_filtered = data.copy()

    opponent_x_cols = [col for col in df_filtered.columns if col.startswith('x_player_opponent')]
    opponent_positions = []
    for x_col in opponent_x_cols:
        suffix = x_col.replace("x_player_opponent_", "")
        y_col = "y_player_opponent_" + suffix
        if y_col in df_filtered.columns:
            # Include additional columns if available:
            cols = [x_col, y_col]
            if 'team_name' in df_filtered.columns:
                cols.append('team_name')
            if 'player_name' in df_filtered.columns:
                cols.append('player_name')
            
            tmp = df_filtered[cols].dropna()
            if not tmp.empty:
                # Rename x and y columns to consistent "x" and "y"
                rename_dict = {x_col: "x", y_col: "y"}
                tmp = tmp.rename(columns=rename_dict)
                opponent_positions.append(tmp)
    if opponent_positions:
        df_opponents = pd.concat(opponent_positions, ignore_index=True)
    else:
        # Create empty DataFrame with proper columns
        cols = ["x", "y"]
        if 'team_name' in df_filtered.columns:
            cols.append("team_name")
        if 'player_name' in df_filtered.columns:
            cols.append("player_name")
        df_opponents = pd.DataFrame(columns=cols)

    # --- Aggregate teammate positions ---
    teammate_x_cols = [col for col in df_filtered.columns if col.startswith('x_player_teammate')]
    teammate_positions = []
    for x_col in teammate_x_cols:
        suffix = x_col.replace("x_player_teammate_", "")
        y_col = "y_player_teammate_" + suffix
        if y_col in df_filtered.columns:
            cols = [x_col, y_col]
            if 'team_name' in df_filtered.columns:
                cols.append('team_name')
            if 'player_name' in df_filtered.columns:
                cols.append('player_name')
            
            tmp = df_filtered[cols].dropna()
            if not tmp.empty:
                tmp = tmp.rename(columns={x_col: "x", y_col: "y"})
                teammate_positions.append(tmp)
    if teammate_positions:
        df_teammates = pd.concat(teammate_positions, ignore_index=True)
    else:
        cols = ["x", "y"]
        if 'team_name' in df_filtered.columns:
            cols.append("team_name")
        if 'player_name' in df_filtered.columns:
            cols.append("player_name")
        df_teammates = pd.DataFrame(columns=cols)

    # Split shots by outcome
    df_shots_goal = df_filtered[df_filtered['is_goal'] == True].copy()
    df_shots_no_goal = df_filtered[df_filtered['is_goal'] == False].copy()

    fig, axs = plt.subplots(2, 2, figsize=figsize)
    pitch = Pitch(pitch_color='#aabb97', line_color='white', stripe=True)
    # ----- WYKRES 1: Strzały - Gole (zielony) -----
    pitch.draw(ax=axs[0, 0])
    sns.kdeplot(
        data=df_shots_goal,
        x=x,
        y=y,
        fill=True,
        cmap='Greens',  # zielona mapa kolorów
        thresh=thresh,
        alpha=alpha,
        ax=axs[0, 0]
    )
    axs[0, 0].set_title("Shots - Goals")
    
    # ----- WYKRES 2: Strzały - Nietrafione (pomarańczowy) -----
    pitch.draw(ax=axs[0, 1])
    sns.kdeplot(
        data=df_shots_no_goal,
        x=x,
        y=y,
        fill=True,
        cmap='Oranges',  # pomarańczowa mapa kolorów
        thresh=thresh,
        alpha=alpha,
        ax=axs[0, 1]
    )
    axs[0, 1].set_title("Shots - No Goals")
    
    # ----- WYKRES 3: Rozstawienie kolegów (jasny niebieski) -----
    pitch.draw(ax=axs[1, 0])
    sns.kdeplot(
        data=df_teammates,
        x="x",
        y="y",
        fill=True,
        cmap='Blues',  # niebieska mapa kolorów
        thresh=thresh,
        alpha=alpha,
        ax=axs[1, 0]
    )
    axs[1, 0].set_title("Teammates")
    
    # ----- WYKRES 4: Rozstawienie przeciwników (czerwony) -----
    pitch.draw(ax=axs[1, 1])
    sns.kdeplot(
        data=df_opponents,
        x="x",
        y="y",
        fill=True,
        cmap='Reds',  # czerwona mapa kolorów
        thresh=thresh,
        alpha=alpha,
        ax=axs[1, 1]
    )
    axs[1, 1].set_title("Opponents")
    
    # Dla czytelności usuwamy znaczniki osi
    for ax in axs.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    return fig