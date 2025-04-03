import dash
from dash import html, dcc, callback, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from f1_simulation import simulate_race
import joblib
import sqlite3

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Load the trained model
model = joblib.load('f1_model.joblib')
scaler = joblib.load('f1_scaler.joblib')

def load_circuits():
    conn = sqlite3.connect('f1.db')
    query = "SELECT circuitId, name, country FROM circuits"
    circuits = pd.read_sql_query(query, conn)
    conn.close()
    return circuits

def load_drivers():
    conn = sqlite3.connect('f1.db')
    query = """
    SELECT DISTINCT d.driverId, d.forename || ' ' || d.surname as name
    FROM drivers d
    ORDER BY d.surname, d.forename
    """
    drivers = pd.read_sql_query(query, conn)
    conn.close()
    return drivers

def load_constructors():
    conn = sqlite3.connect('f1.db')
    query = "SELECT constructorId, name FROM constructors ORDER BY name"
    constructors = pd.read_sql_query(query, conn)
    conn.close()
    return constructors

# Layout
app.layout = dbc.Container([
    html.H1("F1 Race Simulation and Analysis", className="text-center my-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Race Settings"),
                dbc.CardBody([
                    html.Label("Circuit:"),
                    dcc.Dropdown(
                        id='circuit-dropdown',
                        options=[{'label': f"{row['name']} ({row['country']})", 
                                'value': row['circuitId']} 
                               for _, row in load_circuits().iterrows()],
                        value=1,
                        className="mb-3"
                    ),
                    
                    html.Label("Drivers:"),
                    dcc.Dropdown(
                        id='drivers-dropdown',
                        options=[{'label': row['name'], 'value': row['driverId']} 
                               for _, row in load_drivers().iterrows()],
                        multi=True,
                        value=[1, 2, 3, 4, 5],  # Default to first 5 drivers
                        className="mb-3"
                    ),
                    
                    html.Label("Constructor:"),
                    dcc.Dropdown(
                        id='constructor-dropdown',
                        options=[{'label': row['name'], 'value': row['constructorId']} 
                               for _, row in load_constructors().iterrows()],
                        value=1,
                        className="mb-3"
                    ),
                    
                    html.Div(id='circuit-info', className="mt-3")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Simulation Controls"),
                dbc.CardBody([
                    dbc.Button("Run Simulation", id="run-simulation", 
                              color="primary", className="mb-3"),
                    dbc.Button("Run Visualization", id="run-visualization", 
                              color="success", className="mb-3"),
                    html.Div(id="simulation-status")
                ])
            ])
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Race Prediction"),
                dbc.CardBody([
                    html.Div(id="prediction-results")
                ])
            ])
        ], width=4)
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Race Results"),
                dbc.CardBody([
                    dcc.Graph(id="race-results-graph")
                ])
            ])
        ], width=12)
    ])
], fluid=True)

@callback(
    Output("circuit-info", "children"),
    Input("circuit-dropdown", "value")
)
def update_circuit_info(circuit_id):
    conn = sqlite3.connect('f1.db')
    query = """
    SELECT c.name, c.country, c.lat, c.lng, c.alt,
           COUNT(DISTINCT r.raceId) as num_races,
           AVG(res.fastestLapSpeed) as avg_speed
    FROM circuits c
    LEFT JOIN races r ON c.circuitId = r.circuitId
    LEFT JOIN results res ON r.raceId = res.raceId
    WHERE c.circuitId = ?
    GROUP BY c.circuitId
    """
    info = pd.read_sql_query(query, conn, params=[circuit_id]).iloc[0]
    conn.close()
    
    return html.Div([
        html.P(f"Location: {info['country']}"),
        html.P(f"Coordinates: {info['lat']:.4f}, {info['lng']:.4f}"),
        html.P(f"Altitude: {info['alt']}m"),
        html.P(f"Number of races: {info['num_races']}"),
        html.P(f"Average speed: {info['avg_speed']:.1f} km/h")
    ])

@callback(
    [Output("race-results-graph", "figure"),
     Output("simulation-status", "children")],
    [Input("run-simulation", "n_clicks")],
    [State("circuit-dropdown", "value"),
     State("drivers-dropdown", "value"),
     State("constructor-dropdown", "value")]
)
def run_simulation(n_clicks, circuit_id, driver_ids, constructor_id):
    if n_clicks is None:
        return {}, "Click 'Run Simulation' to start"
    
    if not driver_ids or len(driver_ids) < 2:
        return {}, "Please select at least 2 drivers"
    
    # Run simulation with selected drivers and constructor
    results = simulate_race(circuit_id, driver_ids=driver_ids, constructor_id=constructor_id)
    
    # Create visualization
    fig = go.Figure()
    
    # Add lap times bar chart
    fig.add_trace(go.Bar(
        x=results['position'],
        y=results['lap_times'],
        name='Lap Times',
        marker_color='blue'
    ))
    
    # Add points line
    fig.add_trace(go.Scatter(
        x=results['position'],
        y=results['points'],
        name='Points',
        yaxis='y2',
        line=dict(color='red')
    ))
    
    fig.update_layout(
        title='Race Results',
        xaxis_title='Position',
        yaxis_title='Lap Time (seconds)',
        yaxis2=dict(
            title='Points',
            overlaying='y',
            side='right',
            range=[0, 25]
        ),
        showlegend=True
    )
    
    return fig, "Simulation completed!"

@callback(
    Output("prediction-results", "children"),
    [Input("circuit-dropdown", "value"),
     Input("drivers-dropdown", "value"),
     Input("constructor-dropdown", "value")]
)
def update_prediction(circuit_id, driver_ids, constructor_id):
    try:
        if not driver_ids or len(driver_ids) < 2:
            return html.Div([
                html.H4("Predicted Performance"),
                html.P("Please select at least 2 drivers")
            ])
        
        # Get circuit and driver features
        conn = sqlite3.connect('f1.db')
        query = """
        SELECT 
            r.year,
            r.round,
            c.circuitId,
            d.driverId,
            con.constructorId,
            AVG(res.grid) as grid,
            AVG(res.laps) as laps,
            AVG(res.fastestLap) as fastestLap,
            AVG(res.fastestLapTime) as fastestLapTime,
            AVG(res.fastestLapSpeed) as fastestLapSpeed,
            c.lat,
            c.lng,
            c.alt
        FROM circuits c
        LEFT JOIN races r ON c.circuitId = r.circuitId
        LEFT JOIN results res ON r.raceId = res.raceId
        LEFT JOIN drivers d ON res.driverId = d.driverId
        LEFT JOIN constructors con ON res.constructorId = con.constructorId
        WHERE c.circuitId = ? AND d.driverId IN ({}) AND con.constructorId = ?
        GROUP BY c.circuitId, r.year, r.round, d.driverId, con.constructorId
        ORDER BY r.year DESC, r.round DESC
        LIMIT 1
        """.format(','.join('?' * len(driver_ids)))
        
        params = [circuit_id] + driver_ids + [constructor_id]
        features = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if features.empty:
            return html.Div([
                html.H4("Predicted Performance"),
                html.P("No historical data available for the selected combination.")
            ])
        
        # Convert fastestLapTime to seconds
        def convert_time(time_str):
            if pd.isna(time_str) or time_str == '\\N':
                return np.nan
            try:
                time_parts = time_str.split(':')
                if len(time_parts) == 3:
                    minutes, seconds, milliseconds = time_parts
                    return float(minutes) * 60 + float(seconds) + float(milliseconds) / 1000
                return np.nan
            except:
                return np.nan
        
        features['fastestLapTime'] = features['fastestLapTime'].apply(convert_time)
        
        # Convert numeric columns
        numeric_columns = ['fastestLapSpeed', 'fastestLap', 'grid', 'laps']
        for col in numeric_columns:
            features[col] = pd.to_numeric(features[col].replace('\\N', np.nan), errors='coerce')
        
        # Fill missing values with median
        for col in numeric_columns + ['fastestLapTime']:
            features[col] = features[col].fillna(features[col].median())
        
        # Prepare features for prediction
        X = features[['year', 'round', 'circuitId', 'grid', 'laps', 
                     'fastestLap', 'fastestLapTime', 'fastestLapSpeed',
                     'lat', 'lng', 'alt']]
        X_scaled = scaler.transform(X)
        
        # Make prediction
        predicted_position = model.predict(X_scaled)[0]
        
        # Get prediction confidence using model's feature importances
        feature_importance = model.feature_importances_
        confidence = np.mean(feature_importance) * 100  # Convert to percentage
        
        return html.Div([
            html.H4("Predicted Performance"),
            html.P(f"Expected finishing position: {predicted_position:.1f}"),
            html.P(f"Model confidence: {confidence:.1f}%")
        ])
    except Exception as e:
        return html.Div([
            html.H4("Predicted Performance"),
            html.P(f"Error generating prediction: {str(e)}")
        ])

if __name__ == '__main__':
    app.run(host = "0.0.0.0", debug=True) 