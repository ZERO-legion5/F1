import pandas as pd
import sqlite3
import os

def create_database():
    # Create SQLite database
    conn = sqlite3.connect('f1.db')
    
    # Define the folder path
    folder_path = "dataset/"
    
    # List of CSV files and their corresponding table names
    csv_files = {
        'circuits.csv': 'circuits',
        'constructors.csv': 'constructors',
        'drivers.csv': 'drivers',
        'races.csv': 'races',
        'seasons.csv': 'seasons',
        'status.csv': 'status',
        'qualifying.csv': 'qualifying',
        'results.csv': 'results',
        'constructor_results.csv': 'constructor_results',
        'constructor_standings.csv': 'constructor_standings',
        'driver_standings.csv': 'driver_standings',
        'sprint_results.csv': 'sprint_results',
        'lap_times.csv': 'lap_times',
        'pit_stops.csv': 'pit_stops'
    }
    
    # Load and insert data for each CSV file
    for csv_file, table_name in csv_files.items():
        print(f"Processing {csv_file}...")
        df = pd.read_csv(os.path.join(folder_path, csv_file))
        
        # Convert DataFrame to SQL
        df.to_sql(table_name, conn, if_exists='replace', index=False)
        print(f"Created table {table_name}")
    
    # Create indexes for better query performance
    conn.execute('CREATE INDEX IF NOT EXISTS idx_races_year ON races(year)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_results_race ON results(raceId)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_driver_standings_race ON driver_standings(raceId)')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_constructor_standings_race ON constructor_standings(raceId)')
    
    conn.close()
    print("Database setup completed!")

if __name__ == "__main__":
    create_database() 