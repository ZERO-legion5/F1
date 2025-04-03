import subprocess
import json
import pandas as pd
import numpy as np
import sqlite3

def simulate_race(circuit_id, driver_ids=None, constructor_id=None, num_drivers=20):
    """
    Simulate a race using Julia simulation with selected drivers and constructor.
    
    Args:
        circuit_id (int): The ID of the circuit to simulate
        driver_ids (list): List of driver IDs to include in the simulation
        constructor_id (int): The constructor ID to use for all drivers
        num_drivers (int): Number of drivers to simulate (default: 20)
    """
    # Create temporary Julia script
    with open('temp_simulation.jl', 'w') as f:
        f.write('''
using SQLite
using DataFrames
using JSON
using Statistics
using Random
using Distributions

function get_circuit_info(circuit_id)
    db = SQLite.DB("f1.db")
    query = """
    SELECT name, country, lat, lng, alt
    FROM circuits
    WHERE circuitId = ?
    """
    result = DataFrame(DBInterface.execute(db, query, [circuit_id]))
    close(db)
    return result
end

function get_historical_performance(circuit_id, driver_ids, constructor_id)
    db = SQLite.DB("f1.db")
    query = """
    SELECT 
        r.year,
        r.round,
        d.driverId,
        d.forename || ' ' || d.surname as driver_name,
        con.name as constructor_name,
        res.fastestLapTime,
        res.fastestLapSpeed,
        res.position
    FROM races r
    JOIN results res ON r.raceId = res.raceId
    JOIN drivers d ON res.driverId = d.driverId
    JOIN constructors con ON res.constructorId = con.constructorId
    WHERE r.circuitId = ? 
    AND d.driverId IN ($(join(driver_ids, ',')))
    AND con.constructorId = ?
    ORDER BY r.year DESC, r.round DESC
    """
    result = DataFrame(DBInterface.execute(db, query, [circuit_id, constructor_id]))
    close(db)
    return result
end

function parse_lap_time(time_str)
    if time_str === nothing || time_str == raw"\\N"
        return nothing
    end
    try
        parts = split(time_str, ":")
        if length(parts) == 3
            minutes = parse(Float64, parts[1])
            seconds = parse(Float64, parts[2])
            milliseconds = parse(Float64, parts[3]) / 1000
            return minutes * 60 + seconds + milliseconds
        end
        return nothing
    catch
        return nothing
    end
end

function get_points(position)
    points_table = Dict(
        1 => 25,
        2 => 18,
        3 => 15,
        4 => 12,
        5 => 10,
        6 => 8,
        7 => 6,
        8 => 4,
        9 => 2,
        10 => 1
    )
    return get(points_table, position, 0)
end

function simulate_race(circuit_id, driver_ids, constructor_id)
    # Get circuit information
    circuit_info = get_circuit_info(circuit_id)
    
    # Get historical performance data
    historical_data = get_historical_performance(circuit_id, driver_ids, constructor_id)
    
    # Convert lap times to seconds
    historical_data.lap_time_seconds = parse_lap_time.(historical_data.fastestLapTime)
    
    # Filter out invalid times and calculate base lap time
    valid_times = filter(x -> x !== nothing, historical_data.lap_time_seconds)
    base_lap_time = isempty(valid_times) ? 90.0 : mean(valid_times)
    
    # Generate random performance factors for each driver
    Random.seed!(42)  # For reproducibility
    num_drivers = length(driver_ids)
    performance_factors = rand(Normal(1.0, 0.05), num_drivers)
    
    # Simulate race results
    lap_times = base_lap_time * performance_factors
    sorted_indices = sortperm(lap_times)
    
    # Calculate points based on F1 scoring system
    points = [get_points(i) for i in 1:num_drivers]
    
    # Create results DataFrame
    results = DataFrame(
        position = 1:num_drivers,
        driver_id = driver_ids[sorted_indices],
        lap_times = lap_times[sorted_indices],
        points = points
    )
    
    return results
end

# Get command line arguments
circuit_id = parse(Int, ARGS[1])
driver_ids = parse.(Int, split(ARGS[2], ","))
constructor_id = parse(Int, ARGS[3])

# Run simulation
results = simulate_race(circuit_id, driver_ids, constructor_id)

# Save results to JSON file
open("race_results.json", "w") do f
    JSON.print(f, Dict(
        "circuit_id" => circuit_id,
        "driver_ids" => driver_ids,
        "constructor_id" => constructor_id,
        "results" => results
    ))
end
''')

    # Run Julia simulation
    subprocess.run(['julia', 'temp_simulation.jl', str(circuit_id), 
                   ','.join(map(str, driver_ids)), str(constructor_id)], 
                  check=True)
    
    # Read results
    with open('race_results.json', 'r') as f:
        results = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(results['results'])
    
    # Get driver names
    conn = sqlite3.connect('f1.db')
    driver_query = """
    SELECT driverId, forename || ' ' || surname as name
    FROM drivers
    WHERE driverId IN ({})
    """.format(','.join('?' * len(driver_ids)))
    
    driver_names = pd.read_sql_query(driver_query, conn, params=driver_ids)
    conn.close()
    
    # Merge driver names with results
    df = df.merge(driver_names, left_on='driver_id', right_on='driverId')
    
    return df

if __name__ == '__main__':
    # Test the simulation
    results = simulate_race(1, [1, 2, 3, 4, 5], 1)
    print(results) 