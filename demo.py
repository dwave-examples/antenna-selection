# Copyright 2022 D-Wave Systems Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from itertools import combinations
from math import asin, cos, radians, sin, sqrt

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
from dimod import Binary, ConstrainedQuadraticModel, quicksum
from dwave.system import LeapHybridCQMSampler
from shapely.geometry import Point

try:
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib.use("agg")
    import matplotlib.pyplot as plt

def distance(lat1, long1, lat2, long2):
    ''' Compute the distance (miles) between two lat/long points.
    Args:
        - lat1, long1: float. Lat and long of point 1.
        - lat2, long 2: float. Lat and long of point 2.
    Returns:
        - dist: float. Distance in miles between points.
    '''
    # Taken from https://www.geeksforgeeks.org/program-distance-two-points-earth/

    # The math module contains a function named
    # radians which converts from degrees to radians.
    long1 = radians(long1)
    long2 = radians(long2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula
    dlong = long2 - long1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlong / 2)**2
 
    c = 2 * asin(sqrt(a))
    
    # Radius of earth in miles
    r = 3956
      
    # calculate the result
    return(c * r)

def get_existing_towers(filename):
    ''' Loads existing tower locations.
    Args:
        - filename: string. File name to load.
    Returns:
        - towers: df. Dataframe containing tower information.
    '''

    with open(filename) as f:
        lines = f.readlines()

    points = []
    lats = []
    longs = []
    for line in lines:
        temp = line.split("\t")
        points.append(temp[0])
        lats.append(float(temp[1]))
        longs.append(float(temp[2][:-2]))

    towers = pd.DataFrame({'Name': points, 'Latitude': lats, 'Longitude':longs})

    return towers

def gen_new_points(num_new_points, region_map):
    ''' Generates a random set of new locations in the region.
    Args:
        - num_new_points: int. Number of random points to identify.
        - region_map: gdf. Region of interest.
    Returns:
        - new_locs: list of [float, float]. New points as [lat, long].
    '''

    # Load the map boundaries for the region
    boundary = region_map["geometry"].unary_union
    min_long, min_lat, max_long, max_lat = boundary.bounds

    counter = 0
    new_locs = []

    while counter < num_new_points:
        new_long = (max_long - min_long) * np.random.random() + min_long
        new_lat = (max_lat - min_lat) * np.random.random() + min_lat
        point = Point(new_long, new_lat)

        # Check that new point is within region before appending
        if point.intersects(boundary):
            counter += 1
            new_locs.append([new_lat, new_long])

    return new_locs

def build_cqm(num_to_build, existing_towers, new_locs, radius):
    ''' Builds CQM for scenario.
    Args:
        - num_to_build: int. Number of new antennas to build.
        - existing_towers: df. Existing tower locations.
        - new_locs: List of [float, float]. List of potential build sites lat/long coords.
        - radius: int or float. Distance radius for interference.
    Returns:
        - cqm: ConstrainedQuadraticModel representing the optimization problem.
    '''

    # Initialize model
    cqm = ConstrainedQuadraticModel()

    # Build CQM variables
    tower_vars = {(row['Latitude'],row['Longitude'],row['Name']): Binary(row['Name']) for _, row in existing_towers.iterrows()}
    new_vars = {(new_locs[n][0],new_locs[n][1]): Binary(n) for n in range(len(new_locs))}

    # Make a combined list of all variables to calculate objective
    all_vars = tower_vars.copy()
    all_vars.update(new_vars)

    # Objective: minimize interference / maximize distance
    pair_list = list(combinations(all_vars.keys(), 2))
    dist = [distance(a[0], a[1], b[0], b[1])**2 for (a, b) in pair_list]
    max_dist = max(dist)
    biases = [dist[i] if dist[i] < radius**2 else max_dist for i in range(len(dist))]

    # Set the objective; negate the biases since we want to maximize
    cqm.set_objective(quicksum(-biases[i]*all_vars[pair_list[i][0]]*all_vars[pair_list[i][1]] for i in range(len(pair_list))))

    # Constraint: build exactly num_to_build new sites
    cqm.add_constraint(quicksum(new_vars.values()) == num_to_build)

    # Fix existing sites binary variables equal to 1
    cqm.fix_variables({key[2]: 1.0 for key in tower_vars.keys()})

    return cqm

def visualize(region_map, existing_towers, new_locs, build_sites):
    ''' Visualize the scenario and solution.
    Args:
        - region_map: gdf. Whole region map.
        - existing_towers: df. Dataframe containing tower information.
        - new_locs: list of [float, float]. New points as [lat, long].
        - build_sites: list of [float, float]. Build site points as [lat, long].
    Returns:
        None.
    '''

    print("\nVisualizing scenario and solution...")

    # Initialize figure and axes
    _, (ax, ax_final) = plt.subplots(nrows=1, ncols=2, figsize=(32, 12))
    ax.axis('off')
    ax_final.axis('off')

    # Draw borders or region
    region_map.plot(ax = ax, color = '#d3d3d3', zorder=0)
    region_map.plot(ax = ax_final, color = '#d3d3d3', zorder=0)

    # Draw existing towers
    gdf_towers = gpd.GeoDataFrame(
        existing_towers, geometry=gpd.points_from_xy(existing_towers.Longitude, existing_towers.Latitude))
    gdf_towers.plot(ax=ax, color='r', zorder=2)
    gdf_towers.plot(ax=ax_final, color='r', zorder=2)

    # Draw radius around existing towers
    radius = 30
    towers_radius = gdf_towers.copy()
    towers_radius['geometry'] = towers_radius['geometry'].buffer(radius/111)
    towers_radius.plot(ax=ax, color='r', alpha=0.1, zorder=1)
    towers_radius.plot(ax=ax_final, color='r', alpha=0.1, zorder=1)

    # Draw new potential build sites on map
    new_locations = pd.DataFrame(new_locs, columns=['Latitude','Longitude'])
    gdf_new = gpd.GeoDataFrame(
        new_locations, geometry=gpd.points_from_xy(new_locations.Longitude, new_locations.Latitude))
    gdf_new.plot(ax=ax, color='y', zorder=8)

    # Draw new selected build sites on map
    new_builds = pd.DataFrame(build_sites, columns=['Latitude','Longitude'])
    gdf_builds = gpd.GeoDataFrame(
        new_builds, geometry=gpd.points_from_xy(new_builds.Longitude, new_builds.Latitude))
    gdf_builds.plot(ax=ax_final, color='b', zorder=8)

    # Draw radius around selected build sites
    build_radius = gdf_builds.copy()
    build_radius['geometry'] = build_radius['geometry'].buffer(radius/111)
    build_radius.plot(ax=ax_final, color='b', alpha=0.1, zorder=1)

    # Make the figure look good
    ax.set_title("Potential Sites", fontsize = 24)
    ax_final.set_title("Determined Sites", fontsize = 24)

    # Save the figure
    plot_filename = 'map.png'
    plt.savefig(plot_filename)
    print("\nOutput saved as", plot_filename)


if __name__ == "__main__":

    # Load and draw country map from geojson file
    print("\nLoading map and scenario...")
    filename = "data/germany_states.geojson"
    file = open(filename)
    germany_map = gpd.read_file(file)

    # Load existing towers
    existing_towers = get_existing_towers("data/locations.txt")

    # Select random points within the country borders
    num_new = 100
    new_locs = gen_new_points(num_new, germany_map)
    num_to_build = 10

    print("\nBuilding CQM...")
    cqm = build_cqm(num_to_build, existing_towers, new_locs, radius=75)

    # Initialize the CQM solver
    sampler = LeapHybridCQMSampler()

    # Solve the problem using the CQM solver
    print("\nSending problem to hybrid solver...")
    sampleset = sampler.sample_cqm(cqm, label='Example - TV Towers')
    feasible_sampleset = sampleset.filter(lambda row: row.is_feasible)

    try:
        sample = feasible_sampleset.first.sample
    except:
        print("\nNo feasible solutions found.")
        exit()

    # soln = [key for key, val in sample.items() if val == 1.0]
    build_sites = [new_locs[key] for key, val in sample.items() if val == 1.0]
    print("\nSelected", len(build_sites), "build sites.")

    visualize(germany_map, existing_towers, new_locs, build_sites)