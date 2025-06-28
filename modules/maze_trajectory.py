import numpy as np
import pandas as pd
import json

def maze_to_local_coords(maze_solution, maze_side):
    """
    Converts maze coordinates to coords on a local coordinate system.

    :param maze_solution: List of (row, col) tuples representing the maze solution.
    """

    real_world_path = []

    for (row, col) in maze_solution:
        x = col * maze_side / 8
        y = row * -(maze_side / 8)

        real_world_path.append((x, y))

    return real_world_path

def compute_centroid(coords):
    # Convert list of tuples to numpy array for easy mean calculation
    coords = np.array(coords)
    centroid = np.mean(coords, axis=0)
    return centroid

def main():
    # Example maze solution (row, col)
    #maze_solution = [(2, 0), (2, 1), (3, 1), (4, 1), (4, 2), (3, 2), (3, 3), (3, 4), (3, 5)]
    #maze_solution = [(2, 0), (2, 1), (2, 2), (1, 2), (1, 3), (1, 4), (2, 4), (2, 3), (3, 3), (3, 2), (4, 2), (4, 3), (4, 4), (3, 4), (3, 5)]
    #maze_solution = [(3, 0), (3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (7, 2), (7,3), (6, 3), (5, 3), (5, 4), (5, 5), (5, 6), (5, 7), (5, 8)]
    #maze_solution = [(3, 0), (3, 1), (2, 1), (1, 1), (1, 2), (1, 3), (2, 3), (3, 3), (3, 4), (3, 5), (3, 6), (3, 7), (4, 7), (5, 7), (5, 8)]

    # Load data from a JSON file
    with open('path.json', 'r') as file:
        data_loaded = json.load(file)

    # Convert lists back to tuples
    maze_solution = [tuple(item) for item in data_loaded]

    # Actual side length of maze
    maze_side = 130

    # Convert maze solution to real-world coordinates
    local_path = maze_to_local_coords(maze_solution, maze_side)
    # print("Real-world path:", local_path)

    # Local coordinate system corners
    local_corners = [(0, 0), (maze_side, 0), (maze_side, -maze_side), (0, -maze_side)]

    # Real-world corners (already defined)
    real_world_corners = [
        [-276, -234],  # Top-left corner
        [-289, -344],  # Bottom-left corner
        [-179, -344],  # Bottom-right corner
        [-165.8, -232.8]  # Top-right corner
    ]

    # Calculate the centroids to know where to shift
    local_centroid = compute_centroid(local_corners)
    real_world_centroid = compute_centroid(real_world_corners)

    # Shift each point
    real_world_path = [(coord + real_world_centroid - local_centroid) for coord in local_path]

    # Convert to a dataframe
    path_df = pd.DataFrame(real_world_path, columns=['x', 'y'])

    # Convert DataFrame values into list of tuples
    centers_mm = [(row['x'], row['y']) for index, row in path_df.iterrows()]

    # Save the DataFrame to a CSV file
    path_df.to_csv('real_world_path.csv', index=False, header=False)
    # print(f"Real world path: \n{path_df}")
    print("Real-world path saved to 'real_world_path.csv'.")

    return centers_mm




if __name__ == "__main__":
    main()
