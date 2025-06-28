import cv2
import numpy as np
from collections import deque
import json

def capture_webcam_image():
    """
    Captures an image from the webcam and returns it as a NumPy array.
    """
    cap = cv2.VideoCapture(0)  # Open the webcam (0 is usually the default webcam)
    
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        return None

    print("Press 's' to capture the image and save it, or 'q' to quit.")
    
    while True:
        ret, frame = cap.read()  # Read a frame from the webcam
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        cv2.imshow("Webcam", frame)  # Show the webcam feed in a window

        key = cv2.waitKey(1) & 0xFF  # Wait for a key press
        if key == ord('s'):  # If 's' is pressed, save and return the frame
            print("Image captured!")
            cap.release()
            cv2.destroyAllWindows()
            return frame
        elif key == ord('q'):  # If 'q' is pressed, quit without saving
            print("Exiting without capturing.")
            cap.release()
            cv2.destroyAllWindows()
            return None


def draw_solved_path(image, maze_matrix, solved_path, grid_size=(8, 8)):

    height, width = image.shape[:2]
    num_rows, num_cols = grid_size

    # Calculate cell dimensions
    row_step = height // num_rows
    col_step = width // num_cols

    # Draw the solved path
    for (row, col) in solved_path:
        # Calculate the center of the cell
        center_x = int(col * col_step + col_step / 2)
        center_y = int(row * row_step + row_step / 2)
        
        # Draw a small circle to mark the path
        cv2.circle(image, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

    return image

def create_maze_matrix(image, grid_size=(8, 8), wall_threshold=0.8):

    height, width = image.shape[:2]  # Get the dimensions of the image
    num_rows, num_cols = grid_size
    
    # Calculate the spacing for rows and columns
    row_step = height // num_rows
    col_step = width // num_cols

    # Create initial matrix
    maze_matrix = np.ones((9, 9), dtype=int)

    # Convert image to grayscale for easier analysis
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Draw horizontal lines
    for i in range(0, num_rows+1):
        for j in range(1, num_cols):
            y = i * row_step
            x = j * col_step
            y_start = int(max(0, y-(row_step/4)))
            y_end = int(min(height, y+(row_step/4)))
            x_start = int(x-(col_step/4))
            x_end = int(x+(col_step/4))

            section = gray_image[y_start:y_end, x_start:x_end]
            # If section has a portion that is higher than the threshold that is white, then change the corresponding maze_matrix coordinate to 0.

            light_pixels = np.sum(section > 128)
            total_pixels = section.size

            # If the fraction of dark pixels exceeds the threshold, mark as wall
            if light_pixels / total_pixels > wall_threshold:
                maze_matrix[i, j] = 0  # Open



    return maze_matrix

def solve_maze(maze):
    """
    Solves a 9x9 maze matrix using Breadth-First Search (BFS) algorithm.
    
    :param maze: A 9x9 NumPy array where 0 represents paths and 1 represents walls.
    :return: A list of tuples representing the path from top to bottom or an empty list if no path exists.
    """
    num_rows, num_cols = maze.shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right
    queue = deque([(0, j) for j in range(num_cols) if maze[0][j] == 0])  # Start positions in the first row
    visited = set(queue)
    parent = {pos: None for pos in queue}  # To track the path

    while queue:
        current = queue.popleft()

        # If we reach the last row, reconstruct and return the path
        if current[0] == num_rows - 1:
            path = []
            step = current
            while step is not None:
                path.append(step)
                step = parent[step]
            return path[::-1]  # Return reversed path

        # Explore neighbors
        for direction in directions:
            next_pos = (current[0] + direction[0], current[1] + direction[1])
            if 0 <= next_pos[0] < num_rows and 0 <= next_pos[1] < num_cols and next_pos not in visited and maze[next_pos[0]][next_pos[1]] == 0:
                queue.append(next_pos)
                visited.add(next_pos)
                parent[next_pos] = current

    return []  # Return an empty list if no path is found


def main():
    # Capture the image from the webcam
    image = capture_webcam_image()
    if image is None:
        print("Error: Could not capture image from webcam.")
        return

    # Create and return the maze matrix
    maze_matrix = create_maze_matrix(image)

    # Solve the maze
    solved_path = solve_maze(maze_matrix)
    
    if not solved_path:
        print("No path found through the maze.")
        return

    # Convert tuples in the list to lists for JSON compatibility
    path_json_ready = [list(item) for item in solved_path]

    # Save data to a JSON file
    with open('path.json', 'w') as file:
        json.dump(path_json_ready, file)

    print("Data has been converted to JSON and saved.")

    # Draw the solved path on the original image
    solved_image = draw_solved_path(image, maze_matrix, solved_path)

    # Display the image with the solved path
    cv2.imshow("Solved Maze", solved_image)
    cv2.waitKey(0)  # Wait until a key is pressed
    cv2.destroyAllWindows()




if __name__ == "__main__":
    main()
