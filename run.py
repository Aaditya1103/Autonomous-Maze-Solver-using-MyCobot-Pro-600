from modules import capture_maze
from modules import adjust_img_contrast
from modules import solve_maze_matrix
from modules import maze_trajectory


def main():
    try:
        print("Running capture_maze.main")
        capture_maze.main()
        print("Running adjust_img_contrast.main")
        adjust_img_contrast.main()
        print("Running solve_maze_matrix.main")
        solve_maze_matrix.main()
        print("Running maze_trajectory.main")
        centers_mm = maze_trajectory.main()
        print(centers_mm)
        return centers_mm
    except IOError as e:
        print(e)
    except ValueError as e:
        print(e)