import torch
import numpy as np
import csv
import random
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# User input for scalable grid size
GRID_SIZE = int(input("Enter grid size (5 to 100): "))
assert 5 <= GRID_SIZE <= 100, "Grid size must be between 5 and 100."
INNER_GRID_SIZE = GRID_SIZE - 2  # Exclude borders for goat placement
NUM_GOATS = (INNER_GRID_SIZE ** 2) // 2  # Half of the inner grid occupied by goats

# Initialize the grid with empty cells
grid = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.int32, device="cuda")

# Randomly place goats in the inner grid
goat_positions = []
while len(goat_positions) < NUM_GOATS:
    x, y = random.randint(1, INNER_GRID_SIZE), random.randint(1, INNER_GRID_SIZE)
    if grid[x, y] == 0:
        goat_positions.append((x, y))
        grid[x, y] = 1

# Randomly set exit location on the outer border
exit_row = random.choice([0, GRID_SIZE - 1])
exit_col = random.randint(0, GRID_SIZE - 1) if exit_row in [0, GRID_SIZE - 1] else random.choice([0, GRID_SIZE - 1])
grid[exit_row, exit_col] = 2  # Mark the exit

# Convert goat positions to a tensor
goat_positions_tensor = torch.tensor(goat_positions, dtype=torch.int32, device="cuda")
exited_flags = torch.zeros(NUM_GOATS, dtype=torch.int32, device="cuda")

# Movement log
movement_log = {goat_id: [] for goat_id in range(NUM_GOATS)}

# Function for goat movement in parallel
def move_goats(grid, goat_positions_tensor, exited_flags, exit_row, exit_col):
    directions = torch.randint(0, 4, (NUM_GOATS,), device="cuda")  # Random directions for all goats

    # Get current positions
    x_pos = goat_positions_tensor[:, 0]
    y_pos = goat_positions_tensor[:, 1]

    # Calculate new positions based on random directions
    new_x_pos = x_pos.clone()
    new_y_pos = y_pos.clone()

    new_x_pos[directions == 0] -= (new_x_pos[directions == 0] > 0).long()  # Move up
    new_x_pos[directions == 1] += (new_x_pos[directions == 1] < GRID_SIZE - 1).long()  # Move down
    new_y_pos[directions == 2] -= (new_y_pos[directions == 2] > 0).long()  # Move left
    new_y_pos[directions == 3] += (new_y_pos[directions == 3] < GRID_SIZE - 1).long()  # Move right

    # Check for collisions and exits
    for idx in range(NUM_GOATS):
        if exited_flags[idx] == 0:
            if new_x_pos[idx].item() == exit_row and new_y_pos[idx].item() == exit_col:
                exited_flags[idx] = 1
                grid[x_pos[idx].item(), y_pos[idx].item()] = 0
                goat_positions_tensor[idx] = torch.tensor([-1, -1], device="cuda")
            elif (1 <= new_x_pos[idx].item() <= INNER_GRID_SIZE and
                  1 <= new_y_pos[idx].item() <= INNER_GRID_SIZE and
                  grid[new_x_pos[idx].item(), new_y_pos[idx].item()] == 0):
                grid[x_pos[idx].item(), y_pos[idx].item()] = 0
                grid[new_x_pos[idx].item(), new_y_pos[idx].item()] = 1
                goat_positions_tensor[idx] = torch.tensor([new_x_pos[idx].item(), new_y_pos[idx].item()], device="cuda")

# Run simulation
iteration = 0
while exited_flags.sum().item() < NUM_GOATS:
    for i in range(NUM_GOATS):
        if exited_flags[i] == 0:
            x, y = goat_positions_tensor[i].tolist()
            movement_log[i].append((x, y))

    move_goats(grid, goat_positions_tensor, exited_flags, exit_row, exit_col)

    iteration += 1

# Finalize movement log for exited goats
for i in range(NUM_GOATS):
    if exited_flags[i] == 1:
        movement_log[i].append((-1, -1))

# Save movement log to CSV
with open("goat_movements.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for goat_id, movements in movement_log.items():
        row = [goat_id] + [f"{x},{y}" for x, y in movements]
        writer.writerow(row)

print(f"Simulation completed in {iteration} iterations.")
print("Movement log saved to 'goat_movements.csv'.")

# Visualization functions
def load_movement_log(filename):
    movement_log = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            goat_id = int(row[0])
            movements = [tuple(map(int, pos.split(","))) for pos in row[1:]]
            movement_log[goat_id] = movements
    return movement_log

def visualize_movements(grid_size, movement_log, exit_position, output_file="goat_simulation.gif"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.grid(color="gray", linestyle="--", linewidth=0.5)
    ax.set_title("Goat Simulation")

    # Initialize scatter plot for goats and exit
    goats_plot = ax.scatter([], [], color="blue", label="Goats")
    exit_marker = ax.scatter([exit_position[1]], [grid_size - exit_position[0] - 1], color="red", label="Exit")

    max_frames = max(len(movements) for movements in movement_log.values())

    def init():
        goats_plot.set_offsets(np.empty((0, 2)))  # Ensure it starts as an empty 2D array
        return goats_plot,

    def update(frame):
        positions = []
        for goat_id, movements in movement_log.items():
            if frame < len(movements):
                pos = movements[frame]
                if pos != (-1, -1):
                    positions.append([pos[1], grid_size - pos[0] - 1])
        
        positions = np.array(positions) if positions else np.empty((0, 2))
        goats_plot.set_offsets(positions)
        return goats_plot,

    ani = FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True)

    try:
        ani.save(output_file, fps=5, writer="pillow")
        print(f"Visualization saved as '{output_file}'")
    except Exception as e:
        print(f"Failed to save visualization: {e}")

    plt.close(fig)

# Generate visualization
print("Generating visualization...")
movement_log = load_movement_log("goat_movements.csv")
visualize_movements(GRID_SIZE, movement_log, (exit_row, exit_col))
print("Visualization saved as 'goat_simulation.gif'.")
