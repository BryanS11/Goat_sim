import numpy as np
import torch
import random
import csv

# Define the grid size and number of goats
GRID_SIZE = 13
INNER_GRID_SIZE = GRID_SIZE - 2  # to avoid borders for goats
NUM_GOATS = 61

# Initialize the grid with empty cells
grid = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.int32, device="cuda")

# Randomly place goats within the inner grid
goat_positions = []
while len(goat_positions) < NUM_GOATS:
    x, y = random.randint(1, INNER_GRID_SIZE), random.randint(1, INNER_GRID_SIZE)
    if grid[x, y] == 0:  # Place goat if cell is empty
        goat_positions.append((x, y))
        grid[x, y] = 1  # Mark the grid cell as occupied by a goat

# Randomly set exit location on border
exit_row = random.choice([0, GRID_SIZE - 1])
exit_col = random.randint(0, GRID_SIZE - 1)
grid[exit_row, exit_col] = 2  # Mark the exit with a different value

# Initialize movement log for each goat
movement_log = {goat_id: [] for goat_id in range(NUM_GOATS)}

# Convert goat positions to a tensor
goat_positions_tensor = torch.tensor(goat_positions, dtype=torch.int32, device="cuda").flatten()
exited_flags = torch.zeros(NUM_GOATS, dtype=torch.int32, device="cuda")

# Define exit location for CUDA kernel
exit_row_tensor = torch.tensor(exit_row, dtype=torch.int32, device="cuda")
exit_col_tensor = torch.tensor(exit_col, dtype=torch.int32, device="cuda")

# Function to move goats
def move_goats(grid, goat_positions, exited, exit_row, exit_col):
    num_goats = goat_positions.shape[0] // 2  # Since positions are flattened
    new_positions = goat_positions.clone()

    for idx in range(num_goats):
        if exited[idx] == 1:  # Skip if goat has exited
            continue

        x = goat_positions[2 * idx].item()  # Convert tensor to Python integer
        y = goat_positions[2 * idx + 1].item()

        # Random direction movement: 0=up, 1=down, 2=left, 3=right
        direction = random.randint(0, 3)
        new_x, new_y = x, y
        
        if direction == 0 and x > 0: new_x = x - 1  # up
        elif direction == 1 and x < GRID_SIZE - 1: new_x = x + 1  # down
        elif direction == 2 and y > 0: new_y = y - 1  # left
        elif direction == 3 and y < GRID_SIZE - 1: new_y = y + 1  # right

        # Check for collisions: move if cell is empty
        if grid[new_x, new_y].item() == 0:
            grid[x, y] = 0  # Clear old position
            grid[new_x, new_y] = 1  # Mark new position
            new_positions[2 * idx] = new_x
            new_positions[2 * idx + 1] = new_y

        # Check if the new position is the exit
        if new_x == exit_row and new_y == exit_col:
            exited[idx] = 1  # Mark goat as exited
            grid[x, y] = 0  # Clear old position
            new_positions[2 * idx] = -1  # Remove goat from positions
            new_positions[2 * idx + 1] = -1

    return new_positions

# Run simulation until all goats have exited
iteration = 0
while exited_flags.sum().item() < NUM_GOATS:
    # Record each goat's position in the log
    for i in range(NUM_GOATS):
        if exited_flags[i] == 0:
            movement_log[i].append((goat_positions_tensor[2 * i].item(), goat_positions_tensor[2 * i + 1].item()))

    # Move goats
    goat_positions_tensor = move_goats(grid, goat_positions_tensor, exited_flags, exit_row_tensor, exit_col_tensor)

    iteration += 1

# Record each goat's final position if they exited the grid
for i in range(NUM_GOATS):
    if exited_flags[i] == 1:
        movement_log[i].append((-1, -1))

# Save movement log to a file
with open("goat_movements.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    for goat_id, movements in movement_log.items():
        movement_str = [f"({x},{y})" for x, y in movements]
        writer.writerow([goat_id] + movement_str)

print(f"Simulation completed in {iteration} iterations.")
print("Movement log saved to 'goat_movements.csv'")
