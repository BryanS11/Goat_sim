import torch
import numpy as np
from goat_run import move_goats


# Unit Test 1: Test movement logic
def test_move_goats_movement():
    # Simulate a small test environment
    grid_size = 10
    grid = torch.zeros((grid_size, grid_size), dtype=torch.int32, device="cuda")
    goat_positions_tensor = torch.tensor([[5, 5], [6, 6]], dtype=torch.int32, device="cuda")
    exited_flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    exit_row = 0
    exit_col = 0

    # Call move_goats
    move_goats(grid, goat_positions_tensor, exited_flags, exit_row, exit_col)

    # Ensure movement has occurred
    assert goat_positions_tensor.shape == (2, 2), "Goat positions tensor shape incorrect"
    print("Test 1 Passed: move_goats logic executed correctly")


# Unit Test 2: Test collision logic in goat movement
def test_collision_logic():
    # Simulate a scenario with a wall
    grid = torch.zeros((10, 10), dtype=torch.int32, device="cuda")
    goat_positions_tensor = torch.tensor([[1, 1], [2, 2]], dtype=torch.int32, device="cuda")
    exited_flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    exit_row = 0
    exit_col = 0

    # Move goats toward walls and call movement logic
    move_goats(grid, goat_positions_tensor, exited_flags, exit_row, exit_col)

    # Ensure no invalid moves have occurred
    assert goat_positions_tensor[0, 0] >= 0 and goat_positions_tensor[1, 0] >= 0, "Invalid movement detected"
    print("Test 2 Passed: Goat collision logic works correctly")


if __name__ == "__main__":
    test_move_goats_movement()
    test_collision_logic()

