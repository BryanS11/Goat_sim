import torch
import numpy as np
from goat_run import move_goats


def test_integration_simulation():
    """
    Simulates multiple goats exiting under a controlled environment to ensure simulation logic works as expected.
    """
    # Simulated environment
    grid_size = 10
    grid = torch.zeros((grid_size, grid_size), dtype=torch.int32, device="cuda")
    goat_positions_tensor = torch.tensor([[5, 5], [5, 6], [6, 6]], dtype=torch.int32, device="cuda")
    exited_flags = torch.zeros(3, dtype=torch.int32, device="cuda")
    exit_row = 0
    exit_col = 0

    # Simulate a single movement iteration
    move_goats(grid, goat_positions_tensor, exited_flags, exit_row, exit_col)

    # Test if at least one goat exited
    exited_count = exited_flags.sum().item()
    assert exited_count >= 1, "Integration test failed: No goat exited"
    print(f"Integration Test Passed: {exited_count} goats exited successfully")


if __name__ == "__main__":
    test_integration_simulation()
