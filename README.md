# 🐐 Goat Simulation with GPU Support 🚀

Welcome to the **Goat Simulation** project! This project simulates goat AI behavior in a 2D pen environment. Goats move randomly within the pen, navigating toward a designated exit point while avoiding crossing paths with other goats. The simulation leverages **PyTorch** and GPU acceleration for faster computation.

---

## 📊 Overview

The simulation allows:

- **Multiple goats** to exit a 2D pen by navigating random paths.
- **GPU-powered simulation logic** via PyTorch to ensure faster computations.
- **Visualization & logging** of goat movements for analysis.

When you run the simulation:
- A CSV log (`goat_movements.csv`) is saved with movement data.
- A visualization (`goat_simulation.gif`) is generated showing the exit paths of each goat over time.

---

## 🛠️ Dependencies

Below are all the libraries required to run this simulation. They are necessary for GPU support, mathematical operations, visualization, and testing:

1. **PyTorch (with CUDA)**  
   - Enables GPU support and acceleration for fast simulation logic execution.  
   - Official website: [https://pytorch.org/](https://pytorch.org/)
   
2. **NumPy**  
   - Supports efficient numerical computation operations.  
   - Install via:  
     ```bash
     pip install numpy
     ```

3. **Matplotlib**  
   - Used to create visualizations of goat movements.  
   - Install via:  
     ```bash
     pip install matplotlib
     ```

4. **pytest**  
   - A testing tool for writing and running unit and integration tests.  
   - Install via:  
     ```bash
     pip install pytest
     ```

---

## 🚀 Installation Instructions

Follow these steps to set up the simulation environment and dependencies.

---

### 1️⃣ Clone the Repository
First, clone the project repository:

```bash
git clone (the repository that this readme is)
2️⃣ Set Up the Environment
You can set up dependencies in two ways:

Option 1: Conda Environment
If you use Conda, set up the environment using the provided environment.yml:

bash
conda env create -f environment.yml
conda activate goat-env

Option 2: Manually Install Dependencies
If you don't use Conda, manually install the required libraries:

bash
pip install torch numpy matplotlib pytest
3️⃣ Verify GPU Support
Run the following to ensure GPU support is enabled:

python
import torch
print(torch.cuda.is_available())  # Should return True if CUDA is configured properly.
🏃 How to Run the Simulation
After setting up the environment and dependencies, execute the simulation script:

bash
python goat_run.py
📊 During Execution:
You will be prompted with:


Enter grid size (5 to 100):
Here, enter an integer value between 5 and 100 to set the 2D simulation environment grid size.

🖥️ What to Expect After Running the Simulation:
After the simulation completes:

A movement log will be saved as goat_movements.csv.
This contains all path data for every goat in the simulation.
A visualization will be saved as goat_simulation.gif, showing all paths animated over time.
🏆 Testing Instructions
To ensure everything is running correctly, use the testing framework pytest.

1️⃣ Install pytest:
bash
pip install pytest
2️⃣ Run Tests:
Run the following to execute all unit and integration tests:

bash
pytest
✅ Tests validate:
Unit Tests: Test the logic behind goat movement.
Integration Tests: Validate simulation accuracy with GPU computations.
📊 Visualization Insights
After the simulation ends, you'll find the following files:

goat_movements.csv:

Contains paths for each goat's exit behavior. Open it with any CSV viewer to analyze patterns.
goat_simulation.gif:

A visualization showing each goat's path exiting the pen, animated over time.
To analyze movement patterns or debug behavior, inspect these visualizations.

I apologize. The formatting got out of whack. You were supposed to only be able to copy lines of code that would be needed to do things but then this whole last section became that way and now the readme file corrupts or looks awful when i try to fix this part that got messed up.
