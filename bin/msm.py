import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from molgri.space.fullgrid import FullGrid
from molgri.plotting.fullgrid_plots import (
    plot_array_heatmap,
    plot_violin_position_orientation,
    plot_adjacency_array,
    plot_cartesian_voronoi,
    plot_spherical_voronoi
)
from molgri.io import OneMoleculeReader, EnergyReader, PtWriter
from molgri.molecules.transitions import SQRA, DecompositionTool

from scipy.constants import physical_constants
from scipy.constants import Boltzmann

AVOGADRO_CONSTANT = physical_constants["Avogadro constant"][0]

# Configuration dictionary
config = {
    "grid_identifier": "noRotGrid",
    "num_repeats": 1,
    "params_grid": {
        "num_orientations": 10,
        "num_directions": 10,
        "radial_distances_nm": [0.1, 0.2, 0.3],
        "factor_orientation_to_position": 1.0,
        "position_grid_is_cartesian": True
    },
    "params_sqra": {
        "tau_t": 1e-15,
        "temperature_K": 273,
        "energy_type": "Potential",
        "mass_kg": 6.64e-27,
        "lower_lim_rate_matrix": "None",
        "upper_lim_rate_matrix": "None",
        "tol": 1e-5,
        "maxiter": 40000000,
        "sigma": 1,
        "which": "LM",
        "number_lowest_E_structures": 5
    },
    "structure_extension": "xyz",
    "trajectory_extension": "pt",
}

GRID_ID = config["grid_identifier"]
PATH_THIS_GRID = f"./grids/{GRID_ID}/"  # Local directory for grid data
os.makedirs(PATH_THIS_GRID, exist_ok=True)

def run_grid():
    """Create a full grid and its geometric parameters."""
    tgrid=[2,5,20]
    t_grid_name = "linspace(" + str(tgrid[0] /10) + "," + str(tgrid[1] / 10) + str(tgrid[2]) + ")"
    fg = FullGrid("zero","ico_80",t_grid_name, factor=2, position_grid_cartesian=False)
    full_array = fg.get_full_grid_as_array()
    adjacency = fg.get_full_adjacency()
    adjacency_only_position = fg.get_full_adjacency(only_position=True)
    adjacency_only_orientation = fg.get_full_adjacency(only_orientation=True)
    borders = fg.get_full_borders()
    distances = fg.get_full_distances()
    volumes = fg.get_total_volumes()
    return full_array, adjacency, adjacency_only_position, adjacency_only_orientation, borders, distances, volumes


def LJCluster(pts):
    x = pts[0]
    y = pts[1]
    z = pts[2]
    return 0 + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (-1.7320508075688772 - y)**2 + (0.0 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (-1.7320508075688772 - y)**2 + (0.0 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (0.5773502691896257 - y)**2 + (-1.6329931618554518 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (0.5773502691896257 - y)**2 + (-1.6329931618554518 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((-2.0 - x)**2 + (0.0 - y)**2 + (0.0 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((-2.0 - x)**2 + (0.0 - y)**2 + (0.0 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (0.5773502691896257 - y)**2 + (1.6329931618554518 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (0.5773502691896257 - y)**2 + (1.6329931618554518 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (1.7320508075688772 - y)**2 + (0.0 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((-1.0 - x)**2 + (1.7320508075688772 - y)**2 + (0.0 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((0.0 - x)**2 + (-1.1547005383792517 - y)**2 + (-1.6329931618554518 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((0.0 - x)**2 + (-1.1547005383792517 - y)**2 + (-1.6329931618554518 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((1.0 - x)**2 + (-1.7320508075688772 - y)**2 + (0.0 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((1.0 - x)**2 + (-1.7320508075688772 - y)**2 + (0.0 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((0.0 - x)**2 + (-1.1547005383792517 - y)**2 + (1.6329931618554518 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((0.0 - x)**2 + (-1.1547005383792517 - y)**2 + (1.6329931618554518 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((1.0 - x)**2 + (0.5773502691896257 - y)**2 + (-1.6329931618554518 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((1.0 - x)**2 + (0.5773502691896257 - y)**2 + (-1.6329931618554518 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((0.0 - x)**2 + (0.0 - y)**2 + (0.0 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((0.0 - x)**2 + (0.0 - y)**2 + (0.0 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((1.0 - x)**2 + (0.5773502691896257 - y)**2 + (1.6329931618554518 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((1.0 - x)**2 + (0.5773502691896257 - y)**2 + (1.6329931618554518 - z)**2)))**6) + 4 * 1.0 * ((1.7817974362806785/(np.sqrt((1.0 - x)**2 + (1.7320508075688772 - y)**2 + (0.0 - z)**2)))**12 - (1.7817974362806785/(np.sqrt((1.0 - x)**2 + (1.7320508075688772 - y)**2 + (0.0 - z)**2)))**6) 

def evaluate_energies(full_array):
    """Evaluate energies based on a simple analytical potential."""
    pts = full_array[:,0:3]
    energies = [LJCluster(p) for p in pts]
    return energies

def run_sqra(energies, volumes, distances, borders):
    """Build the rate matrix using the SQRA method."""
    T = float(config["params_sqra"]["temperature_K"])
    m_h2o = float(config["params_sqra"]["mass_kg"])
    tau = float(config["params_sqra"].get("tau_t", 0.1))  # Default tau if not provided
    k_B = Boltzmann
    D = (k_B * T * tau) / m_h2o 
    D = D / ( 5.29e-11/ (2.42e-17)**2) # convert to atomic units


    sqra = SQRA(energies, volumes, distances, borders)
    rate_matrix = sqra.get_rate_matrix(D, T)
    lower_limit = float(config["params_sqra"]["lower_lim_rate_matrix"]) if config["params_sqra"]["lower_lim_rate_matrix"] != "None" else None
    upper_limit = float(config["params_sqra"]["upper_lim_rate_matrix"]) if config["params_sqra"]["upper_lim_rate_matrix"] != "None" else None
    rate_matrix, index_list = sqra.cut_and_merge(rate_matrix, T=T, lower_limit=lower_limit, upper_limit=upper_limit)
    return rate_matrix, index_list

def run_decomposition(rate_matrix):
    """Perform eigendecomposition on the rate matrix."""
    dt = DecompositionTool(rate_matrix)
    all_eigenval, all_eigenvec = dt.get_decomposition(
        tol=config["params_sqra"]["tol"],
        maxiter=config["params_sqra"]["maxiter"],
        which=config["params_sqra"]["which"],
        sigma=float(config["params_sqra"]["sigma"]) if config["params_sqra"]["sigma"] != "None" else None
    )
    return all_eigenval, all_eigenvec

def print_its(eigenvalues):
    """Calculate and print inverse timescales (ITS)."""
    all_its = []
    eigenvals = eigenvalues[1:]  # Dropping the first eigenvalue
    all_its.append([-1 / eigenval for eigenval in eigenvals if eigenval != 0])  # Avoid division by zero
    return all_its


# Step 1: Create the grid
prefix = "data/noRotGridSparse2/"
full_array = np.load(prefix + "_fullgrid.npy")
borders = sparse.load_npz(prefix + "borders_array.npz")
distances = sparse.load_npz(prefix + "distances_array.npz")
volumes = np.load(prefix + "volumes_array.npy")
    
# Step 2: Evaluate energies
energies = evaluate_energies(full_array)
energies = np.nan_to_num(energies, nan=np.infty)
print("Energies evaluated.")

# Step 3: Run SQRA to get rate matrix
rate_matrix, index_list = run_sqra(np.array(energies), np.array(volumes), distances, borders)
print("Rate matrix calculated.")
print(np.shape(rate_matrix))
print(type(np.array(rate_matrix)))
rate_matrix = rate_matrix.toarray()
np.save(prefix + "rate_matrix",rate_matrix)

#rate_matrix[rate_matrix == np.infty] = 1e100
#rate_matrix[rate_matrix == -1 * np.infty] = -1e100
#
#
#
## Step 4: Run eigendecomposition
#eigenvalues, eigenvectors = run_decomposition(rate_matrix)
#print("Eigendecomposition completed.")
##
#np.save(prefix + "eigenvalues.npy", eigenvalues)
#np.save(prefix + "eigenvectors.npy", eigenvectors)
#
#
#from molgri.plotting.transition_plots import PlotlyTransitions
#
#pathEigenvalues = prefix + "eigenvalues.npy"
#pathEigenvectors = prefix + "eigenvectors.npy"
#
#
#pt = PlotlyTransitions(is_msm=True, path_eigenvalues=pathEigenvalues, path_eigenvectors=pathEigenvectors,
#    tau_array=None)
## eigenvectors
#pt.plot_eigenvectors_flat(index_tau=0.1)
#pt.save_to("tmp/eigenvectors.png", height=1200)
## eigenvalues
#pt.plot_eigenvalues(index_tau=0.1)
#pt.save_to("tmp/eigenvalues.png")

## Step 5: Print inverse timescales (ITS)
#all_its = print_its(eigenvalues)
#print("Inverse timescales (ITS) calculated.")
#
## Optionally, save the results to files or print them
#np.savetxt(os.path.join(PATH_THIS_GRID, "eigenvalues.npy"), eigenvalues)
#np.savetxt(os.path.join(PATH_THIS_GRID, "eigenvectors.npy"), eigenvectors)
#np.savetxt(os.path.join(PATH_THIS_GRID, "its.csv"), all_its, delimiter=",", header="ITS [ps]")
#print("Results saved successfully.")