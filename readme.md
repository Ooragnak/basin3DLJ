# Basins of attraction and core-sets for a 3D-Lennard-Jones-Cluster
All code related to the bachelor thesis "Basins of attraction and core-sets for a 3D-Lennard-Jones-Cluster" to be completed in the AG Keller.

## Steps to reproduce

To reproduce the results presented in the thesis the following steps should be sufficient:

1. Open a Julia (>v1.11) REPL in the project root directory
2. Install the dependencies:
<pre>
using Pkg
Pkg.instantiate()
</pre>
3. Execute the script to generate all relevant plots
<pre>
include("../bin/run.jl")
</pre>

### Generating required data from molgri

While the molgri-generated grid data required for the reproduction of all results presented in the thesis is included in the repository it may be desirable to experiment with alternative grids which will have to be generated. 
The pthon environment in which `grid.py` was run can be set-up in the following way:
1. Navigate to the root folder of the project directory and install the dependencies using `pip`:
<pre>
pip install -r requirements.txt
</pre>

2. Execute `grid.py` located in the bin directory.

## Information for developers
Example folder structure of the project directory:
<pre>
└───basin3DLJ
    │   .gitignore
    │   Manifest.toml
    │   Project.toml
    │   readme.md
    |   requirements.txt
    ├───.git
    ├───bin
    ├───data
    ├───plots
    ├───src
    ├───tests
    └───tmp
</pre>