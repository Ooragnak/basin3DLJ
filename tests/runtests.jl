#PATH OF PROJECT DIR; contains folder like src and data
rootPath = pwd()

# This is required to get documentation tooltips to work in VSCode
macro ignore(args...) end

# SYNTAX EXAMPLE:
#@ignore include("../src/EXAMPLE_LIBRARY.jl")
#
#include(joinpath(rootPath, "src/EXAMPLE_LIBRARY.jl"))

include(joinpath(rootPath, "src/theme.jl"))
include(joinpath(rootPath, "src/grid.jl"))

################################################################
using Test

