include("../src/grid.jl")
include("../src/potential.jl")
include("../src/plots.jl")
include("../src/theme.jl")

rootPath = pwd()

################################################################
try 
    ROCArray{Float32}(rand(10))
    global ARRAYTYPE = ROCArray{Float32}
    @info "Using ROCm backend for GPU computation."
catch
    try 
        CuArray{Float32}(rand(10))
        global ARRAYTYPE = CuArray{Float32}
        @info "Using CUDA backend for GPU computation."
    catch
        global ARRAYTYPE = Array{Float32}
        @warn "Neither CUDA nor ROCm driver found, falling back to CPU arrays. Provide ARRAYTYPE manually to use a different default backend."
    end
end

polar300_ringpot_grid = makePolarGrid(range(0.1,5,300),300,ringpot,"Ring Potential",nudge = true)

polar300_ringpot_basin = gradDescent(polar300_ringpot_grid)

plot2DPolarwatersheds(polar300_ringpot_basin,"Polar Model Potential", "Basins of Attraction", "polarRingpot300.pdf",L"V(r,\theta)")





