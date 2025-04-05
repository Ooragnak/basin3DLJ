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


################
# 2D Cartesian Grids 
################
rotatedMullerBrown(x,y) = rotated2DPot(MullerBrown,x,y,-3π/16)

cartesian500_mbpotpot_grid = makeCartesianGrid(range(-2.0,1.25,500),range(-0.5,2.5,500),MullerBrown,"Müller-Brown-Potential",diagonal=true)
rotated_cartesian500_mbpotpot_grid = makeCartesianGrid(range(-2.0,1.25,500),range(-0.5,2.5,500),rotatedMullerBrown,"Müller-Brown-Potential",diagonal=true)

cartesian500_mbpotpot_basin = gradDescent(cartesian500_mbpotpot_grid)
rotated_cartesian500_mbpotpot_basin = gradDescent(rotated_cartesian500_mbpotpot_grid)

cartesian100_mbpotpot_grid = makeCartesianGrid(range(-2.0,1.25,100),range(-0.5,2.5,100),MullerBrown,"Müller-Brown-Potential",diagonal=true)
rotated_cartesian100_mbpotpot_grid = makeCartesianGrid(range(-2.0,1.25,100),range(-0.5,2.5,100),rotatedMullerBrown,"Müller-Brown-Potential",diagonal=true)

cartesian100_mbpotpot_basin = gradDescent(cartesian100_mbpotpot_grid)
rotated_cartesian100_mbpotpot_basin = gradDescent(rotated_cartesian100_mbpotpot_grid)

cartesian1000_mbpotpot_grid = makeCartesianGrid(range(-2.0,1.25,1000),range(-0.5,2.5,1000),MullerBrown,"Müller-Brown-Potential",diagonal=true)
cartesian1000_mbpotpot_basin = gradDescent(cartesian1000_mbpotpot_grid)



plot2DCartesianWatersheds(cartesian500_mbpotpot_basin,"Basins of attraction", "Müller-Brown potential", "cartesian500.pdf",L"V(x,y)",lvl=range(-150,75,50))

plot2DCartesianWatersheds(cartesian100_mbpotpot_basin,"Basins of attraction", "Müller-Brown potential", "cartesianMB100.pdf",L"V(x,y)",lvl=range(-150,75,50), msize = 8)

compare2DCartesianWatersheds([cartesian100_mbpotpot_basin,rotated_cartesian100_mbpotpot_basin,cartesian500_mbpotpot_basin,rotated_cartesian500_mbpotpot_basin],["100x100", "100x100, rotated by 33.75°","500x500", "500x500, rotated by 33.75°"],"compareWatershedsMB.png",lvl=range(-150,75,50),msizes=[8,8,4,4])

compare2DCartesianCoreSets(fill(cartesian100_mbpotpot_basin,4),[L"\epsilon = 2",L"\epsilon = 4",L"\epsilon = 8",L"\epsilon = 16"],"compareCoreSetsMB.pdf",lvl=range(-150,75,50),msize=10,epsilons = [2,4,8,16])

plotMEPs2D(cartesian1000_mbpotpot_basin,"MEPs on Müller-Brown Potential", "Energy along the MEPs","mbMEPS.pdf",lvl=range(-150,75,50))

################
# 2D Polar Grids 
################

polar300_ringpot_grid = makePolarGrid(range(0.1,5,300),300,ringpot,"Ring Potential",nudge = true)

polar300_ringpot_basin = gradDescent(polar300_ringpot_grid)

plot2DPolarwatersheds(polar300_ringpot_basin,0.5,"Basins of attraction", "Polar model potential", "Core-sets(ϵ = 0.5)", "polarRingpot300.pdf",L"V(r,\theta)")

polar100_ringpot_grid = makePolarGrid(range(0.1,5,50),100,ringpot,"Ring Potential",nudge = true)

polar100_ringpot_basin = gradDescent(polar100_ringpot_grid)

plot2DPolarwatersheds(polar100_ringpot_basin,0.5,"Basins of attraction", "Polar model potential", "Core-sets (ϵ = 0.5)", "polarRingpot100.pdf",L"V(r,\theta)")

################
# 3D Modified Ring Potential
################

plot3DPotSlice(ringpot3D,"ringpotProjection.png",(-4,4),3.0,detailed=((-0.2,0.2),(2.9,3.1)))
plot3DPotSlice(ringpot3DAlt,"ringpotAltProjection.png",(-4,4),3.0)

molgri150x50_ringpot3D_grid = parseMolgriGrid("data/noRotGrid/",ringpot3D,"Molgri-imported grid")
molgri150x50_ringpot3D_basin = gradDescent(molgri150x50_ringpot3D_grid)
molgri150x50_ringpot3D_packed = basinPack(molgri150x50_ringpot3D_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

packs = [molgri150x50_ringpot3D_packed,molgri150x50_ringpot3D_packed,molgri150x50_ringpot3D_packed,molgri150x50_ringpot3D_packed]
titles = ["Molgri 7500", "Molgri 7500", "Molgri 7500", "Molgri 7500"]

compareBasins(packs, titles, -2, "compareRingpot3D.png")
