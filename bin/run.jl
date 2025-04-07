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

molgri150x50_ringpot3D_grid = parseMolgriGrid("data/noRotGrid/",ringpot3D,"Ringpot3D on Molgri-imported grid 150x50")
molgri150x50_ringpot3D_basin = gradDescent(molgri150x50_ringpot3D_grid)
molgri150x50_ringpot3D_packed = basinPack(molgri150x50_ringpot3D_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

molgri1000x50_ringpot3D_grid = parseMolgriGrid("data/noRotGridFine2/",ringpot3D,"Ringpot3D on Molgri-imported grid 1000x50")
molgri1000x50_ringpot3D_basin = gradDescent(molgri1000x50_ringpot3D_grid)
molgri1000x50_ringpot3D_packed = basinPack(molgri1000x50_ringpot3D_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

cartesian20x20x20_ringpot3D_grid  = makeCartesianGrid(range(-5.01,5,20),range(-5.01,5,20),range(-5.01,5,20),ringpot3D,"Ringpot3D on Cartesian grid 20x20x20",diagonal=true)
cartesian20x20x20_ringpot3D_basin = gradDescent(cartesian20x20x20_ringpot3D_grid)
cartesian20x20x20_ringpot3D_packed = basinPack(cartesian20x20x20_ringpot3D_basin)

cartesian100x100x100_ringpot3D_grid  = makeCartesianGrid(range(-5.1,5,100),range(-5.1,5,100),range(-5.1,5,100),ringpot3D,"Ringpot3D on Cartesian grid 100x100x100",diagonal=true)
cartesian100x100x100_ringpot3D_basin = gradDescent(cartesian100x100x100_ringpot3D_grid)
cartesian100x100x100_ringpot3D_packed = basinPack(cartesian100x100x100_ringpot3D_basin)

packs = [molgri150x50_ringpot3D_packed,molgri1000x50_ringpot3D_packed,cartesian20x20x20_ringpot3D_packed,cartesian100x100x100_ringpot3D_packed]
titles = ["Molgri 7500", "Molgri 50000", "Cartesian 7500",  "Cartesian 1000000"]

compareBasins(packs, titles, fill(-2.5,4), "compareRingpot3D.png")
compareCoreSets(packs, titles, fill(0.5,4), "compareCoreSetsRingpot3D.png")


molgri150x50_ringpot3DAlt_grid = parseMolgriGrid("data/noRotGrid/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 150x50")
molgri150x50_ringpot3DAlt_basin = gradDescent(molgri150x50_ringpot3DAlt_grid)
molgri150x50_ringpot3DAlt_packed = basinPack(molgri150x50_ringpot3DAlt_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

molgri150x50_ringpot3DAlt_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGrid/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 150x50"),true)
molgri150x50_ringpot3DAlt_diagonal_basin = gradDescent(molgri150x50_ringpot3DAlt_diagonal_grid)
molgri150x50_ringpot3DAlt_diagonal_packed = basinPack(molgri150x50_ringpot3DAlt_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

molgri1000x50_ringpot3DAlt_grid = parseMolgriGrid("data/noRotGridFine2/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 1000x50")
molgri1000x50_ringpot3DAlt_basin = gradDescent(molgri1000x50_ringpot3DAlt_grid)
molgri1000x50_ringpot3DAlt_packed = basinPack(molgri1000x50_ringpot3DAlt_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=100)

molgri1000x50_ringpot3DAlt_diagonal_grid = parseMolgriGrid("data/noRotGridFine2/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 1000x50")
molgri1000x50_ringpot3DAlt_diagonal_basin = gradDescent(molgri1000x50_ringpot3DAlt_diagonal_grid)
molgri1000x50_ringpot3DAlt_diagonal_packed = basinPack(molgri1000x50_ringpot3DAlt_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=100)

cartesian20x20x20_ringpot3DAlt_grid  = makeCartesianGrid(range(-5.01,5,20),range(-5.01,5,20),range(-5.01,5,20),ringpot3DAlt,"ringpot3DAlt on Cartesian grid 20x20x20",diagonal=true)
cartesian20x20x20_ringpot3DAlt_basin = gradDescent(cartesian20x20x20_ringpot3DAlt_grid)
cartesian20x20x20_ringpot3DAlt_packed = basinPack(cartesian20x20x20_ringpot3DAlt_basin)

cartesian100x100x100_ringpot3DAlt_grid  = makeCartesianGrid(range(-5.1,5,100),range(-5.1,5,100),range(-5.1,5,100),ringpot3DAlt,"ringpot3DAlt on Cartesian grid 100x100x100",diagonal=true)
cartesian100x100x100_ringpot3DAlt_basin = gradDescent(cartesian100x100x100_ringpot3DAlt_grid)
cartesian100x100x100_ringpot3DAlt_packed = basinPack(cartesian100x100x100_ringpot3DAlt_basin)

packsAlt = [molgri150x50_ringpot3DAlt_packed,molgri1000x50_ringpot3DAlt_packed,cartesian20x20x20_ringpot3DAlt_packed,cartesian100x100x100_ringpot3DAlt_packed]
titlesAlt = ["Molgri 7500", "Molgri 50000", "Cartesian 7500",  "Cartesian 1000000"]

packsAlt2 = [molgri150x50_ringpot3DAlt_diagonal_packed,molgri1000x50_ringpot3DAlt_diagonal_packed,cartesian20x20x20_ringpot3DAlt_packed,cartesian100x100x100_ringpot3DAlt_packed]
titlesAlt2 = ["Molgri (diagonal) 7500", "Molgri (diagonal) 50000", "Cartesian 7500",  "Cartesian 1000000"]

compareBasins(packsAlt, titlesAlt, fill(-3,4), "compareringpot3DAlt.png")
compareCoreSets(packsAlt, titlesAlt, fill(1.0,4), "compareCoreSetsRingpot3DAlt.png")

compareBasins(packsAlt2, titlesAlt2, fill(-3,4), "compareringpot3DAlt_diag.png", voxels=false)
compareCoreSets(packsAlt2, titlesAlt2, fill(1.0,4), "compareCoreSetsRingpot3DAlt_diag.png", voxels=true)

