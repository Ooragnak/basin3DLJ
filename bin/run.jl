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
# Neighborhood effects
################
xsN = -2:1:2
ysN = -1:0.5:1
normalN = [simplepot(x,y) for x in xsN, y in ysN]
rotN = [rotated2DPot(simplepot,x,y,π/4) for x in xsN, y in ysN]

fNeighbor = Figure(size=(2560,1440), fontsize=40)
axN1 = Axis(fNeighbor[1,2], title = L"f(x,y) = x^2 + 10 \cdot y^2,\text{ rotated by }\frac{π}{4}\text{ around the origin}", yautolimitmargin = (0, 0),xlabel="x",ylabel="y")
axN2 = Axis(fNeighbor[1,1], title = L"f(x,y) = x^2 + 10 \cdot y^2 ", yautolimitmargin = (0, 0),xlabel="x",ylabel="y")


heatmap!(axN1,xs,ys,rotN,colormap=:lipari)
for x in xs, y in ys
    data = rotated2DPot(simplepot,x,y,π/4)
    if x == -1.0 && y == 1.0
        txtcolor =:red
        text!(axN1, "$(round(data, sigdigits = 3))", position = (x, y),
            color = txtcolor, align = (:center, :center),space=:data,fontsize = 60)
    else
        txtcolor = data < 11 ? :white : :black
        text!(axN1, "$(round(data, sigdigits = 3))", position = (x, y),
            color = txtcolor, align = (:center, :center),space=:data)
    end
end

heatmap!(axN2,xs,ys,normalN,colormap=:lipari)
for x in xs, y in ys
    data = simplepot(x,y)
    txtcolor = data < 11 ? :white : :black
    text!(axN2, "$(round(data, sigdigits = 3))", position = (x, y),
        color = txtcolor, align = (:center, :center),space=:data)
end

save(string("plots/","compareNeighborhood.pdf"),fNeighbor,backend=CairoMakie)


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



plot2DCartesianWatersheds(cartesian500_mbpotpot_basin,"Basins of attraction", "Müller-Brown potential", "cartesian500.png",L"V(x,y)",lvl=range(-150,75,50))

plot2DCartesianWatersheds(cartesian100_mbpotpot_basin,"Basins of attraction", "Müller-Brown potential", "cartesianMB100.pdf",L"V(x,y)",lvl=range(-150,75,50), msize = 8)

compare2DCartesianWatersheds([cartesian100_mbpotpot_basin,rotated_cartesian100_mbpotpot_basin,cartesian500_mbpotpot_basin,rotated_cartesian500_mbpotpot_basin],["100x100", "100x100, rotated by 33.75°","500x500", "500x500, rotated by 33.75°"],"compareWatershedsMB.png",lvl=range(-150,75,50),msizes=[8,8,4,4])

compare2DCartesianCoreSets(fill(cartesian100_mbpotpot_basin,4),[L"\epsilon = 2",L"\epsilon = 4",L"\epsilon = 8",L"\epsilon = 16"],"compareCoreSetsMB.pdf",lvl=range(-150,75,50),msize=10,epsilons = [2,4,8,16])

plotMEPs2D(cartesian1000_mbpotpot_basin,"MEPs on Müller-Brown Potential", "Energy along the MEPs","mbMEPS.pdf",lvl=range(-150,75,50))

################
# 2D Polar Grids 
################

polar300_ringpot_grid = makePolarGrid(range(0.1,5,300),300,ringpot,"Ring Potential",nudge = true)

polar300_ringpot_basin = gradDescent(polar300_ringpot_grid)

plot2DPolarwatersheds(polar300_ringpot_basin,0.5,"Basins of attraction", "Polar model potential", "Core-sets(ϵ = 0.5)", "polarRingpot300.png",L"V(r,\theta)")

polar100_ringpot_grid = makePolarGrid(range(0.1,5,50),100,ringpot,"Ring Potential",nudge = true)

polar100_ringpot_basin = gradDescent(polar100_ringpot_grid)

plot2DPolarwatersheds(polar100_ringpot_basin,0.5,"Basins of attraction", "Polar model potential", "Core-sets (ϵ = 0.5)", "polarRingpot100.pdf",L"V(r,\theta)")

################
# 3D Modified Ring Potential
################

plot3DPotSlice(ringpot3D,"ringpotProjection.png",(-4,4),3.0,colorrange=[nothing,(-15,60),(-15,60),(-15,60)])
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

molgri80x20_ringpot3D_grid = parseMolgriGrid("data/noRotGridSparse/",ringpot3D,"Ringpot3D on Molgri-imported grid 80x20")
molgri80x20_ringpot3D_basin = gradDescent(molgri80x20_ringpot3D_grid)
molgri80x20_ringpot3D_packed = basinPack(molgri80x20_ringpot3D_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

molgri80x20_ringpot3D_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGridSparse/",ringpot3D,"Ringpot3D on Molgri-imported grid 80x20"),true)
molgri80x20_ringpot3D_diagonal_basin = gradDescent(molgri80x20_ringpot3D_diagonal_grid)
molgri80x20_ringpot3D_diagonal_packed = basinPack(molgri80x20_ringpot3D_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

molgri150x50_ringpot3D_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGrid/",ringpot3D,"Ringpot3D on Molgri-imported grid 150x50"))
molgri150x50_ringpot3D_diagonal_basin = gradDescent(molgri150x50_ringpot3D_diagonal_grid)
molgri150x50_ringpot3D_diagonal_packed = basinPack(molgri150x50_ringpot3D_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

molgri1000x50_ringpot3D_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGridFine2/",ringpot3D,"Ringpot3D on Molgri-imported grid 1000x50"))
molgri1000x50_ringpot3D_diagonal_basin = gradDescent(molgri1000x50_ringpot3D_diagonal_grid)
molgri1000x50_ringpot3D_diagonal_packed = basinPack(molgri1000x50_ringpot3D_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)

packs = [molgri150x50_ringpot3D_packed,molgri1000x50_ringpot3D_packed,cartesian20x20x20_ringpot3D_packed,cartesian100x100x100_ringpot3D_packed]
titles = ["Spherical 7500", "Spherical 50000", "Cartesian 8000",  "Cartesian 1000000"]

packs2 = [molgri80x20_ringpot3D_packed,molgri80x20_ringpot3D_diagonal_packed,molgri150x50_ringpot3D_diagonal_packed,molgri1000x50_ringpot3D_diagonal_packed]
titles2 = ["Spherical 1600", "Spherical 1600 (diag)", "Spherical 7500 (diag)",  "Spherical 50000 (diag)"]

compareBasins(packs, titles, fill(-2.5,4), "compareRingpot3D.png")
compareCoreSets(packs, titles, fill(0.5,4), "compareCoreSetsRingpot3D.png")

compareCoreSets(packs2, titles2, fill(0.5,4), "compareCoreSetsRingpot3D_diag.png")

compareBasins(packs2, titles2, fill(-2.5,4), "compareRingpot3D_diagA.png")


# MODIFIED POTENTIAL - PROBABLY DEPRECATED
#
#molgri150x50_ringpot3DAlt_grid = parseMolgriGrid("data/noRotGrid/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 150x50")
#molgri150x50_ringpot3DAlt_basin = gradDescent(molgri150x50_ringpot3DAlt_grid)
#molgri150x50_ringpot3DAlt_packed = basinPack(molgri150x50_ringpot3DAlt_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)
#
#molgri150x50_ringpot3DAlt_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGrid/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 150x50"),true)
#molgri150x50_ringpot3DAlt_diagonal_basin = gradDescent(molgri150x50_ringpot3DAlt_diagonal_grid)
#molgri150x50_ringpot3DAlt_diagonal_packed = basinPack(molgri150x50_ringpot3DAlt_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250)
#
#molgri1000x50_ringpot3DAlt_grid = parseMolgriGrid("data/noRotGridFine2/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 1000x50")
#molgri1000x50_ringpot3DAlt_basin = gradDescent(molgri1000x50_ringpot3DAlt_grid)
#molgri1000x50_ringpot3DAlt_packed = basinPack(molgri1000x50_ringpot3DAlt_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=100)
#
#molgri1000x50_ringpot3DAlt_diagonal_grid = parseMolgriGrid("data/noRotGridFine2/",ringpot3DAlt,"ringpot3DAlt on Molgri-imported grid 1000x50")
#molgri1000x50_ringpot3DAlt_diagonal_basin = gradDescent(molgri1000x50_ringpot3DAlt_diagonal_grid)
#molgri1000x50_ringpot3DAlt_diagonal_packed = basinPack(molgri1000x50_ringpot3DAlt_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=100)
#
#cartesian20x20x20_ringpot3DAlt_grid  = makeCartesianGrid(range(-5.01,5,20),range(-5.01,5,20),range(-5.01,5,20),ringpot3DAlt,"ringpot3DAlt on Cartesian grid 20x20x20",diagonal=true)
#cartesian20x20x20_ringpot3DAlt_basin = gradDescent(cartesian20x20x20_ringpot3DAlt_grid)
#cartesian20x20x20_ringpot3DAlt_packed = basinPack(cartesian20x20x20_ringpot3DAlt_basin)
#
#cartesian100x100x100_ringpot3DAlt_grid  = makeCartesianGrid(range(-5.1,5,100),range(-5.1,5,100),range(-5.1,5,100),ringpot3DAlt,"ringpot3DAlt on Cartesian grid 100x100x100",diagonal=true)
#cartesian100x100x100_ringpot3DAlt_basin = gradDescent(cartesian100x100x100_ringpot3DAlt_grid)
#cartesian100x100x100_ringpot3DAlt_packed = basinPack(cartesian100x100x100_ringpot3DAlt_basin)
#
#packsAlt = [molgri150x50_ringpot3DAlt_packed,molgri1000x50_ringpot3DAlt_packed,cartesian20x20x20_ringpot3DAlt_packed,cartesian100x100x100_ringpot3DAlt_packed]
#titlesAlt = ["Molgri 7500", "Molgri 50000", "Cartesian 8000",  "Cartesian 1000000"]
#
#packsAlt2 = [molgri150x50_ringpot3DAlt_diagonal_packed,molgri1000x50_ringpot3DAlt_diagonal_packed,cartesian20x20x20_ringpot3DAlt_packed,cartesian100x100x100_ringpot3DAlt_packed]
#titlesAlt2 = ["Molgri (diagonal) 7500", "Molgri (diagonal) 50000", "Cartesian 8000",  "Cartesian 1000000"]
#
#compareBasins(packsAlt, titlesAlt, fill(-3,4), "compareringpot3DAlt.png")
#compareCoreSets(packsAlt, titlesAlt, fill(1.0,4), "compareCoreSetsRingpot3DAlt.png")
#compareBasins(packsAlt2, titlesAlt2, fill(-3,4), "compareringpot3DAlt_diag.png", voxels=false)
#compareCoreSets(packsAlt2, titlesAlt2, fill(1.0,4), "compareCoreSetsRingpot3DAlt_diag.png", voxels=true)

################
# 3D Lennard-Jones Cluster
################

defaultCluster = vcat(fill((1,2*2^(-1/6)),12),nothing)
ABABcluster = generateCluster(1,defaultCluster)

plot3DPotSlice((x,y,z) -> potential(ABABcluster,x,y,z),"LJClusterProjection.png",(-4,4),3.0,colorrange=(-6,2),azimuth=0.1π,xreversed=true)

#Plot LJ particle cluster
f3d_LJ = Figure(size=(1280,1280), fontsize=40)
ax3d = Axis3(f3d_LJ[1,1], title = "Lennard-Jones particle cluster",xreversed=true,aspect=:equal)
c1 = Makie.wong_colors()[1]
c2 = Makie.wong_colors()[2]
meshscatter!(ax3d,[p.t for p in ABABcluster.particles],markersize=0.3333,color=[c1,c2,c1,c2,c1,c2,c1,c2,c2,c1,c2,c1])
xlims!(ax3d,(3,-3))
ylims!(ax3d,(-3,3))
zlims!(ax3d,(-3,3))
save(string("plots/","LJClusterParticles.png"),f3d_LJ,backend=GLMakie,px_per_unit=2)

molgri1000x50_LJCluster_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGridFine2/",(x,y,z) -> potential(ABABcluster,x,y,z),"Lennard-Jones Cluster on Molgri-imported grid"),true)
molgri1000x50_LJCluster_diagonal_basin = gradDescent(molgri1000x50_LJCluster_diagonal_grid)
molgri1000x50_LJCluster_diagonal_packed = basinPack(molgri1000x50_LJCluster_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=120)

molgri150x50_LJCluster_diagonal_grid = getDiagonalNeighbors(parseMolgriGrid("data/noRotGrid/",(x,y,z) -> potential(ABABcluster,x,y,z),"Lennard-Jones Cluster on Molgri-imported grid"),true)
molgri150x50_LJCluster_diagonal_basin = gradDescent(molgri150x50_LJCluster_diagonal_grid)
molgri150x50_LJCluster_diagonal_packed = basinPack(molgri150x50_LJCluster_diagonal_basin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=120)

cartesian20x20x20_LJCluster_grid  = makeCartesianGrid(range(-5.01,5,20),range(-5.01,5,20),range(-5.01,5,20),(x,y,z) -> potential(ABABcluster,x,y,z),"Ringpot3D on Cartesian grid 20x20x20",diagonal=true)
cartesian20x20x20_LJCluster_basin = gradDescent(cartesian20x20x20_LJCluster_grid)
cartesian20x20x20_LJCluster_packed = basinPack(cartesian20x20x20_LJCluster_basin)

cartesian100x100x100_LJCluster_grid  = makeCartesianGrid(range(-5.1,5,100),range(-5.1,5,100),range(-5.1,5,100),(x,y,z) -> potential(ABABcluster,x,y,z),"Ringpot3D on Cartesian grid 100x100x100",diagonal=true)
cartesian100x100x100_LJCluster_basin = gradDescent(cartesian100x100x100_LJCluster_grid)
cartesian100x100x100_LJCluster_packed = basinPack(cartesian100x100x100_LJCluster_basin)

packsLJ = [molgri150x50_LJCluster_diagonal_packed,molgri1000x50_LJCluster_diagonal_packed,cartesian20x20x20_LJCluster_packed,cartesian100x100x100_LJCluster_packed]
titlesLJ = ["Spherical 7500", "Spherical 50000", "Cartesian 8000",  "Cartesian 1000000"]

compareCoreSets(packsLJ, titlesLJ, fill(0.5,4), "compareCoreSetsLJ.png")

cLJ = vcat(lipari(20)[12],[RGBf(x, x, x) for x in rand(collect((24:200) ./ 255),40)])

packsLJ2 = [molgri1000x50_LJCluster_diagonal_packed,molgri1000x50_LJCluster_diagonal_packed,molgri1000x50_LJCluster_diagonal_packed,molgri1000x50_LJCluster_diagonal_packed]
titlesLJ2 = ["Isovalue = -3.2", "Isovalue = -2.4", "Isovalue = -1.6",  "Isovalue = -0.8"]

compareBasins(packsLJ2, titlesLJ2, [-3.2, -2.4, -1.6, -0.8], "compareBasinLJ_Spherical.png",colors=cLJ,azimuth=1.15π,reversed=true,voxels=true,isorange=0.01)

packsLJ3 = [cartesian100x100x100_LJCluster_packed,cartesian100x100x100_LJCluster_packed,cartesian100x100x100_LJCluster_packed,cartesian100x100x100_LJCluster_packed]
titlesLJ3 = ["Isovalue = -3.2", "Isovalue = -2.4", "Isovalue = -1.6",  "Isovalue = -0.8"]

compareBasins(packsLJ3, titlesLJ3, [-3.2, -2.4, -1.6, -0.8], "compareBasinLJ_Cartesian.png",colors=cLJ,azimuth=1.15π,reversed=true,voxels=true,isorange=0.01)

csSpherical = sort(molgri1000x50_LJCluster_diagonal_basin.minima,by=x->x.energy)
csCartesian = sort(cartesian100x100x100_LJCluster_basin.minima,by=x->x.energy)

occurs = []

csA = copy(csSpherical)
csB = copy(csCartesian)

while !isempty(csA) && !isempty(csB)
    dist = minimum(vec([distance(a,b) for a in csA, b in csB]))
    mindist = argmin(vec([distance(a,b) for a in csA, b in csB]))
    a_ind, b_ind = Tuple(CartesianIndices((length(csA),length(csB)))[mindist])
    a = findfirst(csSpherical .== Ref(csA[a_ind]))
    b = findfirst(csCartesian .== Ref(csB[b_ind]))
    if dist < 0.25
        push!(occurs,(a,b))
    else
        push!(occurs,(0,b))
        push!(occurs,(a,0))
    end
    deleteat!(csA,a_ind)
    deleteat!(csB,b_ind)
end

sx, sy, sz, se, sn= [], [], [], [], []
cx, cy, cz, ce, cn= [], [], [], [], []

for elem in occurs
    a, b = elem
    if b == 0
        push!(sx,csSpherical[a].translation.x)
        push!(sy,csSpherical[a].translation.y)
        push!(sz,csSpherical[a].translation.z)
        push!(se,csSpherical[a].energy)
        push!(sn,getBasinSize(molgri1000x50_LJCluster_diagonal_basin, csSpherical[a]))
        push!(cx,nothing)
        push!(cy,nothing)
        push!(cz,nothing)
        push!(ce,nothing)
        push!(cn,nothing)
    elseif a == 0
        push!(cx,csCartesian[b].translation.x)
        push!(cy,csCartesian[b].translation.y)
        push!(cz,csCartesian[b].translation.z)
        push!(ce,csCartesian[b].energy)
        push!(cn,getBasinSize(cartesian100x100x100_LJCluster_basin, csCartesian[b]))
        push!(sx,nothing)
        push!(sy,nothing)
        push!(sz,nothing)
        push!(se,nothing)
        push!(sn,nothing)
    else
        push!(sx,csSpherical[a].translation.x)
        push!(sy,csSpherical[a].translation.y)
        push!(sz,csSpherical[a].translation.z)
        push!(se,csSpherical[a].energy)
        push!(sn,getBasinSize(molgri1000x50_LJCluster_diagonal_basin, csSpherical[a]))
        push!(cx,csCartesian[b].translation.x)
        push!(cy,csCartesian[b].translation.y)
        push!(cz,csCartesian[b].translation.z)
        push!(ce,csCartesian[b].energy)
        push!(cn,getBasinSize(cartesian100x100x100_LJCluster_basin, csCartesian[b]))
    end
end


sn = replace(normalize(replace(sn,nothing=>0.0)) .* 100, 0.0=>nothing)
cn = replace(normalize(replace(cn,nothing=>0.0)) .* 100, 0.0=>nothing)

coreSets = [se, ce, sn, cn, sx, sy, sz, cx, cy, cz]
perm = sortperm(min.(replace(se,nothing=>Inf),replace(ce,nothing=>Inf)))
coreSetsSorted = [x[perm] for x in coreSets]
rs = toLaTeX(coreSetsSorted)
print(rs)