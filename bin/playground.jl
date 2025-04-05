include("../src/grid.jl")
include("../src/potential.jl")
include("../src/plots.jl")
include("../src/theme.jl")

rootPath = pwd()

using BenchmarkTools
using Base.Threads

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
    

GLMakie.activate!()

#tstgrid = makePolarGrid(range(0.01,2,200),200,mbpolar,"Polar Muller Brown")
tstgrid = makePolarGrid(range(0.1,5,300),300,ringpot,"Ring Potential",nudge = true)

ps = getPoints(tstgrid)


ts = gradDescent(tstgrid)


f1 = Figure(size=(2560,1440), fontsize=40)
ax = PolarAxis(f1[1,1], title = "Polar coordinates")
ax2 = PolarAxis(f1[1,2], title = "Polar coordinates")

#p1 = voronoiplot!(ax,ts.grid.points,colorrange=(-150,75),markersize = 0,highclip = :transparent,colormap = :lipari, strokewidth = 0.01)
#p1 = contourf!(ax,ts.grid.points,colormap = :lipari,levels = range(-150,75,50))
p1 = contourf!(ax,ts.grid.points,colormap = :lipari,levels=75)

translate!(p1,0,0,-1000)

for minimum in ts.minima
    tmin = filter(x -> ts.gridpoints[x][2] == minimum, collect(keys(ts.gridpoints)))
    scatter!(ax2,[t.translation.θ for t in tmin], [t.translation.r for t in tmin],markersize = 8)
end

p2 = scatter!(ax2,[t.translation.θ for t in ts.minima], [t.translation.r for t in ts.minima], color = :red)


Colorbar(f1[1,0],p1)

display(f1)

tpot = makeCartesianGrid(range(-2.0,1.25,500),range(-0.5,2.5,500),MullerBrown,"Test",diagonal=true)

#tpot.distances
#dists = [distance(p1,p2) < 2.0 ? distance(p1,p2) : 0.0 for p1 in tpot.points, p2 in tpot.points]

#dists == Matrix(tpot.distances)

tp = gradDescent(tpot)

f2 = Figure(size=(2560,1440), fontsize=40)
ax = Axis(f2[1,1], title = "Müller-Brown-Potential", yautolimitmargin = (0, 0),)
ax2 = Axis(f2[1,2], title = "Basin of attraction", yautolimitmargin = (0, 0),)

vecs = collect(keys(tp.gridpoints))

#p1 = heatmap!(ax,vecs, colorrange=(-150,75),highclip = :transparent,colormap = :lipari)
p1 = contourf!(ax,vecs,colormap = :lipari,levels = range(-150,75,50), )


for (i,minimum) in enumerate(tp.minima)
    tmin = filter(x -> tp.gridpoints[x][2] == minimum, collect(keys(tp.gridpoints)))
    scatter!(ax2,[t.translation.x for t in tmin], [t.translation.y for t in tmin],markersize = 4)
    scatter!(ax,minimum, markersize = 15, marker=:xcross, label = @sprintf "(%.3f, %.3f)" minimum.translation.x minimum.translation.y)
end


p2 = contour!(ax2,vecs,colormap = :lipari,levels = range(-150,75,50), )

axislegend(ax,"Minima")

#p3 = scatter!(ax,[t.translation.x for t in tp.minima], [t.translation.y for t in tp.minima], color = :red)

#scatter!(ax,[t.translation.x for t in path], [t.translation.y for t in path],markersize = 8)

Colorbar(f2[1,0],p1)
paths = []
transitions = findMinimumEnergyPaths.(Ref(tp),tp.minima)
for t in transitions
    for p in t
        path = reverse(tracePath(tp,p[1]))
        append!(path, tracePath(tp,p[2]))
        scatter!(ax,path,markersize = 4,color=:grey)
        push!(paths,path)
    end
end

display(f2)

tpot2 = makeCartesianGrid(range(-2.0,1.25,500),range(-0.5,2.5,500),MullerBrown,"Test",diagonal=false)

tp2 = gradDescent(tpot2)

f2b = Figure(size=(2560,1440), fontsize=40)
ax = Axis(f2b[1,1], title = "Müller-Brown-Potential", yautolimitmargin = (0, 0),)
ax2 = Axis(f2b[1,2], title = "Basin of attraction", yautolimitmargin = (0, 0),)

vecs = collect(keys(tp2.gridpoints))

#p1 = heatmap!(ax,vecs, colorrange=(-150,75),highclip = :transparent,colormap = :lipari)
p1 = contourf!(ax,vecs,colormap = :lipari,levels = range(-150,75,50), )


for (i,minimum) in enumerate(tp2.minima)
    tmin = filter(x -> tp2.gridpoints[x][2] == minimum, collect(keys(tp2.gridpoints)))
    scatter!(ax2,[t.translation.x for t in tmin], [t.translation.y for t in tmin],markersize = 4)
    scatter!(ax,minimum, markersize = 15, marker=:xcross, label = @sprintf "(%.3f, %.3f)" minimum.translation.x minimum.translation.y)
end


p2 = contour!(ax2,vecs,colormap = :lipari,levels = range(-150,75,50), )

axislegend(ax,"Minima")


Colorbar(f2b[1,0],p1)


display(f2b)





f3 = Figure(size=(2560,1440), fontsize=40)
ax3 = Axis(f3[1,1], title = "Minimum energy paths of grid transition")

for path in paths[[2,3]]
    plot!(ax3,[p.energy for p in path])
end


newpot  = makeCartesianGrid(range(-6.01,6,100),range(-6.01,6,100),range(-6.01,6,100),ringpot3D,"Test",diagonal=true)
newbasin = gradDescent(newpot)

paths3d = []
transitions2 = findMinimumEnergyPaths.(Ref(newbasin),newbasin.minima)

f3d = Figure(size=(2560,1440), fontsize=40)
ax3d = Axis3(f3d[1,1], title = "Minimum energy paths of grid transition")

for m in newbasin.minima 
    scatter!(ax3d,m,markersize = 15, marker=:xcross, label = pretty(m,3))
end

for t in transitions2
    for p in t
        path = reverse(tracePath(newbasin,p[1]))
        append!(path, tracePath(newbasin,p[2]))
        scatter!(ax3d,path,markersize = 4,color=:grey)
        push!(paths3d,path)
    end
end


#parsedGridAlt =  parseMolgriGrid("tmp/noRotGridAlt/",ringpot3D,"Molgri-imported grid")
parsedGrid =  parseMolgriGrid("data/noRotGrid/",ringpot3D,"Molgri-imported grid")
parsedGrid_2 =  parseMolgriGrid("data/noRotGrid/",ringpot3DAlt,"Molgri-imported grid")

#parsedGridFine =  parseMolgriGrid("tmp/norotgridfine/",ringpot3D,"Molgri-imported grid")

parsedBasin = gradDescent(parsedGrid)
parsedBasin_2 = gradDescent(parsedGrid_2)


display(f3d)

f4 = Figure(size=(2560,1440), fontsize=40)
s4 = Slider(f4[1,3], range = -5:0.01:5, startvalue = 0.0,horizontal=false)

plotTitleA = lift(s4.value) do z
    latexstring(L"View of potential at $z = %$(round(z,sigdigits=3)) $",)
end

plotTitleB = lift(s4.value) do z
    latexstring(L"Watersheds of potential at $z = %$(round(z,sigdigits=3)) $",)
end

xsvals = range(-4,4,300)
ysvals = range(-4,4,300)

ax4 = Axis(f4[1,1], title = plotTitleA, yautolimitmargin = (0, 0),xlabel="x",ylabel="y")
ax4b = Axis(f4[1,2], title = plotTitleB, yautolimitmargin = (0, 0),xlabel="x",ylabel="y")

#slice = lift(s4.value) do z
#    [ringpot3D(x,y,z) for x in xsvals, y in ysvals]
#end

slice = lift(s4.value) do z
    [p.energy for p in sliceCartesian(newpot,'z',z)]
end
sliceBasin = lift(s4.value) do z
    [findfirst(Ref(newbasin.gridpoints[p][2]) .== newbasin.minima) for p in sliceCartesian(newpot,'z',z)]
end

c = heatmap!(ax4,xsvals,ysvals,slice,colormap=:lipari,colorrange=(-20,60))
d = heatmap!(ax4b,xsvals,ysvals,sliceBasin,colorrange=(1,length(newbasin.minima)),colormap=Makie.wong_colors()[1:length(newbasin.minima)])

Colorbar(f4[1,0],c)

display(f4)


f5 = Figure(size=(2560,1440), fontsize=40)
s5 = Slider(f5[1,3], range = -5:0.01:5, startvalue = 0.0,horizontal=false)

plotTitleA2 = lift(s5.value) do z
    latexstring(L"View of interpolated potential at $z = %$(round(z,sigdigits=3)) $",)
end


plotTitleB2 = lift(s5.value) do z
    latexstring(L"Basins of interpolated potential at $z = %$(round(z,sigdigits=3)) $",)
end

xsvals = range(-4,4,300)
ysvals = range(-4,4,300)

ax5 = Axis(f5[1,1], title = plotTitleA2, yautolimitmargin = (0, 0),xlabel="x",ylabel="y")
ax5b = Axis(f5[1,2], title = plotTitleB2, yautolimitmargin = (0, 0),xlabel="x",ylabel="y")


slice = lift(s5.value) do z
    Array(interpolateSlice(parsedGrid,xsvals,ysvals,[z],power=10,ArrayType=ARRAYTYPE,closest=true))[:,:,1]
end

sliceBasin = lift(s5.value) do z
    [findfirst(Ref(parsedBasin.gridpoints[p][2]) .== parsedBasin.minima) for p in interpolateSlice(parsedGrid,xsvals,ysvals,[z],power=10,ArrayType=ARRAYTYPE,closest=true,getPoints=true)[:,:,1]]
end


c = heatmap!(ax5,xsvals,ysvals,slice,colormap=:lipari,colorrange=(-20,60))
d = heatmap!(ax5b,xsvals,ysvals,sliceBasin,colorrange=(1,length(parsedBasin.minima)),colormap=Makie.wong_colors()[1:length(parsedBasin.minima)])

Colorbar(f5[1,0],c)

display(f5)

parsedVol = Array(interpolateSlice(parsedGrid,range(-5,5,200),range(-5,5,200),range(-5,5,200),power=12,ArrayType=ARRAYTYPE,closest=true))


volume(-1 .* parsedVol)


f_a = plotBasinsIsosurface(newbasin,energyrange=(-12,12))
f_b = plotBasinsIsosurface(parsedBasin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=250,energyrange=(-12,12))
f_c = plotBasinsIsosurface(parsedBasin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=80,voxels=true,energyrange=(-12,12))
f_d = plotBasinsIsosurface(newbasin,voxels=true,energyrange=(-12,12))

f_g = plotBasinsIsosurface(parsedBasin_2,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=80,energyrange=(-12,12))


# Example benchmarks showing the impact of GPU computing for single and double precision floating point
#julia> @time parsedVol = Array(interpolateSlice(parsedGridAlt,range(-5,5,200),range(-5,5,200),range(-5,5,200),power=12,ArrayType=Array{Float64},closest=false));
# 78.438337 seconds (59.70 k allocations: 552.794 MiB, 0.14% gc time, 1.65% compilation time)
#
#julia> @time parsedVol = Array(interpolateSlice(parsedGridAlt,range(-5,5,200),range(-5,5,200),range(-5,5,200),power=12,ArrayType=ROCArray{Float64},closest=false));
# 27.153468 seconds (449.98 k allocations: 319.309 MiB, 0.65% gc time, 0.03% compilation time)
#
#julia> @time parsedVol = Array(interpolateSlice(parsedGridAlt,range(-5,5,200),range(-5,5,200),range(-5,5,200),power=12,ArrayType=Array{Float32},closest=false));
# 68.281488 seconds (59.19 k allocations: 369.548 MiB, 0.02% gc time, 2.15% compilation time)
#
#julia> @time parsedVol = Array(interpolateSlice(parsedGridAlt,range(-5,5,200),range(-5,5,200),range(-5,5,200),power=12,ArrayType=ROCArray{Float32},closest=false));
#  2.174526 seconds (280 allocations: 336.068 MiB, 5.69% gc time)

display(f1)

display(f2)
display(f3)
display(f2b)

display(f4)
display(f5)

display(f3d)

display(f_a)
display(f_b)
display(f_c)
display(f_d)


defaultCluster = vcat(fill((1,2*2^(-1/6)),12),nothing)
ABABcluster1 = generateCluster(1,defaultCluster)

#LJpot =  parseMolgriGrid("tmp/norotgridfine/",(x,y,z) -> potential(ABABcluster1,x,y,z),"Lennard-Jones Cluster on Molgri-imported grid")
LJpot  = makeCartesianGrid(range(-4.1,4,80),range(-4.1,4,80),range(-4.1,4,80),(x,y,z) -> potential(ABABcluster1,x,y,z),"Test",diagonal=true)
LJbasin = gradDescent(LJpot)

fLJ = plotBasinsIsosurface(LJbasin,energyrange=(-6,-1))
#fLJ = plotBasinsIsosurface(LJbasin,energyrange=(-6,-1),interpolate=[(-4,4),(-4,4),(-4,4)],ArrayType=ARRAYTYPE,interpolationResolution=150)

LJpotB = rotate(LJpot,π/16,π/8)
LJbasinB = gradDescent(LJpotB)

fLJB = plotBasinsIsosurface(LJbasinB,energyrange=(-6,-1),interpolate=[(-4,4),(-4,4),(-4,4)],ArrayType=ARRAYTYPE,interpolationResolution=100)

sortedLJ = sort(LJbasin.minima,by=x -> x.energy)
sortedLJ = sort(LJbasin.grid.points,by=x -> x.energy)


LJtransitions = findMinimumEnergyPaths(LJbasin,sortedLJ[1])

f3d = Figure(size=(2560,1440), fontsize=40)
ax3d = Axis3(f3d[1,1], title = "Minimum energy paths of grid transition")

paths3d = []

for m in LJbasin.minima 
    scatter!(ax3d,m,markersize = 15, marker=:xcross, label = pretty(m,3))
end

meshscatter!(ax3d,[p.t for p in ABABcluster1.particles],markersize=0.05)

for p in LJtransitions
    path = reverse(tracePath(LJbasin,p[1]))
    append!(path, tracePath(LJbasin,p[2]))
    scatter!(ax3d,path,markersize = 4,color=:grey)
    push!(paths3d,path)
end

#ImportedPot =  parseMolgriGrid("data/noRotGridFine/",(x,y,z) -> potential(ABABcluster1,x,y,z),"Lennard-Jones Cluster on Molgri-imported grid")
ImportedPot =  parseMolgriGrid("data/noRotGridFine2/",(x,y,z) -> potential(ABABcluster1,x,y,z),"Lennard-Jones Cluster on Molgri-imported grid")

diagonalSphericalPot = getDiagonalNeighbors(ImportedPot,true)

diagSphericalBasin = gradDescent(diagonalSphericalPot)
importedBasin = gradDescent(ImportedPot)

diagonalSphericalPotB = getDiagonalNeighbors(rotate(ImportedPot,π/4,0),true)
diagSphericalBasinB = gradDescent(diagonalSphericalPotB)

fLJ2 = plotBasinsIsosurface(diagSphericalBasin,energyrange=(-6,-1),interpolate=[(-4,4),(-4,4),(-4,4)],ArrayType=ARRAYTYPE,interpolationResolution=120)
fLJ3 = plotBasinsIsosurface(importedBasin,energyrange=(-6,-1),interpolate=[(-4,4),(-4,4),(-4,4)],ArrayType=ARRAYTYPE,interpolationResolution=120)
fLJ4 = plotBasinsIsosurface(diagSphericalBasinB,energyrange=(-6,-1),interpolate=[(-4,4),(-4,4),(-4,4)],ArrayType=ARRAYTYPE,interpolationResolution=120)

#using IntervalRootFinding, IntervalArithmetic, ForwardDiff
#LJpotential(x) = potential(ABABcluster1,x[1],x[2],x[3],replaceNaN=false)
#rts = roots(x -> ForwardDiff.gradient(LJpotential,x),[interval(-4,4),interval(-4,4),interval(-4,4)],abstol=1e-5)
#tstpot(x) = 4 * 1 * ((2/(x[1]^2+x[2]^2+x[3]^2))^12 - (2/(x[1]^2+x[2]^2+x[3]^2))^6)

xs = -2:1:2
ys = -1:0.5:1
normal = [simplepot(x,y) for x in xs, y in ys]
rot = [rotated2DPot(simplepot,x,y,π/4) for x in xs, y in ys]
 

f7 = Figure(size=(2560,1440), fontsize=40)
ax8 = Axis(f7[1,1], title = L"f(x,y) = x^2 + 10 \cdot y^2 ", yautolimitmargin = (0, 0),xlabel="x",ylabel="y")
ax7 = Axis(f7[1,2], title = L"f(x,y) = x^2 + 10 \cdot y^2,\text{ rotated by }\frac{π}{4} ", yautolimitmargin = (0, 0),xlabel="x",ylabel="y")


heatmap!(ax7,xs,ys,rot,colormap=:lipari)
for x in xs, y in ys
    data = rotated2DPot(simplepot,x,y,π/4)
    txtcolor = data < 11 ? :white : :black
    text!(ax7, "$(round(data, sigdigits = 3))", position = (x, y),
        color = txtcolor, align = (:center, :center),space=:data)
end

heatmap!(ax8,xs,ys,normal,colormap=:lipari)
for x in xs, y in ys
    data = simplepot(x,y)
    txtcolor = data < 11 ? :white : :black
    text!(ax8, "$(round(data, sigdigits = 3))", position = (x, y),
        color = txtcolor, align = (:center, :center),space=:data)
end

f8 = Figure(size=(2560,2560), fontsize=40)
ax2 = Axis3(f8[1,2], title = string("Projection on sphere (r = ",round(3.0,sigdigits=3),")"), yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
ps = [normalize(rand(3) .- 0.5) .* 6 for i in 1:100000]
cs = [ringpot3D(p...) for p in ps]
meshscatter!(ax2,[p[1] for p in ps], [p[2] for p in ps], [p[3] for p in ps],color = cs,markersize=0.01)

n = 100
θ = [0;(0.5:n-0.5)/n;1]
φ = [(0:2n-2)*2/(2n-1);2]
x = [cospi(φ)*sinpi(θ) for θ in θ, φ in φ] .* 3
y = [sinpi(φ)*sinpi(θ) for θ in θ, φ in φ] .* 3
z = [cospi(θ) for θ in θ, φ in φ] .* 3
colors = ringpot3D.(x,y,z)
surface(x,y,z,color = colors)