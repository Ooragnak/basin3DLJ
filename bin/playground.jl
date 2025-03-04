include("../src/grid.jl")
include("../src/potential.jl")
include("../src/plots.jl")
include("../src/theme.jl")

rootPath = pwd()

using BenchmarkTools
using Base.Threads

################################################################
ARRAYTYPE = ROCArray{Float32}

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


display(f2)

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



f3 = Figure(size=(2560,1440), fontsize=40)
ax3 = Axis(f3[1,1], title = "Minimum energy paths of grid transition", yautolimitmargin = (0, 0),)

for path in paths
    plot!(ax3,[p.energy for p in path])
end


newpot  = makeCartesianGrid(range(-6.01,6,100),range(-6.01,6,100),range(-6.01,6,100),ringpot3D,"Test",diagonal=false)
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
#parsedGridFine =  parseMolgriGrid("tmp/norotgridfine/",ringpot3D,"Molgri-imported grid")
display(f3d)

f4 = Figure(size=(2560,1440), fontsize=40)
s4 = Slider(f4[1,2], range = -5:0.01:5, startvalue = 0.0,horizontal=false)

plotTitle = lift(s4.value) do z
    latexstring(L"View of interpolated potential at $z = %$(round(z,sigdigits=3)) $",)
end

xsvals = range(-4,4,400)
ysvals = range(-4,4,400)

ax4 = Axis(f4[1,1], title = plotTitle, yautolimitmargin = (0, 0),xlabel="x",ylabel="y")

#slice = lift(s4.value) do z
#    [ringpot3D(x,y,z) for x in xsvals, y in ysvals]
#end

#slice = lift(s4.value) do z
#    Array(interpolateSlice(parsedGrid,xsvals,ysvals,[z],power=10,ArrayType=ARRAYTYPE,closest=true))[:,:,1]
#end

sliceBasin = lift(s4.value) do z
    [findfirst(Ref(parsedBasin.gridpoints[p][2]) .== parsedBasin.minima) for p in interpolateSlice(parsedGrid,xsvals,ysvals,[z],power=10,ArrayType=ARRAYTYPE,closest=true,getPoints=true)[:,:,1]]
end


#slice = lift(s4.value) do z
#    [findfirst(Ref(newbasin.gridpoints[p][2]) .== newbasin.minima) for p in interpolateSlice(newpot,xsvals,ysvals,[z],power=10,ArrayType=ARRAYTYPE,closest=true,getPoints=true)[:,:,1]]
#end


d = heatmap!(ax4,xsvals,ysvals,sliceBasin,colorrange=(1,length(parsedBasin.minima)),colormap=Makie.wong_colors()[1:length(parsedBasin.minima)])


#c = heatmap!(ax4,xsvals,ysvals,slice,colormap=:lipari)
#c = heatmap!(ax4,xsvals,ysvals,slice,colormap=:lipari,colorrange=(-20,60))
#Colorbar(f4[1,0],c)

display(f4)
empty!(f4)

parsedVol = Array(interpolateSlice(parsedGrid,range(-5,5,500),range(-5,5,500),range(-5,5,500),power=12,ArrayType=ARRAYTYPE,closest=true))


volume(-1 .* parsedVol)


plotBasinsIsosurface(newbasin)
parsedBasin = gradDescent(parsedGrid)
plotBasinsIsosurface(parsedBasin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=400)
plotBasinsIsosurface(parsedBasin,interpolate=[(-5,5),(-5,5),(-5,5)],ArrayType=ARRAYTYPE,interpolationResolution=80,voxels=true)


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