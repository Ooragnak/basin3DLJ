include("../src/grid.jl")
include("../src/potential.jl")
include("../src/plots.jl")
include("../src/theme.jl")

rootPath = pwd()

using BenchmarkTools
using Base.Threads
using StatsBase
using KernelAbstractions
using AMDGPU
using CUDA

################################################################

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

tpot = makeCartesianGrid(range(-2.0,1.25,60),range(-0.5,2.5,60),MullerBrown,"Test",diagonal=true)

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
#ax3d2 = Axis3(f3d[1,2], title = "Isosurface of Potential")

s1 = Slider(f3d[1,2], range = -7:0.01:10, startvalue = 0.0,horizontal=false)

function toArray(ps::PointGrid{Point{Cartesian3D}})
    xlen = length(unique(p.translation.x for p in ps.points))
    ylen = length(unique(p.translation.y for p in ps.points))
    zlen = length(unique(p.translation.z for p in ps.points))
    A = reshape(ps.points,(xlen,ylen,zlen))
    return A
end


isoval = lift(s1.value) do x 
    x
end

for (i,m) in enumerate(newbasin.minima)
    A = toArray(newpot)
    a = [newbasin.gridpoints[p][2] == m ? p.energy : NaN for p in A]
    volume!(ax3d,(-4,4),(-4,4),(-4,4),a, algorithm = :iso, isovalue = isoval, isorange = 1 ,colormap = fill(Makie.wong_colors()[i],100) , interpolate = true)
end

volume!(ax3d,(-4,4),(-4,4),(-4,4),[p.energy for p in toArray(newpot)], algorithm = :iso, isovalue = isoval, isorange = 0.1 ,colormap = :lipari, interpolate = true)

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

display(f3d)


function interpolateSlice(grid::PointGrid{<: Point{T}},xs,ys,z) where {T <: Position}
    slice = [Cartesian3D(x,y,z) for x in xs, y in ys]
    interpolated = zeros(size(slice))
    for (i,col) in enumerate(eachcol(slice))
        currentDist, currentClosest = findmin(distance.(grid.points,Ref(slice[1,i])))
        for (j,p) in enumerate(col)
            closestNeighborDist, closestNeighbor_arg = findmin(distance.(grid.points[first.(grid.distances[grid.points[currentClosest]])],Ref(p)))
            closestNeighbor = first.(grid.distances[grid.points[currentClosest]])[closestNeighbor_arg]
            if distance(grid.points[currentClosest],p) > closestNeighborDist
                currentDist = closestNeighborDist
                currentClosest = closestNeighbor
            end
            interpolated[j,i] = grid.points[currentClosest].energy
        end
        #@assert currentDist == minimum(distance.(grid.points,Ref(slice[end,i]))) string(currentDist,"-",minimum(distance.(grid.points,Ref(slice[end,i]))),"-",i,"-",slice[currentClosest])
    end
    return interpolated
end

function interpolateSliceAlt(grid::PointGrid{<: Point{T}},xs,ys,z) where {T <: Position}
    slice = [Cartesian3D(x,y,z) for x in xs, y in ys]
    interpolated = zeros(size(slice))

    @threads for i in eachindex(slice)
        closest = findClosestGridPoint(grid,Point(slice[i],1.0))
        all = Tuple{Int64, Float64}[]
        secondSphere = Tuple{Int64, Float64}[]
        thirdSphere = Tuple{Int64, Float64}[]
        fourthSphere = Tuple{Int64, Float64}[]

        Sphere = grid.distances[closest]
        for k in grid.points[first.(Sphere)]
            append!(secondSphere,grid.distances[k])
        end

        append!(all,secondSphere)
        for k in grid.points[first.(secondSphere)]
            append!(thirdSphere,grid.distances[k])
        end
        append!(all,thirdSphere)

        for k in grid.points[first.(thirdSphere)]
            append!(fourthSphere,grid.distances[k])
        end
        append!(all,thirdSphere)

        energy = StatsBase.mean([p.energy for p in grid.points[first.(unique(all))]],weights(1 ./ last.(unique(all))))

        interpolated[i] = energy
    end
    return interpolated
end

function interpolateSliceGPU(grid::PointGrid{<: Point{T}},xs,ys,z,power=8) where {T <: Position}
    xs2 = vec([x for x in xs, y in ys])
    ys2 = vec([y for x in xs, y in ys])
    interpolated = zeros(size(xs2))
    ps = grid.points
    xarr = ROCArray([p.translation.x for p in grid.points])
    yarr = ROCArray([p.translation.y for p in grid.points])
    zarr = ROCArray([p.translation.z for p in grid.points])
    distances = ROCArray(zeros(size(grid.points)))
    w = ROCArray(zeros(size(grid.points)))
    energies = ROCArray([p.energy for p in grid.points])

    for i in eachindex(xs2)

        distances .= sqrt.((xarr .- xs2[i]).^2 .+ (yarr .- ys2[i]).^2 .+ (zarr .- z).^2)
        w .= ((1.0 ./ distances) .^ power)
        energy = sum(energies .* w) / sum(w)
        #energy = minimum(distances)
        interpolated[i] = energy
    end

    return reshape(interpolated,(length(xs),length(ys)))
end

function interpolateSliceGPUAlt(grid::PointGrid{<: Point{T}},xs,ys,zs  ;power=8,ArrayType = Array,closest=false) where {T <: Position}
    N = length(xs)*length(ys)*length(zs)
    xs2 = ArrayType([x for x in xs, y in ys, z in zs])
    ys2 = ArrayType([y for x in xs, y in ys, z in zs])
    zs2 = ArrayType([z for x in xs, y in ys, z in zs])

    xarr = ArrayType([p.translation.x for p in grid.points])
    yarr = ArrayType([p.translation.y for p in grid.points])
    zarr = ArrayType([p.translation.z for p in grid.points])

    energyarr = ArrayType([p.energy for p in grid.points])


    energies = ArrayType(zeros(Float32,N))

    backend = get_backend(xarr)
    if !closest 
        kernel! = interpolatePoint!(backend,256)
        kernel!(energies,xs2,ys2,zs2,energyarr,xarr,yarr,zarr,length(xarr),power;ndrange = N)
    else
        kernel! = closestPoint!(backend,256)
        kernel!(energies,xs2,ys2,zs2,energyarr,xarr,yarr,zarr,length(xarr);ndrange = N)
    end

    return reshape(energies,(length(xs),length(ys),length(zs)))
end



@kernel function interpolatePoint!(out,xvals,yvals,zvals,energies,px,py,pz,n,power)
    j = @index(Global, Linear)
    weights = 0.0
    energy = 0.0
    weight = 0.0
    for i in 1:n
        weight = (1 / sqrt((xvals[j] - px[i])^2 + (yvals[j] - py[i])^2 + (zvals[j] - pz[i])^2))^power
        weights += weight
        energy += energies[i] * weight
    end
    out[j] = energy / weights
end

@kernel function closestPoint!(out,xvals,yvals,zvals,energies,px,py,pz,n)
    j = @index(Global, Linear)
    smallestDistance = Inf
    distance = 0.0
    energy = 0.0
    for i in 1:n
        distance = sqrt((xvals[j] - px[i])^2 + (yvals[j] - py[i])^2 + (zvals[j] - pz[i])^2) 
        if distance < smallestDistance
            smallestDistance = distance
            energy = energies[i]
        end
    end
    out[j] = energy
end


parsedGrid =  parseMolgriGrid("tmp/norotgrid/",ringpot3D,"Molgri-imported grid")
parsedGridFine =  parseMolgriGrid("tmp/norotgridfine/",ringpot3D,"Molgri-imported grid")


f4 = Figure(size=(2560,1440), fontsize=40)
s4 = Slider(f4[1,2], range = -5:0.01:5, startvalue = 0.0,horizontal=false)

plotTitle = lift(s4.value) do z
    latexstring(L"View of interpolated potential at $z = %$(round(z,sigdigits=3)) $",)
end

xsvals = range(-5,5,400)
ysvals = range(-5,5,400)


ax4 = Axis(f4[1,1], title = plotTitle, yautolimitmargin = (0, 0),xlabel="x",ylabel="y")

#slice = lift(s4.value) do z
#    [ringpot3D(x,y,z) for x in xsvals, y in ysvals]
#end

slice = lift(s4.value) do z
    Array(interpolateSliceGPUAlt(parsedGrid,xsvals,xsvals,[z],power=12,ArrayType=ROCArray,closest=false))[:,:,1]
end


c = heatmap!(ax4,xsvals,ysvals,slice,colormap=:lipari)
Colorbar(f4[1,0],c)
empty!(f4)

@benchmark parsedVol = Array(interpolateSliceGPUAlt(parsedGrid,range(-5,5,1200),range(-5,5,1200),[1],power=12,ArrayType=ROCArray,closest=false))
volume(-1 .* parsedVol)