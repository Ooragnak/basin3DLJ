using LinearAlgebra
using Printf
using ProgressMeter

abstract type Grid{T} end

abstract type AbstractPoint{T} end

abstract type Position end

struct Spherical <: Position
    r::Float64
    ϕ::Float64
    θ::Float64
end

struct Polar <: Position
    r::Float64
    θ::Float64
end

struct Cartesian2D <: Position
    x::Float64
    y::Float64
end

struct Cartesian3D <: Position
    x::Float64
    y::Float64
    z::Float64
end

struct PointRot{T} <: AbstractPoint{T}
    rotation::Tuple{Float64,Float64,Float64,Float64}
    translation::T
    energy::Float64
end


struct Point{T} <: AbstractPoint{T}
    translation::T
    energy::Float64
end

struct Index{Cartesian2D} <: AbstractPoint{Cartesian2D}
    x::Int
    y::Int
end

struct PointGrid{K <: AbstractPoint} <: Grid{K}
    dim::Float64
    points::Vector{K}
    distances::AbstractDict{K,AbstractVector{Tuple{Int64,Float64}}}
    properties::AbstractString
end

"""
Data structure encoding the basins of attraction \n
Gridpoints: AbstractPoint => (Next Point, Reached Minimum)
"""
mutable struct Basin{T <: AbstractPoint}
    const grid::Grid
    minima::Vector{T}
    gridpoints::Dict{T,Tuple{T,T}} 
end

Base.show(io::IO, ::MIME"text/plain",   k::Polar) = print(io, "Polar:\n   ", k)
Base.show(io::IO,                       k::Polar) = print(io, pretty(k))
Base.show(io::IO, ::MIME"text/plain",   k::Spherical) = print(io, "Spherical:\n   ", k)
Base.show(io::IO,                       k::Spherical) = print(io, pretty(k))
Base.show(io::IO, ::MIME"text/plain",   k::Cartesian2D) = print(io, "Cartesian2D(x,y):\n   ", k)
Base.show(io::IO,                       k::Cartesian2D) = print(io, pretty(k))
Base.show(io::IO, ::MIME"text/plain",   k::Cartesian3D) = print(io, "Cartesian3D(x,y,k):\n   ", k)
Base.show(io::IO,                       k::Cartesian3D) = print(io, pretty(k))
Base.show(io::IO,                       k::Index) = print(io, "($(k.x), $(k.y))")

Base.show(io::IO, ::MIME"text/plain",   k::T) where {T <: AbstractPoint} = print(io, "$T:\n   ", k)
Base.show(io::IO,                       k::T) where {T <: AbstractPoint} = print(io, "(E = $(k.energy), $(k.translation))")

Base.show(io::IO, ::MIME"text/plain",   k::PointGrid) = print(io, "$(typeof(k)): $(k.properties) \npoints:    ", pretty(k.points),"\ndistances: $(typeof(k.distances)) with $(length(keys(k.distances))) entries")
Base.show(io::IO, ::MIME"text/plain",   k::Basin) = print(io, "$(typeof(k)) with $(length(k.minima)) basins \ngrid:          $(typeof(k.grid)) \n ├─properties: ", k.grid.properties,"\n ├─points:     ", pretty(k.grid.points),"\n └─distances:  $(typeof(k.grid.distances)) with $(length(keys(k.grid.distances))) entries \ngridpoints:    $(typeof(k.gridpoints)) with $(length(keys(k.gridpoints))) entries \nminima:      ", string(["\n$(getBasinSize(k,m)) points -> $(pretty(m,6))" for m in k.minima]...))

``

pretty(k::Polar, precision = 18) = "(r = $(round(k.r,sigdigits=precision)), θ = $(round(k.θ,sigdigits=precision)))"
pretty(k::Spherical, precision = 16) = "(r = $(round(k.r,sigdigits=precision)), θ = $(round(k.θ,sigdigits=precision)), ϕ = $(round(k.ϕ,sigdigits=precision)))"
pretty(k::Cartesian3D, precision = 16) = "(x = $(round(k.x,sigdigits=precision)), y = $(round(k.y,sigdigits=precision)), z = $(round(k.z,sigdigits=precision)))"
pretty(k::Cartesian2D, precision = 16) = "(x = $(round(k.x,sigdigits=precision)), y = $(round(k.y,sigdigits=precision)))"
pretty(k::Point, precision = 16) = "(E = $(round(k.energy,sigdigits=precision)), $(pretty(k.translation,precision))"


pretty(ks::AbstractArray{T}) where {T <: AbstractPoint} = "$(length(ks))-element $(typeof(ks)) with global energy minimum E = $(minimum(k.energy for k in ks)), $(pretty([k.translation for k in ks]))" 
pretty(ks::AbstractArray{T}) where {T <: Union{Polar, Spherical}} = "contains $(length(unique([k.r for k in ks]))) radii between $(minimum(unique([k.r for k in ks]))) and $(maximum(unique([k.r for k in ks])))" 
pretty(ks::AbstractArray{T}) where {T <: Union{Cartesian2D}} = "contains $(length(unique([k.x for k in ks]))) x-values between $(minimum(unique([k.x for k in ks]))) and $(maximum(unique([k.x for k in ks]))), contains $(length(unique([k.y for k in ks]))) y-values between $(minimum(unique([k.y for k in ks]))) and $(maximum(unique([k.y for k in ks])))" 






function distance(t1::Spherical,t2::Spherical)
    return sqrt(t1.r^2 + t2.r^2 - 2*t1.r*t2.r*(sin(t1.θ) * sin(t2.θ) * cos(t1.ϕ - t2.ϕ) + cos(t1.θ) * cos(t2.θ)))
end

function distance(t1::Polar,t2::Polar)
    return sqrt(t1.r^2 + t2.r^2 - 2*t1.r*t2.r * cos(t1.θ - t2.θ))
end

distance(t1::Cartesian2D,t2::Cartesian2D) = sqrt((t1.x - t2.x)^2 + (t1.y - t2.y)^2)
distance(t1::Cartesian3D,t2::Cartesian3D) = sqrt((t1.x - t2.x)^2 + (t1.y - t2.y)^2 + + (t1.z - t2.z)^2)
distance(p1::Point,p2::AbstractPoint) = distance(p1.translation,p2.translation)

toCartesian(t::Polar) = Cartesian2D(t.r*cos(t.θ),t.r*sin(t.θ))
toCartesian(t::Spherical) = Cartesian3D(t.r*sin(t.θ)*cos(t.ϕ),t.r*sin(t.θ)*sin(t.ϕ),t.r*cos(t.θ))
toPolar(t::Cartesian2D) = Polar(hypot(t.x,t.y),atan2(y,x))
toCartesian(t::Cartesian2D) = t
toCartesian(t::Cartesian3D) = t
toPolar(t::Polar) = t

function toSpherical(x,y,z)
    r = sqrt(x^2 + y^2 + z^2)
    if r == 0
        return Cartesian3D(0,0,0)
    end
    θ = acos(z/r)
    ϕ = sign(y)*acos(x/sqrt(x^2+y^2))

    #This check is needed as sign(0) returns 0 which then causes ϕ to be 0 instead of π
    if y == 0
        ϕ = acos(x/sqrt(x^2+y^2))
    end
    if ϕ <= 0
        ϕ = 2π+ϕ
    end
    if x == 0 == y
        ϕ = 0
    end
    @assert 0 <= θ <= π "θ = $θ, x = $x, y=$y, z=$z"
    @assert 0 <= ϕ <= 2π "ϕ = $ϕ, x = $x, y=$y, z=$z"
    return r,θ,ϕ
end

function distance(p1::PointRot,p2::PointRot)
    return distance(p1.translation,p2.translation) + acos(2*dot(p1.rotation,p2.rotation) -1)
end

function getPoints(grid::PointGrid)
    return grid.points
end

function getNeighbors(grid::PointGrid,point::AbstractPoint)
    g = grid.distances[point]
    return grid.points[first.(g)], last.(g)
end


function findClosestGridPoint(grid::PointGrid,point::AbstractPoint)
    @assert eltype(grid.points) == typeof(point)
    closest = argmin(distance.(grid.points,Ref(point)))
    return grid.points[closest]
end

function tracePath(basin::Basin,startingPosition::T) where {T <: AbstractPoint}
    current = findClosestGridPoint(basin.grid, startingPosition)
    path = T[]
    next = basin.gridpoints[current][1]
    while current != next
        push!(path,current)
        current = next
        next = basin.gridpoints[current][1]
    end
    return path
end

function gradDescent(grid::Grid{T}) where {T <: AbstractPoint}
    minima = T[]
    gridpoints = Dict{T,Tuple{T,T}}()
    sizehint!(gridpoints,length(getPoints(grid)))

    trajectory = Tuple{T,T}[]

    progress = Progress(length(getPoints(grid)),desc = "Descending to minimum:", dt=0.2, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    generate_progress(i) = [("Found basin for",@sprintf "%6i / %6i points." i length(getPoints(grid)))]
    for (i,p) in enumerate(getPoints(grid))
        current = p
        while true
            if haskey(gridpoints,current)
                minim = gridpoints[current][2]
                for step in trajectory
                    push!(gridpoints,step[1] => (step[2],minim))
                end
                trajectory = Tuple{T,T}[]
                break
            end
            neighbors, distances = getNeighbors(grid,current)
            
            grads = [(neighbors[i].energy - current.energy) / d for (i,d) in enumerate(distances)]
            grad = minimum(grads)
            min = argmin(grads)
            grad == 0 && @warn "Energy difference is exactly zero at $current \n grads = $grads"
            if grad >= 0
                push!(minima,current)
                for step in trajectory
                    push!(gridpoints,step[1] => (step[2],current))
                end
                push!(gridpoints,current => (current,current))
                trajectory = Tuple{T,T}[]
            else
                push!(trajectory,(current, neighbors[min]))
                current = neighbors[min]
            end
        end
        next!(progress; showvalues = generate_progress(i))
    end
    
    return Basin(grid,minima,gridpoints)
end

"""
Generate a cartesion grid from vectors of evenly spaced points. 
"""
function makeCartesianGrid(xs,ys,V,properties;diagonal=true)
    xdim = length(xs)
    ydim = length(ys)

    points = [Point(Cartesian2D(x,y),V(x,y)) for x in xs, y in ys]
    distances = Dict{Point{Cartesian2D},AbstractVector{Tuple{Int64,Float64}}}()
    sizehint!(distances,xdim*ydim)

    progress = Progress(xdim*ydim,desc = "Generating grid:", dt=0.2, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    generate_progress(i) = [("Found adjacencies for",@sprintf "%6i / %6i points." i xdim*ydim)]

    for x1 in 1:1:length(xs), y1 in 1:1:length(ys)
        arg1 = xdim*(y1-1)+x1
        neighbors = []
        for x2 in x1-1:1:x1+1, y2 in y1-1:1:y1+1
            arg2 = xdim*(y2-1)+x2
            if diagonal
                if !(x2 == x1 && y2 == y1) && (1 <= x2 <= xdim && 1 <= y2 <= ydim)
                    push!(neighbors,(arg2,distance(points[arg2],points[arg1])))
                end
            elseif xor(x2 == x1,y2 == y1) && (1 <= x2 <= xdim && 1 <= y2 <= ydim)
                    push!(neighbors,(arg2,distance(points[arg2],points[arg1])))
            end
        end
        push!(distances,points[x1,y1] => neighbors)

        next!(progress; showvalues = generate_progress(arg1))
    end

    return PointGrid{Point{Cartesian2D}}(2,vec(points),distances,properties)
end

"""
Generate a polar grid from vectors of evenly spaced points in r and theta. 
"""
function makePolarGrid(rs,θlen,V,properties)
    θs = range(0,2π,θlen)[1:end-1]
    rdim = length(rs)
    θdim = length(θs)

    points = [Point(Polar(r,θ),V(r,θ)) for r in rs, θ in θs]
    distances = Dict{Point{Polar},AbstractVector{Tuple{Int64,Float64}}}()
    sizehint!(distances,rdim*θdim)


    progress = Progress(rdim*θdim,desc = "Generating grid:", dt=0.2, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    generate_progress(i) = [("Found adjacencies for",@sprintf "%6i / %6i points." i rdim*θdim)]

    for r1 in 1:1:length(rs), θ1 in 1:1:length(θs)
        arg1 = rdim*(θ1-1)+r1
        neighbors = []

        for r2 in r1-1:1:r1+1, θ2 in mod1.(θ1-1:1:θ1+1,θdim)
            arg2 = rdim*(θ2-1)+r2
 
            if !(r2 == r1 && θ2 == θ1) && (1 <= r2 <= rdim)
                push!(neighbors,(arg2,distance(points[arg2],points[arg1])))
            end
        end
        push!(distances,points[r1,θ1] => neighbors)

        next!(progress; showvalues = generate_progress(arg1))
    end

    return PointGrid{Point{Polar}}(2,vec(points),distances,properties)
end

function getBasinSize(basin::Basin,minimum::AbstractPoint)
    size = 0
    for g in keys(basin.gridpoints)
        if basin.gridpoints[g][2] == minimum
            size += 1
        end
    end
    return size
end

function findMinimumEnergyPaths(basin::Basin,minimum::AbstractPoint)
    @assert minimum in basin.minima
    branchBorder = [minimum]
    currentPoint = minimum
    visitedPoints = [minimum]

    # Array of Tuple of point in basin of starting minimum and (startingBasinPoint,borderBasinPoint)
    foundTransitionPoints = []
    totalPointsCovered = 0

    progress = ProgressUnknown(desc = "Iteration",dt=0.2)
    generate_progress(i,j,k,l) = [("Checked points",@sprintf "%6i (max = %6i)" i length(basin.gridpoints)), ("Current energy",@sprintf "%.4f" j),("Active branches",k),("Found transitions",l)]

    while !isempty(branchBorder)
        #choosing InsertionSort as it is by far the most efficient for almost sorted arrays
        sort!(branchBorder,by=x -> x.energy,alg=InsertionSort)
        currentPoint = branchBorder[1]
        neighbors = getNeighbors(basin.grid,currentPoint)[1]
        previousPoint = neighbors[argmin([basin.gridpoints[n][2] == minimum ? n.energy : Inf for n in neighbors])]
        currentBasin = basin.gridpoints[currentPoint][2]
        foundBasins = [basin.gridpoints[x[2]][2] for x in foundTransitionPoints]
        currentTransition_arg = findfirst(Ref(currentBasin) .== foundBasins)
        if currentBasin == minimum
            for n in neighbors
                # Checking if the neighbor has already been visited by its energy
                if n.energy > currentPoint.energy && !insorted(n,branchBorder,by=x -> x.energy)
                    push!(branchBorder,n)
                # Checking if the neighbor lies in a different basin as it could therefore have a lower energy, yet it might not have been visited
                elseif basin.gridpoints[n][2] != minimum && !insorted(n,branchBorder, by=x -> x.energy)
                    push!(branchBorder,n)
                end
            end
        elseif currentBasin in foundBasins
            if currentPoint.energy < foundTransitionPoints[currentTransition_arg][2].energy
                @assert previousPoint.energy > foundTransitionPoints[currentTransition_arg][1].energy
            end
        else 
            @assert currentBasin != minimum
            push!(foundTransitionPoints,(previousPoint,currentPoint))
        end
        popfirst!(branchBorder)
        totalPointsCovered += 1
        push!(visitedPoints,currentPoint)

        next!(progress; showvalues = generate_progress(totalPointsCovered,currentPoint.energy,length(branchBorder),length(foundTransitionPoints)))
    end
    finish!(progress)
    return foundTransitionPoints
end