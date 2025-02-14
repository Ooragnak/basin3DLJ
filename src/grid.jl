using SparseArrays
using LinearAlgebra
using Printf
using ProgressMeter

abstract type Grid{T} end

abstract type Point end

abstract type Position end

struct Spherical <: Position
    radius::Float64
    azimuth::Float64
    polar::Float64
end

struct Polar <: Position
    radius::Float64
    polar::Float64
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

struct Point4D <: Point
    rotation::Tuple{Float64,Float64,Float64,Float64}
    translation::Position
    energy::Float64
end

struct Point3D <: Point
    translation::Position
    energy::Float64
end

struct Point2D <: Point
    translation::Position
    energy::Float64
end

struct Cartesian2DIndex <: Point
    x::Int
    y::Int
end

struct PointGrid{T <: Point} <: Grid{T}
    dim::Float64
    points::Vector{T}
    distances::AbstractDict{T,AbstractVector{Tuple{Int64,Float64}}}
    properties::AbstractString
end


"""
Data structure encoding the basins of attraction \n
Gridpoints: Point => (Next Point, Reached Minimum)
"""
mutable struct Basin{T <: Point}
    const grid::Grid
    minima::Vector{T}
    gridpoints::Dict{T,Tuple{T,T}} 
end

function distance(t1::Spherical,t2::Spherical)
    return sqrt(t1.radius^2 + t2.radius^2 - 2*t1.radius*t2.radius*(sin(t1.polar) * sin(t2.polar) * cos(t1.azimuth - t2. azimuth) + cos(t1.polar) * cos(t2.polar)))
end

function distance(t1::Polar,t2::Polar)
    return sqrt(t1.radius^2 + t2.radius^2 - 2*t1.radius*t2.radius * cos(t1.polar - t2.polar))
end

distance(t1::Cartesian2D,t2::Cartesian2D) = sqrt((t1.x - t2.x)^2 + (t1.y - t2.y)^2)
distance(t1::Cartesian3D,t2::Cartesian3D) = sqrt((t1.x - t2.x)^2 + (t1.y - t2.y)^2 + + (t1.z - t2.z)^2)
distance(p1::Point,p2::Point) = distance(p1.translation,p2.translation)

toCartesian(t::Polar) = Cartesian2D(t.radius*cos(t.polar),t.radius*sin(t.polar))
toCartesian(t::Spherical) = Cartesian3D(t.radius*sin(t.polar)*cos(t.azimuth),t.radius*sin(t.polar)*sin(t.azimuth),t.radius*cos(t.polar))
toPolar(t::Cartesian2D) = Polar(hypot(t.x,t.y),atan2(y,x))

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

function distance(p1::Point4D,p2::Point4D)
    return distance(p1.translation,p2.translation) + acos(2*dot(p1.rotation,p2.rotation) -1)
end

function getPoints(grid::PointGrid)
    return grid.points
end

function getNeighbors(grid::PointGrid,point::Point)
    g = grid.distances[point]
    return grid.points[first.(g)], last.(g)
end


function findClosestGridPoint(grid::PointGrid,point::Point)
    @assert eltype(grid.points) == typeof(point)
    closest = argmin(distance.(grid.points,Ref(point)))
    return grid.points[closest]
end

function tracePath(basin::Basin,startingPosition::T) where {T <: Point}
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

function gradDescent(grid::Grid{T}) where {T <: Point}
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

    points = [Point2D(Cartesian2D(x,y),V(x,y)) for x in xs, y in ys]
    distances = Dict{Point2D,AbstractVector{Tuple{Int64,Float64}}}()
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

    return PointGrid{Point2D}(2,vec(points),distances,properties)
end

"""
Generate a polar grid from vectors of evenly spaced points in r and theta. 
"""
function makePolarGrid(rs,θlen,V,properties)
    θs = range(0,2π,θlen)[1:end-1]
    rdim = length(rs)
    θdim = length(θs)

    points = [Point2D(Polar(r,θ),V(r,θ)) for r in rs, θ in θs]
    distances = Dict{Point2D,AbstractVector{Tuple{Int64,Float64}}}()
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

    return PointGrid{Point2D}(2,vec(points),distances,properties)
end

function findMinimumEnergyPaths(basin::Basin,minimum::Point)
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