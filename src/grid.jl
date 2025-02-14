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

struct PolarGrid{T <: Point} <: Grid{T}
    dim::Float64
    rs::Vector{Float64}
    thetas::Vector{Float64}
    potential::Function
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

function getPoints(grid::PolarGrid)
    return [Point2D(Polar(r,theta),grid.potential(r,theta)) for r in grid.rs, theta in grid.thetas]
end

function getNeighbors(grid::PolarGrid,point::Point)
    @assert point.translation.radius in grid.rs "Provided point not in given radial steps"
    @assert point.translation.polar in grid.thetas "Provided angle (polar) not in given angular steps"

    radial_arg = findfirst(point.translation.radius .== grid.rs)

    if radial_arg == 1
        radial_neighbors = [1,2]
    elseif radial_arg == length(grid.rs)
        radial_neighbors = [length(grid.rs)-1,length(grid.rs)]
    else 
        radial_neighbors = [radial_arg-1, radial_arg, radial_arg+1]
    end

    angular_arg = findfirst(point.translation.polar .== grid.thetas)
    angular_neighbors = [mod1(angular_arg-1,length(grid.thetas)), angular_arg,  mod1(angular_arg+1,length(grid.thetas))]

    neighbors = [Point2D(Polar(r,theta),grid.potential(r,theta)) for r in grid.rs[radial_neighbors], theta in grid.thetas[angular_neighbors]]
    withoutSelf = findall(x -> x != point,vec(neighbors))
    return neighbors[withoutSelf], [distance(n,point) for n in neighbors][withoutSelf]
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
            grad == 0 && @warn "Energy difference is exactly zero"
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
function makeCartesianGrid(xs,ys,potential,properties;diagonal=true)
    xdim = length(xs)
    ydim = length(ys)

    points = [Point2D(Cartesian2D(x,y),potential(x,y)) for x in xs, y in ys]
    distances = Dict{Point2D,AbstractVector{Tuple{Int64,Float64}}}()

    progress = Progress(xdim*ydim,desc = "Generating grid:", dt=0.2, barglyphs=BarGlyphs("[=> ]"), barlen=50)
    generate_progress(i) = [("Found adjacencies for",@sprintf "%6i / %6i points." i xdim*ydim)]

    for xref in 1:1:length(xs), yref in 1:1:length(ys)
        neighbors = []
        for xarg in xref-1:1:xref+1, yarg in yref-1:1:yref+1
            if diagonal
                if !(xarg == xref && yarg == yref) && (1 <= xarg <= xdim && 1 <= yarg <= ydim)
                    push!(neighbors,(ydim*(yarg-1)+xarg,distance(points[ydim*(yarg-1)+xarg],points[ydim*(yref-1)+xref])))
                end
            elseif xor(xarg == xref,yarg == yref) && (1 <= xarg <= xdim && 1 <= yarg <= ydim)
                    push!(neighbors,(ydim*(yarg-1)+xarg,distance(points[ydim*(yarg-1)+xarg],points[ydim*(yref-1)+xref])))
            end
        end
        push!(distances,points[xref,yref] => neighbors)

        next!(progress; showvalues = generate_progress(ydim*(yref-1)+xref))
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
                #@warn "Entered dead path - Previous pair: $(foundTransitionPoints[currentTransition_arg])"
                #foundTransitionPoints[currentTransition_arg] = (previousPoint,currentPoint)
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