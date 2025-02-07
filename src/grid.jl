using SparseArrays
using LinearAlgebra

abstract type Grid end

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

struct PointGrid <: Grid 
    dim::Float64
    points::Vector{Point}
    distances::AbstractMatrix{Float64}
end

struct PolarGrid <: Grid
    dim::Float64
    rs::Vector{Float64}
    thetas::Vector{Float64}
    potential::Function
end

mutable struct Basin{T <: Point}
    const grid::Grid
    minima::Vector{T}
    # Key is the point, the tuple contains next point and reached minima
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
    findfirst(point .== grid.points)
    distances = grid.distances[findfirst,:]
    args_neighbor = .!iszero.(distances)
    return grid.points[args_neighbor], distances[args_neighbor]
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
    closest = argmin(distance.(grid.points,point))
    return grid.points[closest]
end

function tracePath(basin::Basin,startingPosition::Point)
    startingPoint = findClosestGridPoint(basin.grid,startingPosition)
    findfirst([b[2][1] == startingPoint for b in basin.gridpoints])
end

function gradDescent(grid::Grid)
    minima = Point[]
    gridpoints = Dict{Point,Tuple{Point,Point}}()
    trajectory = Tuple{Point,Point}[]

    function getStep(point)
        if haskey(gridpoints,point)
            minimum = gridpoints[point][2]
            for step in trajectory
                push!(gridpoints,step[1] => (step[2],minimum))
            end
            trajectory = Tuple{Point,Point}[]
            return 0
        end
        neighbors, distances = getNeighbors(grid,point)
        grads = [(neighbors[i].energy - point.energy) / d for (i,d) in enumerate(distances)] 
        min = argmin(grads)
        grads[min] == 0 && @warn "Energy difference is exactly zero"
        if grads[min] >= 0
            push!(minima,point)
            for step in trajectory
                push!(gridpoints,step[1] => (step[2],point))
            end
            push!(gridpoints,point => (point,point))
            trajectory = Tuple{Point,Point}[]
        else
            push!(trajectory,(point, neighbors[min]))
            getStep(neighbors[min])
        end
    end

    for p in getPoints(grid)
        getStep(p) 
    end
    return Basin(grid,minima,gridpoints)
end