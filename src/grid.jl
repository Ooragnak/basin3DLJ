abstract type Grid end

abstract type Point end

struct Spherical
    radius::Number
    azimuth::Number
    polar::Number
end

struct Polar
    radius::Number
    polar::Number
end

struct Cartesian2D
    radius::Number
    polar::Number
end

function distance(t1::Spherical,t2::Spherical)
    return sqrt(t1.radius^2 + t2.radius^2 - 2*t1.radius*t2.radius*(sin(t1.polar) * sin(t2.polar) * cos(t1.azimuth - t2. azimuth) + cos(t1.polar) * cos(t2.polar)))
end

function distance(t1::Polar,t2::Polar)
    return sqrt(t1.radius^2 + t2.radius^2 - 2*t1.radius*t2.radius * cos(t1.polar - t2.polar))
end


struct Point4D{T <: Number} <: Point
    rotation::Tuple{T,T,T,T}
    translation::Spherical
    energy::Number
end

struct Point3D <: Point
    translation::Spherical
    energy::Number
end

struct Point2D <: Point
    translation::Polar
    energy::Number
end

mutable struct Basin{T <: Point}
    const grid::Grid
    minima::Vector{T}
    # Key is the point, the tuple contains next point and reached minima
    gridpoints::Dict{T,Tuple{T,T}} 
end

struct PolarGrid <: Grid
    dim::Number
    rs::Vector{Number}
    thetas::Vector{Number}
    potential::Function
end

function getPoints(grid::PolarGrid)
    return [Point2D(Polar(r,theta),grid.potential(r,theta)) for r in grid.rs, theta in grid.thetas]
end

function getNeighbors(grid::PolarGrid,point::Point2D)
    @assert point.translation.radius in grid.rs "Provided point not in given radial steps"
    @assert point.translation.polar in grid.thetas "Provided angle (polar) not in given angular steps"

    radial_arg = findfirst(point.translation.radius .== grid.rs)

    if radial_arg == 1
        radial_neighbors = [2]
    elseif radial_arg == length(rs)
        radial_neighbors = [length(rs)-1]
    else 
        radial_neighbors = [radial_arg-1, radial_arg+1]
    end

    angular_arg = findfirst(point.translation.polar .== grid.thetas)
    angular_neighbors = [mod1(radial_arg-1,length(grid.thetas)), mod1(radial_arg+1,length(grid.thetas))]

    return [Point2D(Polar(r,theta),potential(r,theta)) for r in grid.rs[radial_neighbors], theta in grid.thetas[angular_neighbors]]
end


function gradDescent(grid::Grid)
    minima = []
    gridpoints = Dict()
    trajectory = []

    function getStep(point)
        if haskey(gridpoints,point)
            minimum = get(gridpoints,point)
            for step in trajectory
                push!(gridpoints,step[1] => (step[2],minimum))
            end
            trajectory = []
            return 0
        end
        neighbors = getNeighbors(grid,point)
        grads = [(n.energy - point.energy) / distance(n,point) for n in neighbors] 
        min = argmin(grads)
        grads[min] == 0 && @warn "Energy difference is exactly zero"
        if grads[min] >= 0
            push!(minima,point)
            for step in trajectory
                push!(gridpoints,step[1] => (step[2],minimum))
            end
            trajectory = []
        else
            push!(trajectory,(point, neighbors[min]))
            getStep(neighbors(min))
        end
    end

    for p in getPoints(grid)
        getStep(p) 
    end
    return Basin(grid,minima,gridpoints)
end