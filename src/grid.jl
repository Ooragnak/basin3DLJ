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

struct Point3D{T <: Number} <: Point
    translation::Spherical
    energy::Number
end

struct Point2D{T <: Number} <: Point
    translation::Polar
    energy::Number
end

mutable struct Basin{T <: Point}
    const grid::Grid
    minima::Vector{T}
    # Key is the point, the tuple contains next point and reached minima
    gridpoints::Dict{T,Tuple{T,T}} 
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

    for p in grid.points
        getStep(p) 
    end

end