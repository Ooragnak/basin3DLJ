# Contains structs / functions / code snippets no longer used / useful and to be kept for archival purposes / as a basis for possible modification

struct PolarGrid{T <: Point} <: Grid{T}
    dim::Float64
    rs::Vector{Float64}
    thetas::Vector{Float64}
    potential::Function
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

function interpolateSliceGPUAlt(grid::PointGrid{<: Point{T}},xs,ys,zs  ;power=8,ArrayType = Array,closest=false,splitsize = 100) where {T <: Position}
    xs2 = ArrayType([x for x in xs, y in ys, z in zs])
    ys2 = ArrayType([y for x in xs, y in ys, z in zs])
    zs2 = ArrayType([z for x in xs, y in ys, z in zs])

    xarr = ArrayType([p.translation.x for p in grid.points])
    yarr = ArrayType([p.translation.y for p in grid.points])
    zarr = ArrayType([p.translation.z for p in grid.points])

    energyarr = ArrayType([p.energy for p in grid.points])

    weights = ArrayType(zeros(Float64,(length(xs),length(ys),length(zs))))
    energies = ArrayType(zeros(Float64,(length(xs),length(ys),length(zs))))

    backend = get_backend(xarr)
    if !closest 
        kernel! = interpolatePointAlt!(backend)
        kernel!(weights,energies,xs2,ys2,zs2,energyarr,xarr,yarr,zarr,power,splitsize;ndrange = (length(xs),length(ys),length(zs),length(energyarr)))
    else
        kernel! = closestPoint!(backend)
        kernel!(energies,xs2,ys2,zs2,energyarr,xarr,yarr,zarr,length(xarr);ndrange = (length(xs)*length(ys)*length(zs)))
    end
    if !closest
        return energies ./ weights
    else
        return energies
    end
end


@kernel function interpolatePointAlt!(weights,energy,@Const(xvals),@Const(yvals),@Const(zvals),@Const(energies),@Const(px),@Const(py),@Const(pz),power,splitsize)
    k,l,m,i = @index(Global, NTuple)
    weight = (1 / sqrt((xvals[k,l,m] - px[i])^2 + (yvals[k,l,m] - py[i])^2 + (zvals[k,l,m] - pz[i])^2))^power
    Atomix.@atomic weights[k,l,m] += weight
    Atomix.@atomic energy[k,l,m] += energies[i] * weight
end

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