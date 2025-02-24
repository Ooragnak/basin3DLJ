include("../src/theme.jl")

using KernelAbstractions
using AMDGPU
using CUDA
################################################################

# Array based conversions

Makie.convert_arguments(P::T, ps::AbstractVector{<: Point}) where {T <: Union{PointBased, GridBased}} = 
    convert_arguments(P, [p.translation for p in ps], [p.energy for p in ps])

Makie.convert_arguments(P::T, ts::AbstractVector{Polar}, args...) where {T <: Union{PointBased, GridBased}}= 
    convert_arguments(P, [t.θ for t in ts], [t.r for t in ts], args...)

Makie.convert_arguments(P::T, ts::AbstractVector{Cartesian2D}, args...) where {T <: Union{PointBased, GridBased}}= 
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

Makie.convert_arguments(P::T, ts::AbstractVector{Cartesian3D}, args...) where {T <: Union{PointBased, GridBased}}= 
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], [t.z for t in ts], args...)

Makie.convert_arguments(P::Type{<:Scatter}, ps::AbstractVector{<: Point}) = 
    convert_arguments(P, [p.translation for p in ps])

Makie.convert_arguments(P::Type{<:Surface}, ts::AbstractVector{Cartesian2D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

# Point based convesions

Makie.convert_arguments(P::PointBased, p::Type{<: Point}) =
    convert_arguments(P, p.translation)

Makie.convert_arguments(P::PointBased, t::Polar, args...) =
    convert_arguments(P, t.θ, t.r, args...)

Makie.convert_arguments(P::PointBased, t::Cartesian2D, args...) =
    convert_arguments(P,t.x, t.y, args...)

Makie.convert_arguments(P::PointBased, t::Cartesian3D, args...) =
    convert_arguments(P,t.x, t.y, t.z, args...)

Makie.convert_arguments(P::Type{<:Scatter}, p::Point, args...) =
    convert_arguments(P, p.translation, args...)

function interpolateSlice(grid::PointGrid{<: Point{T}},xs,ys,zs ;power=8,ArrayType = Array,closest=false,getPoints=false) where {T <: Position}
    N = length(xs)*length(ys)*length(zs)
    xarr  = ArrayType([x for x in xs, y in ys, z in zs])
    yarr  = ArrayType([y for x in xs, y in ys, z in zs])
    zarr  = ArrayType([z for x in xs, y in ys, z in zs])
    pxarr = ArrayType([p.translation.x for p in grid.points])
    pyarr = ArrayType([p.translation.y for p in grid.points])
    pzarr = ArrayType([p.translation.z for p in grid.points])

    energyarr = ArrayType([p.energy for p in grid.points])
    if !getPoints
        energies = ArrayType(zeros(eltype(xarr),size(xarr)))
        interpolateScalar!(energies, pxarr, pyarr, pzarr, energyarr, xarr, yarr, zarr; power= power, closest= closest)
        return reshape(energies,(length(xs),length(ys),length(zs)))
    else
        pIndices = ArrayType(zeros(eltype(xarr),size(xarr)))
        interpolateScalar!(pIndices, pxarr, pyarr, pzarr, energyarr, xarr, yarr, zarr; power= power, closest= closest,getIndex=true)
        return [grid.points[i] for i in Int64.(Array(pIndices))]
    end
end


function interpolateScalar!(scalar::U, pxs::K, pys::K, pzs::K, pscalar::K, xs::T, ys::T, zs::T; power=12, closest=false,getIndex=false) where {U <: AbstractArray, T <: AbstractArray, K <: AbstractArray}
    @assert length(xs) == length(ys) == length(zs)
    @assert length(pxs) == length(pys) == length(pzs) == length(pscalar)

    N = length(xs)
    M = length(pxs)

    backend = get_backend(pxs)
    if !closest 
        kernel! = interpolatePoint!(backend,256)
        kernel!(scalar,xs,ys,zs,pscalar,pxs,pys,pzs,M,power;ndrange = N)
    elseif getIndex
        kernel! = closestPointIndex!(backend,256)
        kernel!(scalar,xs,ys,zs,pscalar,pxs,pys,pzs,M;ndrange = N)
    else
        kernel! = closestPoint!(backend,256)
        kernel!(scalar,xs,ys,zs,pscalar,pxs,pys,pzs,M;ndrange = N)
    end
end

@kernel function interpolatePoint!(out,@Const(xvals),@Const(yvals),@Const(zvals),@Const(scalar),@Const(px),@Const(py),@Const(pz),n,power)
    j = @index(Global, Linear)
    weights = zero(eltype(out))
    energy = zero(eltype(out))
    weight = zero(eltype(out))
    for i in 1:n
        weight = (1 / sqrt((xvals[j] - px[i])^2 + (yvals[j] - py[i])^2 + (zvals[j] - pz[i])^2))^power
        weights += weight
        energy += scalar[i] * weight
    end
    out[j] = energy / weights
end



@kernel function closestPoint!(out,@Const(xvals),@Const(yvals),@Const(zvals),@Const(scalar),@Const(px),@Const(py),@Const(pz),n)
    j = @index(Global, Linear)
    smallestDistance = typemax(eltype(out))
    distance = zero(eltype(out))
    energy = zero(eltype(out))
    for i in 1:n
        distance = (xvals[j] - px[i])^2 + (yvals[j] - py[i])^2 + (zvals[j] - pz[i])^2
        if distance < smallestDistance
            smallestDistance = distance
            energy = scalar[i]
        end
    end
    out[j] = energy
end

@kernel function closestPointIndex!(out,@Const(xvals),@Const(yvals),@Const(zvals),@Const(scalar),@Const(px),@Const(py),@Const(pz),n)
    j = @index(Global, Linear)
    smallestDistance = typemax(eltype(scalar))
    distance = zero(eltype(scalar))
    closestIndex = zero(eltype(out))
    for i in 1:n
        distance = (xvals[j] - px[i])^2 + (yvals[j] - py[i])^2 + (zvals[j] - pz[i])^2
        if distance < smallestDistance
            smallestDistance = distance
            closestIndex = Float32(i)
        end
    end
    out[j] = closestIndex
end

function plotBasinsIsosurface(basin;interpolate=nothing,ArrayType=nothing,energyrange=nothing,figsize=(2560,1440),fontsize=40,isorange = 1,interpolationResolution = 100)
    grid = basin.grid
    @assert grid.dim == 3 "Grid must be 3-dimensional"
    @assert !isnothing(interpolate) || grid.isCartesian "Missing required interpolation arguments for non-Cartesian grid."
    @assert isnothing(interpolate) || !isnothing(interpolate) "Missing required interpolation arguments - Provide \"ArrayType\" for interpolation."

    if !isnothing(interpolate)
        xlimits = extrema(interpolate[1])
        ylimits = extrema(interpolate[2])
        zlimits = extrema(interpolate[3])
        points = Array(interpolateSlice(grid,range(xlimits...,interpolationResolution),range(xlimits...,interpolationResolution),range(xlimits...,interpolationResolution),ArrayType=ArrayType,closest=true,getPoints=true))
    else
        xs = unique(p.translation.x for p in grid.points)
        ys = unique(p.translation.y for p in grid.points)
        zs = unique(p.translation.z for p in grid.points)
        xlimits = extrema(xs)
        ylimits = extrema(ys)
        zlimits = extrema(zs)
        points = reshape([p for p in grid.points],(length(xs),length(ys),length(zs)))
    end

    energies = [p.energy for p in points]

    if isnothing(energyrange)
        energyrange = (extrema(energies))
    end

    f = Figure(size=figsize, fontsize=fontsize)

    s = Slider(f[1,2], range = range(energyrange...,500), startvalue = sum(energyrange)/2,horizontal=false)

    isoval = lift(s.value) do z
        z
    end

    plotTitle = lift(s.value) do z
        latexstring(L"Basins of attraction with isosurface value $E = %$(round(z,sigdigits=3)) $")
    end

    ax = Axis3(f[1,1], title = plotTitle)

    for (i,m) in enumerate(basin.minima)
        basinPoints = [basin.gridpoints[p][2] == m ? p.energy : NaN for p in points]
        volume!(ax,xlimits,ylimits,zlimits,basinPoints , algorithm = :iso, isovalue = isoval, isorange = isorange ,colormap = fill(Makie.wong_colors()[i],100) , interpolate = true)
        #basinPoints = [basin.gridpoints[p][2] == m ? p.energy : NaN for p in points]
        #isOutside = @lift x -> !(x <= $isoval)
        #voxels!(ax,xlimits,ylimits,zlimits,basinPoints, colormap = fill(Makie.wong_colors()[i],100),is_air = isOutside)
    end
    return f
end