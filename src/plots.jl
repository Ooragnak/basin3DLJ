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

function interpolateSlice(grid::PointGrid{<: Point{T}},xs,ys,zs ;power=8,ArrayType = Array,closest=false) where {T <: Position}
    N = length(xs)*length(ys)*length(zs)
    xarr  = ArrayType([x for x in xs, y in ys, z in zs])
    yarr  = ArrayType([y for x in xs, y in ys, z in zs])
    zarr  = ArrayType([z for x in xs, y in ys, z in zs])
    pxarr = ArrayType([p.translation.x for p in grid.points])
    pyarr = ArrayType([p.translation.y for p in grid.points])
    pzarr = ArrayType([p.translation.z for p in grid.points])

    energyarr = ArrayType([p.energy for p in grid.points])
    energies = ArrayType(zeros(eltype(xarr),size(xarr)))

    interpolateScalar!(energies, pxarr, pyarr, pzarr, energyarr, xarr, yarr, zarr; power= power, closest= closest)

    return reshape(energies,(length(xs),length(ys),length(zs)))
end

function interpolateScalar!(scalar::T, pxs::K, pys::K, pzs::K, pscalar::K, xs::T, ys::T, zs::T; power=12, closest=false) where {T <: AbstractArray, K <: AbstractArray}
    @assert length(xs) == length(ys) == length(zs)
    @assert length(pxs) == length(pys) == length(pzs) == length(pscalar)

    N = length(xs)
    M = length(pxs)

    backend = get_backend(pxs)
    if !closest 
        kernel! = interpolatePoint!(backend,256)
        kernel!(scalar,xs,ys,zs,pscalar,pxs,pys,pzs,M,power;ndrange = N)
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