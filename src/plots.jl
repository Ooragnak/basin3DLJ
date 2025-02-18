include(joinpath(rootPath, "src/theme.jl"))

################################################################

# Array based conversions

Makie.convert_arguments(P::GridBased, ps::AbstractVector{<: Point}) = 
    convert_arguments(P, [p.translation for p in ps], [p.energy for p in ps])

Makie.convert_arguments(P::T, ps::AbstractVector{<: Point}) where {T <: Union{PointBased, GridBased}}= 
    convert_arguments(P, [p.translation for p in ps], [p.energy for p in ps])

Makie.convert_arguments(P::PointBased, ts::AbstractVector{Polar}, args...) =
    convert_arguments(P, [t.θ for t in ts], [t.r for t in ts], args...)

Makie.convert_arguments(P::PointBased, ts::AbstractVector{Cartesian2D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

Makie.convert_arguments(P::PointBased, ts::AbstractVector{Cartesian3D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], [t.z for t in ts], args...)

Makie.convert_arguments(P::Type{<:Scatter}, ps::AbstractVector{<: Point}) = 
    convert_arguments(P, [p.translation for p in ps])

Makie.convert_arguments(P::GridBased, ts::AbstractVector{Cartesian2D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

Makie.convert_arguments(P::GridBased, ts::AbstractVector{Cartesian3D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts],[t.z for t in ts], args...)

Makie.convert_arguments(P::GridBased, ts::AbstractVector{Polar}, args...) =
    convert_arguments(P, [t.θ for t in ts], [t.r for t in ts], args...)

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
