#PATH OF PROJECT DIR; contains folder like src and data
rootPath = pwd()

# This is required to get documentation tooltips to work in VSCode
macro ignore(args...) end

# SYNTAX EXAMPLE:
#@ignore include("../src/EXAMPLE_LIBRARY.jl")
#
#include(joinpath(rootPath, "src/EXAMPLE_LIBRARY.jl"))

include(joinpath(rootPath, "src/theme.jl"))
include(joinpath(rootPath, "src/grid.jl"))

using BenchmarkTools
using Base.Threads

################################################################
function MullerBrown(x,y)
    As = [-200,-100,-170,15]
    as = [-1,-1,-6.5,0.7]
    bs = [0,0,11,0.6]
    cs = [-10,-10,-6.5,0.7]
    x0s = [1,0,-0.5,-1]
    y0s = [0,0.5,1.5,1]
    return sum(As.*exp.(as.*(x.-x0s).^2 .+ bs.*(x.-x0s) .* (y.-y0s) .+ cs.*(y.-y0s).^2))
end

mb(x) = MullerBrown(x...)

mbpolar(r,theta) = MullerBrown(r*cos(theta)-0.3,r*sin(theta)+1)

ringpot(r,θ, α = 3.0, γ = 3.0, χ₁ = 2.25, χ₂ = 4.5 ) = α * (r-γ)^2 + χ₁ * cos(2θ) -χ₂ * cos(4θ)



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


GLMakie.activate!()

#tstgrid = makePolarGrid(range(0.01,2,200),200,mbpolar,"Polar Muller Brown")
tstgrid = makePolarGrid(range(0.1,5,300),300,ringpot,"Ring Potential",nudge = true)

ps = getPoints(tstgrid)


ts = gradDescent(tstgrid)


f1 = Figure(size=(2560,1440), fontsize=40)
ax = PolarAxis(f1[1,1], title = "Polar coordinates")
ax2 = PolarAxis(f1[1,2], title = "Polar coordinates")

#p1 = voronoiplot!(ax,ts.grid.points,colorrange=(-150,75),markersize = 0,highclip = :transparent,colormap = :lipari, strokewidth = 0.01)
#p1 = contourf!(ax,ts.grid.points,colormap = :lipari,levels = range(-150,75,50))
p1 = contourf!(ax,ts.grid.points,colormap = :lipari,levels=75)

translate!(p1,0,0,-1000)

for minimum in ts.minima
    tmin = filter(x -> ts.gridpoints[x][2] == minimum, collect(keys(ts.gridpoints)))
    scatter!(ax2,[t.translation.θ for t in tmin], [t.translation.r for t in tmin],markersize = 8)
end

p2 = scatter!(ax2,[t.translation.θ for t in ts.minima], [t.translation.r for t in ts.minima], color = :red)


Colorbar(f1[1,0],p1)

display(f1)

tpot = makeCartesianGrid(range(-2.0,1.25,60),range(-0.5,2.5,60),MullerBrown,"Test",diagonal=true)

#tpot.distances
#dists = [distance(p1,p2) < 2.0 ? distance(p1,p2) : 0.0 for p1 in tpot.points, p2 in tpot.points]

#dists == Matrix(tpot.distances)

tp = gradDescent(tpot)

f2 = Figure(size=(2560,1440), fontsize=40)
ax = Axis(f2[1,1], title = "Müller-Brown-Potential", yautolimitmargin = (0, 0),)
ax2 = Axis(f2[1,2], title = "Basin of attraction", yautolimitmargin = (0, 0),)

vecs = collect(keys(tp.gridpoints))

#p1 = heatmap!(ax,vecs, colorrange=(-150,75),highclip = :transparent,colormap = :lipari)
p1 = contourf!(ax,vecs,colormap = :lipari,levels = range(-150,75,50), )


for (i,minimum) in enumerate(tp.minima)
    tmin = filter(x -> tp.gridpoints[x][2] == minimum, collect(keys(tp.gridpoints)))
    scatter!(ax2,[t.translation.x for t in tmin], [t.translation.y for t in tmin],markersize = 4)
    scatter!(ax,minimum, markersize = 15, marker=:xcross, label = @sprintf "(%.3f, %.3f)" minimum.translation.x minimum.translation.y)
end


p2 = contour!(ax2,vecs,colormap = :lipari,levels = range(-150,75,50), )

axislegend(ax,"Minima")

#p3 = scatter!(ax,[t.translation.x for t in tp.minima], [t.translation.y for t in tp.minima], color = :red)

#scatter!(ax,[t.translation.x for t in path], [t.translation.y for t in path],markersize = 8)

Colorbar(f2[1,0],p1)

display(f2)

paths = []
transitions = findMinimumEnergyPaths.(Ref(tp),tp.minima)
for t in transitions
    for p in t
        path = reverse(tracePath(tp,p[1]))
        append!(path, tracePath(tp,p[2]))
        scatter!(ax,path,markersize = 4,color=:grey)
        push!(paths,path)
    end
end



f3 = Figure(size=(2560,1440), fontsize=40)
ax3 = Axis(f3[1,1], title = "Minimum energy paths of grid transition", yautolimitmargin = (0, 0),)

for path in paths
    plot!(ax3,[p.energy for p in path])
end

function pot3(x,y,z) 
    r, θ, ϕ = toSpherical(x,y,z)
    return ringpot(r,θ) * (sin(ϕ)^2 + 1)
end

newpot  = makeCartesianGrid(range(-6.01,6,100),range(-6.01,6,100),range(-6.01,6,100),pot3,"Test",diagonal=false)
newbasin = gradDescent(newpot)

paths3d = []
transitions2 = findMinimumEnergyPaths.(Ref(newbasin),newbasin.minima)

f3d = Figure(size=(2560,1440), fontsize=40)
ax3d = Axis3(f3d[1,1], title = "Minimum energy paths of grid transition")
#ax3d2 = Axis3(f3d[1,2], title = "Isosurface of Potential")

s1 = Slider(f3d[1,2], range = -7:0.01:10, startvalue = 0.0,horizontal=false)

function toArray(ps::PointGrid{Point{Cartesian3D}})
    xlen = length(unique(p.translation.x for p in ps.points))
    ylen = length(unique(p.translation.y for p in ps.points))
    zlen = length(unique(p.translation.z for p in ps.points))
    A = reshape(ps.points,(xlen,ylen,zlen))
    return A
end


isoval = lift(s1.value) do x 
    x
end

for (i,m) in enumerate(newbasin.minima)
    A = toArray(newpot)
    a = [newbasin.gridpoints[p][2] == m ? p.energy : NaN for p in A]
    volume!(ax3d,(-4,4),(-4,4),(-4,4),a, algorithm = :iso, isovalue = isoval, isorange = 1 ,colormap = fill(Makie.wong_colors()[i],100) , interpolate = true)
end

volume!(ax3d,(-4,4),(-4,4),(-4,4),[p.energy for p in toArray(newpot)], algorithm = :iso, isovalue = isoval, isorange = 0.1 ,colormap = :lipari, interpolate = true)

for m in newbasin.minima 
    scatter!(ax3d,m,markersize = 15, marker=:xcross, label = pretty(m,3))
end

for t in transitions2
    for p in t
        path = reverse(tracePath(newbasin,p[1]))
        append!(path, tracePath(newbasin,p[2]))
        scatter!(ax3d,path,markersize = 4,color=:grey)
        push!(paths3d,path)
    end
end

display(f3d)