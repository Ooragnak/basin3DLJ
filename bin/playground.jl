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


tstpot(r,theta) = (cos(5r/π)+r) * cos(theta)
tstgrid = PolarGrid{Point2D}(2,collect(range(0.01,2,60)),collect(range(0.0,2π,60)[1:end-1]),mbpolar)
ps = getPoints(tstgrid)


ts = gradDescent(tstgrid)

Makie.convert_arguments(P::GridBased, ps::AbstractVector{<: Point}) = 
    convert_arguments(P, [p.translation for p in ps], [p.energy for p in ps])

Makie.convert_arguments(P::PointBased, ps::AbstractVector{<: Point}) = 
    convert_arguments(P, [p.translation for p in ps], [p.energy for p in ps])

Makie.convert_arguments(P::PointBased, ts::AbstractVector{Polar}, args...) =
    convert_arguments(P, [t.polar for t in ts], [t.radius for t in ts], args...)

Makie.convert_arguments(P::PointBased, ts::AbstractVector{Cartesian2D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

Makie.convert_arguments(P::PointBased, p::Type{<: Point}) =
    convert_arguments(P, p.translation)

Makie.convert_arguments(P::PointBased, t::Polar, args...) =
    convert_arguments(P, t.polar, t.radius, args...)

Makie.convert_arguments(P::PointBased, t::Cartesian2D, args...) =
    convert_arguments(P,t.x, t.y, args...)

Makie.convert_arguments(P::Type{<:Scatter}, ps::AbstractVector{<: Point}) = 
    convert_arguments(P, [p.translation for p in ps])

Makie.convert_arguments(P::GridBased, ts::AbstractVector{Cartesian2D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

Makie.convert_arguments(P::Type{<:Surface}, ts::AbstractVector{Cartesian2D}, args...) =
    convert_arguments(P, [t.x for t in ts], [t.y for t in ts], args...)

Makie.convert_arguments(P::Type{<:Scatter}, p::Point2D, args...) =
    convert_arguments(P, [p.translation.x], [p.translation.y], args...)


GLMakie.activate!()

f1 = Figure(size=(1600,900), fontsize=40)
ax = PolarAxis(f1[1,1], title = "Polar coordinates")
ax2 = PolarAxis(f1[1,2], title = "Polar coordinates")

p1 = voronoiplot!(ax,collect(keys(ts.gridpoints)),colorrange=(-150,75),markersize = 0,highclip = :transparent,colormap = :lipari, strokewidth = 0.01)
translate!(p1,0,0,-1000)

for minimum in ts.minima
    tmin = filter(x -> ts.gridpoints[x][2] == minimum, collect(keys(ts.gridpoints)))
    scatter!(ax2,[t.translation.polar for t in tmin], [t.translation.radius for t in tmin],markersize = 8)
end

p2 = scatter!(ax2,[t.translation.polar for t in ts.minima], [t.translation.radius for t in ts.minima], color = :red)


Colorbar(f1[1,3],p1)

display(f1)

tpot = makeCartesianGrid(range(-2.0,1.25,300),range(-0.5,2.5,300),MullerBrown,"Test",diagonal=false)

#tpot.distances
#dists = [distance(p1,p2) < 2.0 ? distance(p1,p2) : 0.0 for p1 in tpot.points, p2 in tpot.points]

#dists == Matrix(tpot.distances)

tp = gradDescent(tpot)

f2 = Figure(size=(1600,900), fontsize=40)
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

tspoint = findClosestGridPoint(tpot,Point2D(Cartesian2D(-2,-0.5),1))
path = tracePath(tp,tspoint)

#scatter!(ax,[t.translation.x for t in path], [t.translation.y for t in path],markersize = 8)

Colorbar(f2[1,0],p1)

display(f2)

transitions = findMinimumEnergyPaths.(Ref(tp),tp.minima)
for t in transitions
    for p in t
        path = reverse(tracePath(tp,p[1]))
        append!(path, tracePath(tp,p[2]))
        scatter!(ax,path,markersize = 4,color=:grey)
    end
end




scatter!(ax,[t.translation.x for t in transition[2]], [t.translation.y for t in transition[2]],markersize = 8)
f3,ax3,p = plot([p.energy for p in path1])
plot!(ax3,[p.energy for p in path2])

