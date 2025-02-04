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

mbpolar(r,theta) = MullerBrown(r*cos(theta),r*sin(theta))


tstpot(r,theta) = (cos(5r/π)+r) * cos(theta)
tstgrid = PolarGrid(2,collect(range(0.01,2,30)),collect(range(0.0,2π,31)[1:end-1]),mbpolar)
ps = getPoints(tstgrid)

getNeighbors(tstgrid,ps[112])

ts = gradDescent(tstgrid)


Makie.convert_arguments(P::PointBased, ps::AbstractVector{Point}) =
    convert_arguments(P::PointBased, [p.translation.polar for p in ps], [p.translation.radius for p in ps], [p.energy for p in ps])

GLMakie.activate!()

f1 = Figure(size=(2000,1500), fontsize=48)
ax = PolarAxis(f1[1,1], title = "Polar coordinates")
p1 = voronoiplot!(ax,collect(keys(ts.gridpoints)),colorrange=(-150,75),markersize = 0)

for minimum in ts.minima
    tmin = filter(x -> ts.gridpoints[x][2] == minimum, collect(keys(ts.gridpoints)))
    scatter!(ax,[t.translation.polar for t in tmin], [t.translation.radius for t in tmin])
end

p2 = scatter!(ax,[t.translation.polar for t in ts.minima], [t.translation.radius for t in ts.minima], color = :red)


Colorbar(f1[1,2])