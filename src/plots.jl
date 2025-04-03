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

function sliceCartesian(grid,direction,sliceval)
    @assert grid.isCartesian

    xs = unique([p.translation.x for p in getPoints(grid)])
    ys = unique([p.translation.y for p in getPoints(grid)])
    zs = unique([p.translation.z for p in getPoints(grid)])

    if direction == 'x'
        slice = argmin(abs.(xs .- sliceval))
        val = xs[slice]
        return reshape(getPoints(grid)[findall(p -> p.translation.x == val,getPoints(grid))],length(ys),length(zs))
    elseif direction == 'y'
        slice = argmin(abs.(ys .- sliceval))
        val = ys[slice]
        return reshape(getPoints(grid)[findall(p -> p.translation.y == val,getPoints(grid)),length(xs)],length(zs))
    else
        slice = argmin(abs.(zs .- sliceval))
        val = zs[slice]
        return reshape(getPoints(grid)[findall(p -> p.translation.z == val,getPoints(grid))],length(xs),length(ys))
    end
end

function plotBasinsIsosurface(basin;interpolate=nothing,ArrayType=nothing,energyrange=nothing,figsize=(2560,1440),fontsize=40,isorange = 1,interpolationResolution = 100, voxels=false)
    grid = basin.grid
    @assert grid.dim == 3 "Grid must be 3-dimensional"
    @assert !isnothing(interpolate) || grid.isCartesian "Missing required interpolation arguments for non-Cartesian grid."
    @assert isnothing(interpolate) || !isnothing(ArrayType) "Missing required interpolation arguments - Provide \"ArrayType\" for interpolation."

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
        if voxels
            basinPoints = [basin.gridpoints[p][2] == m ? p.energy : NaN for p in points]
            isOutside = @lift x -> !(x <= $isoval)
            voxels!(ax,xlimits,ylimits,zlimits,basinPoints, colormap = fill(Makie.wong_colors()[mod1(i,7)],100),is_air = isOutside,colorrange=(-1,1))
        else 
            basinPoints = [basin.gridpoints[p][2] == m ? p.energy : NaN for p in points]
            volume!(ax,xlimits,ylimits,zlimits,basinPoints , algorithm = :iso, isovalue = isoval, isorange = isorange ,colormap = fill(Makie.wong_colors()[mod1(i,7)],100) , interpolate = true)
        end
    end
    return f
end

function plot2DPolarwatersheds(basin,coreDelta,basinTitle,potTitle,setTitle,filename,zlabel)
    CairoMakie.activate!()
    f = Figure(size=(2560,2560), fontsize=40)

    gridTop = f[1,1]
    gridBot = f[2,1]
    lgridTop = gridTop[1,1]
    rgridTop = gridTop[1,2]
    
    ax = PolarAxis(lgridTop[1,1], title = potTitle)


    ax2 = PolarAxis(gridBot[1,1], title = basinTitle)
    ax3 = PolarAxis(gridBot[1,2], title = setTitle)
    rlims!(ax3,0,maximum([k.translation.r for k in keys(basin.gridpoints)]))

    l1 = []
    l2 = []

    p1 = contourf!(ax,basin.grid.points,colormap = :lipari,levels=75)

    translate!(p1,0,0,-1000)

    for minimum in basin.minima
        tmin = filter(x -> basin.gridpoints[x][2] == minimum, collect(keys(basin.gridpoints)))
        tmincore = filter(x -> basin.gridpoints[x][2] == minimum && x.energy - minimum.energy < coreDelta, collect(keys(basin.gridpoints)))

        scatter!(ax2,[t.translation.θ for t in tmin], [t.translation.r for t in tmin],markersize = 8)
        scatter!(ax3,[t.translation.θ for t in tmincore], [t.translation.r for t in tmincore],markersize = 8)

        minleg = scatter!(ax,minimum, markersize = 18, marker=:xcross)
        push!(l1,minleg)
        push!(l2,@sprintf "(r = %.3f, θ = %.3f)" minimum.translation.r minimum.translation.θ)
    end

    l = Legend(rgridTop[1,2:4], l1, l2, "Local Minima")


    p2 = scatter!(ax2,[t.translation.θ for t in basin.minima], [t.translation.r for t in basin.minima], color = :red)


    Colorbar(rgridTop[1,1],p1,label=zlabel)

    save(string("plots/",filename),f)
end

function plot2DCartesianWatersheds(basin,basinTitle,potTitle,filename,zlabel;lvl = 75, msize = 4)

    f = Figure(size=(2560,1440), fontsize=40)
    ax = Axis(f[1,1], title = potTitle, yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax2 = Axis(f[1,2], title = basinTitle, yautolimitmargin = (0, 0), xlabel="x", ylabel="y")

    vecs = collect(keys(basin.gridpoints))

    p1 = contourf!(ax,vecs,colormap = :lipari,levels = lvl, )


    for (i,minimum) in enumerate(basin.minima)
        tmin = filter(x -> basin.gridpoints[x][2] == minimum, collect(keys(basin.gridpoints)))
        scatter!(ax2,[t.translation.x for t in tmin], [t.translation.y for t in tmin],markersize = msize)
        scatter!(ax,minimum, markersize = 15, marker=:xcross, label = @sprintf "(%.3f, %.3f)" minimum.translation.x minimum.translation.y)
    end

    p2 = contour!(ax2,vecs,colormap = :lipari,levels = lvl, )

    axislegend(ax,"Minima")


    Colorbar(f[1,0],p1, label = zlabel)

    save(string("plots/",filename),f)
end

function compare2DCartesianWatersheds(basins,titles,filename;lvl = 75, msizes = fill(4,4))
    @assert length(basins) == length(titles) == 4
    
    f = Figure(size=(2560,2560), fontsize=40)
    ax1 = Axis(f[1,1], title = titles[1], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax2 = Axis(f[1,2], title = titles[2], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax3 = Axis(f[2,1], title = titles[3], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax4 = Axis(f[2,2], title = titles[4], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")

    symbols = [:xcross, :circle, :utriangle, :star5, :rect, :diamond, :hexagon, :cross, :star4] 

    for (j,ax) in enumerate([ax1,ax2,ax3,ax4])

        vecs = collect(keys(basins[j].gridpoints))


        for (i,minimum) in enumerate(basins[j].minima)
            tmin = filter(x -> basins[j].gridpoints[x][2] == minimum, collect(keys(basins[j].gridpoints)))
            scatter!(ax,[t.translation.x for t in tmin], [t.translation.y for t in tmin],markersize = msizes[j])
            scatter!(ax,minimum, markersize = 16, marker=symbols[i], color=:black,  label = @sprintf "(%.3f, %.3f)" minimum.translation.x minimum.translation.y)
        end

        p2 = contour!(ax,vecs,colormap = :lipari,levels = lvl, )

        axislegend(ax,"Minima")

    end

    save(string("plots/",filename),f)
end

function compare2DCartesianCoreSets(basins,titles,filename;lvl = 75, epsilons = fill(1,4), msize = 8)
    CairoMakie.activate!()
    @assert length(basins) == length(titles) == 4
    
    f = Figure(size=(2560,2560), fontsize=40)
    ax1 = Axis(f[1,1], title = titles[1], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax2 = Axis(f[1,2], title = titles[2], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax3 = Axis(f[2,1], title = titles[3], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax4 = Axis(f[2,2], title = titles[4], yautolimitmargin = (0, 0), xlabel="x", ylabel="y")

    symbols = [:xcross, :circle, :utriangle, :star5, :rect, :diamond, :hexagon, :cross, :star4] 

    for (j,ax) in enumerate([ax1,ax2,ax3,ax4])

        vecs = collect(keys(basins[j].gridpoints))

        for (i,minimum) in enumerate(basins[j].minima)
            tmin = filter(x -> basins[j].gridpoints[x][2] == minimum && x.energy - minimum.energy < epsilons[j], collect(keys(basins[j].gridpoints)))
            scatter!(ax,[t.translation.x for t in tmin], [t.translation.y for t in tmin],markersize = msize)
            scatter!(ax,minimum, markersize = 16, marker=symbols[i], color=:black,  label = @sprintf "(%.3f, %.3f)" minimum.translation.x minimum.translation.y)
        end

        p2 = contour!(ax,vecs,colormap = :lipari,levels = lvl, )

        axislegend(ax,"Minima")

    end

    save(string("plots/",filename),f)
end

function plotMEPs2D(basin,title,mepTitle,filename;lvl = 75, basinSmall = basin)    
    f = Figure(size=(2560,1440), fontsize=40)
    ax = Axis(f[1,1], title = title, yautolimitmargin = (0, 0), xlabel="x", ylabel="y")
    ax2 = Axis(f[1,2], title = mepTitle,xlabel = L"Q" ,ylabel=L"V", xautolimitmargin = (0.05, 0.05))

    symbols = [:xcross, :circle, :utriangle, :star5, :rect, :diamond, :hexagon, :cross, :star4] 

    vecs = collect(keys(basin.gridpoints))

    p1 = contourf!(ax,vecs,colormap = :lipari,levels =lvl, )
    p2 = contour!(ax,vecs,colormap = :lipari,levels = lvl, linewidth = 0.001, colorrange = extrema(lvl))



    symbols = ['A','B','C']

    for (i,minimum) in enumerate(basin.minima)
        scatter!(ax,minimum, markersize = 40, marker=symbols[i], color=:gray,  label = @sprintf ": V(%.3f, %.3f) = %.3f" minimum.translation.x minimum.translation.y minimum.energy)
    end

    axislegend(ax,"Minima")

    Colorbar(f2[1,0],p1)
    paths = []

    transitions = findMinimumEnergyPaths.(Ref(basin),basin.minima)

    for t in transitions
        for (i,p) in enumerate(t)
            path = reverse(tracePath(basin,p[1]))
            append!(path, tracePath(basin,p[2]))
            scatter!(ax,path,markersize = 4,color=Makie.wong_colors()[3+i])
            push!(paths,path)
        end
    end

    ABpath = reverse(paths[3])
    BCpath = paths[2]
    B_ind = length(ABpath)+2
    C_ind = length(ABpath)+length(BCpath)+3

    ax2.xticks = ([1,B_ind,C_ind],["A","B","C"])

    ABCpath = vcat(basin.minima[1],ABpath,basin.minima[2],BCpath,basin.minima[3])

    scatter!(ax2,2:1:(B_ind-1),[p.energy for p in ABpath],color=Makie.wong_colors()[5])
    scatter!(ax2,(B_ind+1):1:(C_ind-1),[p.energy for p in BCpath],color=Makie.wong_colors()[4])
    scatter!(ax2,[1,B_ind,C_ind],[basin.minima[1].energy,basin.minima[2].energy,basin.minima[3].energy],color=:black)

    tsAB = 1 + argmax([p.energy for p in ABpath])
    scatter!([tsAB],[ABCpath[tsAB].energy], color=Makie.wong_colors()[5],marker=:xcross,markersize=20,label=latexstring(L"V(TS_{A \rightarrow B}) = ",round(ABCpath[tsAB].energy,sigdigits = 5)))

    tsBC = B_ind + argmax([p.energy for p in BCpath])
    scatter!([tsBC],[ABCpath[tsBC].energy], color=Makie.wong_colors()[4],marker=:star4,markersize=20,label=latexstring(L"V(TS_{B \rightarrow C}) = ",round(ABCpath[tsBC].energy,sigdigits = 5)))

    axislegend(ax2)

    save(string("plots/",filename),f)
end