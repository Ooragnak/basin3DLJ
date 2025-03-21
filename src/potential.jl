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

function ringpot3D(x,y,z) 
    r, θ, ϕ = toSpherical(x,y,z)
    return ringpot(r,θ) * (sin(ϕ)^2 + 1)
end


function rotated2DPot(pot,x,y,rotation)
    r, θ = toPolar(x,y)
    x2, y2 = toCartesian(r,θ+rotation)
    return pot(x2,y2)
end

simplepot(x,y) = x^2 + 10*y^2

struct LJParticle
    t::Position
    ϵ::Float64
    σ::Float64
end

struct LJCluster
    particles::AbstractArray{LJParticle}
end

function potential(p::LJParticle,x,y,z;replaceNaN=false)
    t1 = convert(Cartesian3D,p.t)
    r = sqrt((t1.x - x)^2 + (t1.y - y)^2 + (t1.z - z)^2)
    if iszero(r) && replaceNaN
        return Inf64
    else
        return 4 * p.ϵ * ((p.σ/r)^12 - (p.σ/r)^6)
    end
end

function potential(c::LJCluster,t::Position)
    p = convert(Cartesian3D,t)
    return potential(c,p.x,p.y,p.z)
end

function potential(c::LJCluster,x,y,z;kwargs...)
    #pot = 0
    #for p in c.particles
    #    pot += potential(p,x,y,z;kwargs...)
    #end
    return sum(potential.(c.particles,x,y,z;kwargs...))
end

function generateFunctionString(c::LJCluster)
    output = "LJCluster(x,y,z) = 0 "
    for p in c.particles
        t1 = convert(Cartesian3D,p.t)
        output = output * "+ 4 * $(p.ϵ) * (($(p.σ)/(sqrt(($(t1.x) - x)^2 + ($(t1.y) - y)^2 + ($(t1.z) - z)^2)))^12 - ($(p.σ)/(sqrt(($(t1.x) - x)^2 + ($(t1.y) - y)^2 + ($(t1.z) - z)^2)))^6) "
    end
    return output
end

function generateSpherePacking(r,is,js,ks) 
    centers = [Cartesian3D(r*(2i + mod(j+k,2)), r*(sqrt(3) * (j + 1/3*mod(k,2))), r*(sqrt(24)/3 * k)) for k in ks, j in js, i in is]
    return centers
end


LJCluster(x,y,z) = 0 + 4 * 1.0 * ((1.7817974362806785/(sqrt((-1.0 - x)^2 + (-1.7320508075688772 - y)^2 + (0.0 - z)^2)))^12 - (1.7817974362806785/(sqrt((-1.0 - x)^2 + (-1.7320508075688772 - y)^2 + (0.0 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((-1.0 - x)^2 + (0.5773502691896257 - y)^2 + (-1.6329931618554518 - z)^2)))^12 - (1.7817974362806785/(sqrt((-1.0 - x)^2 + (0.5773502691896257 - y)^2 + (-1.6329931618554518 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((-2.0 - x)^2 + (0.0 - y)^2 + (0.0 - z)^2)))^12 - (1.7817974362806785/(sqrt((-2.0 - x)^2 + (0.0 - y)^2 + (0.0 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((-1.0 - x)^2 + (0.5773502691896257 - y)^2 + (1.6329931618554518 - z)^2)))^12 - (1.7817974362806785/(sqrt((-1.0 - x)^2 + (0.5773502691896257 - y)^2 + (1.6329931618554518 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((-1.0 - x)^2 + (1.7320508075688772 - y)^2 + (0.0 - z)^2)))^12 - (1.7817974362806785/(sqrt((-1.0 - x)^2 + (1.7320508075688772 - y)^2 + (0.0 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((0.0 - x)^2 + (-1.1547005383792517 - y)^2 + (-1.6329931618554518 - z)^2)))^12 - (1.7817974362806785/(sqrt((0.0 - x)^2 + (-1.1547005383792517 - y)^2 + (-1.6329931618554518 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((1.0 - x)^2 + (-1.7320508075688772 - y)^2 + (0.0 - z)^2)))^12 - (1.7817974362806785/(sqrt((1.0 - x)^2 + (-1.7320508075688772 - y)^2 + (0.0 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((0.0 - x)^2 + (-1.1547005383792517 - y)^2 + (1.6329931618554518 - z)^2)))^12 - (1.7817974362806785/(sqrt((0.0 - x)^2 + (-1.1547005383792517 - y)^2 + (1.6329931618554518 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((1.0 - x)^2 + (0.5773502691896257 - y)^2 + (-1.6329931618554518 - z)^2)))^12 - (1.7817974362806785/(sqrt((1.0 - x)^2 + (0.5773502691896257 - y)^2 + (-1.6329931618554518 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((0.0 - x)^2 + (0.0 - y)^2 + (0.0 - z)^2)))^12 - (1.7817974362806785/(sqrt((0.0 - x)^2 + (0.0 - y)^2 + (0.0 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((1.0 - x)^2 + (0.5773502691896257 - y)^2 + (1.6329931618554518 - z)^2)))^12 - (1.7817974362806785/(sqrt((1.0 - x)^2 + (0.5773502691896257 - y)^2 + (1.6329931618554518 - z)^2)))^6) + 4 * 1.0 * ((1.7817974362806785/(sqrt((1.0 - x)^2 + (1.7320508075688772 - y)^2 + (0.0 - z)^2)))^12 - (1.7817974362806785/(sqrt((1.0 - x)^2 + (1.7320508075688772 - y)^2 + (0.0 - z)^2)))^6) 

function generateCluster(r,LJTypes)
    pack = generateSpherePacking(1,-1:1,-1:1,-1:1)
    center = pack[2,2,2]
    neighbors = findall(x -> distance(x,center) <= r*2.1,pack)
    close = pack[neighbors]
    LJParticles = []
    for (i,c) in enumerate(close)
        if !isnothing(LJTypes[i])
            push!(LJParticles,LJParticle(c,LJTypes[i]...))
        end
    end
    return LJCluster(LJParticles)
end

