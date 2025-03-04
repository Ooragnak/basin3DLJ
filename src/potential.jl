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


struct LJParticle
    t::Position
    ϵ::Float64
    σ::Float64
end

struct LJCluster
    particles::AbstractArray{LJParticle}
end

function potential(p::LJParticle,t::Position)
    r = distance(p.t,t)
    return 4 * p.ϵ * ((p.σ/r)^12 - (p.σ/r)^6)
end

function potential(c::LJCluster,t::Position)
    pot = 0
    for p in c
        pot += potential(p,t)
    end
    return pot
end

function potential(c::LJCluster,x,y,z)
    pot = 0
    for p in c
        pot += potential(p,Cartesian3D(x,y,z))
    end
    return pot
end

function generateSpherePacking(r,is,js,ks) 
    centers = [Cartesian3D(r*(2i + mod(j+k,2)), r*(sqrt(3) * (j + 1/3*mod(k,2))), r*(sqrt(24)/3 * k)) for k in ks, j in js, i in is]
    return centers
end

function generateCluster(r,)
    pack = generateSpherePacking(1,1:3,1:3,1:3)
    center = pack[2,2,2]
    neighbors = findall(x -> distance(x,center) <= r*2.1,pack)
    close = pack[neighbors]
end