" Computes the moving average over array vs with window size n.
    the axis chosen will be reduced in length by (n-1)
"
function moving_average(vs, n::Int)
    data = [(sum(vs[i:(i+n-1)],dims=1)/n) for i in 1:(size(vs,1)-(n-1))]
    if length(data) > 0
        return floor.(reduce(hcat,data))
    end
    return Vector{Float64}[]
end

" Computes the rate that two adjacent atoms in occs are in the excited states 
    occs is a Matrix composed of either 0 or 1
"
function blockade_rate(occs, dims)
    data = reshape(occs, (size(occs,1),dims...))
    if length(dims) == 1
        ignored_blockade = moving_average(permutedims(occs,(2,1)), 2)
        return sum(ignored_blockade)/length(ignored_blockade)
    else
        moving_av_up = moving_average(permutedims(data,(2,1,3)), 2)
        moving_av_right = moving_average(permutedims(data,(3,2,1)), 2)
        return (sum(moving_av_up, init=0) + sum(moving_av_right, init=0))/(length(moving_av_right) + length(moving_av_up))
    end
end
" Computes the energy "
function compute_energy(ns, β, energy_shift)
    energy(x) = -x / β + energy_shift  # The energy shift here ensures that all matrix elements are non-negative. See Merali et al for details.
    BE = LogBinner(energy.(ns)) # Binning analysis
    τ_energy = tau(BE)
    ratio = 2 * τ_energy + 1
    return measurement(mean(BE), std_error(BE)*sqrt(ratio))
end

" Given an array of samples occs with size (# samples, # atoms per sample), compute the order
    parameter.
"
function compute_order_parameter(occs)
    spin_array = 2 .* occs .- 1 # map 0,1 to -1,1

    MCS = size(occs,1)
    order_param = zeros(MCS)
    for i in 1:MCS
        spin = spin_array[i,:]
        order_param[i] = abs(sum(spin[1:2:end]) - sum(spin[2:2:end]))/length(spin) # order param is accurate for any # of dimensions
    end
    return order_param
end

function run_qmc(atoms, Ω, Δ, β; M=50, EQ_MCS::Int=100, MCS::Int=100_000, seed::Int=3214)
    h = rydberg_h(atoms; Δ, Ω)
    h_qmc = rydberg_qmc(h);
    ts = BinaryThermalState(h_qmc, M);
    d = Diagnostics();
    rng = MersenneTwister(seed);

    [mc_step_beta!(rng, ts, h_qmc,β, d, eq=true) for i in 1:EQ_MCS] # equilibration phase

    densities_QMC = zeros(length(atoms))
    occs = zeros(MCS, length(atoms))
    ns = zeros(MCS)

    for i in 1:MCS # Monte Carlo Steps
        ns[i] = mc_step_beta!(rng, ts, h_qmc,β, d, eq=false) do lsize, ts, h_qmc
            SSE_slice = sample(h_qmc,ts, 1)
            occs[i, :] = ifelse.(SSE_slice .== true, 1.0, 0.0)
        end
    end

    energy = compute_energy(ns, β, h_qmc.energy_shift)       

    return occs, energy, ns
end


"
Folder structure
├── code/
│   └── qc-temp/
│       ├── simulation.jl
│       └── simulation_main.jl
└── data/
    └── qc-temp/
        ├── YxY,Rb=XXX,Δ=XXX,β=XX_data/
        │   ├── samples.csv
        │   ├── metrics.csv
        │   └── meta_data.json
        ├── Δ=XXX_Δ=XXX_β=XX_data/
        │   └── ...
        └── ...
"
function save_data(occs, ns, order_parameter, data_dict; gridname=nothing)
    nx = data_dict["nx"]
    ny = data_dict["ny"]
    Ω = data_dict["Ω"]
    Rb = data_dict["Rb_per_a"]
    Δ = data_dict["Δ_per_Ω"]
    β = data_dict["βΩ"]
    if isnothing(gridname)
        folder = @sprintf("../../data/qc-temp/%dx%d,Rb=%.2f,Δ=%.2f,β=%.6f_data", nx, ny, Rb, Δ, β)
    else
        folder = @sprintf("../../data/qc-temp/%dx%d,Rb=%.2f,Δ=%.2f,β=%.6f,Ω=%.2f_%sdata", nx, ny, Rb, Δ, β, Ω, gridname)
    end
    save("$folder/data.jld", "occs", occs, "ns", ns, "order_param", order_parameter)
    open("$folder/meta_data.json","w") do f
        JSON.print(f, data_dict) # save meta data
    end
end

"""
Generates a grid of atoms like this: 
1111   1111
1111   1111
1111   1111
1111   1111



1111   1111
1111   1111
1111   1111
1111   1111
n is the number of atoms in a square side, a is the distance between atoms
and L is the distance between corners. Squares are placed to be as far from 
each other as possible.
"""
function generate_corner_grids(n, a; L=75)
    return generate_corner_grids(n, n, a, a; L=L)
end

function generate_corner_grids(nx, ny, ax, ay; L=75)
    i = 1
    atom_list = Vector{Tuple{Float64,Float64}}(undef, nx*ny*4)
    for (xcorner,ycorner) in [(0,0), (L,0), (0,L),(L,L)]
        for x = 0:nx-1
            for y = 0:ny-1
                atom_list[i] = (abs(xcorner - ax*x), abs(ycorner - ay*y))
                i += 1
            end
        end
    end
    return atom_list
end