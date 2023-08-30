using Bloqade
using Yao: mat, ArrayReg
using LinearAlgebra
using Measurements
using Measurements: value, uncertainty
using Statistics
using Distributed

using BloqadeQMC
using Random
using Plots
using Plots: bar

using BinningAnalysis
using JLD
using JSON
using ArgParse
using Printf

include("simulation.jl")

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--split"
            help = "Input two parameters. Second is number of ways to divide data into, 
                    and the first number indexes them (starting at 0)"
            nargs = 2
            arg_type = Int
    end

    return parse_args(s)
end


function main(ARGS)
    println("BEGINNING OF JOB")
    println("----------------")
    
    parsed = parse_commandline()
    (split_number, total_split) = parsed["split"]
    @show ARGS

    M = 50;
    EQ_MCS = 2_000; # number of iterations to equilibrate
    MCS = 10_000_000; # number of samples
    seed = 3215;

    @show M, EQ_MCS, MCS, seed
    
    Ω_list = [1]
    Δ_per_Ω_list = LinRange(1.5,2.7,12)
    Rb_per_a_list = [1.3]
    nx_list = [4]
    βΩ_list = LinRange(2, 3, 10)

    for (i, (nx, Ω, Δ_per_Ω, Rb_per_a, βΩ)) in enumerate(Iterators.product(nx_list, Ω_list, Δ_per_Ω_list,Rb_per_a_list, βΩ_list))
        if i % total_split != split_number
            continue
        end
        println()
        println("STARTING NEW ITERATION: $i")
        println("----------------------")
        
        data_dict = Dict()
        
        data_dict["MCS"] = MCS
        data_dict["seed"] = seed
        data_dict["EQ_MCS"] = EQ_MCS
        data_dict["M"] = M

        @show Rb_per_a
        @show Δ_per_Ω
        @show Ω
        @show βΩ
        data_dict["Rb_per_a"] = Rb_per_a
        data_dict["Δ_per_Ω"] = Δ_per_Ω
        data_dict["Ω"] = Ω
        data_dict["βΩ"] = βΩ

        C = 2π * 862690;
        Rb = (C/Ω)^(1/6);
        a = Rb/Rb_per_a;
        Δ = Ω*Δ_per_Ω;
        β = βΩ/Ω;

        data_dict["C"] = C
        data_dict["Rb"] = Rb
        data_dict["a"] = a
        data_dict["Δ"] = Δ
        data_dict["β"] = β

        @show C
        @show Rb
        @show a
        @show Δ

        ny = nx;
        data_dict["nx"] = nx
        data_dict["ny"] = ny

        # data_dict["lattice"] = "CornerSquareLattice"
        # atoms = AtomList(generate_corner_grids(nx, a))
        data_dict["lattice"] = "SquareLattice"
        atoms = generate_sites(SquareLattice(), nx, ny, scale = a)

        @show nx, ny
        
        t = @elapsed begin
            occs, energy, ns = run_qmc(atoms, Ω, Δ, β; M=M, EQ_MCS=EQ_MCS, MCS=MCS, seed=seed)       
            order_parameter = compute_order_parameter(occs)
            data_dict["energy"] = energy.val
            data_dict["energy_error"] = energy.err
            save_data(occs,ns, order_parameter, data_dict; gridname="SquareLattice") 
        end
        @show t
    end
end

main(ARGS)
