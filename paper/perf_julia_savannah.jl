# Time Hydrograd.jl (the original Julia implementation) on the Savannah River cases.
#
# Usage (from paper/, Julia depot as in hpc/roar/julia_dump.sbatch):
#     julia --project=$HG/Hydrograd.jl perf_julia_savannah.jl FWD_CASE UDE_CASE OUT.json
#
# FWD_CASE: copy of examples/SWE_2D/forward_simulation/Savannah_River (nSave set to 1).
# UDE_CASE: copy of examples/SWE_2D/UDE/ManningN/Savannah_River_ks_h_Umag with two control
#           files, run_control_1.json and run_control_4.json (1 and 4 Adam iterations).
#
# Timings (s), all wall-clock times returned by Hydrograd.solve_swe_2D:
#   fwd_first_s, fwd_s     forward run (200 s simulated, Tsit5), first call (with JIT
#                          compilation) and second call;
#   ude1_first_s           UDE training with 1 iteration, first call (compilation);
#   ude1_s, ude4_s         compiled UDE training with 1 and 4 iterations;
#   grad_s = (ude4_s - ude1_s) / 3, the cost of one training iteration (one forward-mode
#                          ForwardDiff gradient of the loss plus the optimizer update).
using Hydrograd
using Dates

fwd_case, ude_case, out_file = ARGS[1], ARGS[2], ARGS[3]

# Run solve_swe_2D on one control file inside its case folder; return the wall time (s).
# solve_swe_2D always reads run_control.json from the current folder (its argument only
# sets the folder), so the requested control file is first copied to run_control.json.
function timed_run(case_dir::String, control::String)
    if control != "run_control.json"
        cp(joinpath(case_dir, control), joinpath(case_dir, "run_control.json"); force=true)
    end
    cd(case_dir) do
        t = @elapsed Hydrograd.solve_swe_2D(joinpath(case_dir, "run_control.json"))
        return t
    end
end

# 1. Forward run: compile, then time.
fwd_first = timed_run(fwd_case, "run_control.json")
fwd = timed_run(fwd_case, "run_control.json")

# 2. UDE training: 1 iteration (compile + timed), then 4 iterations.
ude1_first = timed_run(ude_case, "run_control_1.json")
ude1 = timed_run(ude_case, "run_control_1.json")
ude4 = timed_run(ude_case, "run_control_4.json")

# 3. Write the results as JSON by hand (no extra package needed).
open(out_file, "w") do io
    println(io, "{")
    println(io, "  \"date\": \"$(Dates.now())\",")
    println(io, "  \"julia_version\": \"$(VERSION)\",")
    println(io, "  \"threads\": $(Threads.nthreads()),")
    println(io, "  \"cpu\": \"$(Sys.cpu_info()[1].model)\",")
    println(io, "  \"fwd_first_s\": $fwd_first,")
    println(io, "  \"fwd_s\": $fwd,")
    println(io, "  \"ude1_first_s\": $ude1_first,")
    println(io, "  \"ude1_s\": $ude1,")
    println(io, "  \"ude4_s\": $ude4,")
    println(io, "  \"grad_s\": $((ude4 - ude1) / 3)")
    println(io, "}")
end
println(read(out_file, String))
