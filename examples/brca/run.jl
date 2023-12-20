using ParProx, Printf, Statistics # load the packages
using CSV, DataFrames, CodecZlib, Mmap # packages for data reading. GZip is used to read the gzipped text file.
using Serialization
using NPZ
using Pickle
using JSON
using CUDA, Adapt
using Dates


sort_order = npzread("sort_order.npz")["data"];
survival = npzread("survival.npz");
survival_event = survival["pfi"];
survival_time = survival["pfi_time"];

# group_to_variables = [convert(Vector{Int64}, x) for x in Pickle.load("only_mrna/group_to_variables_index.pkl")];
# array_predictors = npzread("only_mrna/array_predictors.npz")["data"];
# json_file_path = "only_mrna/index_to_group.json"
# index_to_group = JSON.parsefile(json_file_path);

# json_file_path = "only_mrna/index_to_variable.json"
# index_to_variable = JSON.parsefile(json_file_path);

# json_file_path = "only_mrna/variable_to_index.json"
# variable_to_index = JSON.parsefile(json_file_path);

group_to_variables = [convert(Vector{Int64}, x) for x in Pickle.load("full/group_to_variables_index.pkl")];
array_predictors = npzread("full/array_predictors.npz")["data"];
json_file_path = "full/index_to_group.json"
index_to_group = JSON.parsefile(json_file_path);

json_file_path = "full/index_to_variable.json"
index_to_variable = JSON.parsefile(json_file_path);

json_file_path = "full/variable_to_index.json"
variable_to_index = JSON.parsefile(json_file_path);

X = array_predictors[:, 1:end-5];
X_unpen = array_predictors[:, end-4:end];

lambdas = 10 .^ (range(-8, stop=-10, length=21))
lambda = 1e-5

CUDA.device!(0)
CUDA.allowscalar(true)

validation_step = 100
max_steps = 200000
split = 10

T = Float64
A = CuArray
U = ParProx.COXUpdate(; maxiter=div(max_steps, split), step=validation_step, tol=1e-12, verbose=true)
V = ParProx.COXVariables{T}(
    adapt(A{T}, X),
    adapt(A{T}, X_unpen),
    adapt(A{T}, survival_event),
    adapt(A{T}, survival_time),
    lambda,
    group_to_variables,
    eval_obj=true
)

now_time = now()
folder_name_format = "yyyymmdd_HHMMSS"
result_dir = Dates.format(now_time, folder_name_format)
mkdir(result_dir)

for i in 1:split
    ParProx.fit!(U, V)
    open("$(result_dir)/V_$(i*validation_step).jls", "w") do file
        serialize(file, V)
    end

    _, grpmat, _ = ParProx.mapper_mat_idx(group_to_variables, length(index_to_variable));
    β_orig = vcat(grpmat * collect(V.β[1:end-5]), collect(V.β)[end-4:end]);
    variable_names_replicated = String[]
    
    for i in 1:length(group_to_variables)
        for v in group_to_variables[i]
            variable_name = index_to_variable[string(v)]
            push!(variable_names_replicated, "$variable_name")
        end
    end

    variable_names_replicated = [variable_names_replicated; "age_at_initial_pathologic_diagnosis"; "gender"; "BLACK OR AFRICAN AMERICAN"; "ASIAN"; "AMERICAN INDIAN OR ALASKA NATIVE"];
    df_variable_name = DataFrame(index = collect(keys(index_to_variable)), value = collect(values(index_to_variable)))
    df_variable_name.index = parse.(Int, df_variable_name.index);
    sort!(df_variable_name, :index);
    variable_names = [df_variable_name.value; "age_at_initial_pathologic_diagnosis"; "gender"; "BLACK OR AFRICAN AMERICAN"; "ASIAN"; "AMERICAN INDIAN OR ALASKA NATIVE"];

    open("$(result_dir)/var_$(i*validation_step).txt", "w") do file  # 수정된 부분: 파일 확장자를 .txt로 변경
        for (v, β) in zip(variable_names_replicated[V.β .!= 0], V.β[V.β .!= 0])
            println(file, "$v\t$β")
        end
    end
end