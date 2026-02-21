# Remove Project and Manifests from the notebooks to force updates.

notebooks_dir = joinpath(dirname(@__DIR__), "posts", "notebooks")
notebook_paths = readdir(notebooks_dir; join=true)
filter!(endswith(".jl"), notebook_paths)

start_line = "# ╔═╡ 00000000-0000-0000-0000-000000000001"
end_line = "# ╔═╡ Cell order:"

anything = raw"[\s\S]*"
rx = Regex("($start_line\n$anything)$end_line\n")

for path in notebook_paths
    @info "Starting to remove the Project and Manifest in $path"

    text = read(path, String)
    m = match(rx, text)
    if isnothing(m) || length(m) == 0
        @info "  Found no matches in $path. Skipping"
        continue
    else
        @info "  Removing manifest for $path."
    end

    matched_text = string(m[1])
    replacers = [
        matched_text => '\n',
        # Project in cell order.
        "# ╟─00000000-0000-0000-0000-000000000001" => "",
        # Manifest in cell order.
        "# ╟─00000000-0000-0000-0000-000000000002" => ""
    ]

    updated_text = replace(text, replacers...)
    write(path, updated_text)
end
