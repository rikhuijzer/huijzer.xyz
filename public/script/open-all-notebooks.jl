using Pluto: Pluto

session = Pluto.ServerSession()

notebooks_dir = joinpath(dirname(@__DIR__), "posts", "notebooks")
notebook_paths = readdir(notebooks_dir; join=true)
filter!(endswith(".jl"), notebook_paths)

server_task = @async Pluto.run(session)

# Give the async task time to start.
sleep(10)

for path in notebook_paths
    @info "Opening notebook at $path"
    Pluto.SessionActions.open(session, path)
end

wait(server_task)
