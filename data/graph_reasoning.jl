module GraphReasoning

using Graphs
using Random
using LinearAlgebra

export GraphProblem, GraphDataset, generate_graph_dataset,
       problem_to_tensors, adjacency_features, shortest_path_labels

struct SimpleWeightedDiGraph
    n::Int
    adj::Matrix{Float64}  # adj[i,j] > 0 means edge i→j with that weight
end

function SimpleWeightedDiGraph(n::Int)
    SimpleWeightedDiGraph(n, zeros(n, n))
end

Graphs.nv(g::SimpleWeightedDiGraph) = g.n
Graphs.ne(g::SimpleWeightedDiGraph) = count(>(0), g.adj)

"""
A single graph reasoning problem: find the shortest path
between `src` and `dst` in a weighted graph.
"""
struct GraphProblem
    graph::SimpleWeightedDiGraph
    weights::Matrix{Float64}
    src::Int
    dst::Int
    shortest_path::Vector{Int}
    shortest_dist::Float64
end

function add_edge!(g::SimpleWeightedDiGraph, i::Int, j::Int, w::Float64)
    g.adj[i, j] = w
end

function neighbors(g::SimpleWeightedDiGraph, v::Int)
    [j for j in 1:g.n if g.adj[v, j] > 0]
end

"""
Dijkstra's algorithm on our simple weighted digraph.
Returns (distances, predecessors).
"""
function dijkstra(g::SimpleWeightedDiGraph, src::Int)
    n = g.n
    dist = fill(Inf, n)
    prev = fill(0, n)
    visited = falses(n)
    dist[src] = 0.0

    for _ in 1:n
        u = 0
        min_d = Inf
        for v in 1:n
            if !visited[v] && dist[v] < min_d
                min_d = dist[v]
                u = v
            end
        end
        u == 0 && break
        visited[u] = true

        for v in neighbors(g, u)
            alt = dist[u] + g.adj[u, v]
            if alt < dist[v]
                dist[v] = alt
                prev[v] = u
            end
        end
    end
    dist, prev
end

function reconstruct_path(prev::Vector{Int}, src::Int, dst::Int)
    path = Int[]
    v = dst
    while v != 0 && v != src
        pushfirst!(path, v)
        v = prev[v]
    end
    v == src ? pushfirst!(path, src) : Int[]
    path
end

"""
Generate a random connected graph with `n` nodes and edge probability `p`.
Edge weights are drawn from Uniform(0.1, 1.0).
"""
function random_weighted_graph(rng::AbstractRNG, n::Int, p::Float64)
    g = SimpleWeightedDiGraph(n)
    for i in 1:n, j in 1:n
        i == j && continue
        if rand(rng) < p
            w = 0.1 + 0.9 * rand(rng)
            add_edge!(g, i, j, w)
        end
    end
    # Ensure connectivity: add a random spanning path
    perm = randperm(rng, n)
    for k in 1:(n-1)
        i, j = perm[k], perm[k+1]
        if g.adj[i, j] == 0.0
            add_edge!(g, i, j, 0.1 + 0.9 * rand(rng))
        end
    end
    g
end

"""
Generate a single graph problem with known shortest path.
"""
function generate_problem(rng::AbstractRNG; n_nodes::Int=10, edge_prob::Float64=0.3)
    g = random_weighted_graph(rng, n_nodes, edge_prob)
    src = rand(rng, 1:n_nodes)
    dst = rand(rng, setdiff(1:n_nodes, src))

    dist, prev = dijkstra(g, src)
    path = reconstruct_path(prev, src, dst)

    GraphProblem(g, g.adj, src, dst, path, dist[dst])
end

struct GraphDataset
    problems::Vector{GraphProblem}
end

Base.length(ds::GraphDataset) = length(ds.problems)
Base.getindex(ds::GraphDataset, i::Int) = ds.problems[i]

"""
Generate a dataset of graph reasoning problems.
"""
function generate_graph_dataset(;
    n::Int=1000,
    n_nodes_range::Tuple{Int,Int}=(8, 20),
    edge_prob::Float64=0.3,
    seed::Int=42,
)
    rng = MersenneTwister(seed)
    problems = GraphProblem[]
    for _ in 1:n
        n_nodes = rand(rng, n_nodes_range[1]:n_nodes_range[2])
        prob = generate_problem(rng; n_nodes, edge_prob)
        if !isempty(prob.shortest_path) && isfinite(prob.shortest_dist)
            push!(problems, prob)
        end
    end
    GraphDataset(problems)
end

"""
Convert a graph problem to tensor features suitable for the encoder.
Returns (node_features, adjacency, src_onehot, dst_onehot, max_nodes).
Pads to `max_n` nodes.
"""
function problem_to_tensors(prob::GraphProblem; max_n::Int=20)
    n = prob.graph.n
    adj = zeros(Float32, max_n, max_n)
    adj[1:n, 1:n] .= Float32.(prob.weights)

    src_oh = zeros(Float32, max_n)
    dst_oh = zeros(Float32, max_n)
    src_oh[prob.src] = 1f0
    dst_oh[prob.dst] = 1f0

    node_feats = zeros(Float32, max_n, 3)
    for i in 1:n
        node_feats[i, 1] = Float32(sum(prob.weights[i, :]))  # out-degree weight
        node_feats[i, 2] = Float32(sum(prob.weights[:, i]))  # in-degree weight
        node_feats[i, 3] = 1f0  # node exists mask
    end

    (; node_features=node_feats, adjacency=adj, src_onehot=src_oh, dst_onehot=dst_oh, n_nodes=n)
end

"""
Create path labels as a sequence of node indices (padded).
"""
function shortest_path_labels(prob::GraphProblem; max_len::Int=20)
    path = prob.shortest_path
    labels = zeros(Int, max_len)
    for (i, v) in enumerate(path)
        i > max_len && break
        labels[i] = v
    end
    (; path=labels, length=min(length(path), max_len))
end

end # module
