import jijmodeling as jm
from qamomile.core.converters.qaoa import QAOAConverter
from qamomile.cudaq.transpiler import CudaqTranspiler

# ============================================================
# Part 1: Jijmodeling Frontend (same logic as PyQUBO version)
# ============================================================

# Define placeholders
V = jm.Placeholder("V")  # number of vertices
E = jm.Placeholder("E", ndim=2)  # edges as 2D array [num_edges, 2]
W = jm.Placeholder("W", ndim=1)  # edge weights

# Define binary variables (0, 1) for each node
x = jm.BinaryVar("x", shape=(V,))

# Define iteration elements
e = jm.Element("e", belong_to=E)  # iterate over edges

# Create problem
problem = jm.Problem("MaxCut")

# Objective: maximize the sum over cut edges (multiplied by -1 for minimization)
# Original PyQUBO formula: -weight * (x_i + x_j - 2*x_i*x_j)
# x_i + x_j - 2*x_i*x_j equals 1 when nodes are in different partitions (cut edge)
i = jm.Element("i", belong_to=(0, jm.size(E, 0)))  # edge index
objective = jm.sum(i, (-1) * W[i] * (x[E[i, 0]] + x[E[i, 1]] - 2 * x[E[i, 0]] * x[E[i, 1]]))
problem += objective

# ============================================================
# Prepare instance data from graph G (assuming G is a NetworkX graph)
# ============================================================
# Example usage (uncomment and replace with your actual graph):
"""
import networkx as nx

# Assuming G is your graph with nodes and weighted edges
G = nx.Graph()
G.add_edge(0, 1, weight=1.0)
G.add_edge(1, 2, weight=2.0)
G.add_edge(0, 2, weight=1.5)

# Convert graph to instance data
edge_list = [[i, j] for i, j in G.edges()]
weights = [G[i][j]['weight'] for i, j in G.edges()]

instance_data = {
    "V": G.number_of_nodes(),
    "E": edge_list,
    "W": weights
}

# Compile the model
interpreter = jm.Interpreter(instance_data)
compiled_instance = interpreter.eval_problem(problem)

# ============================================================
# Part 2: Convert to CUDA-Q Hamiltonian using Qamomile
# ============================================================

# Step 1: Create QAOA converter
qaoa_converter = QAOAConverter(compiled_instance)

# Step 2: Get cost Hamiltonian (Ising encoding)
cost_hamiltonian = qaoa_converter.get_cost_hamiltonian()

# Step 3: Convert to CUDA-Q Hamiltonian
transpiler = CudaqTranspiler()
cudaq_hamiltonian = transpiler.transpile_hamiltonian(cost_hamiltonian)

print("CUDA-Q Hamiltonian:")
print(cudaq_hamiltonian)
"""
