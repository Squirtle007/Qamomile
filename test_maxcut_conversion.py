"""
Test script for MaxCut conversion from PyQUBO to Jijmodeling
Demonstrates the same logic with a simple example graph
"""

import jijmodeling as jm
import networkx as nx
from qamomile.core.converters.qaoa import QAOAConverter

# Try to import CUDA-Q transpiler (optional dependency)
try:
    from qamomile.cudaq.transpiler import CudaqTranspiler
    CUDAQ_AVAILABLE = True
except ImportError:
    CUDAQ_AVAILABLE = False
    print("Note: CUDA-Q not installed. Install with: pip install cuda-quantum")
    print("Continuing with Jijmodeling conversion only...\n")

# ============================================================
# Part 1: Jijmodeling Frontend (same logic as PyQUBO version)
# ============================================================

# Define placeholders
V = jm.Placeholder("V")  # number of vertices
E = jm.Placeholder("E", ndim=2)  # edges as 2D array [num_edges, 2]
W = jm.Placeholder("W", ndim=1)  # edge weights

# Define binary variables (0, 1) for each node
x = jm.BinaryVar("x", shape=(V,))

# Get number of edges and define iteration element
num_edges = E.len_at(0, latex="m")
i = jm.Element("i", belong_to=(0, num_edges))  # iterate over edge indices

# Create problem
problem = jm.Problem("MaxCut")

# Objective: maximize the sum over cut edges (multiplied by -1 for minimization)
# PyQUBO formula: -weight * (x_i + x_j - 2*x_i*x_j)
# This equals 1 when nodes are in different partitions (cut edge)
objective = jm.sum(i, (-1) * W[i] * (x[E[i, 0]] + x[E[i, 1]] - 2 * x[E[i, 0]] * x[E[i, 1]]))
problem += objective

print("=" * 60)
print("MaxCut Problem - PyQUBO to Jijmodeling Conversion")
print("=" * 60)

# ============================================================
# Create a simple test graph
# ============================================================
G = nx.Graph()
G.add_edge(0, 1, weight=1.0)
G.add_edge(1, 2, weight=2.0)
G.add_edge(2, 3, weight=1.5)
G.add_edge(3, 0, weight=1.0)
G.add_edge(0, 2, weight=0.5)  # diagonal edge

print(f"\nTest Graph:")
print(f"  Nodes: {list(G.nodes())}")
print(f"  Edges with weights:")
for i, j in G.edges():
    print(f"    ({i}, {j}): weight = {G[i][j]['weight']}")

# ============================================================
# Prepare instance data from graph G
# ============================================================
edge_list = [[i, j] for i, j in G.edges()]
weights = [G[i][j]['weight'] for i, j in G.edges()]

instance_data = {
    "V": G.number_of_nodes(),
    "E": edge_list,
    "W": weights
}

print(f"\nInstance Data:")
print(f"  V (num vertices): {instance_data['V']}")
print(f"  E (edges): {instance_data['E']}")
print(f"  W (weights): {instance_data['W']}")

# Compile the model
interpreter = jm.Interpreter(instance_data)
compiled_instance = interpreter.eval_problem(problem)

print(f"\nCompiled Instance:")
print(f"  Objective: {compiled_instance.objective}")

# ============================================================
# Part 2: Convert to CUDA-Q Hamiltonian using Qamomile
# ============================================================

# Step 1: Create QAOA converter
qaoa_converter = QAOAConverter(compiled_instance)

# Step 2: Get cost Hamiltonian (Ising encoding)
cost_hamiltonian = qaoa_converter.get_cost_hamiltonian()
print("\n" + "=" * 60)
print("Qamomile Hamiltonian (Ising encoding):")
print("=" * 60)
print(cost_hamiltonian)

# Step 3: Convert to CUDA-Q Hamiltonian (if available)
if CUDAQ_AVAILABLE:
    print("\n" + "=" * 60)
    print("CUDA-Q Hamiltonian:")
    print("=" * 60)
    transpiler = CudaqTranspiler()
    cudaq_hamiltonian = transpiler.transpile_hamiltonian(cost_hamiltonian)
    print(cudaq_hamiltonian)
else:
    print("\n" + "=" * 60)
    print("CUDA-Q Conversion (Not Available)")
    print("=" * 60)
    print("To convert to CUDA-Q Hamiltonian, install CUDA-Q:")
    print("  pip install cuda-quantum")

print("\n" + "=" * 60)
print("Conversion Complete!")
print("=" * 60)
