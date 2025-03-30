import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt

# ----------------------------
# Genetic Algorithm Functions
# ----------------------------

def calculate_total_distance(route, distance_matrix):
    total_distance = 0
    n = len(route)
    for i in range(n):
        total_distance += distance_matrix[route[i]][route[(i+1) % n]]
    return total_distance

def create_route(n):
    route = list(range(n))
    random.shuffle(route)
    return route

def initial_population(pop_size, n):
    return [create_route(n) for _ in range(pop_size)]

def tournament_selection(population, fitness, k=3):
    selected_indices = random.sample(range(len(population)), k)
    best_index = min(selected_indices, key=lambda idx: fitness[idx])
    return population[best_index]

def order_crossover(parent1, parent2):
    size = len(parent1)
    child = [None] * size
    start, end = sorted(random.sample(range(size), 2))
    # Copy the segment from parent1
    child[start:end+1] = parent1[start:end+1]
    # Fill the rest with parent2 in order skipping the cities already in child.
    pointer = 0
    for city in parent2:
        if city not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = city
    return child

def swap_mutation(route, mutation_rate):
    new_route = route.copy()
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(new_route)), 2)
        new_route[a], new_route[b] = new_route[b], new_route[a]
    return new_route

def genetic_algorithm(distance_matrix, pop_size, generations, mutation_rate=0.2, tournament_k=3, elitism_count=1):
    n = len(distance_matrix)
    population = initial_population(pop_size, n)
    
    best_per_gen = []  # To store best distance per generation
    best_routes = []   # To store best route per generation

    # Evaluate initial fitness
    fitness = [calculate_total_distance(route, distance_matrix) for route in population]
    best_index = np.argmin(fitness)
    best_route = population[best_index]
    best_distance = fitness[best_index]
    best_per_gen.append(best_distance)
    best_routes.append(best_route)

    # Main loop
    for gen in range(generations):
        new_population = []
        # Elitism: directly carry over best individuals
        sorted_indices = np.argsort(fitness)
        for idx in sorted_indices[:elitism_count]:
            new_population.append(population[idx].copy())
        
        # Create the rest of the new generation
        while len(new_population) < pop_size:
            # Select two parents using tournament selection (without replacement within each tournament)
            parent1 = tournament_selection(population, fitness, k=tournament_k)
            parent2 = parent1
            while parent2 == parent1:
                parent2 = tournament_selection(population, fitness, k=tournament_k)
            # Crossover
            child = order_crossover(parent1, parent2)
            # Mutation
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        fitness = [calculate_total_distance(route, distance_matrix) for route in population]
        best_index = np.argmin(fitness)
        best_route = population[best_index]
        best_distance = fitness[best_index]
        best_per_gen.append(best_distance)
        best_routes.append(best_route)
    
    return best_routes, best_per_gen

# ----------------------------
# Streamlit App
# ----------------------------
st.title("TSP Solver using Genetic Algorithm")

st.sidebar.header("Input Parameters")

# User input for city names
cities_input = st.sidebar.text_input("Enter city names (comma separated)", "A,B,C,D,E")
cities = [c.strip() for c in cities_input.split(",")]
n = len(cities)

# Input for distance matrix
st.sidebar.subheader("Distance Matrix")
st.sidebar.write("Enter the distance matrix with rows separated by newlines and values by commas.")
default_matrix = "\n".join([", ".join(map(str, row)) for row in 
                           [[0, 10, 8, 9, 7],
                            [10, 0, 10, 5, 6],
                            [8, 10, 0, 8, 9],
                            [9, 5, 8, 0, 6],
                            [7, 6, 9, 6, 0]]])
matrix_input = st.sidebar.text_area("Distance Matrix", default_matrix, height=200)

# Parse the distance matrix
try:
    distance_matrix = np.array([[float(num) for num in row.split(",")] 
                                for row in matrix_input.strip().split("\n")])
    if distance_matrix.shape != (n, n):
        st.sidebar.error(f"Distance matrix must be of shape ({n},{n}).")
except Exception as e:
    st.sidebar.error("Error parsing distance matrix. Please check your input format.")
    st.stop()

# Input for GA parameters
pop_size = st.sidebar.number_input("Population Size", min_value=2, value=10)
generations = st.sidebar.number_input("Number of Generations", min_value=1, value=50)
mutation_rate = st.sidebar.slider("Mutation Rate", 0.0, 1.0, 0.2)
tournament_k = st.sidebar.number_input("Tournament Size", min_value=2, value=3)
elitism_count = st.sidebar.number_input("Elitism Count", min_value=0, max_value=pop_size, value=1)

# Run the GA when button is pressed
if st.sidebar.button("Run Genetic Algorithm"):
    st.header("GA Results")
    st.write("**Cities:**", cities)
    
    best_routes, best_distances = genetic_algorithm(distance_matrix, int(pop_size), int(generations), 
                                                    mutation_rate, int(tournament_k), int(elitism_count))
    
    # Display best route and distance for each generation in a table
    results = {"Generation": list(range(generations+1)),
               "Best Distance": best_distances,
               "Best Route": [" -> ".join([cities[i] for i in route]) for route in best_routes]}
    st.table(results)
    
    # Plot improvement graph
    fig, ax = plt.subplots()
    ax.plot(range(generations+1), best_distances, marker="o", linestyle="-")
    ax.set_title("Improvement in Best Distance Over Generations")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Distance")
    st.pyplot(fig)
    
    # Display final best route
    final_best_route = best_routes[-1]
    final_best_distance = best_distances[-1]
    st.subheader("Final Best Route")
    st.write("Route:", " -> ".join([cities[i] for i in final_best_route]))
    st.write("Total Distance:", final_best_distance)
