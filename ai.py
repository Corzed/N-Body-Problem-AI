import pygame
import math
import sys
import neat
import os
import pickle
import random
import textwrap
import numpy as np
import matplotlib.pyplot as plt

# Constants
WIDTH, HEIGHT = 800, 600
G = 6.67430e-11     # Gravitational constant
SCALE = 1e-7        # Scale factor: pixels per meter

# Global Variables
time_step = 3600    # Initial time step in seconds
MIN_TIME_STEP = 1e-2
MAX_TIME_STEP = 1e5
TOTAL_SIMULATION_TIME = 10000000000000000000000000000  # Total desired simulation time in seconds

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("N-Body Problem Simulation with NEAT AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# Planet Class
class Planet:
    def __init__(self, x, y, radius, mass, vx=0, vy=0):
        self.x = x        # Position in meters
        self.y = y
        self.radius = radius  # For drawing in pixels
        self.mass = mass
        self.vx = vx      # Velocity in m/s
        self.vy = vy

    def attract(self, others):
        self.ax = 0
        self.ay = 0
        for other in others:
            if other == self:
                continue
            dx = other.x - self.x
            dy = other.y - self.y
            distance = math.hypot(dx, dy)
            if distance == 0:
                continue
            force = G * self.mass * other.mass / distance**2
            fx = force * dx / distance
            fy = force * dy / distance
            self.ax += fx / self.mass
            self.ay += fy / self.mass

    def update_position(self, time_step):
        # Symplectic Euler method
        self.vx += self.ax * time_step
        self.vy += self.ay * time_step
        self.x += self.vx * time_step
        self.y += self.vy * time_step

    def draw(self, screen, center_x, center_y):
        # Convert position to screen coordinates
        x_pix = center_x + self.x * SCALE
        y_pix = center_y - self.y * SCALE
        pygame.draw.circle(screen, WHITE, (int(x_pix), int(y_pix)), self.radius)

def calculate_center_of_mass(planets):
    total_mass = sum(p.mass for p in planets)
    center_x = sum(p.x * p.mass for p in planets) / total_mass
    center_y = sum(p.y * p.mass for p in planets) / total_mass
    return center_x, center_y

def run_simulation(genome, config):
    global time_step
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    planets = []

    # Neural network input: a constant bias
    inputs = [1.0]  # Constant input

    # Neural network outputs: positions and velocities
    outputs = net.activate(inputs)

    POSITION_SCALE = 1e9    # Scale for positions
    VELOCITY_SCALE = 1e4    # Scale for velocities
    earth_mass = 5.972e24   # kg

    # Create planets based on neural network outputs
    for i in range(num_planets):
        idx = 4 * i
        x = outputs[idx] * POSITION_SCALE
        y = outputs[idx + 1] * POSITION_SCALE
        vx = outputs[idx + 2] * VELOCITY_SCALE
        vy = outputs[idx + 3] * VELOCITY_SCALE
        planet = Planet(x, y, 10, earth_mass, vx, vy)
        planets.append(planet)

    steps = 0
    max_steps = int(TOTAL_SIMULATION_TIME / time_step)
    if max_steps < 1:
        max_steps = 1

    for step in range(max_steps):
        for p in planets:
            p.attract(planets)
        for p in planets:
            p.update_position(time_step)

        # Check for collisions or escapes
        collision = False
        escape = False
        for i, p1 in enumerate(planets):
            if math.hypot(p1.x, p1.y) > 3e9:
                escape = True
                break
            for j, p2 in enumerate(planets):
                if i >= j:
                    continue
                dx = p1.x - p2.x
                dy = p1.y - p2.y
                distance = math.hypot(dx, dy)
                if distance <= (p1.radius + p2.radius) / SCALE:
                    collision = True
                    break
            if collision or escape:
                break
        if collision or escape:
            break
        steps += 1

    total_simulation_time = steps * time_step

    # Calculate initial condition penalties
    position_penalty = sum(math.hypot(p.x, p.y) for p in planets) / (num_planets * POSITION_SCALE)
    velocity_penalty = sum(math.hypot(p.vx, p.vy) for p in planets) / (num_planets * VELOCITY_SCALE)

    # Scaling factors to reduce fitness magnitude
    TIME_SCALE = 1e-6
    PENALTY_SCALE = 1e-3

    # Calculate fitness primarily based on time alive
    fitness = total_simulation_time * TIME_SCALE

    # Optionally include scaled penalties
    # Uncomment the following line if you want to include penalties
    # fitness -= (position_penalty * PENALTY_SCALE + velocity_penalty * PENALTY_SCALE)

    # Ensure fitness is non-negative
    fitness = max(fitness, 0.0)

    return fitness

def eval_genomes(genomes, config):
    global best_genome, best_fitness, generation, avg_fitness, genome_records, best_genome_id, fitness_history
    generation += 1
    total_fitness = 0
    genome_records = []
    max_fitness_in_generation = -math.inf

    for genome_id, genome in genomes:
        fitness = run_simulation(genome, config)
        genome.fitness = fitness
        total_fitness += genome.fitness
        genome_records.append((genome_id, genome, fitness))
        if genome.fitness >= best_fitness:
            best_fitness = genome.fitness
            best_genome = genome
            best_genome_id = genome_id
        if genome.fitness > max_fitness_in_generation:
            max_fitness_in_generation = genome.fitness

    avg_fitness = total_fitness / len(genomes)
    fitness_history.append((generation, avg_fitness, best_fitness))
    print(f"Generation {generation} - Average Fitness: {avg_fitness:.6f}, Max Fitness: {max_fitness_in_generation:.6f}")
    print(f"Best Fitness so far: {best_fitness:.6f}\n")

    # Visualize the best genome of this generation
    visualize_simulation(best_genome, config, generation, best_genome_id)

    # Save checkpoint every 5 generations
    if generation % 5 == 0:
        save_checkpoint(population, generation, best_genome, best_fitness, fitness_history)

def visualize_simulation(genome, config, generation, genome_id):
    global time_step
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    planets = []

    inputs = [1.0]  # Constant input
    outputs = net.activate(inputs)

    POSITION_SCALE = 1e9    # Scale for positions
    VELOCITY_SCALE = 1e4    # Scale for velocities
    earth_mass = 5.972e24   # kg

    for i in range(num_planets):
        idx = 4 * i
        x = outputs[idx] * POSITION_SCALE
        y = outputs[idx + 1] * POSITION_SCALE
        vx = outputs[idx + 2] * VELOCITY_SCALE
        vy = outputs[idx + 3] * VELOCITY_SCALE
        planet = Planet(x, y, 10, earth_mass, vx, vy)
        planets.append(planet)

    running = True
    steps = 0
    fitness = 0
    max_steps = int(TOTAL_SIMULATION_TIME / time_step)
    if max_steps < 1:
        max_steps = 1
    paused = False

    while running and steps < max_steps:
        screen.fill(BLACK)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    time_step *= 1.1  # Increase simulation speed
                elif event.key == pygame.K_DOWN:
                    time_step /= 1.1  # Decrease simulation speed

                # Limit time_step within bounds
                if time_step > MAX_TIME_STEP:
                    time_step = MAX_TIME_STEP
                elif time_step < MIN_TIME_STEP:
                    time_step = MIN_TIME_STEP

                elif event.key == pygame.K_SPACE:
                    paused = not paused

        if not paused:
            for p in planets:
                p.attract(planets)
            for p in planets:
                p.update_position(time_step)

            # Check for collisions or escapes
            collision = False
            escape = False
            for i, p1 in enumerate(planets):
                if math.hypot(p1.x, p1.y) > 3e9:
                    escape = True
                    break
                for j, p2 in enumerate(planets):
                    if i >= j:
                        continue
                    dx = p1.x - p2.x
                    dy = p1.y - p2.y
                    distance = math.hypot(dx, dy)
                    if distance <= (p1.radius + p2.radius) / SCALE:
                        collision = True
                        break
                if collision or escape:
                    break

            steps += 1

            # Scaling factors
            TIME_SCALE = 1e-6
            PENALTY_SCALE = 1e-3

            # Calculate fitness primarily based on time alive
            fitness = steps * time_step * TIME_SCALE

            # Optionally include scaled penalties
            # Uncomment the following lines if you want to include penalties
            # fitness -= (position_penalty * PENALTY_SCALE + velocity_penalty * PENALTY_SCALE)

            # Ensure fitness is non-negative
            fitness = max(fitness, 0.0)

            if collision or escape:
                running = False

        # Draw planets
        cam_x, cam_y = calculate_center_of_mass(planets)
        center_x = WIDTH / 2 - cam_x * SCALE
        center_y = HEIGHT / 2 + cam_y * SCALE
        for p in planets:
            p.draw(screen, center_x, center_y)

        # Display status
        gen_text = font.render(f"Generation: {generation}", True, GREEN)
        genome_id_text = font.render(f"Genome ID: {genome_id}", True, GREEN)
        fitness_text = font.render(f"Fitness: {fitness:.6f}", True, GREEN)
        steps_text = font.render(f"Steps: {steps}", True, GREEN)
        speed_text = font.render(f"Time Step: {time_step:.2f} s", True, GREEN)
        best_fit_text = font.render(f"Best Fitness: {best_fitness:.6f}", True, GREEN)
        avg_fit_text = font.render(f"Average Fitness: {avg_fitness:.6f}", True, GREEN)
        num_planets_text = font.render(f"Number of Planets: {num_planets}", True, GREEN)
        instructions_text = font.render("Up/Down: Adjust Speed | Space: Pause/Resume", True, GREEN)

        screen.blit(gen_text, (10, 10))
        screen.blit(genome_id_text, (10, 30))
        screen.blit(fitness_text, (10, 50))
        screen.blit(steps_text, (10, 70))
        screen.blit(speed_text, (10, 90))
        screen.blit(best_fit_text, (10, 110))
        screen.blit(avg_fit_text, (10, 130))
        screen.blit(num_planets_text, (10, 150))
        screen.blit(instructions_text, (10, 170))

        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS

def replay_genome(genome_id, config):
    # Find the genome with the given ID
    genome = None
    for gid, g, fitness in genome_records:
        if gid == genome_id:
            genome = g
            break
    if genome is None:
        print(f"Genome ID {genome_id} not found.")
        return

    # Replay the genome
    print(f"Replaying Genome ID {genome_id} with fitness {genome.fitness}")
    visualize_simulation(genome, config, generation, genome_id)

def save_checkpoint(population, generation, best_genome, best_fitness, fitness_history):
    checkpoint = {
        'population': population,
        'generation': generation,
        'best_genome': best_genome,
        'best_fitness': best_fitness,
        'fitness_history': fitness_history
    }
    filename = f'checkpoint_gen_{generation}.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved at generation {generation}.")

def load_checkpoint(filename):
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint

def plot_fitness(fitness_history):
    generations = [x[0] for x in fitness_history]
    avg_fitnesses = [x[1] for x in fitness_history]
    best_fitnesses = [x[2] for x in fitness_history]

    plt.figure()
    plt.plot(generations, avg_fitnesses, label='Average Fitness')
    plt.plot(generations, best_fitnesses, label='Best Fitness')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness over Generations')
    plt.legend()
    plt.show()

def run_neat(config_file, checkpoint_file=None):
    global population, best_genome, best_fitness, generation, avg_fitness, genome_records, best_genome_id, fitness_history
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    if checkpoint_file:
        # Load from checkpoint
        checkpoint = load_checkpoint(checkpoint_file)
        population = checkpoint['population']
        generation = checkpoint['generation']
        best_genome = checkpoint['best_genome']
        best_fitness = checkpoint['best_fitness']
        fitness_history = checkpoint['fitness_history']
        print(f"Loaded checkpoint from generation {generation}.")
    else:
        # Start new population
        population = neat.Population(config)
        generation = 0
        best_genome = None
        best_genome_id = None
        best_fitness = -math.inf
        fitness_history = []

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    genome_records = []

    # Run indefinitely until user stops
    try:
        while True:
            population.run(eval_genomes, n=1)
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Save the best genome
    with open('best_genome.pkl', 'wb') as f:
        pickle.dump(best_genome, f)

    # Plot fitness history
    plot_fitness(fitness_history)

    # After NEAT run, allow user to replay genomes
    while True:
        choice = input("Enter 'r' to replay a genome, 'b' to replay best genome, 'p' to plot fitness, or 'q' to quit: ")
        if choice.lower() == 'r':
            try:
                gid = int(input("Enter Genome ID to replay: "))
                replay_genome(gid, config)
            except ValueError:
                print("Invalid Genome ID.")
        elif choice.lower() == 'b':
            print(f"Replaying Best Genome with ID {best_genome_id} and fitness {best_fitness}")
            visualize_simulation(best_genome, config, generation, best_genome_id)
        elif choice.lower() == 'p':
            plot_fitness(fitness_history)
        elif choice.lower() == 'q':
            break
        else:
            print("Invalid choice.")

if __name__ == '__main__':
    # User selects the number of planets
    while True:
        try:
            num_planets = int(input("Enter the number of planets: "))
            if num_planets < 1:
                print("Number of planets must be at least 1.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    num_inputs = 1  # Constant input
    num_outputs = 4 * num_planets  # x, y, vx, vy for each planet

    # Generate NEAT configuration
    config_path = 'config-feedforward.txt'
    config_text = textwrap.dedent(f"""
    [NEAT]
    fitness_criterion     = max
    fitness_threshold     = 1000
    pop_size              = 1000
    reset_on_extinction   = False


    [DefaultGenome]
    num_inputs                      = {num_inputs}
    num_outputs                     = {num_outputs}
    num_hidden                      = 0
    initial_connection              = full_direct
    feed_forward                    = True

    # Activation functions
    activation_default              = tanh
    activation_mutate_rate          = 0.2
    activation_options              = tanh relu sigmoid

    # Aggregation functions
    aggregation_default             = sum
    aggregation_mutate_rate         = 0.0
    aggregation_options             = sum

    # Bias parameters
    bias_init_mean                  = 0.0
    bias_init_stdev                 = 1.0
    bias_max_value                  = 30.0
    bias_min_value                  = -30.0
    bias_mutate_power               = 0.5
    bias_mutate_rate                = 0.7
    bias_replace_rate               = 0.1

    # Response parameters
    response_init_mean              = 1.0
    response_init_stdev             = 0.0
    response_max_value              = 30.0
    response_min_value              = -30.0
    response_mutate_power           = 0.0
    response_mutate_rate            = 0.0
    response_replace_rate           = 0.0

    # Compatibility coefficients
    compatibility_disjoint_coefficient = 1.0
    compatibility_weight_coefficient   = 0.5

    # Connection parameters
    conn_add_prob                   = 0.5
    conn_delete_prob                = 0.2
    enabled_default                 = True
    enabled_mutate_rate             = 0.01

    # Node mutation parameters
    node_add_prob                   = 0.2
    node_delete_prob                = 0.1

    # Weight mutation parameters
    weight_init_mean                = 0.0
    weight_init_stdev               = 1.0
    weight_max_value                = 30
    weight_min_value                = -30
    weight_mutate_power             = 0.5
    weight_mutate_rate              = 0.8
    weight_replace_rate             = 0.1

    [DefaultSpeciesSet]
    compatibility_threshold         = 2.5

    [DefaultStagnation]
    species_fitness_func            = max
    max_stagnation                  = 20
    species_elitism                 = 2

    [DefaultReproduction]
    elitism                         = 2
    survival_threshold              = 0.2
    """)
    with open(config_path, 'w') as f:
        f.write(config_text)

    # Check for existing checkpoints
    checkpoint_files = sorted([f for f in os.listdir('.') if f.startswith('checkpoint_gen_')],
                              key=lambda x: int(x.split('_')[-1].split('.pkl')[0]))
    if checkpoint_files:
        print("Existing checkpoints found:")
        for i, file in enumerate(checkpoint_files):
            print(f"{i}: {file}")
        choice = input("Enter the number of the checkpoint to load, or 'n' to start a new training session: ")
        if choice.lower() == 'n':
            run_neat(config_path)
        else:
            try:
                index = int(choice)
                if 0 <= index < len(checkpoint_files):
                    checkpoint_file = checkpoint_files[index]
                    run_neat(config_path, checkpoint_file)
                else:
                    print("Invalid index. Starting a new training session.")
                    run_neat(config_path)
            except (ValueError, IndexError):
                print("Invalid choice. Starting a new training session.")
                run_neat(config_path)
    else:
        run_neat(config_path)
