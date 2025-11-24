"""
ga_optimize.py

simple genetic-style optimizer for ourcontroller's fuzzy and movement parameters.

it:
- initializes a small population of parameter sets (chromosomes)
- evaluates each by running kesslergame several times
- keeps the best and mutates it over a few generations
- prints the best params at the end so you can paste them into our_controller.py
"""

import random
import math
import json
from typing import Dict, List, Tuple

from kesslergame import KesslerGame, Scenario, GraphicsType
from test_controller import TestController
from our_controller import OurController


# -----------------------------
# 1. parameter space definition
# -----------------------------
# we are tuning:
#   - turn_scale: scales ship_turn (deg/s)
#   - fire_threshold: threshold on fuzzy fire output in [-1, 1]
#   - escape_close_dist / escape_med_dist: distance thresholds for thrust
#   - thrust_close / thrust_med: thrust levels
#   - danger_dist / danger_time: when to switch from fight to escape
PARAM_BOUNDS = {
    "turn_scale": (0.5, 2.0),        # lower = slower turn, higher = more aggressive
    "fire_threshold": (-0.5, 0.5),   # lower = fires more often, higher = only when sure

    # movement / escape parameters
    "escape_close_dist": (250.0, 500.0),
    "escape_med_dist":   (450.0, 900.0),
    "thrust_close":      (200.0, 320.0),
    "thrust_med":        (120.0, 260.0),

    "danger_dist":       (300.0, 700.0),
    "danger_time":       (0.5, 1.4),
}

def append_chromosome_json(chrom, fitness, path="chromosomes.jsonl"):
    entry = {
        "fitness": fitness,
        "params": chrom
    }
    with open(path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def random_chromosome() -> Dict[str, float]:
    """create a random parameter set within bounds."""
    return {
        name: random.uniform(lo, hi)
        for name, (lo, hi) in PARAM_BOUNDS.items()
    }


def mutate(
    chrom: Dict[str, float],
    mutation_rate: float = 0.3,
    mutation_scale: float = 0.2,
) -> Dict[str, float]:
    """return a mutated copy of `chrom`."""
    new = chrom.copy()
    for name, (lo, hi) in PARAM_BOUNDS.items():
        if random.random() < mutation_rate:
            span = hi - lo
            delta = random.uniform(-mutation_scale, mutation_scale) * span
            new[name] = min(hi, max(lo, new[name] + delta))
    return new


# -----------------------------
# 2. scenario and game helpers
# -----------------------------

def make_scenario() -> Scenario:
    """create a small test scenario similar to the lab setup."""
    return Scenario(
        name="ga training scenario",
        num_asteroids=8,
        ship_states=[
            {"position": (400, 400), "angle": 90, "lives": 3, "team": 1},
            {"position": (600, 400), "angle": 90, "lives": 3, "team": 2},
        ],
        map_size=(1000, 800),
        time_limit=40,
        ammo_limit_multiplier=0,
        stop_if_no_ammo=False,
    )


def make_game(no_graphics: bool = True) -> KesslerGame:
    """create a kesslergame instance. for ga we usually want no graphics."""
    if no_graphics:
        gtype = (
            GraphicsType.NoGraphics
            if hasattr(GraphicsType, "NoGraphics")
            else GraphicsType.None_
        )
    else:
        gtype = GraphicsType.Tkinter

    settings = {
        "perf_tracker": True,
        "graphics_type": gtype,
        "realtime_multiplier": 100,  # run fast if graphics are on
        "graphics_obj": None,
    }
    return KesslerGame(settings=settings)


# -----------------------------
# 3. fitness evaluation
# -----------------------------

def evaluate_chromosome(chrom: Dict[str, float], runs: int = 3) -> float:
    """
    evaluate one parameter set by running several games and averaging team 2's score.

    team mapping:
      - testcontroller  -> team 1
      - ourcontroller   -> team 2 (the one we care about)
    """
    total_score = 0.0

    for _ in range(runs):
        game = make_game(no_graphics=True)
        scenario = make_scenario()

        # instantiate controllers
        c1 = TestController()
        c2 = OurController(params=chrom)

        score, perf_data = game.run(
            scenario=scenario,
            controllers=[c1, c2],
        )

        # score.teams is a list of teamscore objects; team index 1 is ourcontroller
        my_team = score.teams[1]

        # simple fitness: reward hits and accuracy, penalize deaths
        episode_fitness = (
            100 * my_team.asteroids_hit
            + 100 * my_team.accuracy
            - 200 * my_team.deaths
        )

        total_score += episode_fitness

    return total_score / runs


# -----------------------------
# 4. genetic loop
# -----------------------------

def run_ga(
    population_size: int = 8,
    generations: int = 5,
    elite_frac: float = 0.25,
    mutation_rate: float = 0.3,
    mutation_scale: float = 0.2,
) -> Tuple[Dict[str, float], float]:
    """
    simple ga:
      - start with random population
      - evaluate fitness
      - keep the top elite_frac as parents
      - fill the rest by mutating elites
    """
    pop: List[Dict[str, float]] = [random_chromosome() for _ in range(population_size)]

    best_overall: Dict[str, float] = None
    best_fitness = -math.inf

    for gen in range(generations):
        print(f"\n=== generation {gen} ===")
        scored: List[Tuple[Dict[str, float], float]] = []

        for i, chrom in enumerate(pop):
            fit = evaluate_chromosome(chrom)
            scored.append((chrom, fit))
            print(f"  individual {i}: params={chrom}, fitness={fit:.2f}")

            if fit > best_fitness:
                best_fitness = fit
                best_overall = chrom

        # sort by fitness, highest first
        scored.sort(key=lambda x: x[1], reverse=True)
        elites_count = max(1, int(population_size * elite_frac))
        elites = [c for (c, f) in scored[:elites_count]]

        print(f"  best this gen: {scored[0][0]} -> {scored[0][1]:.2f}")
        print(f"  best overall so far: {best_overall} -> {best_fitness:.2f}")

        # build next generation from elites + mutations
        new_pop: List[Dict[str, float]] = elites.copy()
        while len(new_pop) < population_size:
            parent = random.choice(elites)
            child = mutate(parent, mutation_rate=mutation_rate, mutation_scale=mutation_scale)
            new_pop.append(child)

        pop = new_pop

    return best_overall, best_fitness


if __name__ == "__main__":
    # fixed seed so runs are repeatable
    random.seed(42)

    best_params, best_fit = run_ga(
        population_size=8,
        generations=5,
        elite_frac=0.25,
        mutation_rate=0.3,
        mutation_scale=0.2,
    )

    print("\n=== ga finished ===")
    print(f"best params: {best_params}")
    print(f"best fitness: {best_fit:.2f}")

    append_chromosome_json(best_params, best_fit)
    
    print("\nSaved into chromosomes.jsonl")
