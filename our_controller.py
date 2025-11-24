from typing import Dict, Tuple
from kesslergame import KesslerController
import math
import json
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class OurController(KesslerController):
    def fetch_best_chromosome(self, path="chromosomes.jsonl"):
        best = None
        best_fitness = float("-inf")
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
            entry = json.loads(line)
            # Json loaded as dict of fitness and params dict
            fitness = entry["fitness"]
            if fitness > best_fitness:
                best_fitness = fitness
                best = entry["params"]

        return best


    def __init__(self, params=None):
        # used for optional performance tracking
        self.eval_frames = 0

        # ga-tunable parameters (with defaults)
        self.params = params
        if self.params is None:
            self.params = self.fetch_best_chromosome()
            if self.params is None:
                raise ValueError("No chromosome found in jsonl file")


        # self.params = {
        #     # scales the defuzzified ship_turn output
        #     "turn_scale": params.get("turn_scale", 1.1329),
        #     # threshold on ship_fire in [-1, 1]; >= threshold -> fire
        #     "fire_threshold": params.get("fire_threshold", -0.47),

        #     # movement / escape tuning
        #     "escape_close_dist": params.get("escape_close_dist", 304.66),
        #     "escape_med_dist":   params.get("escape_med_dist", 677.41),
        #     "thrust_close":      params.get("thrust_close", 263.18),
        #     "thrust_med":        params.get("thrust_med", 147.84),

        #     # when to switch from fighting to escaping
        #     "danger_dist":       params.get("danger_dist", 100.0),
        #     "danger_time":       params.get("danger_time", 0.92),
        # }

        # fuzzy targeting system
        # inputs:
        #   bullet_time : predicted time (s) for bullet to reach intercept
        #   theta_delta : angle error (rad) between ship heading and firing azimuth
        # outputs:
        #   ship_turn   : turn rate (deg/s)
        #   ship_fire   : scalar in [-1, 1] -> thresholded to bool

        bullet_time = ctrl.Antecedent(np.arange(0, 1.0, 0.002), "bullet_time")
        theta_delta = ctrl.Antecedent(
            np.arange(-1 * math.pi / 30, math.pi / 30, 0.001),
            "theta_delta",
        )
        ship_turn = ctrl.Consequent(np.arange(-180, 180, 1), "ship_turn")
        ship_fire = ctrl.Consequent(np.arange(-1, 1, 0.1), "ship_fire")

        # bullet_time memberships
        bullet_time["S"] = fuzz.trimf(bullet_time.universe, [0, 0, 0.05])
        bullet_time["M"] = fuzz.trimf(bullet_time.universe, [0, 0.05, 0.1])
        bullet_time["L"] = fuzz.smf(bullet_time.universe, 0.0, 0.1)

        # theta_delta memberships
        theta_delta["NL"] = fuzz.zmf(theta_delta.universe, -1 * math.pi / 30, -2 * math.pi / 90)
        theta_delta["NM"] = fuzz.trimf(
            theta_delta.universe,
            [-1 * math.pi / 30, -2 * math.pi / 90, -1 * math.pi / 90],
        )
        theta_delta["NS"] = fuzz.trimf(
            theta_delta.universe,
            [-2 * math.pi / 90, -1 * math.pi / 90, math.pi / 90],
        )
        theta_delta["PS"] = fuzz.trimf(
            theta_delta.universe,
            [-1 * math.pi / 90, math.pi / 90, 2 * math.pi / 90],
        )
        theta_delta["PM"] = fuzz.trimf(
            theta_delta.universe,
            [math.pi / 90, 2 * math.pi / 90, math.pi / 30],
        )
        theta_delta["PL"] = fuzz.smf(theta_delta.universe, 2 * math.pi / 90, math.pi / 30)

        # ship_turn memberships
        ship_turn["NL"] = fuzz.trimf(ship_turn.universe, [-180, -180, -120])
        ship_turn["NM"] = fuzz.trimf(ship_turn.universe, [-180, -120, -60])
        ship_turn["NS"] = fuzz.trimf(ship_turn.universe, [-120, -60, 60])
        ship_turn["PS"] = fuzz.trimf(ship_turn.universe, [-60, 60, 120])
        ship_turn["PM"] = fuzz.trimf(ship_turn.universe, [60, 120, 180])
        ship_turn["PL"] = fuzz.trimf(ship_turn.universe, [120, 180, 180])

        # ship_fire memberships
        ship_fire["N"] = fuzz.trimf(ship_fire.universe, [-1, -1, 0.0])
        ship_fire["Y"] = fuzz.trimf(ship_fire.universe, [0.0, 1, 1])

        # fuzzy rules (based on scott dick's example controller)
        rules = [
            ctrl.Rule(bullet_time["L"] & theta_delta["NL"], (ship_turn["NL"], ship_fire["N"])),
            ctrl.Rule(bullet_time["L"] & theta_delta["NM"], (ship_turn["NM"], ship_fire["N"])),
            ctrl.Rule(bullet_time["L"] & theta_delta["NS"], (ship_turn["NS"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["L"] & theta_delta["PS"], (ship_turn["PS"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["L"] & theta_delta["PM"], (ship_turn["PM"], ship_fire["N"])),
            ctrl.Rule(bullet_time["L"] & theta_delta["PL"], (ship_turn["PL"], ship_fire["N"])),

            ctrl.Rule(bullet_time["M"] & theta_delta["NL"], (ship_turn["NL"], ship_fire["N"])),
            ctrl.Rule(bullet_time["M"] & theta_delta["NM"], (ship_turn["NM"], ship_fire["N"])),
            ctrl.Rule(bullet_time["M"] & theta_delta["NS"], (ship_turn["NS"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["M"] & theta_delta["PS"], (ship_turn["PS"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["M"] & theta_delta["PM"], (ship_turn["PM"], ship_fire["N"])),
            ctrl.Rule(bullet_time["M"] & theta_delta["PL"], (ship_turn["PL"], ship_fire["N"])),

            ctrl.Rule(bullet_time["S"] & theta_delta["NL"], (ship_turn["NL"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["S"] & theta_delta["NM"], (ship_turn["NM"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["S"] & theta_delta["NS"], (ship_turn["NS"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["S"] & theta_delta["PS"], (ship_turn["PS"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["S"] & theta_delta["PM"], (ship_turn["PM"], ship_fire["Y"])),
            ctrl.Rule(bullet_time["S"] & theta_delta["PL"], (ship_turn["PL"], ship_fire["Y"])),
        ]

        self.targeting_control = ctrl.ControlSystem()
        for r in rules:
            self.targeting_control.addrule(r)

    # -------------------------------------------------------
    # main action loop
    # -------------------------------------------------------
    def actions(self, ship_state: Dict, game_state: Dict) -> Tuple[float, float, bool, bool]:
        # if no asteroids, just sit still
        if not game_state["asteroids"]:
            self.eval_frames += 1
            return 0.0, 0.0, False, False

        # ship position and heading
        sx, sy = ship_state["position"]
        svx, svy = ship_state["velocity"]
        heading_deg = ship_state["heading"]

        # find the closest asteroid
        danger_target = None
        danger_dist = float("inf")
        min_dist = float("inf")
        min_collision_time = float("inf")
        sspeed = math.hypot(svx, svy)

        for a in game_state["asteroids"]:
            ax, ay = a["position"]
            vx, vy = a["velocity"]

            dx = sx - ax
            dy = sy - ay

            dist = math.hypot(dx, dy)

            if  dist < min_dist:
                min_dist = dist
                shoot_target = a

            rel_vx = vx - ship_state["velocity"][0]
            rel_vy = vy - ship_state["velocity"][1]

            rel_speed = max(math.hypot(rel_vx, rel_vy), 1e-7)
            collision_time  = dist/rel_speed

            if collision_time < min_collision_time:
                min_collision_time = collision_time
                danger_target = a
                danger_dist = dist

        ax, ay = shoot_target["position"]
        vx, vy = shoot_target["velocity"]
        D = min_dist

        asteroid_ship_x = sx - ax
        asteroid_ship_y = sy - ay
        asteroid_ship_theta = math.atan2(asteroid_ship_y, asteroid_ship_x)

        asteroid_direction = math.atan2(vy, vx)

        theta2 = asteroid_ship_theta - asteroid_direction
        cos_theta2 = math.cos(theta2)

        asteroid_speed = math.sqrt(vx * vx + vy * vy)
        bullet_speed = 800.0

        det = (
            (-2 * D * asteroid_speed * cos_theta2) ** 2
            - 4 * (asteroid_speed**2 - bullet_speed**2) * (D**2)
        )

        if det < 0:
            # no real intercept, just pick a small positive time
            bullet_t = 0.1
        else:
            sqrt_det = math.sqrt(det)
            denom = 2 * (asteroid_speed**2 - bullet_speed**2)
            t1 = (2 * D * asteroid_speed * cos_theta2 + sqrt_det) / denom
            t2 = (2 * D * asteroid_speed * cos_theta2 - sqrt_det) / denom
            candidates = [t for t in (t1, t2) if t >= 0]
            bullet_t = min(candidates) if candidates else 0.1

        # predicted intercept point
        intrcpt_x = ax + vx * (bullet_t + 1 / 30)
        intrcpt_y = ay + vy * (bullet_t + 1 / 30)

        # angle error between current heading and firing direction
        theta1 = math.atan2(intrcpt_y - sy, intrcpt_x - sx)
        shooting_theta = (theta1 - math.radians(heading_deg) + math.pi) % (2 * math.pi) - math.pi

        # --- fuzzy targeting ---
        sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        sim.input["bullet_time"] = max(0.0, min(bullet_t, 0.999))
        sim.input["theta_delta"] = shooting_theta
        sim.compute()

        # fuzzy outputs for turning and firing
        base_turn = float(sim.output["ship_turn"])
        thrust = 0.0
        turn_rate = max(-180, min(180, base_turn * self.params["turn_scale"]))
        fire = float(sim.output["ship_fire"]) >= self.params["fire_threshold"]

        # --------------------------------------------------
        # geometry for escape logic
        # --------------------------------------------------
        dx2 = sx - danger_target["position"][0]
        dy2 = sy - danger_target["position"][1]

        # angle from ship to dangerous asteroid
        angle_to_danger = math.atan2(dy2, dx2)

        # escape direction (straight away from dangerous asteroid)
        escape_angle = angle_to_danger

        # smallest signed difference between heading and escape direction
        ship_heading_rad = math.radians(heading_deg)
        escape_error = (escape_angle - ship_heading_rad + math.pi) % (2 * math.pi) - math.pi
        escape_error_abs = abs(escape_error)

        time_to_ship = min_collision_time

        # --------------------------------------------------
        # danger check: can we kill it before it hits us?
        # --------------------------------------------------
        # if impact is soon and bullet_t is not clearly smaller, switch to hard escape
        escape_mode = False
        kp = 1000  # proportional gain for steering toward escape
        max_turn = 180.0
        anglethresh = math.radians(270)
        if danger_dist < self.params["danger_dist"] or time_to_ship < self.params["danger_time"]:
            # hard escape: turn toward escape direction and burn
            turn_rate = max(-max_turn, min(max_turn, kp * escape_error))

            if abs(escape_error) < anglethresh:
                thrust = self.params["thrust_close"]
            else:
                thrust = 0.0
            fire = False
            escape_mode = True
        elif danger_dist < self.params["escape_med_dist"] or time_to_ship > self.params["danger_time"]:
            turn_rate = max(-180, min(180, base_turn * self.params["turn_scale"]))

            if sspeed > (self.params["thrust_med"]/3):
                thrust = -self.params["thrust_close"]
            else:
                thrust = 0.0
            fire = float(sim.output["ship_fire"]) >= self.params["fire_threshold"]

        # --------------------------------------------------
        # safe last-ditch mine logic
        # --------------------------------------------------
        # mines damage the ship too, so only drop when:
        #   - asteroid is very close
        #   - impact is soon
        #   - we are already escaping and pointed almost exactly away
        drop_mine = False
        CLOSE_DIST = 230.0
        CRITICAL_TIME = 0.5
        ESCAPE_ANGLE = math.radians(180)

        if (
            escape_mode
            and D < CLOSE_DIST
            and time_to_ship < CRITICAL_TIME
            and escape_error_abs < ESCAPE_ANGLE
        ):
            drop_mine = False

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "EULER MY GOAT"