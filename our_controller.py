from typing import Dict, Tuple
from kesslergame import KesslerController
import math
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl


class OurController(KesslerController):
    def __init__(self, params=None):
        # used for optional performance tracking
        self.eval_frames = 0

        # ga-tunable parameters (with defaults)
        if params is None:
            params = {}
        self.params = {
            # scales the defuzzified ship_turn output
            "turn_scale": params.get("turn_scale", 1.1329),
            # threshold on ship_fire in [-1, 1]; >= threshold -> fire
            "fire_threshold": params.get("fire_threshold", -0.47),

            # movement / escape tuning
            "escape_close_dist": params.get("escape_close_dist", 304.66),
            "escape_med_dist":   params.get("escape_med_dist", 677.41),
            "thrust_close":      params.get("thrust_close", 203.18),
            "thrust_med":        params.get("thrust_med", 147.84),

            # when to switch from fighting to escaping
            "danger_dist":       params.get("danger_dist", 559.95),
            "danger_time":       params.get("danger_time", 0.99),
        }

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
        heading_deg = ship_state["heading"]

        # find closest asteroid
        closest = None
        for a in game_state["asteroids"]:
            dx = sx - a["position"][0]
            dy = sy - a["position"][1]
            dist = math.sqrt(dx * dx + dy * dy)
            if (closest is None) or (dist < closest["dist"]):
                closest = {"aster": a, "dist": dist}

        a = closest["aster"]
        D = closest["dist"]

        # intercept math (same idea as scott's controller)
        ax, ay = a["position"]
        vx, vy = a["velocity"]

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

        # angle from ship to intercept point
        theta1 = math.atan2(intrcpt_y - sy, intrcpt_x - sx)

        # angle error between current heading and firing direction
        shooting_theta = theta1 - math.radians(heading_deg)
        shooting_theta = (shooting_theta + math.pi) % (2 * math.pi) - math.pi

        # --- fuzzy targeting ---
        sim = ctrl.ControlSystemSimulation(self.targeting_control, flush_after_run=1)
        sim.input["bullet_time"] = max(0.0, min(bullet_t, 0.999))
        sim.input["theta_delta"] = shooting_theta
        sim.compute()

        # fuzzy outputs for turning and firing
        base_turn = float(sim.output["ship_turn"])
        turn_rate = self.params["turn_scale"] * base_turn

        # clamp turn_rate into the game’s allowed range
        max_turn = 180.0
        turn_rate = max(-max_turn, min(max_turn, turn_rate))

        fire_scalar = float(sim.output["ship_fire"])
        fire = fire_scalar >= self.params["fire_threshold"]

        # --------------------------------------------------
        # geometry for escape logic
        # --------------------------------------------------
        ship_heading_rad = math.radians(heading_deg)

        # angle from ship to asteroid
        angle_to_asteroid = math.atan2(ay - sy, ax - sx)

        # escape direction (straight away from asteroid)
        escape_angle = angle_to_asteroid + math.pi

        # smallest signed difference between heading and escape direction
        escape_error = (escape_angle - ship_heading_rad + math.pi) % (2 * math.pi) - math.pi
        escape_error_abs = abs(escape_error)

        # approximate time until asteroid reaches ship
        rel_speed = max(asteroid_speed, 1e-3)  # avoid divide-by-zero
        time_to_ship = D / rel_speed

        # pull ga-tuned movement parameters
        escape_close_dist = self.params["escape_close_dist"]
        escape_med_dist = self.params["escape_med_dist"]
        thrust_close = self.params["thrust_close"]
        thrust_med = self.params["thrust_med"]
        danger_dist = self.params["danger_dist"]
        danger_time = self.params["danger_time"]

        # --------------------------------------------------
        # baseline escape movement (ga-tunable)
        # --------------------------------------------------
        thrust = 0.0
        if D < escape_close_dist and escape_error_abs < math.radians(60):
            # close and mostly pointed away → strong thrust
            thrust = thrust_close
        elif D < escape_med_dist and escape_error_abs < math.radians(60):
            # medium distance and mostly pointed away → smaller thrust
            thrust = thrust_med
        else:
            thrust = 0.0

        # --------------------------------------------------
        # danger check: can we kill it before it hits us?
        # --------------------------------------------------
        # if impact is soon and bullet_t is not clearly smaller, switch to hard escape
        escape_mode = False
        if D < danger_dist and time_to_ship < danger_time and bullet_t > time_to_ship:
            # hard escape: turn toward escape direction and burn
            kp = 220.0  # proportional gain for steering toward escape
            turn_rate = kp * escape_error
            turn_rate = max(-max_turn, min(max_turn, turn_rate))
            thrust = 260.0
            fire = False
            escape_mode = True

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
        ESCAPE_ANGLE = math.radians(20)

        if (
            escape_mode
            and D < CLOSE_DIST
            and time_to_ship < CRITICAL_TIME
            and escape_error_abs < ESCAPE_ANGLE
        ):
            drop_mine = True

        self.eval_frames += 1
        return thrust, turn_rate, fire, drop_mine

    @property
    def name(self) -> str:
        return "KEPPLER MY GOAT"
