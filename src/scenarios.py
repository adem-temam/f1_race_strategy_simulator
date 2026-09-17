"""Phase 6: Scenario Analysis & Counterfactual What-If Engine.

Evaluates how optimal strategy choices, stint durations, pit windows, and race
times shift when race conditions deviate from baseline nominal parameters.
Addresses the 6 core research questions from project.md:
1. When does an additional pit stop become beneficial?
2. How does tyre degradation affect optimal stint length?
3. How sensitive is the optimal strategy to pit-stop duration?
4. How much performance advantage is required for a softer compound to justify an additional stop?
5. How does uncertainty in lap time affect strategy selection?
6. How accurately does the model reproduce historical decisions across scenarios?
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from src.config import create_race_model, get_circuit_compounds
from src.fuel import FuelModel
from src.model import CircuitConfig, CircuitTrafficConfig, RaceModel
from src.montecarlo import MonteCarloResult, simulate_monte_carlo
from src.optimization import OptimizationObjective, StrategyOptimizer
from src.pitstop import PitStopModel
from src.simulation import simulate_race
from src.stochastic import StochasticParameters
from src.strategies import Strategy
from src.tyres import TyreCompound


class ScenarioCategory(str, Enum):
    """Categorization of what-if scenarios."""

    PIT_STOP = "pit_stop"
    TYRE_DEGRADATION = "tyre_degradation"
    COMPOUND_PERFORMANCE = "compound_performance"
    RACE_DISTANCE = "race_distance"
    ENVIRONMENTAL = "environmental"
    REGULATORY = "regulatory"
    SAFETY_CAR = "safety_car"
    UNCERTAINTY = "uncertainty"


@dataclass
class ScenarioPerturbation:
    """Quantitative perturbation parameters to apply to a baseline race model.

    Attributes:
        pit_loss_delta: Additive change to pit transit loss in seconds (e.g. +2.0s for slow stop).
        stationary_stop_delta: Additive change to stationary service time in seconds.
        deg_multiplier_factor: Multiplicative scaling on tyre degradation (e.g. 1.20 for +20% deg).
        soft_delta_adjustment: Additive change to Soft tyre pace advantage (e.g. -0.4s faster).
        hard_delta_adjustment: Additive change to Hard tyre pace offset (e.g. +0.3s slower).
        fuel_penalty_multiplier: Multiplier on fuel weight sensitivity.
        total_laps_delta: Additive change to total race distance (e.g. -38 for Sprint, +10 for extended).
        total_laps_override: Absolute override of total race laps.
        traffic_penalty_multiplier: Scaling factor on dirty air lap time and tyre wear penalties.
        safety_car_lap: Optional lap number where Safety Car neutralizes pit loss to SC delta (~11s).
        safety_car_pit_loss: Net pit loss under Safety Car conditions (default ~11.0s).
        lap_noise_std_multiplier: Scaling factor on stochastic lap time standard deviation.
    """

    pit_loss_delta: float = 0.0
    stationary_stop_delta: float = 0.0
    deg_multiplier_factor: float = 1.0
    soft_delta_adjustment: float = 0.0
    hard_delta_adjustment: float = 0.0
    fuel_penalty_multiplier: float = 1.0
    total_laps_delta: int = 0
    total_laps_override: Optional[int] = None
    traffic_penalty_multiplier: float = 1.0
    safety_car_lap: Optional[int] = None
    safety_car_pit_loss: float = 11.0
    lap_noise_std_multiplier: float = 1.0


@dataclass
class Scenario:
    """Represents a named what-if scenario with metadata and perturbation rules.

    Attributes:
        id: Unique machine identifier (e.g. 'slow_pit_stop').
        name: Human-readable title (e.g. 'Slow Pit Stop (+2.5s Delay)').
        category: ScenarioCategory enum.
        description: Detailed explanation of the scenario hypothesis and rationale.
        research_question: Associated research question from project specifications.
        perturbation: Quantitative perturbation values.
    """

    id: str
    name: str
    category: ScenarioCategory
    description: str
    research_question: str
    perturbation: ScenarioPerturbation = field(default_factory=ScenarioPerturbation)


@dataclass
class ScenarioStrategyOutcome:
    """Evaluation metrics for a specific strategy under scenario conditions.

    Attributes:
        strategy_name: Name of the strategy.
        stops: Number of pit stops.
        baseline_time: Total race time under nominal conditions (s).
        scenario_time: Total race time under scenario conditions (s).
        time_impact: scenario_time - baseline_time (s).
        gap_to_winner: Gap to the scenario-optimal strategy (s).
    """

    strategy_name: str
    stops: int
    baseline_time: float
    scenario_time: float
    time_impact: float
    gap_to_winner: float

    @property
    def formatted_scenario_time(self) -> str:
        mins = int(self.scenario_time // 60)
        secs = self.scenario_time % 60
        return f"{mins}:{secs:05.2f}"

    @property
    def formatted_impact(self) -> str:
        sign = "+" if self.time_impact >= 0 else ""
        return f"{sign}{self.time_impact:.2f}s"


@dataclass
class ScenarioResult:
    """Comprehensive output of a scenario evaluation against baseline conditions.

    Attributes:
        scenario: Evaluated Scenario.
        circuit_name: Target circuit name.
        baseline_model: Nominal unperturbed RaceModel.
        perturbed_model: Perturbed RaceModel.
        baseline_optimal_strategy: Best strategy under nominal conditions.
        baseline_optimal_time: Best time under nominal conditions.
        scenario_optimal_strategy: Best strategy discovered under scenario conditions.
        scenario_optimal_time: Best time under scenario conditions.
        strategy_pivoted: True if the winning strategy changed under this scenario.
        stops_pivoted: True if the optimal stop count changed (e.g. 1-stop to 2-stop).
        strategic_regret: Time lost if the team stubbornly stuck with baseline strategy:
            T(S*_base; theta') - T(S*_scenario; theta').
        strategy_outcomes: List of evaluation outcomes for candidate strategies.
        summary_insight: Causal engineering summary of the strategic shift.
    """

    scenario: Scenario
    circuit_name: str
    baseline_model: RaceModel
    perturbed_model: RaceModel
    baseline_optimal_strategy: Strategy
    baseline_optimal_time: float
    scenario_optimal_strategy: Strategy
    scenario_optimal_time: float
    strategy_pivoted: bool
    stops_pivoted: bool
    strategic_regret: float
    strategy_outcomes: list[ScenarioStrategyOutcome]
    summary_insight: str

    @property
    def formatted_regret(self) -> str:
        return f"{self.strategic_regret:.2f}s"

    @property
    def formatted_baseline_time(self) -> str:
        mins = int(self.baseline_optimal_time // 60)
        secs = self.baseline_optimal_time % 60
        return f"{mins}:{secs:05.2f}"

    @property
    def formatted_scenario_time(self) -> str:
        mins = int(self.scenario_optimal_time // 60)
        secs = self.scenario_optimal_time % 60
        return f"{mins}:{secs:05.2f}"


# -------------------------------------------------------------------------
# Standard Preset Scenario Catalog
# -------------------------------------------------------------------------

PRESET_SCENARIOS: dict[str, Scenario] = {
    "slow_stop": Scenario(
        id="slow_stop",
        name="Slow Pit Stop (+2.0s Delay)",
        category=ScenarioCategory.PIT_STOP,
        description="Simulates pit-crew delay or stationary traffic hold adding +2.0s per pit stop.",
        research_question="How sensitive is the optimal strategy to pit-stop duration?",
        perturbation=ScenarioPerturbation(pit_loss_delta=2.0),
    ),
    "botched_stop": Scenario(
        id="botched_stop",
        name="Botched Pit Stop (+4.5s Wheel Nut Hang)",
        category=ScenarioCategory.PIT_STOP,
        description="Simulates a major pit stop error (cross-threaded nut) adding +4.5s pit loss.",
        research_question="When does an execution error destroy a multi-stop advantage?",
        perturbation=ScenarioPerturbation(pit_loss_delta=4.5),
    ),
    "high_deg": Scenario(
        id="high_deg",
        name="Extreme Tyre Degradation (+25%)",
        category=ScenarioCategory.TYRE_DEGRADATION,
        description="Simulates hot track temperatures or severe tyre overheating increasing degradation by 25%.",
        research_question="At what degradation rate does an additional stop become mandatory?",
        perturbation=ScenarioPerturbation(deg_multiplier_factor=1.25),
    ),
    "low_deg": Scenario(
        id="low_deg",
        name="Low Tyre Degradation (-25%)",
        category=ScenarioCategory.TYRE_DEGRADATION,
        description="Simulates overcast cool weather and heavy track rubbering reducing wear by 25%.",
        research_question="When does a 1-stop strategy become unassailable?",
        perturbation=ScenarioPerturbation(deg_multiplier_factor=0.75),
    ),
    "soft_pace_boost": Scenario(
        id="soft_pace_boost",
        name="Soft Tyre Advantage (-0.4s Faster)",
        category=ScenarioCategory.COMPOUND_PERFORMANCE,
        description="Simulates Pirelli C3/C4 Soft compound offering an additional -0.4s/lap pace advantage.",
        research_question="How much pace advantage is required for a softer compound to justify an additional stop?",
        perturbation=ScenarioPerturbation(soft_delta_adjustment=-0.40),
    ),
    "hard_pace_deficit": Scenario(
        id="hard_pace_deficit",
        name="Hard Tyre Mismatch (+0.4s Slower)",
        category=ScenarioCategory.COMPOUND_PERFORMANCE,
        description="Simulates Hard compound failing to reach its thermal operating window, losing +0.4s/lap.",
        research_question="How does compound working window mismatch alter strategy viability?",
        perturbation=ScenarioPerturbation(hard_delta_adjustment=0.40),
    ),
    "sprint_distance": Scenario(
        id="sprint_distance",
        name="Sprint Race Distance (19 Laps)",
        category=ScenarioCategory.RACE_DISTANCE,
        description="Simulates a 100km Saturday Sprint race distance (~19 laps) with zero mandatory stops.",
        research_question="How does race distance scale optimal stint length?",
        perturbation=ScenarioPerturbation(total_laps_override=19),
    ),
    "extended_distance": Scenario(
        id="extended_distance",
        name="Extended Distance (+10 Laps)",
        category=ScenarioCategory.RACE_DISTANCE,
        description="Simulates red-flag restart extension or high-lap circuits (+10 laps distance).",
        research_question="How does stint length scaling force strategy transitions on long distances?",
        perturbation=ScenarioPerturbation(total_laps_delta=10),
    ),
    "safety_car_lap18": Scenario(
        id="safety_car_lap18",
        name="Safety Car Neutralization (Lap 18)",
        category=ScenarioCategory.SAFETY_CAR,
        description="Simulates a full Safety Car deployed on Lap 18, reducing pit transit loss from ~22s to 11s.",
        research_question="How much strategic advantage is unlocked by an opportunistic pit stop under Safety Car?",
        perturbation=ScenarioPerturbation(safety_car_lap=18, safety_car_pit_loss=11.0),
    ),
    "high_lap_variance": Scenario(
        id="high_lap_variance",
        name="High Lap Pace Noise (σ_lap = 0.8s)",
        category=ScenarioCategory.UNCERTAINTY,
        description="Simulates tricky mixed conditions, traffic congestion, or unpredictable driver variance (2.5x noise).",
        research_question="How does uncertainty in lap time affect strategy selection and risk?",
        perturbation=ScenarioPerturbation(lap_noise_std_multiplier=2.5),
    ),
    "regulations_2026": Scenario(
        id="regulations_2026",
        name="2026 Technical Regulations Era",
        category=ScenarioCategory.REGULATORY,
        description="Simulates 2026 FIA regulations: 75kg fuel capacity and active aero halving dirty air wake penalty.",
        research_question="How will future power unit and aerodynamic regulations alter race strategy?",
        perturbation=ScenarioPerturbation(
            fuel_penalty_multiplier=0.75,
            traffic_penalty_multiplier=0.50,
        ),
    ),
}


# -------------------------------------------------------------------------
# Scenario Analysis Engine
# -------------------------------------------------------------------------

class ScenarioEngine:
    """Engine for executing and comparing counterfactual race strategy scenarios."""

    def __init__(self, circuit_key: str = "bahrain"):
        """Initialize scenario engine for a specific Grand Prix circuit.

        Args:
            circuit_key: Circuit preset key ('bahrain', 'barcelona', 'monza').
        """
        self.circuit_key = circuit_key.lower()
        self.baseline_model = create_race_model(self.circuit_key)
        self.compounds = get_circuit_compounds(self.circuit_key)

    def create_perturbed_model(
        self, perturbation: ScenarioPerturbation
    ) -> tuple[RaceModel, dict[str, TyreCompound]]:
        """Construct a new RaceModel and compound dict with applied perturbations.

        Args:
            perturbation: ScenarioPerturbation dataclass.

        Returns:
            Tuple of (perturbed_race_model, perturbed_compounds_dict).
        """
        base_cfg = self.baseline_model.circuit
        new_laps = base_cfg.total_laps
        if perturbation.total_laps_override is not None:
            new_laps = max(5, perturbation.total_laps_override)
        elif perturbation.total_laps_delta != 0:
            new_laps = max(5, base_cfg.total_laps + perturbation.total_laps_delta)

        # Traffic adjustments
        base_traffic = base_cfg.traffic_config
        new_traffic = CircuitTrafficConfig(
            field_spread_rate=base_traffic.field_spread_rate,
            dirty_air_time_penalty=base_traffic.dirty_air_time_penalty * perturbation.traffic_penalty_multiplier,
            dirty_air_deg_multiplier=1.0 + (base_traffic.dirty_air_deg_multiplier - 1.0) * perturbation.traffic_penalty_multiplier,
            overtake_difficulty=base_traffic.overtake_difficulty,
        )

        new_circuit = CircuitConfig(
            name=f"{base_cfg.name} (Scenario Perturbed)",
            total_laps=new_laps,
            base_lap_time=base_cfg.base_lap_time,
            track_evolution_total=base_cfg.track_evolution_total,
            tyre_degradation_multiplier=base_cfg.tyre_degradation_multiplier * perturbation.deg_multiplier_factor,
            traffic_config=new_traffic,
        )

        # Pit Stop adjustments
        base_pit = self.baseline_model.pitstop_model
        new_pit_loss = max(base_pit.stationary_time + 1.0, base_pit.pit_loss + perturbation.pit_loss_delta)
        new_stationary = min(new_pit_loss - 0.5, max(1.5, base_pit.stationary_time + perturbation.stationary_stop_delta))
        new_pit = PitStopModel(
            pit_loss=new_pit_loss,
            stationary_time=new_stationary,
            vsc_pit_loss=min(new_pit_loss, max(new_stationary, base_pit.vsc_pit_loss)),
            sc_pit_loss=min(new_pit_loss, max(new_stationary, perturbation.safety_car_pit_loss if perturbation.safety_car_pit_loss else base_pit.sc_pit_loss)),
        )

        # Fuel adjustments
        base_fuel = self.baseline_model.fuel_model
        new_fuel = FuelModel(
            initial_fuel_kg=base_fuel.initial_fuel_kg,
            fuel_penalty_per_kg=base_fuel.fuel_penalty_per_kg * perturbation.fuel_penalty_multiplier,
            reserve_fuel_kg=base_fuel.reserve_fuel_kg,
        )

        # Compound adjustments
        perturbed_compounds: dict[str, TyreCompound] = {}
        for c_name, comp in self.compounds.items():
            base_delta = comp.base_delta
            if comp.name.lower() == "soft" and perturbation.soft_delta_adjustment != 0:
                base_delta += perturbation.soft_delta_adjustment
            elif comp.name.lower() == "hard" and perturbation.hard_delta_adjustment != 0:
                base_delta += perturbation.hard_delta_adjustment

            perturbed_compounds[c_name] = TyreCompound(
                name=comp.name,
                base_delta=base_delta,
                alpha=comp.alpha,
                beta=comp.beta,
                cliff_lap=comp.cliff_lap,
                cliff_coefficient=comp.cliff_coefficient,
            )

        perturbed_model = RaceModel(
            circuit=new_circuit,
            fuel_model=new_fuel,
            pitstop_model=new_pit,
        )
        return perturbed_model, perturbed_compounds

    def evaluate_scenario(
        self,
        scenario: Scenario,
        candidate_strategies: Optional[list[Strategy]] = None,
    ) -> ScenarioResult:
        """Run a comprehensive evaluation of a scenario versus baseline conditions.

        Args:
            scenario: The Scenario definition to test.
            candidate_strategies: Optional explicit strategies to benchmark.

        Returns:
            ScenarioResult containing optimization, regret, and strategy outcomes.
        """
        perturbed_model, perturbed_compounds = self.create_perturbed_model(scenario.perturbation)

        # 1. Discover baseline optimal strategy
        base_optimizer = StrategyOptimizer(self.baseline_model, self.compounds)
        base_opt_res = base_optimizer.optimize(
            max_stops=2,
            objective=OptimizationObjective.DETERMINISTIC_TIME,
        )
        base_opt_strat = base_opt_res.optimal_strategy
        base_opt_time = base_opt_res.optimal_race_time

        # 2. Discover scenario optimal strategy
        scen_optimizer = StrategyOptimizer(perturbed_model, perturbed_compounds)
        scen_opt_res = scen_optimizer.optimize(
            max_stops=2,
            objective=OptimizationObjective.DETERMINISTIC_TIME,
        )
        scen_opt_strat = scen_opt_res.optimal_strategy
        scen_opt_time = scen_opt_res.optimal_race_time

        # 3. Handle Safety Car opportunistic pit stop evaluation if specified
        if scenario.perturbation.safety_car_lap is not None:
            sc_lap = scenario.perturbation.safety_car_lap
            sc_pit_loss = scenario.perturbation.safety_car_pit_loss
            scen_opt_strat, scen_opt_time = self._evaluate_safety_car_adaptation(
                sc_lap, sc_pit_loss, perturbed_model, perturbed_compounds
            )

        # 4. Benchmark candidate strategies
        if candidate_strategies is None:
            candidates_dict: dict[str, Strategy] = {
                "Baseline Optimal": base_opt_strat,
                "Scenario Optimal": scen_opt_strat,
            }
            for top_strat, _ in base_opt_res.ranked_strategies[:3]:
                candidates_dict[top_strat.name or top_strat.description] = top_strat
            candidate_list = list(candidates_dict.values())
        else:
            candidate_list = candidate_strategies

        outcomes: list[ScenarioStrategyOutcome] = []
        for strat in candidate_list:
            if strat.total_laps != perturbed_model.circuit.total_laps:
                adapted_strat = self._rescale_strategy(strat, perturbed_model.circuit.total_laps, perturbed_compounds)
            else:
                adapted_strat = self._rebind_compounds(strat, perturbed_compounds)

            base_res = simulate_race(strat, self.baseline_model) if strat.total_laps == self.baseline_model.circuit.total_laps else None
            scen_res = simulate_race(adapted_strat, perturbed_model)

            base_t = base_res.total_time if base_res else scen_res.total_time
            scen_t = scen_res.total_time
            impact = scen_t - base_t
            gap = max(0.0, scen_t - scen_opt_time)

            outcomes.append(
                ScenarioStrategyOutcome(
                    strategy_name=strat.name or strat.description,
                    stops=len(strat.stints) - 1,
                    baseline_time=base_t,
                    scenario_time=scen_t,
                    time_impact=impact,
                    gap_to_winner=gap,
                )
            )

        # 5. Strategic Regret calculation
        adapted_base = self._rescale_strategy(base_opt_strat, perturbed_model.circuit.total_laps, perturbed_compounds)
        regret_res = simulate_race(adapted_base, perturbed_model)
        strategic_regret = max(0.0, regret_res.total_time - scen_opt_time)

        strategy_pivoted = (
            base_opt_strat.description != scen_opt_strat.description
            or len(base_opt_strat.stints) != len(scen_opt_strat.stints)
        )
        stops_pivoted = len(base_opt_strat.stints) != len(scen_opt_strat.stints)

        summary_insight = self._generate_scenario_insight(
            scenario,
            base_opt_strat,
            scen_opt_strat,
            stops_pivoted,
            strategic_regret,
        )

        return ScenarioResult(
            scenario=scenario,
            circuit_name=self.baseline_model.circuit.name,
            baseline_model=self.baseline_model,
            perturbed_model=perturbed_model,
            baseline_optimal_strategy=base_opt_strat,
            baseline_optimal_time=base_opt_time,
            scenario_optimal_strategy=scen_opt_strat,
            scenario_optimal_time=scen_opt_time,
            strategy_pivoted=strategy_pivoted,
            stops_pivoted=stops_pivoted,
            strategic_regret=strategic_regret,
            strategy_outcomes=outcomes,
            summary_insight=summary_insight,
        )

    def run_all_presets(self) -> dict[str, ScenarioResult]:
        """Execute all standard preset scenarios and return a dictionary of results."""
        results: dict[str, ScenarioResult] = {}
        for scen_id, scenario in PRESET_SCENARIOS.items():
            results[scen_id] = self.evaluate_scenario(scenario)
        return results

    def _evaluate_safety_car_adaptation(
        self,
        sc_lap: int,
        sc_pit_loss: float,
        model: RaceModel,
        compounds: dict[str, TyreCompound],
    ) -> tuple[Strategy, float]:
        """Evaluate strategy adaptation when a Safety Car occurs on lap `sc_lap`."""
        total_laps = model.circuit.total_laps
        standard_pit_loss = model.pitstop_model.pit_loss
        discount = standard_pit_loss - sc_pit_loss

        best_strat: Optional[Strategy] = None
        best_time = float("inf")

        from src.strategies import Stint

        if 5 <= sc_lap <= total_laps - 5:
            for c1_name, c1 in compounds.items():
                for c2_name, c2 in compounds.items():
                    if c1.name == c2.name:
                        continue
                    s = Strategy(
                        [Stint(c1, sc_lap), Stint(c2, total_laps - sc_lap)],
                        name=f"SC-1Stop ({c1.name[0]}{sc_lap}-{c2.name[0]}{total_laps - sc_lap})",
                    )
                    res = simulate_race(s, model)
                    discounted_time = res.total_time - discount
                    if discounted_time < best_time:
                        best_time = discounted_time
                        best_strat = s

        if best_strat is None:
            opt = StrategyOptimizer(model, compounds)
            res = opt.optimize(max_stops=2)
            return res.optimal_strategy, res.optimal_race_time

        return best_strat, best_time

    def _rebind_compounds(
        self, strategy: Strategy, new_compounds: dict[str, TyreCompound]
    ) -> Strategy:
        """Create a copy of the strategy with compound instances rebound to new models."""
        from src.strategies import Stint
        new_stints = []
        for stint in strategy.stints:
            c_name = stint.compound.name.capitalize()
            target_comp = new_compounds.get(c_name, stint.compound)
            new_stints.append(Stint(target_comp, stint.laps))
        return Strategy(new_stints, name=strategy.name)

    def _rescale_strategy(
        self,
        strategy: Strategy,
        target_total_laps: int,
        new_compounds: dict[str, TyreCompound],
    ) -> Strategy:
        """Rescale stint lengths proportionally to match target_total_laps."""
        from src.strategies import Stint
        if strategy.total_laps == target_total_laps:
            return self._rebind_compounds(strategy, new_compounds)

        scale = target_total_laps / strategy.total_laps
        new_stints = []
        assigned_laps = 0
        num_stints = len(strategy.stints)

        for idx, stint in enumerate(strategy.stints):
            c_name = stint.compound.name.capitalize()
            target_comp = new_compounds.get(c_name, stint.compound)
            if idx == num_stints - 1:
                stint_laps = target_total_laps - assigned_laps
            else:
                stint_laps = max(3, int(round(stint.laps * scale)))
                assigned_laps += stint_laps
            new_stints.append(Stint(target_comp, stint_laps))

        return Strategy(
            new_stints,
            name=f"{strategy.name or 'Strategy'} (Rescaled {target_total_laps}L)",
        )

    def _generate_scenario_insight(
        self,
        scenario: Scenario,
        base_strat: Strategy,
        scen_strat: Strategy,
        stops_pivoted: bool,
        regret: float,
    ) -> str:
        """Formulate a concise engineering diagnostic for the scenario outcome."""
        base_stops = len(base_strat.stints) - 1
        scen_stops = len(scen_strat.stints) - 1

        if stops_pivoted:
            return (
                f"Under '{scenario.name}', the optimal stop count shifted from {base_stops}-stop "
                f"({base_strat.description}) to {scen_stops}-stop ({scen_strat.description}). "
                f"Failing to pivot incurs a strategic regret penalty of {regret:.2f}s."
            )
        elif base_strat.description != scen_strat.description:
            return (
                f"Optimal stop count remained at {scen_stops}-stop under '{scenario.name}', but "
                f"tactical stint allocation adapted to {scen_strat.description} (Regret: {regret:.2f}s)."
            )
        else:
            return (
                f"Strategy remained fully robust under '{scenario.name}'. The nominal {base_stops}-stop "
                f"({base_strat.description}) remains mathematically optimal with 0.00s strategic regret."
            )
