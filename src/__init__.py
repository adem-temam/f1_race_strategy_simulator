"""F1 Race Strategy Simulator & Analysis."""

from src.tyres import TyreCompound
from src.fuel import FuelModel
from src.pitstop import PitStopModel
from src.strategies import Stint, Strategy
from src.model import CircuitConfig, LapRecord, RaceModel
from src.simulation import RaceResult, simulate_race
from src.optimization import (
    OptimizationObjective,
    OptimizationResult,
    PitWindow,
    StrategyOptimizer,
)
from src.sensitivity import (
    CrossoverFinder,
    CrossoverPoint,
    ParameterSweep1D,
    PhaseDiagram2D,
    PhaseDiagramResult2D,
    RobustnessAnalyzer,
    SensitivityParameter,
    SweepResult1D,
)

__all__ = [
    "TyreCompound",
    "FuelModel",
    "PitStopModel",
    "Stint",
    "Strategy",
    "CircuitConfig",
    "LapRecord",
    "RaceModel",
    "RaceResult",
    "simulate_race",
    "OptimizationObjective",
    "OptimizationResult",
    "PitWindow",
    "StrategyOptimizer",
    "SensitivityParameter",
    "CrossoverPoint",
    "SweepResult1D",
    "PhaseDiagramResult2D",
    "ParameterSweep1D",
    "CrossoverFinder",
    "PhaseDiagram2D",
    "RobustnessAnalyzer",
    "HistoricalLap",
    "HistoricalStint",
    "HistoricalPitStop",
    "HistoricalRaceData",
    "load_historical_data",
    "filter_clean_laps",
    "FittedCompoundParam",
    "EmpiricalCircuitParameters",
    "estimate_empirical_parameters",
    "FuelCorrectionEstimator",
    "TyreDegradationFitter",
    "ValidationMetrics",
    "DiscrepancyDiagnostic",
    "RaceBacktester",
    "DiscrepancyAnalyzer",
    "plot_empirical_degradation_fit",
    "plot_lap_residuals",
    "plot_strategy_backtest_gantt",
    "Scenario",
    "ScenarioEngine",
    "ScenarioPerturbation",
    "ScenarioResult",
    "PRESET_SCENARIOS",
]

from src.scenarios import (
    PRESET_SCENARIOS,
    Scenario,
    ScenarioEngine,
    ScenarioPerturbation,
    ScenarioResult,
)

from src.data_pipeline import (
    HistoricalLap,
    HistoricalPitStop,
    HistoricalRaceData,
    HistoricalStint,
    filter_clean_laps,
    load_historical_data,
)
from src.estimation import (
    EmpiricalCircuitParameters,
    FittedCompoundParam,
    FuelCorrectionEstimator,
    TyreDegradationFitter,
    estimate_empirical_parameters,
)
from src.validation import (
    DiscrepancyAnalyzer,
    DiscrepancyDiagnostic,
    RaceBacktester,
    ValidationMetrics,
)
from src.visualization import (
    plot_empirical_degradation_fit,
    plot_lap_residuals,
    plot_strategy_backtest_gantt,
)
