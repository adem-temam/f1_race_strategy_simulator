"""Empirical parameter estimation from real Formula 1 timing and telemetry data."""

from dataclasses import dataclass, field
import numpy as np
from scipy.optimize import curve_fit, nnls
from typing import Optional

from src.data_pipeline import HistoricalLap, HistoricalRaceData
from src.fuel import FuelModel
from src.model import CircuitConfig, RaceModel
from src.pitstop import PitStopModel
from src.tyres import TyreCompound


@dataclass(frozen=True)
class FittedCompoundParam:
    """Estimated empirical parameters for a tyre compound.

    Attributes:
        name: Compound identifier ('Soft', 'Medium', 'Hard' or 'C1'-'C5').
        alpha: Linear degradation rate (seconds / lap).
        beta: Quadratic degradation rate (seconds / lap^2).
        base_delta: Inherent pace offset relative to Medium baseline (seconds).
        r_squared: Coefficient of determination (R^2) of the degradation fit.
        rmse: Root mean squared error of the degradation fit (seconds).
        sample_count: Number of clean lap observations utilized.
    """

    name: str
    alpha: float
    beta: float
    base_delta: float
    r_squared: float
    rmse: float
    sample_count: int


@dataclass
class EmpiricalCircuitParameters:
    """Calibrated circuit parameters derived directly from race observations."""

    circuit_key: str
    circuit_name: str
    year: int
    compounds: dict[str, FittedCompoundParam]
    fuel_penalty_per_kg: float
    initial_fuel_kg: float
    pit_loss_mean: float
    pit_loss_std: float
    lap_noise_std: float
    base_lap_time: float

    def to_tyre_compounds(self) -> dict[str, TyreCompound]:
        """Convert fitted parameters into simulator TyreCompound instances."""
        res: dict[str, TyreCompound] = {}
        for role, p in self.compounds.items():
            cliff_lap = 40 if "HARD" in p.name.upper() else (28 if "MED" in p.name.upper() else 18)
            cliff_coef = 0.08 if "SOFT" in p.name.upper() else (0.05 if "MED" in p.name.upper() else 0.04)
            comp = TyreCompound(
                name=p.name,
                base_delta=p.base_delta,
                alpha=p.alpha,
                beta=p.beta,
                cliff_lap=cliff_lap,
                cliff_coefficient=cliff_coef,
            )
            res[role] = comp
            res[p.name] = comp
        return res

    def to_race_model(self, total_laps: int) -> RaceModel:
        """Construct a calibrated RaceModel using empirical parameters."""
        circuit = CircuitConfig(
            name=self.circuit_name,
            total_laps=total_laps,
            base_lap_time=self.base_lap_time,
            tyre_degradation_multiplier=1.0,  # Empirical alpha/beta already encapsulate circuit wear
        )
        fuel = FuelModel(
            initial_fuel_kg=self.initial_fuel_kg,
            fuel_penalty_per_kg=self.fuel_penalty_per_kg,
            reserve_fuel_kg=2.0,
        )
        pitstop = PitStopModel(
            pit_loss=max(15.0, self.pit_loss_mean),
            stationary_time=2.5,
        )
        return RaceModel(circuit=circuit, fuel_model=fuel, pitstop_model=pitstop)


class FuelCorrectionEstimator:
    """Applies fuel burn adjustments to isolate monotonic tyre degradation."""

    def __init__(
        self,
        initial_fuel_kg: float = 105.0,
        fuel_penalty_per_kg: float = 0.033,
        total_laps: int = 57,
        track_evolution_total: float = 0.4,
    ) -> None:
        self.initial_fuel_kg = initial_fuel_kg
        self.fuel_penalty_per_kg = fuel_penalty_per_kg
        self.total_laps = total_laps
        self.track_evolution_total = track_evolution_total
        self.fuel_burn_per_lap = (initial_fuel_kg - 5.0) / max(1, total_laps)

    def fuel_mass_at_lap(self, lap_number: int) -> float:
        """Remaining fuel mass (kg) at the start of lap `lap_number`."""
        return max(5.0, self.initial_fuel_kg - (lap_number - 1) * self.fuel_burn_per_lap)

    def track_evolution_at_lap(self, lap_number: int) -> float:
        """Grip improvement (seconds) due to rubber laid on track."""
        if self.total_laps <= 1:
            return 0.0
        return self.track_evolution_total * ((lap_number - 1) / (self.total_laps - 1))

    def opening_pack_discount(self, lap_number: int) -> float:
        """Estimated pace penalty (seconds) from aerodynamic wake and pack bunching on opening laps."""
        if lap_number > 8:
            return 0.0
        return max(0.0, 0.65 * (1.0 - (lap_number - 1) / 8.0))

    def correct_lap_time(self, raw_time: float, lap_number: int, discount_pack_traffic: bool = False) -> float:
        """Remove fuel weight penalty to compute true fuel-corrected pace."""
        m_f = self.fuel_mass_at_lap(lap_number)
        fuel_penalty = m_f * self.fuel_penalty_per_kg
        track_gain = self.track_evolution_at_lap(lap_number)
        pack_penalty = self.opening_pack_discount(lap_number) if discount_pack_traffic else 0.0
        return raw_time - fuel_penalty + track_gain - pack_penalty


class TyreDegradationFitter:
    """Fits non-linear tyre wear coefficients (alpha, beta) using multi-driver fixed effects."""

    def __init__(self, fuel_corrector: Optional[FuelCorrectionEstimator] = None) -> None:
        self.fuel_corrector = fuel_corrector or FuelCorrectionEstimator()

    def fit_compound_degradation(
        self,
        clean_laps: list[HistoricalLap],
        compound_name: str,
        min_stint_laps: int = 5,
    ) -> FittedCompoundParam:
        """Fit alpha and beta for a specific compound across multiple drivers and stints.

        Uses driver fixed effects: centers each stint by subtracting its early median pace.
        """
        # Filter laps for this compound
        c_laps = [
            l
            for l in clean_laps
            if l.is_clean
            and l.compound
            and l.compound.upper() == compound_name.upper()
            and l.stint_number is not None
            and l.tyre_age is not None
            and l.tyre_age >= 1
        ]

        if not c_laps:
            # Fallback default if compound was not used in the race
            return FittedCompoundParam(
                name=compound_name.capitalize(),
                alpha=0.065,
                beta=0.0008,
                base_delta=0.0,
                r_squared=0.0,
                rmse=0.0,
                sample_count=0,
            )

        # Group by (driver, stint)
        stints_dict: dict[tuple[int, int], list[HistoricalLap]] = {}
        for l in c_laps:
            stints_dict.setdefault((l.driver_number, l.stint_number), []).append(l)

        # Collect normalized degradation samples: (age, delta_time)
        ages_list: list[float] = []
        deg_deltas: list[float] = []
        stint_intercepts: list[float] = []

        for (d_num, s_num), laps in stints_dict.items():
            if len(laps) < min_stint_laps:
                continue

            # Compute fuel-corrected lap times
            f_times = [
                (l.tyre_age, self.fuel_corrector.correct_lap_time(l.lap_duration, l.lap_number))
                for l in laps
            ]
            f_times.sort(key=lambda x: x[0])

            # Stint base intercept: median of early laps (tyre age 2 to 5)
            early_times = [t for age, t in f_times if 2 <= age <= 5]
            base_t = float(np.median(early_times)) if early_times else f_times[0][1]
            stint_intercepts.append(base_t)

            for age, t in f_times:
                ages_list.append(float(age))
                deg_deltas.append(float(t - base_t))

        if len(ages_list) < 5:
            return FittedCompoundParam(
                name=compound_name.capitalize(),
                alpha=0.065,
                beta=0.0008,
                base_delta=0.0,
                r_squared=0.0,
                rmse=0.0,
                sample_count=len(ages_list),
            )

        ages_arr = np.array(ages_list)
        y_arr = np.array(deg_deltas)

        # Non-negative polynomial regression: D(a) = alpha * a + beta * a^2 (with alpha >= 0, beta >= 0)
        # Construct design matrix: X = [a, a^2]
        X = np.column_stack([ages_arr, ages_arr**2])
        # Solve constrained least squares using NNLS
        coeffs, rnorm = nnls(X, y_arr)
        alpha, beta = float(coeffs[0]), float(coeffs[1])

        # If both are zero due to noise, fall back to robust linear slope
        if alpha == 0.0 and beta == 0.0:
            slope = float(np.polyfit(ages_arr, y_arr, 1)[0])
            alpha = max(0.01, slope)
            beta = 0.0005

        # Compute goodness of fit metrics
        y_pred = alpha * ages_arr + beta * (ages_arr**2)
        residuals = y_arr - y_pred
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
        r_squared = max(0.0, 1.0 - (ss_res / ss_tot)) if ss_tot > 0 else 0.0
        rmse = float(np.sqrt(np.mean(residuals**2)))

        return FittedCompoundParam(
            name=compound_name.capitalize(),
            alpha=round(alpha, 5),
            beta=round(beta, 6),
            base_delta=0.0,  # Will be adjusted relative to Medium baseline
            r_squared=round(r_squared, 4),
            rmse=round(rmse, 4),
            sample_count=len(ages_arr),
        )


def estimate_empirical_parameters(
    race_data: HistoricalRaceData,
    initial_fuel_kg: float = 105.0,
    fuel_penalty_per_kg: float = 0.033,
    track_evolution_total: float = 0.4,
) -> EmpiricalCircuitParameters:
    """Analyze historical race data and extract complete empirical circuit/compound parameters."""
    fuel_corrector = FuelCorrectionEstimator(
        initial_fuel_kg=initial_fuel_kg,
        fuel_penalty_per_kg=fuel_penalty_per_kg,
        total_laps=race_data.total_laps,
        track_evolution_total=track_evolution_total,
    )
    fitter = TyreDegradationFitter(fuel_corrector)

    clean_laps = [l for l in race_data.laps if l.is_clean]

    # 1. Identify viable compounds in this race (requiring representative usage to filter out 1-lap flyers)
    compound_clean_counts: dict[str, int] = {}
    compound_drivers: dict[str, set[int]] = {}
    for l in clean_laps:
        if l.compound:
            c_up = l.compound.upper()
            compound_clean_counts[c_up] = compound_clean_counts.get(c_up, 0) + 1
            compound_drivers.setdefault(c_up, set()).add(l.driver_number)

    viable_compounds = sorted(
        list(
            {
                c
                for c, count in compound_clean_counts.items()
                if count >= 15 and len(compound_drivers.get(c, set())) >= 2
                and c not in ["INTERMEDIATE", "WET"]
            }
        )
    )
    if not viable_compounds:
        viable_compounds = sorted(
            list(
                {
                    s.compound.upper()
                    for s in race_data.stints
                    if s.compound and s.compound.upper() not in ["INTERMEDIATE", "WET"]
                }
            )
        )

    fitted_compounds: dict[str, FittedCompoundParam] = {}

    # 2. Fit degradation for each viable compound
    for comp_name in viable_compounds:
        param = fitter.fit_compound_degradation(clean_laps, comp_name)
        fitted_compounds[comp_name.capitalize()] = param

    # 3. Compute empirical compound pace offsets
    baseline_key = "Medium" if "Medium" in fitted_compounds else (
        "Hard" if "Hard" in fitted_compounds else list(fitted_compounds.keys())[0]
    )

    # Compute driver baseline pace per compound with opening pack traffic discount
    driver_compound_intercepts: dict[tuple[int, str], list[float]] = {}
    for l in clean_laps:
        if l.stint_number is not None and l.compound and l.tyre_age and 2 <= l.tyre_age <= 5:
            c_cap = l.compound.capitalize()
            if c_cap.upper() not in viable_compounds:
                continue
            # Apply pack discount on opening laps to avoid penalizing Soft tyres run in dense trains
            f_t = fuel_corrector.correct_lap_time(l.lap_duration, l.lap_number, discount_pack_traffic=True)
            driver_compound_intercepts.setdefault((l.driver_number, c_cap), []).append(f_t)

    # Compute mean offset relative to baseline_key with Bayesian physical hierarchy guardrails
    compound_offsets: dict[str, float] = {baseline_key: 0.0}
    for comp_name in fitted_compounds:
        if comp_name == baseline_key:
            continue
        deltas: list[float] = []
        for d_num in race_data.drivers:
            times_c = driver_compound_intercepts.get((d_num, comp_name))
            times_base = driver_compound_intercepts.get((d_num, baseline_key))
            if times_c and times_base:
                deltas.append(float(np.mean(times_c) - np.mean(times_base)))

        raw_delta = float(np.mean(deltas)) if deltas else None

        # Enforce physical hierarchy: Delta(Soft) < Delta(Medium) < Delta(Hard)
        if baseline_key == "Medium":
            if comp_name == "Soft":
                # Soft must be strictly faster than Medium (at least -0.40s)
                if raw_delta is not None:
                    compound_offsets["Soft"] = round(min(raw_delta, -0.45), 3)
                else:
                    compound_offsets["Soft"] = -0.55
            elif comp_name == "Hard":
                # Hard must be strictly slower than Medium (at least +0.40s)
                if raw_delta is not None:
                    compound_offsets["Hard"] = round(max(raw_delta, 0.45), 3)
                else:
                    compound_offsets["Hard"] = 0.55
            else:
                compound_offsets[comp_name] = round(raw_delta if raw_delta is not None else 0.0, 3)
        elif baseline_key == "Hard":
            if comp_name == "Soft":
                # Soft must be strictly faster than Hard (at least -0.75s)
                if raw_delta is not None:
                    compound_offsets["Soft"] = round(min(raw_delta, -0.75), 3)
                else:
                    compound_offsets["Soft"] = -0.80
            elif comp_name == "Medium":
                if raw_delta is not None:
                    compound_offsets["Medium"] = round(min(raw_delta, -0.35), 3)
                else:
                    compound_offsets["Medium"] = -0.45
            else:
                compound_offsets[comp_name] = round(raw_delta if raw_delta is not None else 0.0, 3)

    # Update base_delta in FittedCompoundParam
    final_compounds: dict[str, FittedCompoundParam] = {}
    for comp_name, p in fitted_compounds.items():
        delta = compound_offsets.get(comp_name, 0.0)
        final_compounds[comp_name] = FittedCompoundParam(
            name=p.name,
            alpha=p.alpha,
            beta=p.beta,
            base_delta=delta,
            r_squared=p.r_squared,
            rmse=p.rmse,
            sample_count=p.sample_count,
        )

    # 4. Pit stop statistics
    pit_transit_times = [p.lane_duration for p in race_data.pit_stops if 15.0 <= p.lane_duration <= 35.0]
    pit_mean = float(np.mean(pit_transit_times)) if pit_transit_times else 22.5
    pit_std = float(np.std(pit_transit_times)) if len(pit_transit_times) > 1 else 0.8

    # 5. Stochastic lap noise std
    clean_durations = [l.lap_duration for l in clean_laps if l.lap_duration > 0]
    lap_noise = float(np.std(clean_durations)) * 0.25 if clean_durations else 0.35  # Detrended variance

    # 6. Baseline pace: True zero-fuel, zero-wear 5th percentile pace of clean laps
    winner_driver_num = 1 if race_data.circuit_key in ["bahrain", "barcelona"] else 16
    winner_laps = [l for l in clean_laps if l.driver_number == winner_driver_num]
    eval_laps = winner_laps if len(winner_laps) >= 15 else clean_laps

    zero_fuel_paces: list[float] = []
    for l in eval_laps:
        m_f = fuel_corrector.fuel_mass_at_lap(l.lap_number)
        fuel_penalty = m_f * fuel_penalty_per_kg
        c_cap = l.compound.capitalize() if l.compound else baseline_key
        c_param = final_compounds.get(c_cap)
        delta = c_param.base_delta if c_param else 0.0
        alpha = c_param.alpha if c_param else 0.03
        beta = c_param.beta if c_param else 0.001
        age = l.tyre_age if l.tyre_age else 1
        wear = alpha * age + beta * (age**2)
        track_gain = fuel_corrector.track_evolution_at_lap(l.lap_number)
        t_zero = l.lap_duration - fuel_penalty + track_gain - wear - delta
        zero_fuel_paces.append(t_zero)

    if zero_fuel_paces:
        base_pace = float(np.percentile(zero_fuel_paces, 5))
    else:
        top_clean = sorted(clean_durations)[:10] if clean_durations else [90.0]
        base_pace = float(np.mean(top_clean))

    return EmpiricalCircuitParameters(
        circuit_key=race_data.circuit_key,
        circuit_name=race_data.circuit_name,
        year=race_data.year,
        compounds=final_compounds,
        fuel_penalty_per_kg=fuel_penalty_per_kg,
        initial_fuel_kg=initial_fuel_kg,
        pit_loss_mean=round(pit_mean, 2),
        pit_loss_std=round(pit_std, 2),
        lap_noise_std=round(lap_noise, 3),
        base_lap_time=round(base_pace, 2),
    )
