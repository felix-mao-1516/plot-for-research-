"""Zhou (2014): a transparent single-fracture Darcy/time-superposition model.

Default: 3 panels on one wing, vertices numbered TIP -> WELL, with the
opposite symmetric wing included in the reservoir Green function.
Unknowns: [p0,...,pN, qf1,...,qfN, Q1,...,QN].

This is an independently implemented, simplified case of the paper's method,
not its original software. Reservoir: 2-D laterally infinite, fully penetrating
fracture, constant properties. Fracture: quasi-steady Darcy, no storage.
Inputs/outputs use field units; all equations and convolution use SI units.

Run: python zhou_three_panel.py --times 1 2 3 5 10 60
     python zhou_three_panel.py --control rate --rate 25 --out results_rate
"""

# %% Imports and physical parameters
from __future__ import annotations

import argparse
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf, exp1

FT = 0.3048
PSI = 6894.757293168
MD = 9.869233e-16
STB = 0.158987294928
DAY = 86400.0
HOUR = 3600.0


@dataclass(frozen=True)
class Parameters:
    # All values below except pwf, discretization and control are from Table 1.
    k_md: float = 0.1
    phi: float = 0.1
    ct_per_psi: float = 3.0e-6
    mu_cp: float = 0.6
    h_ft: float = 50.0
    xf_ft: float = 210.0
    conductivity_md_ft: float = 420.0
    pi_psi: float = 4200.0
    B_rb_per_stb: float = 1.273
    n_panels: int = 3
    control: str = "pressure"       # "pressure" or "rate"
    pwf_psi: float = 3000.0          # DEMONSTRATION choice, not Table 1
    total_rate_stb_day: float = 25.0 # Whole well, BOTH wings if symmetric
    include_other_wing: bool = True

    def __post_init__(self):
        positive = (self.k_md, self.phi, self.ct_per_psi, self.mu_cp,
                    self.h_ft, self.xf_ft, self.conductivity_md_ft,
                    self.pi_psi, self.B_rb_per_stb)
        if not all(np.isfinite(v) and v > 0 for v in positive):
            raise ValueError("Rock/fluid/geometry parameters must be finite and positive.")
        if self.phi > 1 or not isinstance(self.n_panels, int) or self.n_panels < 1:
            raise ValueError("Require 0 < phi <= 1 and an integer n_panels >= 1.")
        if self.control not in ("pressure", "rate"):
            raise ValueError("control must be 'pressure' or 'rate'.")
        if not np.isfinite(self.pwf_psi) or self.pwf_psi <= 0:
            raise ValueError("pwf_psi must be finite and positive.")
        if self.control == "pressure" and self.pwf_psi >= self.pi_psi:
            raise ValueError("This production example requires pwf_psi < pi_psi.")
        if not np.isfinite(self.total_rate_stb_day) or self.total_rate_stb_day < 0:
            raise ValueError("total_rate_stb_day must be finite and nonnegative.")

    @property
    def k(self):
        return self.k_md * MD

    @property
    def mu(self):
        return self.mu_cp * 1e-3

    @property
    def h(self):
        return self.h_ft * FT

    @property
    def xf(self):
        return self.xf_ft * FT

    @property
    def ell(self):
        return self.xf / self.n_panels

    @property
    def eta(self):
        # ct must be converted from 1/psi to 1/Pa.
        return self.k / (self.phi * self.mu * (self.ct_per_psi / PSI))

    @property
    def Rf(self):
        # qf and Q use reservoir VOLUME, so no density appears here.
        return self.mu / (self.conductivity_md_ft * MD * FT * self.h)

    @property
    def wing_factor(self):
        return 2 if self.include_other_wing else 1

    @property
    def stocktank_to_reservoir_rate(self):
        return self.B_rb_per_stb * STB / DAY  # (m3/s) / (STB/day)

    @property
    def vertices_m(self):
        return np.linspace(0.0, self.xf, self.n_panels + 1)

    @property
    def centers_m(self):
        v = self.vertices_m
        return (v[:-1] + v[1:]) / 2


# %% Reservoir step-response A_ij(t): exact spatial integration on y = 0
def _e1_primitive(offset_m, diffusion_length_m):
    """Primitive of E1((u/a)^2): u*E1((u/a)^2) + a*sqrt(pi)*erf(u/a).

    Its value at u=0 is exactly 0 by continuity. This integrates the
    logarithmic self-panel singularity without choosing an arbitrary radius.
    """
    u, a = np.broadcast_arrays(offset_m, diffusion_length_m)
    z = (u / a) ** 2
    first_term = np.zeros_like(z)
    nonzero = u != 0
    first_term[nonzero] = u[nonzero] * exp1(z[nonzero])
    return first_term + a * np.sqrt(np.pi) * erf(u / a)


def step_response(p: Parameters, age_s):
    r"""A_ij(age), mapping constant qf_j [m2/s] to drawdown [Pa].

    A_ij(t) = mu/(4*pi*k*h) * integral_panel_j E1(r^2/(4*eta*t)) dxi.
    For the other wing, add the source interval reflected about x = xf.
    Input shape (...) -> output shape (..., N, N). A(0) = 0.
    qf includes the entire height and BOTH reservoir faces of one panel.
    """
    age = np.asarray(age_s, dtype=float)
    if np.any(~np.isfinite(age)) or np.any(age < 0):
        raise ValueError("Green-function ages must be finite and nonnegative.")
    safe_age = np.where(age > 0, age, 1.0)
    a = 2 * np.sqrt(p.eta * safe_age)[..., None, None]
    x = p.centers_m[:, None]
    left, right = p.vertices_m[:-1][None, :], p.vertices_m[1:][None, :]
    integral = (_e1_primitive(x - left, a)
                - _e1_primitive(x - right, a))
    if p.include_other_wing:
        # Mirror EACH panel; adding the other wing is not just doubling A.
        mirror_left, mirror_right = 2 * p.xf - right, 2 * p.xf - left
        integral += (_e1_primitive(x - mirror_left, a)
                     - _e1_primitive(x - mirror_right, a))
    # Far-field subtraction can produce tiny negative roundoff near zero.
    integral = np.maximum(integral, 0.0)
    out = p.mu / (4 * np.pi * p.k * p.h) * integral
    return np.where((age > 0)[..., None, None], out, 0.0)


# %% Assemble the same matrix as the user's 10-by-10 matrix (N = 3)
def unknown_names(n):
    return ([f"p{i}" for i in range(n + 1)]
            + [f"qf{i+1}" for i in range(n)]
            + [f"Q{i+1}" for i in range(n)])


def equation_names(n):
    return (["tip_Q1_zero"]
            + [f"continuity_node_{i}" for i in range(1, n)]
            + ["well_control"]
            + [f"Darcy_panel_{i+1}" for i in range(n)]
            + [f"center_coupling_{i+1}" for i in range(n)])


def assemble_system(p: Parameters, A_current, history_pa):
    """Build M x = b in SI, with absolute vertex pressures.

    For N=3, rows are:
      Q1=0; Q2-Q1-ell*qf1=0; Q3-Q2-ell*qf2=0; p3=pwf;
      3 Darcy pressure drops; 3 center-pressure matching equations.
    Rate control changes ONLY the well-control row to QN+ell*qfN=Qhalf.
    """
    n, ell, R = p.n_panels, p.ell, p.Rf
    qstart, Qstart = n + 1, 2 * n + 1
    M = np.zeros((3 * n + 1, 3 * n + 1))
    b = np.zeros(3 * n + 1)
    M[0, Qstart] = 1.0
    for i in range(1, n):
        M[i, Qstart + i] = 1.0
        M[i, Qstart + i - 1] = -1.0
        M[i, qstart + i - 1] = -ell
    if p.control == "pressure":
        M[n, n] = 1.0
        b[n] = p.pwf_psi * PSI
    else:
        M[n, Qstart + n - 1] = 1.0
        M[n, qstart + n - 1] = ell
        b[n] = (p.total_rate_stb_day * p.stocktank_to_reservoir_rate
                / p.wing_factor)

    D_Q, D_q = R * ell, R * ell**2 / 2
    C_Q, C_q = R * ell / 2, R * ell**2 / 8
    for j in range(n):
        r = n + 1 + j
        M[r, j], M[r, j + 1] = 1.0, -1.0
        M[r, qstart + j], M[r, Qstart + j] = -D_q, -D_Q
        r = 2 * n + 1 + j
        M[r, j] = 1.0
        M[r, qstart:qstart + n] = A_current[j]
        M[r, qstart + j] -= C_q
        M[r, Qstart + j] = -C_Q
        b[r] = p.pi_psi * PSI - history_pa[j]
    return M, b


def solve_scaled(p, M, b):
    """Row/column scaling only; the returned x still solves the original Mx=b."""
    n = p.n_panels
    pressure_scale = max(abs(p.pi_psi - p.pwf_psi), 100.0) * PSI
    rate_scale = 4 * np.pi * p.k * p.h * pressure_scale / p.mu
    column_scale = np.r_[np.full(n + 1, pressure_scale),
                         np.full(n, rate_scale / p.xf),
                         np.full(n, rate_scale)]
    MC = M * column_scale[None, :]
    row_scale = np.max(np.abs(MC), axis=1)
    Ms, bs = MC / row_scale[:, None], b / row_scale
    y = np.linalg.solve(Ms, bs)  # Do not explicitly invert the matrix.
    x = column_scale * y
    relative_residual = np.linalg.norm(Ms @ y - bs, np.inf) / max(
        1.0, np.linalg.norm(bs, np.inf))
    return x, float(relative_residual)


# %% Time grid and history convolution
def make_time_grid(report_times_h, points_per_decade=60, first_step_h=1e-6):
    """Internal integration times are finer than requested display times."""
    reports = np.unique(np.asarray(report_times_h, dtype=float))
    if reports.ndim != 1 or len(reports) == 0 or np.any(~np.isfinite(reports)):
        raise ValueError("report_times_h must be a finite, nonempty 1-D array.")
    if np.any(reports <= 0) or points_per_decade < 1 or first_step_h <= 0:
        raise ValueError("Times, first_step_h and points_per_decade must be positive.")
    start = min(first_step_h, reports[0])
    count = max(2, int(np.ceil(np.log10(reports[-1] / start)
                              * points_per_decade)) + 1)
    base = np.geomspace(start, reports[-1], count)
    # Avoid nearly duplicate endpoints that could create negligible time steps.
    near_report = np.any(np.isclose(base[:, None], reports[None, :],
                                   rtol=1e-10, atol=0.0), axis=1)
    times_h = np.unique(np.r_[0.0, base[~near_report], reports])
    return times_h * HOUR, reports


@dataclass
class Solution:
    parameters: Parameters
    edges_s: np.ndarray
    reports_h: np.ndarray
    x: np.ndarray
    history_pa: np.ndarray
    max_scaled_residual: float

    @property
    def time_h(self):
        return self.edges_s[1:] / HOUR

    @property
    def p_pa(self):
        return self.x[:, :self.parameters.n_panels + 1]

    @property
    def qf_m2_s(self):
        n = self.parameters.n_panels
        return self.x[:, n + 1:2 * n + 1]

    @property
    def Q_m3_s(self):
        return self.x[:, 2 * self.parameters.n_panels + 1:]

    @property
    def center_darcy_pa(self):
        p = self.parameters
        return (self.p_pa[:, :-1] - p.Rf * p.ell / 2 * self.Q_m3_s
                - p.Rf * p.ell**2 / 8 * self.qf_m2_s)

    @property
    def center_reservoir_pa(self):
        A = step_response(self.parameters, np.diff(self.edges_s))
        current = np.einsum("nij,nj->ni", A, self.qf_m2_s)
        return self.parameters.pi_psi * PSI - self.history_pa - current

    @property
    def panel_rate_stb_day(self):
        p = self.parameters
        return self.qf_m2_s * p.ell / p.stocktank_to_reservoir_rate

    @property
    def total_rate_stb_day(self):
        return self.parameters.wing_factor * self.panel_rate_stb_day.sum(axis=1)

    @property
    def report_indices(self):
        return np.array([np.argmin(abs(self.time_h - t)) for t in self.reports_h])


def simulate(p=None, report_times_h=(1, 2, 3, 5, 10, 60),
             points_per_decade=60, first_step_h=1e-6):
    r"""Implicit piecewise-constant panel influx, exact convolution weights.

    On interval m: qf(t) = qf[m] for t_edges[m] < t <= t_edges[m+1].
    At t_n:
      H_n = sum_{m<n} [A(t_n-t_{m-1}) - A(t_n-t_m)] @ qf_m
      p_center = pi - H_n - A(dt_n) @ qf_n.
    Each step solves a fresh M_n x_n = b_n, retaining all previous qf_m.
    """
    p = Parameters() if p is None else p
    edges, reports = make_time_grid(report_times_h, points_per_decade, first_step_h)
    count, n = len(edges) - 1, p.n_panels
    states = np.zeros((count, 3 * n + 1))
    histories = np.zeros((count, n))
    worst_residual = 0.0
    for step in range(count):
        now = edges[step + 1]
        if step:
            # The old interval has both a start and an end: both ages matter.
            older = step_response(p, now - edges[:step])
            younger = step_response(p, now - edges[1:step + 1])
            previous_qf = states[:step, n + 1:2 * n + 1]
            histories[step] = np.einsum("mij,mj->i", older - younger, previous_qf)
        A_current = step_response(p, now - edges[step])
        M, b = assemble_system(p, A_current, histories[step])
        states[step], residual = solve_scaled(p, M, b)
        worst_residual = max(worst_residual, residual)
    return Solution(p, edges, reports, states, histories, worst_residual)


# %% Inspect and export results, including the actual matrix at a report time
def result_table(sol, reports_only=True):
    p = sol.parameters
    selected = sol.report_indices if reports_only else np.arange(len(sol.time_h))
    data = {"time_h": sol.time_h[selected],
            "well_rate_total_STB_day": sol.total_rate_stb_day[selected]}
    for j in range(p.n_panels + 1):
        data[f"p{j}_psi"] = sol.p_pa[selected, j] / PSI
    for j in range(p.n_panels):
        data[f"pc{j+1}_psi"] = sol.center_darcy_pa[selected, j] / PSI
        data[f"qf{j+1}_STB_day_ft"] = (sol.qf_m2_s[selected, j] * FT
                                       / p.stocktank_to_reservoir_rate)
        data[f"panel{j+1}_halfwing_STB_day"] = sol.panel_rate_stb_day[selected, j]
        data[f"Q{j+1}_STB_day"] = sol.Q_m3_s[selected, j] / p.stocktank_to_reservoir_rate
    return pd.DataFrame(data)


def system_at_report(sol, report_number=0, field_units=True):
    """Return M,b,x at a requested output time; default units psi/STB/day/ft."""
    k = sol.report_indices[report_number]
    p, n = sol.parameters, sol.parameters.n_panels
    A = step_response(p, sol.edges_s[k + 1] - sol.edges_s[k])
    M, b = assemble_system(p, A, sol.history_pa[k])
    x = sol.x[k].copy()
    if field_units:
        q_unit = p.stocktank_to_reservoir_rate
        columns = np.r_[np.full(n + 1, PSI), np.full(n, q_unit / FT),
                        np.full(n, q_unit)]
        rows = np.r_[np.full(n, q_unit), PSI, np.full(2 * n, PSI)]
        if p.control == "rate":
            rows[n] = q_unit
        M, b, x = M * columns[None, :] / rows[:, None], b / rows, x / columns
    return M, b, x


def diagnostics(sol):
    p = sol.parameters
    outflow = sol.Q_m3_s + p.ell * sol.qf_m2_s
    continuity = outflow[:, :-1] - sol.Q_m3_s[:, 1:]
    denominator = max(float(np.max(abs(outflow))), 1e-30)
    darcy_drop = (p.Rf * p.ell * sol.Q_m3_s
                  + p.Rf * p.ell**2 / 2 * sol.qf_m2_s)
    reverse_influx = np.any(sol.panel_rate_stb_day < -1e-9, axis=1)
    overshoot = np.any(sol.p_pa / PSI > p.pi_psi + 1e-7, axis=1)
    return {
        "internal_time_steps": len(sol.time_h),
        "eta_ft2_hour": p.eta * HOUR / FT**2,
        "Fcd": p.conductivity_md_ft / (p.k_md * p.xf_ft),
        "max_scaled_linear_residual": sol.max_scaled_residual,
        "max_center_pressure_mismatch_psi": float(np.max(abs(
            sol.center_darcy_pa - sol.center_reservoir_pa)) / PSI),
        "max_Darcy_mismatch_psi": float(np.max(abs(
            sol.p_pa[:, :-1] - sol.p_pa[:, 1:] - darcy_drop)) / PSI),
        "max_continuity_relative_error": (float(np.max(abs(continuity)))
                                          / denominator if continuity.size else 0.0),
        "tip_rate_relative_error": float(np.max(abs(sol.Q_m3_s[:, 0]))) / denominator,
        "max_well_balance_relative_error": float(np.max(abs(
            outflow[:, -1] - p.ell * sol.qf_m2_s.sum(axis=1)))) / denominator,
        "minimum_vertex_pressure_psi": float(np.min(sol.p_pa) / PSI),
        "minimum_panel_rate_STB_day": float(np.min(sol.panel_rate_stb_day)),
        "last_reverse_influx_time_h": (float(sol.time_h[reverse_influx][-1])
                                        if reverse_influx.any() else None),
        "last_vertex_pressure_overshoot_time_h": (float(sol.time_h[overshoot][-1])
                                                  if overshoot.any() else None),
        "minimum_panel_rate_at_requested_times_STB_day": float(np.min(
            sol.panel_rate_stb_day[sol.report_indices])),
    }


def plot_results(sol):
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    available = {f.name for f in font_manager.fontManager.ttflist}
    font = "Times New Roman" if "Times New Roman" in available else "STIXGeneral"
    style = {"font.family": font, "mathtext.fontset": "stix", "font.size": 12,
             "axes.labelsize": 13, "axes.titlesize": 13, "axes.linewidth": 1.4,
             "lines.linewidth": 2.0, "xtick.direction": "out", "ytick.direction": "out"}
    p = sol.parameters
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
        colors = plt.cm.viridis(np.linspace(0.08, 0.84, len(sol.reports_h)))
        vertices = p.vertices_m / FT
        for k, color in zip(sol.report_indices, colors):
            label = f"{sol.time_h[k]:g} h"
            # Pressure is quadratic WITHIN each panel, not piecewise linear.
            xx, pp = [], []
            for j in range(p.n_panels):
                u = np.linspace(0, p.ell, 35)
                xx.extend((p.vertices_m[j] + u) / FT)
                pp.extend((sol.p_pa[k, j] - p.Rf * (sol.Q_m3_s[k, j] * u
                                  + sol.qf_m2_s[k, j] * u**2 / 2)) / PSI)
            axes[0, 0].plot(xx, pp, color=color, label=label)
            axes[0, 0].scatter(vertices, sol.p_pa[k] / PSI, color=color, s=20)
            qf_field = sol.qf_m2_s[k] * FT / p.stocktank_to_reservoir_rate
            axes[0, 1].stairs(qf_field, vertices, color=color, label=label, baseline=None)
        axes[0, 0].set(title="Fracture pressure along one wing", ylabel="Pressure (psi)")
        axes[0, 1].set(title="Reservoir influx per unit fracture length",
                       ylabel="Panel influx (STB/day/ft)")
        displayed_flux = (sol.qf_m2_s[sol.report_indices] * FT
                          / p.stocktank_to_reservoir_rate)
        if np.min(displayed_flux) >= 0:
            axes[0, 1].set_ylim(0, np.max(displayed_flux) * 1.35)
        for ax in axes[0]:
            ax.set_xlabel("Distance from tip toward well (ft)")
            ax.set_xlim(0, p.xf_ft)
            ax.legend(ncol=2, fontsize=10, frameon=False)
        axes[1, 0].loglog(sol.time_h, sol.total_rate_stb_day, color="#204c70")
        axes[1, 0].scatter(sol.reports_h, sol.total_rate_stb_day[sol.report_indices],
                           c=colors, s=38, zorder=3)
        axes[1, 0].set(title=f"Whole-well rate ({p.wing_factor} wing(s))",
                       xlabel="Time (hours)", ylabel="Rate (STB/day)")
        for j in range(p.n_panels + 1):
            label = f"p{j}" + (" (tip)" if j == 0 else " (well)" if j == p.n_panels else "")
            axes[1, 1].semilogx(sol.time_h, sol.p_pa[:, j] / PSI, label=label)
        axes[1, 1].set(title="Vertex pressure history", xlabel="Time (hours)",
                       ylabel="Pressure (psi)")
        axes[1, 1].legend(frameon=False, fontsize=10)
        # Focus the history plots on the requested times. Very early values
        # remain in full_time_history.csv and diagnostics (coarse-grid startup
        # oscillations are explained in the notebook, not clipped in the solver).
        plot_start_h = max(sol.time_h[0], sol.reports_h[0] / 10)
        visible = sol.time_h >= plot_start_h
        for ax in axes[1]:
            ax.set_xlim(plot_start_h, sol.time_h[-1] * 1.06)
        rates = sol.total_rate_stb_day[visible]
        if np.min(rates) > 0:
            axes[1, 0].set_ylim(np.min(rates) * 0.85, np.max(rates) * 1.15)
            from matplotlib.ticker import LogLocator, ScalarFormatter
            axes[1, 0].yaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
            axes[1, 0].yaxis.set_major_formatter(ScalarFormatter())
        pressure_values = sol.p_pa[visible] / PSI
        pressure_pad = max(10.0, np.ptp(pressure_values) * 0.05)
        axes[1, 1].set_ylim(pressure_values.min() - pressure_pad,
                            pressure_values.max() + pressure_pad)
        for ax in axes.ravel():
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
        control = (f"Fixed BHP = {p.pwf_psi:g} psi" if p.control == "pressure"
                   else f"Fixed total rate = {p.total_rate_stb_day:g} STB/day")
        fig.suptitle(f"{p.n_panels}-panel Darcy model | {control}\n"
                     "Infinite lateral reservoir; symmetric opposite wing included"
                     if p.include_other_wing else
                     f"{p.n_panels}-panel Darcy model | {control}\nInfinite reservoir; single wing",
                     fontsize=16)
    return fig


def save_results(sol, directory="results"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result_table(sol).to_csv(directory / "requested_times.csv", index=False)
    result_table(sol, reports_only=False).to_csv(directory / "full_time_history.csv", index=False)
    p = sol.parameters
    M, b, x = system_at_report(sol)
    matrix = pd.DataFrame(M, columns=unknown_names(p.n_panels),
                          index=equation_names(p.n_panels))
    matrix["rhs_b"] = b
    matrix.to_csv(directory / "matrix_and_rhs_first_output.csv")
    pd.DataFrame({"unknown": unknown_names(p.n_panels), "value": x}).to_csv(
        directory / "solution_first_output.csv", index=False)
    (directory / "parameters.json").write_text(json.dumps(asdict(p), indent=2), encoding="utf-8")
    (directory / "diagnostics.json").write_text(json.dumps(diagnostics(sol), indent=2), encoding="utf-8")
    fig = plot_results(sol)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180)
    (directory / "time_evolution.png").write_bytes(buffer.getvalue())
    return fig


# %% Command-line entry point
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", choices=["pressure", "rate"], default="pressure")
    parser.add_argument("--pwf", type=float, default=3000.0, help="Fixed BHP, psi")
    parser.add_argument("--rate", type=float, default=25.0, help="Whole-well rate, STB/day")
    parser.add_argument("--panels", type=int, default=3, help="Panels per wing")
    parser.add_argument("--times", nargs="+", type=float, default=[1, 2, 3, 5, 10, 60],
                        help="Requested output times, HOURS")
    parser.add_argument("--points-per-decade", type=int, default=60)
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    p = Parameters(n_panels=args.panels, control=args.control, pwf_psi=args.pwf,
                   total_rate_stb_day=args.rate)
    sol = simulate(p, args.times, points_per_decade=args.points_per_decade)
    save_results(sol, args.out)
    columns = ["time_h", "well_rate_total_STB_day", "p0_psi", f"p{p.n_panels}_psi"]
    print(result_table(sol)[columns].to_string(index=False, float_format=lambda v: f"{v:.6g}"))
    print(json.dumps(diagnostics(sol), indent=2))
    print(f"Results: {Path(args.out).resolve()}")


if __name__ == "__main__":
    main()
