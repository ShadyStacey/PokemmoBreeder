"""
PokeMMO breeding planner.

This script accepts a desired target Pokémon (IVs, nature, and egg moves) and
produces a step-by-step breeding route starting from single 31 IV parents or a
nature parent. The calculations are based on PokeMMO mechanics (braces to lock
IVs, Everstone for nature, and egg moves being carried through the line).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set, Tuple
import argparse
import logging
import sys
import textwrap
import tkinter as tk
from tkinter import ttk

STATS = ["hp", "atk", "def", "spatk", "spdef", "spe"]


@dataclass
class PokemonPlan:
    """Represents a parent or child in the breeding tree."""

    name: str
    perfect_stats: Set[str] = field(default_factory=set)
    nature: Optional[str] = None
    egg_moves: List[str] = field(default_factory=list)
    ability: Optional[str] = None

    def label(self) -> str:
        stats_label = "/".join(sorted(self.perfect_stats)) if self.perfect_stats else "no 31s"
        nature_label = f", {self.nature} nature" if self.nature else ""
        ability_label = f", ability: {self.ability}" if self.ability else ""
        moves_label = f", moves: {', '.join(self.egg_moves)}" if self.egg_moves else ""
        return f"{self.name} ({stats_label}{nature_label}{ability_label}{moves_label})"


@dataclass
class BreedingStep:
    """Instruction for a single breeding action."""

    parent_a: PokemonPlan
    parent_b: PokemonPlan
    child: PokemonPlan
    notes: List[str]

    def describe(self, idx: int) -> str:
        bullet = f"Step {idx}:"
        notes_block = "\n".join(f"    - {note}" for note in self.notes)
        return f"{bullet} Breed {self.parent_a.label()} with {self.parent_b.label()} -> {self.child.label()}\n{notes_block}\n"


@dataclass
class PlanResult:
    """Container for the full breeding roadmap."""

    steps: List[BreedingStep]
    base_parents: List[PokemonPlan]
    final_child: Optional[PokemonPlan]


def parse_ivs(raw: str) -> Dict[str, int]:
    logging.debug("Parsing IVs from raw input: %s", raw)
    ivs = {stat: 0 for stat in STATS}
    if not raw:
        logging.debug("No IV string provided; defaulting all IVs to 0")
        return ivs
    for part in raw.split(","):
        key, value = part.split("=")
        key = key.strip().lower()
        if key not in STATS:
            raise ValueError(f"Unknown IV stat '{key}'. Valid stats: {', '.join(STATS)}")
        ivs[key] = int(value)
        logging.debug("Parsed IV %s=%s", key, ivs[key])
    return ivs


def find_required_perfects(target_ivs: Dict[str, int]) -> Set[str]:
    required = {stat for stat, value in target_ivs.items() if value == 31}
    logging.debug("Required perfect IV stats: %s", required)
    return required


def build_base_parents(required_perfects: Set[str], nature: Optional[str], egg_moves: List[str]) -> List[PokemonPlan]:
    parents: List[PokemonPlan] = []
    for stat in sorted(required_perfects):
        parents.append(PokemonPlan(name=f"Single 31 {stat.upper()}", perfect_stats={stat}))
        logging.debug("Added base parent for stat %s", stat)

    nature_parent_needed = bool(nature)
    if nature_parent_needed:
        parents.append(PokemonPlan(name="Nature parent", nature=nature, perfect_stats=set()))
        logging.debug("Added nature parent for %s", nature)

    if egg_moves:
        parents.append(PokemonPlan(name="Egg move carrier", egg_moves=egg_moves.copy()))
        logging.debug("Added egg move carrier with moves: %s", egg_moves)

    return parents


def merge_plans(plan_a: PokemonPlan, plan_b: PokemonPlan, target_nature: Optional[str]) -> Tuple[PokemonPlan, BreedingStep]:
    merged_stats = plan_a.perfect_stats | plan_b.perfect_stats
    egg_moves = plan_a.egg_moves if plan_a.egg_moves else plan_b.egg_moves

    child = PokemonPlan(
        name=f"{'/'.join(sorted(merged_stats)) or 'Nature'} child",
        perfect_stats=merged_stats,
        nature=target_nature if (plan_a.nature or plan_b.nature) and target_nature else None,
        egg_moves=egg_moves,
    )

    braces = []
    if plan_a.perfect_stats:
        braces.append(f"Brace {', '.join(sorted(plan_a.perfect_stats))} on parent A")
    if plan_b.perfect_stats:
        braces.append(f"Brace {', '.join(sorted(plan_b.perfect_stats))} on parent B")

    notes = []
    if braces:
        notes.extend(braces)
    if target_nature and (plan_a.nature or plan_b.nature):
        notes.append("Give Everstone to the nature parent to lock nature")
    if child.egg_moves:
        notes.append("Make sure the egg moves are on the male parent before breeding")

    logging.debug(
        "Merged %s and %s into %s with notes %s",
        plan_a.label(),
        plan_b.label(),
        child.label(),
        notes,
    )

    return child, BreedingStep(plan_a, plan_b, child, notes)


def build_breeding_plan(
    target_ivs: Dict[str, int], nature: Optional[str], egg_moves: List[str], ability: Optional[str]
) -> PlanResult:
    required_perfects = find_required_perfects(target_ivs)
    parents = build_base_parents(required_perfects, nature, egg_moves)
    base_parents = parents.copy()

    logging.info("Building breeding plan")
    logging.info("Starting parents: %s", [p.label() for p in parents])

    if not parents:
        return PlanResult([], [], None)

    steps: List[BreedingStep] = []
    # Greedy merge: always combine the two smallest parents to minimize cost.
    while len(parents) > 1:
        parents.sort(key=lambda p: (len(p.perfect_stats), p.nature is None, len(p.egg_moves)))
        plan_a, plan_b = parents[0], parents[1]
        logging.debug("Selected parents to merge: %s and %s", plan_a.label(), plan_b.label())
        child, step = merge_plans(plan_a, plan_b, nature)
        steps.append(step)

        # Remove used parents and add new child
        parents = parents[2:]
        parents.append(child)
        logging.debug("Parents after merge: %s", [p.label() for p in parents])

    final_child = parents[0] if parents else None
    if final_child:
        final_child.ability = ability
    return PlanResult(steps, base_parents, final_child)


def format_flowchart(plan: PlanResult) -> str:
    """Render a pyramid shaped diagram from target to base parents."""

    if not plan.final_child:
        return "No steps required."

    step_lookup = {id(step.child): step for step in plan.steps}

    def collect_levels(node: PokemonPlan) -> List[List[PokemonPlan]]:
        if id(node) not in step_lookup:
            return [[node]]

        step = step_lookup[id(node)]
        left_levels = collect_levels(step.parent_a)
        right_levels = collect_levels(step.parent_b)

        merged_children: List[List[PokemonPlan]] = []
        max_depth = max(len(left_levels), len(right_levels))
        for i in range(max_depth):
            level: List[PokemonPlan] = []
            if i < len(left_levels):
                level.extend(left_levels[i])
            if i < len(right_levels):
                level.extend(right_levels[i])
            merged_children.append(level)

        return [[node]] + merged_children

    levels = collect_levels(plan.final_child)
    labels = [[p.label() for p in level] for level in levels]
    max_label = max((len(label) for level in labels for label in level), default=0)
    total_levels = len(labels)

    lines: List[str] = ["Pyramid route", "-------------"]
    for depth, level_labels in enumerate(labels):
        indent_units = (total_levels - depth - 1) * 2
        indent = " " * indent_units
        gap = " " * (4 + (total_levels - depth))
        line = indent + gap.join(label.center(max_label) for label in level_labels)
        lines.append(line)
        if depth < total_levels - 1:
            connector = indent + gap.join("  ↓  ".center(max_label) for _ in level_labels)
            lines.append(connector)

    return "\n".join(lines)


def summarize_resources(plan: PlanResult, nature: Optional[str], ability: Optional[str]) -> Tuple[str, Dict[str, int]]:
    brace_counts: Dict[str, int] = {stat: 0 for stat in STATS}
    everstone_count = 0
    for step in plan.steps:
        for parent in (step.parent_a, step.parent_b):
            for stat in parent.perfect_stats:
                brace_counts[stat] += 1
        if nature and (step.parent_a.nature or step.parent_b.nature):
            everstone_count += 1

    ability_pills = 1 if ability else 0

    items_lines = ["Required items:"]
    total_braces = 0
    for stat, count in brace_counts.items():
        if count:
            items_lines.append(f"- {count}x {stat.upper()} brace")
            total_braces += count
    if everstone_count:
        items_lines.append(f"- {everstone_count}x Everstone")
    if ability_pills:
        items_lines.append(f"- {ability_pills}x Ability Pill (set ability to {ability})")
    if not items_lines[1:]:
        items_lines.append("- None")

    pokemon_lines = ["Starting parents (assumed 1x31 or nature-only):"]
    for parent in plan.base_parents:
        pokemon_lines.append(f"- {parent.label()}")

    summary = "\n".join(items_lines + ["", "Required Pokémon:"] + pokemon_lines)
    return summary, {"total_braces": total_braces, "everstones": everstone_count, "ability_pills": ability_pills}


def estimate_costs(costs: Dict[str, int], counts: Dict[str, int], base_parents: Iterable[PokemonPlan]) -> Tuple[str, int]:
    brace_total = counts.get("total_braces", 0) * costs.get("brace", 0)
    everstone_total = counts.get("everstones", 0) * costs.get("everstone", 0)
    pill_total = counts.get("ability_pills", 0) * costs.get("ability_pill", 0)

    base_list = list(base_parents)
    parent_total = 0
    for parent in base_list:
        if parent.nature:
            parent_total += costs.get("nature_parent", 0)
        elif parent.egg_moves:
            parent_total += costs.get("egg_carrier", 0)
        elif parent.perfect_stats:
            parent_total += costs.get("stat_parent", 0)

    total = brace_total + everstone_total + pill_total + parent_total
    breakdown = ["Cost estimate:"]
    breakdown.append(f"- Braces: {counts.get('total_braces', 0)} x {costs.get('brace', 0)} = {brace_total}")
    breakdown.append(f"- Everstones: {counts.get('everstones', 0)} x {costs.get('everstone', 0)} = {everstone_total}")
    if counts.get("ability_pills", 0):
        breakdown.append(
            f"- Ability Pills: {counts.get('ability_pills', 0)} x {costs.get('ability_pill', 0)} = {pill_total}"
        )
    breakdown.append(f"- Base parents: {len(base_list)} x (custom) = {parent_total}")
    breakdown.append(f"Total estimated cost: {total}")
    return "\n".join(breakdown), total


def format_plan(
    plan: PlanResult,
    target_ivs: Dict[str, int],
    nature: Optional[str],
    egg_moves: List[str],
    ability: Optional[str],
    cost_summary: Optional[str] = None,
) -> str:
    header = ["Breeding roadmap", "================", ""]
    header.append("Target summary:")
    header.append("- IVs: " + ", ".join(f"{stat.upper()}={value}" for stat, value in target_ivs.items()))
    header.append(f"- Nature: {nature or 'any'}")
    header.append(f"- Ability: {ability or 'any'} (Ability Pill after breeding)")
    header.append(f"- Egg moves: {', '.join(egg_moves) if egg_moves else 'none'}")
    header.append("")

    if not plan.steps:
        header.append("No steps required. You already have the target Pokémon!")
        if ability:
            header.append(f"Use an Ability Pill to set the ability to {ability} if needed.")
        return "\n".join(header)

    body = [step.describe(idx + 1) for idx, step in enumerate(plan.steps)]
    flowchart = format_flowchart(plan)
    resources_text, counts = summarize_resources(plan, nature, ability)
    cost_block = cost_summary or ""
    ability_note = f"Use an Ability Pill on the final Pokémon to set its ability to {ability}." if ability else ""
    tips = textwrap.dedent(
        """
        Extra tips:
        - If you prefer cheaper braces, merge stats in pairs (e.g., HP/DEF with ATK/SPDEF) before combining into quads and sextuple 31s.
        - Consider using gender selection to keep egg move inheritance straightforward.
        - Keep track of costs; each brace is 10k BP or 20k Pokéyen in PokeMMO.
        """
    )

    sections = header + body + [flowchart, resources_text]
    if ability_note:
        sections.append("")
        sections.append(ability_note)
    if cost_block:
        sections.append("")
        sections.append(cost_block)
    sections.append(tips)
    return "\n".join(sections)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.debug("Logging initialized. Verbose=%s", verbose)


def run_gui() -> None:
    window = tk.Tk()
    window.title("PokeMMO Breeding Planner")

    main_frame = ttk.Frame(window, padding=10)
    main_frame.grid(row=0, column=0, sticky="nsew")
    window.columnconfigure(0, weight=1)
    window.rowconfigure(0, weight=1)

    # Inputs
    ttk.Label(main_frame, text="Target IVs (hp=31,atk=31,...):").grid(row=0, column=0, sticky="w")
    ivs_var = tk.StringVar(value="hp=31,atk=31,def=31,spatk=31,spdef=31,spe=31")
    ttk.Entry(main_frame, textvariable=ivs_var, width=60).grid(row=1, column=0, sticky="ew")

    ttk.Label(main_frame, text="Nature (optional):").grid(row=2, column=0, sticky="w", pady=(8, 0))
    nature_var = tk.StringVar()
    ttk.Entry(main_frame, textvariable=nature_var, width=30).grid(row=3, column=0, sticky="ew")

    ttk.Label(main_frame, text="Egg moves (comma-separated):").grid(row=4, column=0, sticky="w", pady=(8, 0))
    egg_var = tk.StringVar()
    ttk.Entry(main_frame, textvariable=egg_var, width=60).grid(row=5, column=0, sticky="ew")

    ttk.Label(main_frame, text="Ability (optional, set with Ability Pill):").grid(row=6, column=0, sticky="w", pady=(8, 0))
    ability_var = tk.StringVar()
    ttk.Entry(main_frame, textvariable=ability_var, width=30).grid(row=7, column=0, sticky="ew")

    # Cost inputs
    ttk.Label(main_frame, text="Brace price:").grid(row=8, column=0, sticky="w", pady=(8, 0))
    brace_price_var = tk.StringVar(value="20000")
    ttk.Entry(main_frame, textvariable=brace_price_var, width=20).grid(row=9, column=0, sticky="w")

    ttk.Label(main_frame, text="Everstone price:").grid(row=10, column=0, sticky="w", pady=(4, 0))
    everstone_price_var = tk.StringVar(value="20000")
    ttk.Entry(main_frame, textvariable=everstone_price_var, width=20).grid(row=11, column=0, sticky="w")

    ttk.Label(main_frame, text="Ability Pill price:").grid(row=12, column=0, sticky="w", pady=(4, 0))
    ability_pill_price_var = tk.StringVar(value="35000")
    ttk.Entry(main_frame, textvariable=ability_pill_price_var, width=20).grid(row=13, column=0, sticky="w")

    ttk.Label(main_frame, text="Base parent prices (31 IV / nature / egg carrier):").grid(row=14, column=0, sticky="w", pady=(8, 0))
    stat_parent_price_var = tk.StringVar(value="0")
    nature_parent_price_var = tk.StringVar(value="0")
    egg_carrier_price_var = tk.StringVar(value="0")
    prices_frame = ttk.Frame(main_frame)
    prices_frame.grid(row=15, column=0, sticky="w")
    ttk.Entry(prices_frame, textvariable=stat_parent_price_var, width=10).grid(row=0, column=0, padx=(0, 4))
    ttk.Entry(prices_frame, textvariable=nature_parent_price_var, width=10).grid(row=0, column=1, padx=(0, 4))
    ttk.Entry(prices_frame, textvariable=egg_carrier_price_var, width=10).grid(row=0, column=2, padx=(0, 4))

    output = tk.Text(main_frame, width=80, height=25, wrap="word")
    output.grid(row=17, column=0, sticky="nsew", pady=(10, 0))

    main_frame.rowconfigure(17, weight=1)
    main_frame.columnconfigure(0, weight=1)

    def on_plan() -> None:
        logging.info("GUI: Plan requested")
        try:
            target_ivs = parse_ivs(ivs_var.get())
            egg_moves = [move.strip() for move in egg_var.get().split(",") if move.strip()]
            plan = build_breeding_plan(target_ivs, nature_var.get() or None, egg_moves, ability_var.get() or None)
            _, counts = summarize_resources(plan, nature_var.get() or None, ability_var.get() or None)
            costs = {
                "brace": int(brace_price_var.get() or 0),
                "everstone": int(everstone_price_var.get() or 0),
                "ability_pill": int(ability_pill_price_var.get() or 0),
                "stat_parent": int(stat_parent_price_var.get() or 0),
                "nature_parent": int(nature_parent_price_var.get() or 0),
                "egg_carrier": int(egg_carrier_price_var.get() or 0),
            }
            cost_summary, _ = estimate_costs(costs, counts, plan.base_parents)
            plan_text = format_plan(
                plan,
                target_ivs,
                nature_var.get() or None,
                egg_moves,
                ability_var.get() or None,
                cost_summary,
            )
        except Exception as exc:  # noqa: BLE001 - surface any GUI errors directly
            logging.exception("Error while building plan")
            plan_text = f"Error: {exc}"

        output.delete("1.0", tk.END)
        output.insert(tk.END, plan_text)
        output.see(tk.END)

    ttk.Button(main_frame, text="Generate Plan", command=on_plan).grid(row=16, column=0, sticky="e", pady=(8, 0))

    window.mainloop()


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PokeMMO breeding planner")
    parser.add_argument("--ivs", default="", help="Comma list of stat=value (hp,atk,def,spatk,spdef,spe). Example: hp=31,atk=31,def=31")
    parser.add_argument("--nature", default=None, help="Desired nature (e.g., Adamant).")
    parser.add_argument("--ability", default=None, help="Desired ability to set with an Ability Pill after breeding.")
    parser.add_argument("--egg-moves", default="", help="Comma-separated egg moves to preserve")
    parser.add_argument("--brace-price", type=int, default=20000, help="Price per brace in Pokéyen")
    parser.add_argument("--everstone-price", type=int, default=20000, help="Price per Everstone in Pokéyen")
    parser.add_argument("--ability-pill-price", type=int, default=35000, help="Price per Ability Pill in Pokéyen")
    parser.add_argument("--stat-parent-price", type=int, default=0, help="Price per 1x31 base parent")
    parser.add_argument("--nature-parent-price", type=int, default=0, help="Price for the nature parent")
    parser.add_argument("--egg-carrier-price", type=int, default=0, help="Price for the egg move carrier")
    parser.add_argument("--gui", action="store_true", help="Launch the visual planner interface")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose debug logging")
    args = parser.parse_args(argv)

    launch_gui = args.gui or (argv is None and len(sys.argv) == 1)
    setup_logging(args.verbose)
    logging.info("Application started. GUI=%s", launch_gui)

    if launch_gui:
        run_gui()
        return

    target_ivs = parse_ivs(args.ivs)
    egg_moves = [move.strip() for move in args.egg_moves.split(",") if move.strip()]

    logging.info("Target IVs: %s", target_ivs)
    logging.info("Target nature: %s", args.nature)
    logging.info("Target ability: %s", args.ability)
    logging.info("Egg moves: %s", egg_moves)

    plan = build_breeding_plan(target_ivs, args.nature, egg_moves, args.ability)
    _, counts = summarize_resources(plan, args.nature, args.ability)
    costs = {
        "brace": args.brace_price,
        "everstone": args.everstone_price,
        "ability_pill": args.ability_pill_price,
        "stat_parent": args.stat_parent_price,
        "nature_parent": args.nature_parent_price,
        "egg_carrier": args.egg_carrier_price,
    }
    cost_summary, _ = estimate_costs(costs, counts, plan.base_parents)
    print(format_plan(plan, target_ivs, args.nature, egg_moves, args.ability, cost_summary))


if __name__ == "__main__":
    main()
