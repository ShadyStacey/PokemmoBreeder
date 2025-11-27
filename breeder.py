"""
PokeMMO breeding planner.

This script accepts a desired target Pokémon (IVs, nature, and egg moves) and
produces a step-by-step breeding route starting from single 31 IV parents or a
nature parent. The calculations are based on PokeMMO mechanics (braces to lock
IVs, Everstone for nature, and egg moves being carried through the line).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import argparse
import textwrap

STATS = ["hp", "atk", "def", "spatk", "spdef", "spe"]


@dataclass
class PokemonPlan:
    """Represents a parent or child in the breeding tree."""

    name: str
    perfect_stats: Set[str] = field(default_factory=set)
    nature: Optional[str] = None
    egg_moves: List[str] = field(default_factory=list)

    def label(self) -> str:
        stats_label = "/".join(sorted(self.perfect_stats)) if self.perfect_stats else "no 31s"
        nature_label = f", {self.nature} nature" if self.nature else ""
        moves_label = f", moves: {', '.join(self.egg_moves)}" if self.egg_moves else ""
        return f"{self.name} ({stats_label}{nature_label}{moves_label})"


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


def parse_ivs(raw: str) -> Dict[str, int]:
    ivs = {stat: 0 for stat in STATS}
    if not raw:
        return ivs
    for part in raw.split(","):
        key, value = part.split("=")
        key = key.strip().lower()
        if key not in STATS:
            raise ValueError(f"Unknown IV stat '{key}'. Valid stats: {', '.join(STATS)}")
        ivs[key] = int(value)
    return ivs


def find_required_perfects(target_ivs: Dict[str, int]) -> Set[str]:
    return {stat for stat, value in target_ivs.items() if value == 31}


def build_base_parents(required_perfects: Set[str], nature: Optional[str], egg_moves: List[str]) -> List[PokemonPlan]:
    parents: List[PokemonPlan] = []
    for stat in sorted(required_perfects):
        parents.append(PokemonPlan(name=f"Single 31 {stat.upper()}", perfect_stats={stat}))

    nature_parent_needed = bool(nature)
    if nature_parent_needed:
        parents.append(PokemonPlan(name="Nature parent", nature=nature, perfect_stats=set()))

    if egg_moves:
        parents.append(PokemonPlan(name="Egg move carrier", egg_moves=egg_moves.copy()))

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

    return child, BreedingStep(plan_a, plan_b, child, notes)


def build_breeding_plan(target_ivs: Dict[str, int], nature: Optional[str], egg_moves: List[str]) -> List[BreedingStep]:
    required_perfects = find_required_perfects(target_ivs)
    parents = build_base_parents(required_perfects, nature, egg_moves)

    if not parents:
        return []

    steps: List[BreedingStep] = []
    # Greedy merge: always combine the two smallest parents to minimize cost.
    while len(parents) > 1:
        parents.sort(key=lambda p: (len(p.perfect_stats), p.nature is None, len(p.egg_moves)))
        plan_a, plan_b = parents[0], parents[1]
        child, step = merge_plans(plan_a, plan_b, nature)
        steps.append(step)

        # Remove used parents and add new child
        parents = parents[2:]
        parents.append(child)

    return steps


def format_plan(steps: List[BreedingStep], target_ivs: Dict[str, int], nature: Optional[str], egg_moves: List[str]) -> str:
    header = ["Breeding roadmap", "================", ""]
    header.append("Target summary:")
    header.append("- IVs: " + ", ".join(f"{stat.upper()}={value}" for stat, value in target_ivs.items()))
    header.append(f"- Nature: {nature or 'any'}")
    header.append(f"- Egg moves: {', '.join(egg_moves) if egg_moves else 'none'}")
    header.append("")

    if not steps:
        header.append("No steps required. You already have the target Pokémon!")
        return "\n".join(header)

    body = [step.describe(idx + 1) for idx, step in enumerate(steps)]
    tips = textwrap.dedent(
        """
        Extra tips:
        - If you prefer cheaper braces, merge stats in pairs (e.g., HP/DEF with ATK/SPDEF) before combining into quads and sextuple 31s.
        - Consider using gender selection to keep egg move inheritance straightforward.
        - Keep track of costs; each brace is 10k BP or 20k Pokéyen in PokeMMO.
        """
    )

    return "\n".join(header + body + [tips])


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="PokeMMO breeding planner")
    parser.add_argument("--ivs", default="", help="Comma list of stat=value (hp,atk,def,spatk,spdef,spe). Example: hp=31,atk=31,def=31")
    parser.add_argument("--nature", default=None, help="Desired nature (e.g., Adamant).")
    parser.add_argument("--egg-moves", default="", help="Comma-separated egg moves to preserve")
    args = parser.parse_args(argv)

    target_ivs = parse_ivs(args.ivs)
    egg_moves = [move.strip() for move in args.egg_moves.split(",") if move.strip()]

    steps = build_breeding_plan(target_ivs, args.nature, egg_moves)
    print(format_plan(steps, target_ivs, args.nature, egg_moves))


if __name__ == "__main__":
    main()
