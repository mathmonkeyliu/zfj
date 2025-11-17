from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Iterable, List, Optional, Tuple

from .constants import BOARD_SIZE, PLANES_PER_SIDE
from .plane import Orientation, Plane, Segment, Coordinate


class AttackOutcome(Enum):
    MISS = auto()
    HIT = auto()
    HEAD = auto()


@dataclass
class AttackResult:
    coordinate: Coordinate
    outcome: AttackOutcome
    segment: Optional[Segment]
    plane_index: Optional[int]
    plane_destroyed: bool
    repeat: bool = False


class Board:
    def __init__(self, seed: Optional[int] = None) -> None:
        self._rng = random.Random(seed)
        self.planes: List[Plane] = []
        self._occupancy: Dict[Coordinate, Tuple[int, Segment]] = {}
        self._history: Dict[Coordinate, AttackResult] = {}
        self._destroyed: set[int] = set()
        self.reset()

    def reset(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng.seed(seed)
        self.planes.clear()
        self._occupancy.clear()
        self._history.clear()
        self._destroyed.clear()

        attempts = 0
        while len(self.planes) < PLANES_PER_SIDE:
            attempts += 1
            if attempts > 10_000:
                raise RuntimeError("Failed to place all planes on board, please retry or adjust parameters.")

            orientation = self._rng.choice(Plane.all_orientations())
            head = (self._rng.randrange(BOARD_SIZE), self._rng.randrange(BOARD_SIZE))
            plane = Plane(head=head, orientation=orientation)
            if not plane.within_bounds():
                continue

            plane_cells = plane.covered_cells()
            if any(cell in self._occupancy for cell in plane_cells):
                continue

            self._register_plane(plane, plane_cells)

    def _register_plane(self, plane: Plane, cells: Dict[Coordinate, Segment]) -> None:
        idx = len(self.planes)
        self.planes.append(plane)
        for coord, segment in cells.items():
            self._occupancy[coord] = (idx, segment)

    def attack(self, coord: Coordinate) -> AttackResult:
        if coord in self._history:
            result = self._history[coord]
            return AttackResult(
                coordinate=coord,
                outcome=result.outcome,
                segment=result.segment,
                plane_index=result.plane_index,
                plane_destroyed=result.plane_destroyed,
                repeat=True,
            )

        if coord not in self._occupancy:
            result = AttackResult(
                coordinate=coord,
                outcome=AttackOutcome.MISS,
                segment=None,
                plane_index=None,
                plane_destroyed=False,
            )
            self._history[coord] = result
            return result

        plane_index, segment = self._occupancy[coord]
        destroyed = False
        outcome = AttackOutcome.HIT

        if segment == Segment.HEAD:
            outcome = AttackOutcome.HEAD
            destroyed = True
            self._destroyed.add(plane_index)

        result = AttackResult(
            coordinate=coord,
            outcome=outcome,
            segment=segment,
            plane_index=plane_index,
            plane_destroyed=destroyed,
        )
        self._history[coord] = result
        return result

    def remaining_planes(self) -> int:
        return PLANES_PER_SIDE - len(self._destroyed)

    def all_planes_destroyed(self) -> bool:
        return self.remaining_planes() == 0

    def disclosed_cells(self) -> Dict[Coordinate, Tuple[int, Segment]]:
        return dict(self._occupancy)

    def attack_history(self) -> Dict[Coordinate, AttackResult]:
        return dict(self._history)

    def plane_cells(self, plane_index: int) -> Dict[Coordinate, Segment]:
        plane = self.planes[plane_index]
        return plane.covered_cells()
