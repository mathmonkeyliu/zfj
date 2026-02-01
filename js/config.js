// js/config.js
export const GRID_SIZE = 10;
export const STATE = {
    UNKNOWN: 0,
    VOID: 1,
    BODY: 2,
    HEAD: 3
};

// Python: UP=1, RIGHT=2, DOWN=3, LEFT=4
// JS Logic (0-based for arrays): 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
export const DIRECTIONS = [
    { name: 'UP', dx: -1, dy: 0 },
    { name: 'RIGHT', dx: 0, dy: 1 },
    { name: 'DOWN', dx: 1, dy: 0 },
    { name: 'LEFT', dx: 0, dy: -1 }
];

// Relative coords for T-shape body parts relative to head (0,0)
// Base shape points UP (Head at bottom, body extending UP)
// Coordinates are (row_offset, col_offset)
export const RAW_BODY_OFFSETS = [
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2], // Wings
    [-2, 0], // Fuselage
    [-3, -1], [-3, 0], [-3, 1] // Tail
];
