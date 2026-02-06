// MinAvg logic module

// Constants
const GRID_SIZE = 10;
const GridState = {
    UNKNOWN: 0,
    VOID: 1,
    BODY: 2,
    HEAD: 3
};

const Direction = {
    UP: 1,
    RIGHT: 2,
    DOWN: 3,
    LEFT: 4
};

const RELATIVE_COORDS = [
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [-2, 0],
    [-3, -1], [-3, 0], [-3, 1],
];

class MinAvgEngine {
    constructor() {
        this.policy = null; // topk_2.json
        this.layouts = null; // All valid layouts
        this.isValid = false;
        
        this.currentState = null; // Array(100)
        this.history = []; // Stack of states for undo
        this.validLayoutIndices = null; // Set of indices into this.layouts
        
        // Start preloading immediately
        this.preloadPromise = this.preloadData();
    }

    async preloadData() {
        try {
            console.log("Preloading MinAvg data...");
            // Load policy
            const policyResp = await fetch('./topk_2.json');
            if (!policyResp.ok) throw new Error(`Failed to load topk_2.json: ${policyResp.status}`);
            this.policy = await policyResp.json();

            // Load layouts
            const layoutsResp = await fetch('./layouts.jsonl');
            if (!layoutsResp.ok) throw new Error(`Failed to load layouts.jsonl: ${layoutsResp.status}`);
            const layoutsText = await layoutsResp.text();
            this.layouts = this.parseLayouts(layoutsText);
            
            console.log("MinAvg data preloaded successfully.");
            return true;
        } catch (e) {
            console.error("Preload failed:", e);
            throw e;
        }
    }

    async init() {
        if (this.isValid) return;
        
        try {
            console.time("MinAvgLoad");
            // Wait for the preloading to finish
            await this.preloadPromise;
            
            this.isValid = true;
            console.log("MinAvg Engine ready. Layouts:", this.layouts.length);
            console.timeEnd("MinAvgLoad");
        } catch (e) {
            console.error("Failed to init MinAvg Engine:", e);
            throw e;
        }
    }

    rotatePoint(dx, dy, direction) {
        if (direction === Direction.UP) return [-dx, -dy];
        if (direction === Direction.DOWN) return [dx, dy];
        if (direction === Direction.LEFT) return [dy, -dx];
        if (direction === Direction.RIGHT) return [-dy, dx];
        return [dx, dy];
    }

    // Parse layouts.jsonl into flat Uint8Arrays representing the grid
    parseLayouts(text) {
        const lines = text.trim().split('\n');
        const grids = [];

        for (const line of lines) {
            try {
                const data = JSON.parse(line);
                const heads = data.h; // [[x,y], [x,y], [x,y]]
                const directions = data.d; // [[d1, d2, d3], ...]

                for (const d of directions) {
                    const grid = new Uint8Array(GRID_SIZE * GRID_SIZE).fill(GridState.VOID);
                    
                    for (let i = 0; i < heads.length; i++) {
                        const [hx, hy] = heads[i];
                        const dir = d[i];
                        
                        // Set head
                        grid[hx * GRID_SIZE + hy] = GridState.HEAD;
                        
                        // Set body
                        for (const [ox, oy] of RELATIVE_COORDS) {
                            const [rx, ry] = this.rotatePoint(ox, oy, dir);
                            const bx = hx + rx;
                            const by = hy + ry;
                            if (bx >= 0 && bx < GRID_SIZE && by >= 0 && by < GRID_SIZE) {
                                grid[bx * GRID_SIZE + by] = GridState.BODY;
                            }
                        }
                    }
                    grids.push(grid);
                }
            } catch (e) {
                console.error("Error parsing layout line", e);
            }
        }
        return grids;
    }

    startGame() {
        this.currentState = new Array(GRID_SIZE * GRID_SIZE).fill(GridState.UNKNOWN);
        this.history = [];
        // Reset valid candidates to all layouts
        this.validLayoutIndices = new Set();
        for(let i=0; i<this.layouts.length; i++) {
            this.validLayoutIndices.add(i);
        }
    }

    // Convert current state to string key for policy lookup
    getStateKey() {
        return this.currentState.join('');
    }

    // Get the next move recommendation
    getNextMove() {
        // 1. Check Policy
        const key = this.getStateKey();
        if (this.policy[key] !== undefined) {
            return {
                type: 'suggest',
                index: this.policy[key]
            };
        }

        // 2. If not in policy, check remaining candidates
        if (this.validLayoutIndices.size === 1) {
            const idx = this.validLayoutIndices.values().next().value;
            return {
                type: 'solved',
                layout: this.layouts[idx]
            };
        } else if (this.validLayoutIndices.size === 0) {
            return {
                type: 'error',
                message: 'No valid layouts match current state.'
            };
        } else {
            // Ambiguous state (multiple layouts remain, but policy has no suggestion)
            // Check if all remaining candidates share the same HEAD positions (or we found 3 heads)
            
            // Check if we found 3 heads in current state
            let headsFound = 0;
            this.currentState.forEach(s => { if(s === GridState.HEAD) headsFound++; });
            
            if (headsFound >= 3) {
                // If 3 heads are found, we treat it as solved.
                // We construct a composite layout for display.
                // 1. Use the first candidate as base
                // 2. Compute intersection of bodies
                
                const indices = Array.from(this.validLayoutIndices);
                const firstLayout = this.layouts[indices[0]];
                
                // Clone layout to avoid mutating original
                const compositeLayout = new Uint8Array(firstLayout);
                
                // If multiple candidates, find intersection
                if (indices.length > 1) {
                    for (let i = 0; i < GRID_SIZE * GRID_SIZE; i++) {
                        // User request: If not unique layout (just unique heads), 
                        // do not show any green body hints.
                        if (compositeLayout[i] === GridState.BODY) {
                            compositeLayout[i] = GridState.VOID; 
                        }
                    }
                }
                
                return {
                    type: 'solved',
                    layout: compositeLayout,
                    message: `Found 3 heads! (${indices.length} orientation variants)`
                };
            }

            return {
                type: 'unknown',
                message: `Ambiguous state (${this.validLayoutIndices.size} candidates remain), but not in policy.`
            };
        }
    }

    // Apply user feedback
    updateState(index, status) {
        // Save history
        this.history.push({
            state: [...this.currentState],
            validIndices: new Set(this.validLayoutIndices)
        });

        // Update grid
        this.currentState[index] = status;

        // Filter candidates
        const nextIndices = new Set();
        for (const idx of this.validLayoutIndices) {
            const layout = this.layouts[idx];
            // Check consistency: 
            // layout[index] must match status (if status is specific)
            // Note: User provides VOID(1), BODY(2), HEAD(3).
            // Layout has VOID(1), BODY(2), HEAD(3).
            if (layout[index] === status) {
                nextIndices.add(idx);
            }
        }
        this.validLayoutIndices = nextIndices;
    }

    undo() {
        if (this.history.length === 0) return false;
        const last = this.history.pop();
        this.currentState = last.state;
        this.validLayoutIndices = last.validIndices;
        return true;
    }

    getGrid() {
        return this.currentState;
    }
}

export const minAvgEngine = new MinAvgEngine();
