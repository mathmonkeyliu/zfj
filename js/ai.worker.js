// js/ai.worker.js
// Standalone worker, imports config manually or duplicates constants to avoid ES module complexity in simple workers
// (Browsers support module workers, but for max compatibility we can just inline or use importScripts if needed.
//  However, importScripts doesn't work well with ES6 modules without specific server headers.
//  We will use a classic worker and define constants inside.)

const GRID_SIZE = 10;
const STATE = {
    UNKNOWN: 0,
    VOID: 1,
    BODY: 2,
    HEAD: 3
};

const RAW_BODY_OFFSETS = [
    [-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
    [-2, 0],
    [-3, -1], [-3, 0], [-3, 1]
];

const Engine = {
    singles: [],
    layouts: [], // { grid: Int8Array(100), headHash: number }

    rotatePoint(dx, dy, dirIdx) {
        // 0: UP (-dx, -dy)
        // 1: RIGHT (-dy, dx)
        // 2: DOWN (dx, dy)
        // 3: LEFT (dy, -dx)
        // Match Python config logic:
        /*
         UP (1): -dx, -dy
         RIGHT (2): -dy, dx
         DOWN (3): dx, dy
         LEFT (4): dy, -dx
        */
        switch(dirIdx) {
            case 0: return [-dx, -dy];      // UP
            case 1: return [-dy, dx];       // RIGHT
            case 2: return [dx, dy];        // DOWN
            case 3: return [dy, -dx];       // LEFT
        }
        return [dx, dy];
    },

    generateSingles() {
        this.singles = [];
        const size = GRID_SIZE;
        
        for (let d = 0; d < 4; d++) {
            for (let hx = 0; hx < size; hx++) {
                for (let hy = 0; hy < size; hy++) {
                    const body = [];
                    let valid = true;
                    
                    for (const [ox, oy] of RAW_BODY_OFFSETS) {
                        const [rx, ry] = this.rotatePoint(ox, oy, d);
                        const ax = hx + rx;
                        const ay = hy + ry;
                        
                        if (ax < 0 || ax >= size || ay < 0 || ay >= size) {
                            valid = false;
                            break;
                        }
                        body.push(ax * size + ay);
                    }
                    
                    if (valid) {
                        const headIdx = hx * size + hy;
                        // Use BigInt for bitmask (100 bits)
                        let mask = 1n << BigInt(headIdx);
                        for (const idx of body) {
                            mask |= (1n << BigInt(idx));
                        }
                        
                        this.singles.push({
                            head: headIdx,
                            dir: d,
                            mask: mask,
                            gridIndices: [headIdx, ...body]
                        });
                    }
                }
            }
        }
        // console.log(`Generated ${this.singles.length} single planes.`);
    },

    generateLayouts() {
        this.layouts = [];
        const n = this.singles.length;
        
        // Brute force combinations of 3
        // To avoid freezing even the worker for too long, we can break it up,
        // but workers are background threads so blocking is 'okay' as long as we report progress.
        // But for 64 million iterations (worst case, though fewer valid ones), it might take time.
        // Actually valid combos are ~67,000.
        // The loop is O(N^3) ~ 400^3 = 64,000,000. 
        // JS can do ~10^8 ops per second. It should take < 1-2 seconds.
        
        let checked = 0;
        const totalOps = (n * (n-1) * (n-2)) / 6;
        const reportInterval = Math.floor(totalOps / 20); // Report 20 times

        for (let i = 0; i < n; i++) {
            const p1 = this.singles[i];
            
            for (let j = i + 1; j < n; j++) {
                const p2 = this.singles[j];
                if ((p1.mask & p2.mask) !== 0n) continue;
                
                for (let k = j + 1; k < n; k++) {
                    const p3 = this.singles[k];
                    if ((p1.mask & p3.mask) !== 0n || (p2.mask & p3.mask) !== 0n) continue;
                    
                    // Valid Combo
                    const grid = new Int8Array(100).fill(STATE.VOID);
                    
                    // Mark p1
                    grid[p1.head] = STATE.HEAD;
                    p1.gridIndices.forEach(idx => { if(idx !== p1.head) grid[idx] = STATE.BODY; });
                    
                    // Mark p2
                    grid[p2.head] = STATE.HEAD;
                    p2.gridIndices.forEach(idx => { if(idx !== p2.head) grid[idx] = STATE.BODY; });
                    
                    // Mark p3
                    grid[p3.head] = STATE.HEAD;
                    p3.gridIndices.forEach(idx => { if(idx !== p3.head) grid[idx] = STATE.BODY; });
                    
                    const heads = [p1.head, p2.head, p3.head].sort((a,b) => a-b);
                    // headHash
                    const headHash = heads[0] * 10000 + heads[1] * 100 + heads[2];
                    
                    this.layouts.push({ grid, headHash });
                }
            }
            // Simple progress reporting based on outer loop isn't linear but good enough
            const progress = Math.floor((i / n) * 100);
            if (i % 10 === 0) postMessage({ type: 'PROGRESS', value: progress });
        }
        
        postMessage({ type: 'PROGRESS', value: 100 });
        // console.log(`Generated ${this.layouts.length} total layouts.`);
    },

    symmetryMaps: [],

    generateSymmetryMaps() {
        this.symmetryMaps = [];
        const size = GRID_SIZE;
        // Base index grid
        const base = new Int32Array(size * size);
        for(let i=0; i<size*size; i++) base[i] = i;

        const getIdx = (r, c) => r * size + c;
        const getRC = (idx) => [Math.floor(idx / size), idx % size];

        // 8 transforms: 4 rotations + 4 flipped rotations
        // Rotations: 0, 90, 180, 270
        for(let k=0; k<4; k++) {
            const map = new Int32Array(size * size);
            for(let r=0; r<size; r++) {
                for(let c=0; c<size; c++) {
                    let nr = r, nc = c;
                    // Rotate k times
                    for(let i=0; i<k; i++) {
                        // 90 deg: (r, c) -> (c, size-1-r) ? Or (size-1-c, r)?
                        // Let's stick to numpy rot90 logic used in python: (N-1-c, r)
                        const pr = nr, pc = nc;
                        nr = size - 1 - pc;
                        nc = pr;
                    }
                    map[getIdx(r, c)] = getIdx(nr, nc);
                }
            }
            this.symmetryMaps.push(map);
        }

        // Flips: FlipUD then Rotate k times
        for(let k=0; k<4; k++) {
            const map = new Int32Array(size * size);
            for(let r=0; r<size; r++) {
                for(let c=0; c<size; c++) {
                    // Flip UD first: (size-1-r, c)
                    let nr = size - 1 - r;
                    let nc = c;
                    
                    // Rotate k times
                    for(let i=0; i<k; i++) {
                        const pr = nr, pc = nc;
                        nr = size - 1 - pc;
                        nc = pr;
                    }
                    map[getIdx(r, c)] = getIdx(nr, nc);
                }
            }
            this.symmetryMaps.push(map);
        }
    },

    init() {
        this.generateSingles();
        this.generateSymmetryMaps();
        this.generateLayouts();
        postMessage({ type: 'READY' });
    },

    // Check if observed grid is symmetric under transform map
    checkSymmetry(observed, map) {
        for(let i=0; i<100; i++) {
            const j = map[i];
            // If map[i] = j, it means cell at i moves to j.
            // For symmetry (invariance), the value at j must equal the value at i.
            if (observed[i] !== observed[j]) return false;
        }
        return true;
    },

    getPossibleLayouts(observed) {
        const matches = [];
        const knownIndices = [];
        for(let i=0; i<100; i++) {
            if (observed[i] !== STATE.UNKNOWN) knownIndices.push(i);
        }
        
        if (knownIndices.length === 0) return this.layouts;

        for (const layout of this.layouts) {
            let compatible = true;
            for (const idx of knownIndices) {
                if (layout.grid[idx] !== observed[idx]) {
                    compatible = false;
                    break;
                }
            }
            if (compatible) matches.push(layout);
        }
        return matches;
    },

    recommend(observed) {
        const candidates = this.getPossibleLayouts(observed);
        if (candidates.length === 0) return null; // No solution
        
        const unknownIndices = [];
        for(let i=0; i<100; i++) {
            if (observed[i] === STATE.UNKNOWN) unknownIndices.push(i);
        }
        
        if (unknownIndices.length === 0) return []; // All known

        const headSetCounts = new Map();
        for (const c of candidates) {
            headSetCounts.set(c.headHash, (headSetCounts.get(c.headHash) || 0) + 1);
        }
        const totalCandidates = candidates.length;
        
        // Optimization: Single head set remaining
        if (headSetCounts.size === 1) {
            const sample = candidates[0];
            const targets = [];
            for(let i=0; i<100; i++) {
                if(sample.grid[i] === STATE.HEAD && observed[i] === STATE.UNKNOWN) {
                    targets.push({ idx: i, score: 999 });
                }
            }
            if (targets.length === 0) {
                 for(let i=0; i<100; i++) {
                    if(sample.grid[i] === STATE.BODY && observed[i] === STATE.UNKNOWN) {
                        targets.push({ idx: i, score: 500 });
                    }
                }
            }
            return targets;
        }

        const scores = [];
        
        // Entropy calculation
        for (const idx of unknownIndices) {
            const counts = {
                [STATE.VOID]: new Map(),
                [STATE.BODY]: new Map(),
                [STATE.HEAD]: new Map()
            };
            const stateTotals = { [STATE.VOID]: 0, [STATE.BODY]: 0, [STATE.HEAD]: 0 };
            
            for (const layout of candidates) {
                const s = layout.grid[idx];
                const h = layout.headHash;
                stateTotals[s]++;
                const map = counts[s];
                map.set(h, (map.get(h) || 0) + 1);
            }
            
            let weightedEntropy = 0;
            for (const s of [STATE.VOID, STATE.BODY, STATE.HEAD]) {
                const totalInBranch = stateTotals[s];
                if (totalInBranch === 0) continue;
                
                let branchEntropy = 0;
                for (const count of counts[s].values()) {
                    const p = count / totalInBranch;
                    branchEntropy -= p * Math.log2(p);
                }
                const probBranch = totalInBranch / totalCandidates;
                weightedEntropy += probBranch * branchEntropy;
            }
            
            const headProb = stateTotals[STATE.HEAD] / totalCandidates;
            
            scores.push({
                idx: idx,
                entropy: weightedEntropy,
                headProb: headProb
            });
        }
        
        // Sort: Min Entropy, then Max HeadProb
        scores.sort((a, b) => {
            if (Math.abs(a.entropy - b.entropy) > 0.00001) {
                return a.entropy - b.entropy;
            }
            return b.headProb - a.headProb;
        });

        // Symmetry Deduplication
        const validTransforms = [];
        for (const map of this.symmetryMaps) {
            if (this.checkSymmetry(observed, map)) {
                validTransforms.push(map);
            }
        }
        
        const selected = [];
        const seen = new Set();
        
        for (const item of scores) {
            if (seen.has(item.idx)) continue;
            
            selected.push(item);
            if (selected.length >= 3) break;
            
            seen.add(item.idx);
            // Mark all symmetric equivalents as seen
            for (const map of validTransforms) {
                seen.add(map[item.idx]);
            }
        }
        
        return selected;
    },
    
    getRandomLayout() {
        if (this.layouts.length === 0) return null;
        const randIdx = Math.floor(Math.random() * this.layouts.length);
        return this.layouts[randIdx].grid;
    }
};

onmessage = function(e) {
    const { type, payload } = e.data;
    
    switch (type) {
        case 'INIT':
            Engine.init();
            break;
        case 'RECOMMEND':
            try {
                const result = Engine.recommend(payload.observed);
                postMessage({ type: 'SUGGESTION', candidates: result });
            } catch (err) {
                console.error(err);
                postMessage({ type: 'ERROR', message: err.message });
            }
            break;
        case 'GET_RANDOM_LAYOUT':
            const layout = Engine.getRandomLayout();
            postMessage({ type: 'RANDOM_LAYOUT', layout: layout });
            break;
    }
};
