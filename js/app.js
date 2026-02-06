// js/app.js
import { STATE } from './config.js';
import { minAvgEngine } from './min_avg.js';

const App = {
    worker: null,
    board: new Array(100).fill(STATE.UNKNOWN),
    trueLayout: null,
    mode: 'play', // 'play' | 'explore'
    algorithm: 'id3', // 'id3' | 'minavg'
    paletteSelected: STATE.VOID,
    isAIReady: false,
    
    // Min Avg State
    minAvgCurrentSuggestion: null, // index
    
    init() {
        console.log("App initializing...");
        this.renderBoard();
        this.startWorker();
        
        // Resize Listener for Square Board
        window.addEventListener('resize', () => this.handleResize());
        // Call once to set initial size
        requestAnimationFrame(() => this.handleResize());

        // Event Listeners - Common
        const modeLink = document.getElementById('mode-link');
        if (modeLink) {
            modeLink.onclick = (e) => {
                e.preventDefault();
                this.toggleMode();
            };
        }

        // Algorithm Toggles
        const btnId3 = document.getElementById('btn-algo-id3');
        const btnMinAvg = document.getElementById('btn-algo-minavg');
        
        if (btnId3) btnId3.onclick = () => this.setAlgorithm('id3');
        if (btnMinAvg) btnMinAvg.onclick = () => {
            console.log("Min Avg button clicked");
            this.setAlgorithm('minavg');
        };

        // ... rest of init

        // Play Mode Controls (ID3)
        document.getElementById('play-controls').querySelector('.btn-reset').onclick = () => this.resetGame();
        document.getElementById('play-controls').querySelector('.btn-ask').onclick = () => this.askAI();

        // Explore Mode Controls (ID3)
        document.getElementById('explore-controls').querySelector('.btn-clear').onclick = () => this.clearBoard();
        document.getElementById('explore-controls').querySelector('.btn-ask').onclick = () => this.askAI();
        
        // Palette listeners (Explore)
        document.getElementById('pal-1').onclick = () => this.setPalette(STATE.VOID);
        document.getElementById('pal-2').onclick = () => this.setPalette(STATE.BODY);
        document.getElementById('pal-3').onclick = () => this.setPalette(STATE.HEAD);

        // Min Avg Controls
        const minAvgControls = document.getElementById('min-avg-controls');
        minAvgControls.querySelector('.f-void').onclick = () => this.handleMinAvgFeedback(STATE.VOID);
        minAvgControls.querySelector('.f-body').onclick = () => this.handleMinAvgFeedback(STATE.BODY);
        minAvgControls.querySelector('.f-head').onclick = () => this.handleMinAvgFeedback(STATE.HEAD);
        minAvgControls.querySelector('.btn-undo').onclick = () => this.undoMinAvg();
        minAvgControls.querySelector('.btn-reset').onclick = () => this.startMinAvgGame();

        // Initial UI State
        this.setAlgorithm('id3');
    },

    async setAlgorithm(algo) {
        console.log(`Switching to algorithm: ${algo}`);
        if (this.algorithm === algo && this.isAIReady) return;
        this.algorithm = algo;

        // Update Buttons
        document.querySelectorAll('.btn-algo').forEach(b => b.classList.remove('active'));
        const activeBtn = document.getElementById(`btn-algo-${algo}`);
        if (activeBtn) activeBtn.classList.add('active');

        // Hide/Show Logic
        const id3Elements = [
            document.getElementById('mode-switch-container'),
            document.getElementById('play-controls'), 
            document.getElementById('explore-controls') // will be managed by toggleMode logic
        ];
        const minAvgElements = [
            document.getElementById('min-avg-controls'),
            document.getElementById('min-avg-overlay')
        ];

        if (algo === 'minavg') {
            // Switch to Min Avg
            id3Elements.forEach(el => { if(el) el.style.display = 'none'; });
            // Explore controls are hidden by default in minavg
            const exploreControls = document.getElementById('explore-controls');
            if (exploreControls) exploreControls.style.display = 'none';
            
            const minAvgControls = document.getElementById('min-avg-controls');
            if (minAvgControls) minAvgControls.style.display = 'flex';
            
            const modeLink = document.getElementById('mode-link');
            if (modeLink) modeLink.style.display = 'none'; // Hide mode switch in Min Avg
            
            // Init Engine if needed
            if (!minAvgEngine.isValid) {
                this.setStatus("正在加载 Min Avg 数据(约6MB)...");
                const btn = document.getElementById('btn-algo-minavg');
                const originalText = btn.innerText;
                btn.innerText = "加载中...";
                btn.disabled = true;

                try {
                    console.time("MinAvgLoad");
                    await minAvgEngine.init();
                    console.timeEnd("MinAvgLoad");
                } catch (e) {
                    console.error("Min Avg Init Failed:", e);
                    this.setStatus("加载失败: " + e.message);
                    btn.innerText = originalText;
                    btn.disabled = false;
                    return;
                }
                
                btn.innerText = originalText;
                btn.disabled = false;
            }
            this.startMinAvgGame();

        } else {
            // Switch to ID3
            id3Elements.forEach(el => { if(el) el.style.display = ''; });
            const modeLink = document.getElementById('mode-link');
            if (modeLink) modeLink.style.display = 'block';
            
            minAvgElements.forEach(el => { if(el) el.style.display = 'none'; });
            
            // Restore proper ID3 mode state
            this.toggleMode(this.mode, true); // Force update UI
            this.clearOverlay();
        }
        
        // Recalculate layout after UI changes
        requestAnimationFrame(() => this.handleResize());
    },

    // --- Min Avg Logic ---

    startMinAvgGame() {
        minAvgEngine.startGame();
        this.board = minAvgEngine.getGrid(); // Reference to engine's grid
        this.clearOverlay();
        this.updateBoardUI();
        this.updateMinAvgUI();
    },

    updateMinAvgUI() {
        const result = minAvgEngine.getNextMove();
        this.minAvgCurrentSuggestion = null;
        this.clearHints(); // Remove ID3 hints
        this.clearHighlight();

        const undoBtn = document.getElementById('min-avg-controls').querySelector('.btn-undo');
        undoBtn.disabled = minAvgEngine.history.length === 0;

        if (result.type === 'suggest') {
            this.minAvgCurrentSuggestion = result.index;
            this.highlightCell(result.index);
            this.setStatus("请探测高亮格子，并告知结果");
            this.setInstruction("AI 建议探测高亮格子");
            this.enableFeedbackButtons(true);
        } else if (result.type === 'solved') {
            this.setStatus("已找到唯一布局！");
            this.setInstruction("分析完成");
            this.showOverlay(result.layout);
            this.enableFeedbackButtons(false);
        } else if (result.type === 'error') {
            this.setStatus(result.message);
            this.enableFeedbackButtons(false);
        } else {
            this.setStatus(result.message || "未知状态");
        }
        
        // Sync board visualization
        this.board = minAvgEngine.getGrid();
        this.updateBoardUI();
    },

    handleMinAvgFeedback(status) {
        if (this.minAvgCurrentSuggestion === null) return;
        
        minAvgEngine.updateState(this.minAvgCurrentSuggestion, status);
        this.updateMinAvgUI();
    },

    undoMinAvg() {
        if (minAvgEngine.undo()) {
            this.clearOverlay();
            this.updateMinAvgUI();
        }
    },

    enableFeedbackButtons(enabled) {
        const btns = document.querySelectorAll('.btn-feedback');
        btns.forEach(b => b.disabled = !enabled);
    },

    setInstruction(text) {
        document.getElementById('min-avg-instruction').innerText = text;
    },

    highlightCell(idx) {
        const cell = document.getElementById(`cell-${idx}`);
        if (cell) cell.classList.add('highlight');
    },

    clearHighlight() {
        document.querySelectorAll('.cell.highlight').forEach(el => el.classList.remove('highlight'));
    },

    showOverlay(layout) {
        const overlay = document.getElementById('min-avg-overlay');
        overlay.style.display = 'grid';
        overlay.innerHTML = '';
        
        for (let i = 0; i < 100; i++) {
            const div = document.createElement('div');
            div.className = 'overlay-cell';
            if (layout[i] === STATE.HEAD) {
                div.classList.add('head-hint');
            } else if (layout[i] === STATE.BODY) {
                div.classList.add('body-hint');
            }
            overlay.appendChild(div);
        }
    },

    clearOverlay() {
        const overlay = document.getElementById('min-avg-overlay');
        overlay.style.display = 'none';
        overlay.innerHTML = '';
    },


    // --- ID3 Logic (Existing) ---

    startWorker() {
        this.worker = new Worker('js/ai.worker.js');
        
        this.worker.onmessage = (e) => {
            const { type, value, candidates, layout, message } = e.data;
            
            switch (type) {
                case 'PROGRESS':
                    document.getElementById('loading-progress').innerText = `${value}%`;
                    break;
                case 'READY':
                    this.isAIReady = true;
                    document.getElementById('loading').style.display = 'none';
                    document.getElementById('app').style.display = ''; // Let CSS Grid take over
                    
                    // Trigger resize now that app is visible
                    requestAnimationFrame(() => this.handleResize());

                    // Don't auto-reset if in MinAvg mode, but initially we are in ID3
                    if (this.algorithm === 'id3') this.resetGame();
                    break;
                case 'SUGGESTION':
                    if (this.algorithm === 'id3') this.handleAISuggestion(candidates);
                    break;
                case 'RANDOM_LAYOUT':
                    this.trueLayout = layout;
                    this.board.fill(STATE.UNKNOWN);
                    this.clearHints();
                    this.updateBoardUI();
                    this.setStatus("新游戏开始");
                    break;
                case 'ERROR':
                    console.error("Worker Error:", message);
                    this.setStatus("AI 出错: " + message);
                    break;
            }
        };

        // Start init
        this.worker.postMessage({ type: 'INIT' });
    },

    toggleMode(targetMode, force = false) {
        if (!force) {
            this.mode = this.mode === 'play' ? 'explore' : 'play';
        } else if (targetMode) {
            this.mode = targetMode;
        }

        const link = document.getElementById('mode-link');
        const title = document.getElementById('title');
        const playControls = document.getElementById('play-controls');
        const exploreControls = document.getElementById('explore-controls');
        
        if (this.algorithm === 'minavg') {
            // Should not happen as controls are hidden, but safety check
            return;
        }

        if (this.mode === 'play') {
            link.innerText = "进入自由探索模式";
            title.innerText = "实战演练";
            playControls.style.display = 'flex';
            exploreControls.style.display = 'none';
            // Only reset if forcing or switching manually
            if (!force || targetMode === 'play') this.resetGame();
        } else {
            link.innerText = "返回实战演练模式";
            title.innerText = "自由探索";
            playControls.style.display = 'none';
            exploreControls.style.display = 'flex';
            this.clearBoard();
            this.setPalette(STATE.VOID);
        }
        
        // Recalculate layout after UI changes
        requestAnimationFrame(() => this.handleResize());
    },

    resetGame() {
        if (!this.isAIReady || this.algorithm !== 'id3') return;
        this.worker.postMessage({ type: 'GET_RANDOM_LAYOUT' });
    },

    clearBoard() {
        this.board.fill(STATE.UNKNOWN);
        this.clearHints();
        this.updateBoardUI();
        this.setStatus("棋盘已清空");
    },

    setPalette(state) {
        this.paletteSelected = state;
        document.querySelectorAll('.palette-item').forEach(el => el.classList.remove('active'));
        // Map STATE to ID suffix: VOID(1), BODY(2), HEAD(3)
        document.getElementById(`pal-${state}`).classList.add('active');
    },

    handleClick(idx) {
        // ID3 Mode Interaction
        if (this.algorithm !== 'id3') return;

        this.clearHints();

        if (this.mode === 'play') {
            if (this.board[idx] !== STATE.UNKNOWN) return;
            if (!this.trueLayout) return;

            const result = this.trueLayout[idx];
            this.board[idx] = result;
            this.updateCellUI(idx);

            // Win check
            let heads = 0;
            this.board.forEach(s => { if(s === STATE.HEAD) heads++; });
            if (heads === 3) {
                this.setStatus("恭喜！你找到了所有飞机！");
                // Reveal all
                for(let i=0; i<100; i++) {
                    if(this.board[i] === STATE.UNKNOWN) {
                        this.board[i] = this.trueLayout[i];
                        this.updateCellUI(i);
                    }
                }
            } else {
                this.setStatus("...");
            }

        } else {
            // Explore mode
            if (this.board[idx] === this.paletteSelected) {
                this.board[idx] = STATE.UNKNOWN;
            } else {
                this.board[idx] = this.paletteSelected;
            }
            this.updateCellUI(idx);
        }
    },

    askAI() {
        if (!this.isAIReady) return;
        this.clearHints();
        this.setStatus("AI 思考中...");
        
        this.worker.postMessage({ 
            type: 'RECOMMEND', 
            payload: { observed: this.board } 
        });
    },

    handleAISuggestion(candidates) {
        if (!candidates) {
            this.setStatus("无解 / 矛盾的布局");
            return;
        }
        if (candidates.length === 0) {
            this.setStatus("所有格子已知或无更多信息");
            return;
        }

        const top3 = candidates.slice(0, 3);
        top3.forEach((s, i) => {
            const el = document.getElementById(`cell-${s.idx}`);
            if (el) {
                const hint = document.createElement('div');
                hint.className = 'ai-hint';
                hint.innerText = i + 1;
                el.appendChild(hint);
            }
        });

        this.setStatus(`AI 推荐了 ${top3.length} 个位置`);
    },

    renderBoard() {
        const boardEl = document.getElementById('board');
        boardEl.innerHTML = '';
        for (let i = 0; i < 100; i++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.id = `cell-${i}`;
            cell.onclick = () => this.handleClick(i);
            boardEl.appendChild(cell);
        }
    },

    updateBoardUI() {
        for (let i = 0; i < 100; i++) {
            this.updateCellUI(i);
        }
    },

    updateCellUI(idx) {
        const cell = document.getElementById(`cell-${idx}`);
        cell.className = 'cell'; // reset
        const s = this.board[idx];
        if (s === STATE.VOID) cell.classList.add('void');
        if (s === STATE.BODY) cell.classList.add('body');
        if (s === STATE.HEAD) cell.classList.add('head');
        
        // Preserve highlight if exists and logic allows
        if (this.algorithm === 'minavg' && this.minAvgCurrentSuggestion === idx) {
            cell.classList.add('highlight');
        }
    },

    clearHints() {
        document.querySelectorAll('.ai-hint').forEach(e => e.remove());
    },

    setStatus(text) {
        document.getElementById('status').innerText = text;
    },

    handleResize() {
        const boardContainer = document.querySelector('.layout-board');
        const wrapper = document.querySelector('.board-wrapper');
        const status = document.getElementById('status');
        
        if (!boardContainer || !wrapper) return;

        // Get available space in the layout-board container
        // We need to subtract padding and the height of the status text
        const containerStyle = window.getComputedStyle(boardContainer);
        const padX = parseFloat(containerStyle.paddingLeft) + parseFloat(containerStyle.paddingRight);
        const padY = parseFloat(containerStyle.paddingTop) + parseFloat(containerStyle.paddingBottom);
        
        const availableWidth = boardContainer.clientWidth - padX;
        // Subtract status height and margin
        const statusHeight = status ? status.offsetHeight + parseFloat(window.getComputedStyle(status).marginBottom) : 0;
        const availableHeight = boardContainer.clientHeight - padY - statusHeight;

        // The size should be the minimum of width and height to ensure square
        // Use 98% to leave a tiny breathing room/border safety
        const size = Math.floor(Math.min(availableWidth, availableHeight) * 0.98);

        wrapper.style.width = `${size}px`;
        wrapper.style.height = `${size}px`;
        // Force reset flex/max constraints that might be in CSS
        wrapper.style.maxWidth = 'none';
        wrapper.style.maxHeight = 'none';
    }
};

window.onload = () => App.init();
