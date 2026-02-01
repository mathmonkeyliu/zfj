// js/app.js
import { STATE } from './config.js';

const App = {
    worker: null,
    board: new Array(100).fill(STATE.UNKNOWN),
    trueLayout: null,
    mode: 'play', // 'play' | 'explore'
    paletteSelected: STATE.VOID,
    isAIReady: false,
    
    init() {
        this.renderBoard();
        this.startWorker();
        
        // Event Listeners
        document.getElementById('play-controls').querySelector('.btn-reset').onclick = () => this.resetGame();
        document.getElementById('play-controls').querySelector('.btn-ask').onclick = () => this.askAI();
        document.getElementById('explore-controls').querySelector('.btn-clear').onclick = () => this.clearBoard();
        document.getElementById('explore-controls').querySelector('.btn-ask').onclick = () => this.askAI();
        document.getElementById('mode-link').onclick = (e) => {
            e.preventDefault();
            this.toggleMode();
        };

        // Palette listeners
        document.getElementById('pal-1').onclick = () => this.setPalette(STATE.VOID);
        document.getElementById('pal-2').onclick = () => this.setPalette(STATE.BODY);
        document.getElementById('pal-3').onclick = () => this.setPalette(STATE.HEAD);
    },

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
                    document.getElementById('app').style.display = 'flex';
                    this.resetGame();
                    break;
                case 'SUGGESTION':
                    this.handleAISuggestion(candidates);
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

    toggleMode() {
        this.mode = this.mode === 'play' ? 'explore' : 'play';
        const link = document.getElementById('mode-link');
        const title = document.getElementById('title');
        
        if (this.mode === 'play') {
            link.innerText = "进入自由探索模式";
            title.innerText = "实战演练";
            document.getElementById('play-controls').style.display = 'flex';
            document.getElementById('explore-controls').style.display = 'none';
            this.resetGame();
        } else {
            link.innerText = "返回实战演练模式";
            title.innerText = "自由探索";
            document.getElementById('play-controls').style.display = 'none';
            document.getElementById('explore-controls').style.display = 'flex';
            this.clearBoard();
            this.setPalette(STATE.VOID);
        }
    },

    resetGame() {
        if (!this.isAIReady) return;
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
    },

    clearHints() {
        document.querySelectorAll('.ai-hint').forEach(e => e.remove());
    },

    setStatus(text) {
        document.getElementById('status').innerText = text;
    }
};

window.onload = () => App.init();
