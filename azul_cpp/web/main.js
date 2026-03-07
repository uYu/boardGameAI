import AzulModule from './azul.js';

let Module, game;
const COLOR_MAP = { 0: 'blue', 1: 'yellow', 2: 'red', 3: 'black', 4: 'white', 5: 'first' };
const WALL_LAYOUT = [
    [0, 1, 2, 3, 4], [4, 0, 1, 2, 3], [3, 4, 0, 1, 2], [2, 3, 4, 0, 1], [1, 2, 3, 4, 0]
];

let playerConfigs = [{type:'human', name:'P1'}, {type:'ai', name:'P2'}];
let aiThinkingTime = 2.0;
let selectedMove = null;
let historyStack = [];

async function init() {
    try {
        Module = await AzulModule();
        document.getElementById('restart-btn').onclick = startGame;
        document.getElementById('undo-btn').onclick = undo;
        startGame();
    } catch (e) { console.error(e); }
}

function startGame() {
    const p0 = document.getElementById('p0-type').value;
    const p1 = document.getElementById('p1-type').value;
    playerConfigs = [
        { type: p0, name: p0 === 'human' ? '人类 (P1)' : 'AI (P1)' },
        { type: p1, name: p1 === 'human' ? '人类 (P2)' : 'AI (P2)' }
    ];
    aiThinkingTime = parseFloat(document.getElementById('ai-difficulty').value);
    game = Module.createGame(2, Math.floor(Math.random() * 1000000));
    historyStack = [];
    selectedMove = null;
    update();
}

function update() {
    const state = JSON.parse(game.toJSON());
    render(state);
    if (!state.gameOver && playerConfigs[state.currentPlayer].type === 'ai') {
        setTimeout(triggerAI, 600);
    }
}

function triggerAI() {
    const res = JSON.parse(Module.askAI(game, aiThinkingTime));
    if (res && res.best_action) {
        Module.applyMove(game, res.best_action.src, res.best_action.col, res.best_action.row);
        update();
    }
}

function render(state) {
    document.getElementById('game-status').innerText = `轮到: ${playerConfigs[state.currentPlayer].name}`;
    
    // 市场
    const market = document.getElementById('market');
    market.innerHTML = '';
    state.factories.forEach((f, i) => {
        const d = document.createElement('div'); d.className = 'factory';
        f.forEach(c => { if(c!==-1) d.appendChild(createTile(c, i, state.currentPlayer)); });
        market.appendChild(d);
    });
    const cp = document.createElement('div'); cp.className = 'center-pot';
    state.center.forEach(c => cp.appendChild(createTile(c, -1, state.currentPlayer)));
    market.appendChild(cp);

    // 版图
    state.players.forEach((p, i) => {
        const el = document.getElementById(`player-${i}`);
        el.innerHTML = getPlayerHTML(p, i, state);
    });
}

function createTile(color, srcId, curP) {
    const t = document.createElement('div');
    const sel = selectedMove && selectedMove.srcId === srcId && selectedMove.color === color;
    t.className = `tile ${COLOR_MAP[color]} ${sel ? 'selected' : ''}`;
    t.onclick = (e) => {
        if (playerConfigs[curP].type !== 'human') return;
        selectedMove = sel ? null : { srcId, color };
        update();
    };
    return t;
}

function getPlayerHTML(p, pIdx, state) {
    const isAct = state.currentPlayer === pIdx;
    const isHum = playerConfigs[pIdx].type === 'human';

    const patterns = p.patterns.map((line, rIdx) => {
        let canP = false;
        if (isAct && isHum && selectedMove && selectedMove.color < 5) {
            const targetCol = (selectedMove.color + rIdx) % 5;
            canP = p.wall[rIdx][targetCol] === -1 && (line.count === 0 || line.color === selectedMove.color) && line.count < rIdx + 1;
        }
        
        let rowHtml = '';
        // 空位
        for(let i=0; i < (rIdx + 1 - line.count); i++) 
            rowHtml += `<div class="slot ${line.count > 0 ? COLOR_MAP[line.color] : ''}" style="opacity:0.2"></div>`;
        // 砖块
        for(let i=0; i < line.count; i++) 
            rowHtml += `<div class="tile ${COLOR_MAP[line.color]}"></div>`;

        return `<div class="row ${canP ? 'highlight' : ''}" onclick="executeMove(${rIdx}, ${canP})">${rowHtml}</div>`;
    }).join('');

    const wall = p.wall.map((row, rIdx) => row.map((c, cIdx) => 
        c !== -1 ? `<div class="tile ${COLOR_MAP[c]}"></div>` : `<div class="slot ${COLOR_MAP[WALL_LAYOUT[rIdx][cIdx]]}"></div>`
    ).join('')).join('');

    return `
    <div class="player-board ${isAct ? 'active' : ''}">
        <div style="display:flex; justify-content:space-between; font-weight:bold; margin-bottom:10px;">
            <span>${playerConfigs[pIdx].name}</span><span>得分: ${p.score}</span>
        </div>
        <div style="display:flex; gap:20px;">
            <div class="pattern-lines">${patterns}</div>
            <div class="wall-grid">${wall}</div>
        </div>
        <div class="floor-line ${isAct && isHum && selectedMove ? 'highlight' : ''}" 
             onclick="executeMove(-1, ${isAct && isHum && !!selectedMove})" 
             style="display:flex; gap:5px; margin-top:15px; min-height:45px;">
            ${p.floor.map(c => `<div class="tile ${COLOR_MAP[c]}"></div>`).join('')}
        </div>
    </div>`;
}

window.executeMove = (row, ok) => {
    if (!ok || !selectedMove) return;
    Module.applyMove(game, selectedMove.srcId, selectedMove.color, row);
    selectedMove = null;
    update();
};

function undo() { startGame(); } // 简化版重开

init();