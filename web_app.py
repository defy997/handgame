"""掌心战争 Nash AI - Web 版（局域网，手机浏览器直接玩）

运行: python web_app.py
手机访问: http://<本机IP>:5000  (需同一 WiFi)
"""
from flask import Flask, render_template_string, request, jsonify, session
import numpy as np
import os, secrets, socket

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# 导入游戏逻辑（play_nash.py 完全自包含，无 torch 依赖）
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from play_nash import (
    game_step, legal_actions, overtime_winner,
    NUM_SKILLS, MAX_HP, MAX_ENERGY, MAX_ROUNDS, INIT_HP, INIT_EN,
    SKILL_NAMES, SKILL_COST, load_strategy,
)

S1, S2, FINITE = load_strategy()


# ── 游戏逻辑 ──────────────────────────────────────────────────────────────

def _ai_action(php, pen, ahp, aen, t, human_is_p1):
    ai_legal = legal_actions(aen if human_is_p1 else pen)
    ai_raw = (S2[php, pen, ahp, aen, t] if (FINITE and human_is_p1) else
              S1[php, pen, ahp, aen, t] if FINITE else
              S2[php, pen, ahp, aen] if human_is_p1 else
              S1[php, pen, ahp, aen]).copy()
    sg = np.zeros(NUM_SKILLS)
    for a in ai_legal: sg[a] = ai_raw[a]
    total = sg.sum()
    if total < 1e-12:
        for a in ai_legal: sg[a] = 1.0 / len(ai_legal)
    else:
        sg /= total
    return int(np.random.choice(NUM_SKILLS, p=sg))


def _new_gs(wins=None, game_idx=0):
    game_idx += 1
    return {
        'php': INIT_HP, 'pen': INIT_EN,
        'ahp': INIT_HP, 'aen': INIT_EN,
        'rnd': 0, 'done': False, 'winner': 0, 'result': '',
        'human_player': 1 if game_idx % 2 == 1 else 2,
        'wins': wins or {'人类': 0, 'AI': 0, '平局': 0},
        'game_idx': game_idx,
        'log': [],
    }


def _response(gs):
    hp1 = gs['human_player'] == 1
    legal = legal_actions(gs['pen'] if hp1 else gs['aen'])
    return {
        'php': gs['php'], 'pen': gs['pen'],
        'ahp': gs['ahp'], 'aen': gs['aen'],
        'rnd': gs['rnd'], 'done': gs['done'],
        'winner': gs['winner'], 'result': gs['result'],
        'human_player': gs['human_player'],
        'human_is_p1': hp1,
        'legal_actions': legal,
        'wins': gs['wins'],
        'game_idx': gs['game_idx'],
        'log': gs['log'][-25:],
    }


# ── 路由 ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/init', methods=['POST'])
def init():
    gs = session.get('gs')
    if gs is None:
        gs = _new_gs()
        session['gs'] = gs
    return jsonify(_response(gs))


@app.route('/new_game', methods=['POST'])
def new_game():
    gs = session.get('gs', _new_gs())
    gs = _new_gs(gs['wins'], gs['game_idx'])
    session['gs'] = gs
    return jsonify(_response(gs))


@app.route('/move', methods=['POST'])
def move():
    gs = session.get('gs')
    if not gs or gs['done']:
        return jsonify({'error': 'no game'}), 400

    h_sk = int(request.get_json()['skill'])
    hp1 = gs['human_player'] == 1
    t = gs['rnd']

    ai_sk = _ai_action(gs['php'], gs['pen'], gs['ahp'], gs['aen'], t, hp1)
    a1, a2 = (h_sk, ai_sk) if hp1 else (ai_sk, h_sk)

    pre_php, pre_ahp = gs['php'], gs['ahp']
    php, pen, ahp, aen, done, winner = game_step(
        gs['php'], gs['pen'], gs['ahp'], gs['aen'], a1, a2)
    gs.update(php=int(php), pen=int(pen), ahp=int(ahp), aen=int(aen))
    gs['rnd'] += 1

    # 构建日志
    h_pre  = pre_php if hp1 else pre_ahp
    h_post = php     if hp1 else ahp
    ai_pre  = pre_ahp if hp1 else pre_php
    ai_post = ahp     if hp1 else php
    hd, aid = h_post - h_pre, ai_post - ai_pre

    parts = [f"R{gs['rnd']} 你「{SKILL_NAMES[h_sk]}」AI「{SKILL_NAMES[ai_sk]}」"]
    if hd:  parts.append(f"你{'+' if hd>0 else ''}{hd}HP")
    if aid: parts.append(f"AI{'+' if aid>0 else ''}{aid}HP")
    if not hd and not aid: parts.append("无伤")
    gs['log'].append(' · '.join(parts))

    # 结算
    if done or gs['rnd'] >= MAX_ROUNDS:
        if not done:
            winner = overtime_winner(php, ahp)
            h_hp = php if hp1 else ahp
            ai_hp = ahp if hp1 else php
            gs['log'].append(f"⏱ 超时 — 你{h_hp}血 AI{ai_hp}血")
        w = int(winner)
        if w == 1:   result = '人类获胜' if hp1 else 'AI获胜'
        elif w == 2: result = 'AI获胜'   if hp1 else '人类获胜'
        else:        result = '平局'
        key = '人类' if '人类' in result else ('AI' if 'AI' in result else '平局')
        gs['wins'][key] += 1
        gs['done'] = True
        gs['winner'] = w
        gs['result'] = result

    session['gs'] = gs
    session.modified = True
    return jsonify(_response(gs))


# ── HTML 模板 ─────────────────────────────────────────────────────────────

HTML = r'''<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>掌心战争</title>
<style>
*{box-sizing:border-box;margin:0;padding:0;-webkit-tap-highlight-color:transparent}
body{background:#0f0f1a;color:#e0e0e0;font-family:-apple-system,BlinkMacSystemFont,sans-serif;
     max-width:460px;margin:0 auto;padding:12px;min-height:100vh;overflow-x:hidden}
h1{text-align:center;color:#4ecca3;font-size:20px;margin-bottom:10px;letter-spacing:3px}

/* 战绩栏 */
.stats{display:flex;justify-content:space-around;background:#1a1a2e;
       padding:10px 4px;border-radius:10px;margin-bottom:8px}
.stat{text-align:center;font-size:12px;color:#888}
.stat b{display:block;font-size:22px;font-weight:700}
.stat b.g{color:#4ecca3} .stat b.r{color:#e94560} .stat b.y{color:#f5a623}

/* 回合信息 */
.badge{text-align:center;font-size:12px;color:#666;margin-bottom:8px}

/* 玩家卡片 */
.card{background:#1a1a2e;padding:12px;border-radius:12px;margin-bottom:8px}
.card-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:7px;font-size:14px;font-weight:600}
.hp-bar{height:9px;background:#090916;border-radius:5px;overflow:hidden}
.hp-fill{height:100%;border-radius:5px;transition:width .35s ease}
.hp-you{background:linear-gradient(90deg,#e94560,#ff6b9d)}
.hp-ai {background:linear-gradient(90deg,#7b2fd4,#a855f7)}
.en-row{display:flex;gap:4px;margin-top:7px}
.dot{width:15px;height:15px;border-radius:50%;background:#090916;border:1px solid #1e1e35;transition:all .25s}
.dot.on{background:#f5a623;border-color:#f5a623;box-shadow:0 0 5px #f5a623aa}

/* 技能按钮 */
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:8px 0}
.sk{background:#1a1a2e;border:1px solid #252540;color:#ddd;padding:13px 2px;
    border-radius:10px;font-size:15px;cursor:pointer;transition:all .12s;user-select:none;
    display:flex;flex-direction:column;align-items:center;gap:3px}
.sk:not(:disabled):active{background:#e94560;border-color:#e94560;transform:scale(.94)}
.sk:disabled{opacity:.22;cursor:not-allowed}
.sk .cost{font-size:11px;color:#f5a623}
.sk.legal{border-color:#4ecca355}

/* 结算框 */
.over{background:#1a1a2e;border-radius:14px;padding:22px 12px;text-align:center;margin:8px 0}
.over-title{font-size:30px;font-weight:700;margin-bottom:14px}
.over-title.w{color:#4ecca3} .over-title.l{color:#e94560} .over-title.d{color:#f5a623}
.next-btn{background:#4ecca3;color:#0f0f1a;border:none;padding:12px 36px;
          border-radius:10px;font-size:16px;font-weight:700;cursor:pointer}
.next-btn:active{opacity:.8}

/* 日志 */
.log{background:#1a1a2e;padding:8px 10px;border-radius:10px;
     max-height:140px;overflow-y:auto;font-size:12px;margin-top:6px}
.li{padding:3px 0;border-bottom:1px solid #0d0d20;color:#888}
.li:first-child{color:#ccc}
.li:last-child{border:none}
</style>
</head>
<body>
<h1>⚔ 掌心战争</h1>

<div class="stats">
  <div class="stat"><b class="g" id="wh">0</b>人类胜</div>
  <div class="stat"><b class="r" id="wa">0</b>AI胜</div>
  <div class="stat"><b class="y" id="wd">0</b>平局</div>
</div>
<div class="badge" id="badge">连接中...</div>

<div class="card">
  <div class="card-top"><span>🤖 AI</span><span id="ai-ht">HP 4/4</span></div>
  <div class="hp-bar"><div class="hp-fill hp-ai" id="ai-hb" style="width:100%"></div></div>
  <div class="en-row" id="ai-en"></div>
</div>
<div class="card">
  <div class="card-top"><span>👤 你</span><span id="pl-ht">HP 4/4</span></div>
  <div class="hp-bar"><div class="hp-fill hp-you" id="pl-hb" style="width:100%"></div></div>
  <div class="en-row" id="pl-en"></div>
</div>

<div class="grid" id="grid"></div>

<div class="over" id="over" style="display:none">
  <div class="over-title" id="over-t"></div>
  <button class="next-btn" id="nbtn">下一局 →</button>
</div>

<div class="log" id="log"><div style="color:#333;text-align:center;padding:4px">战斗记录</div></div>

<script>
const MH=4,ME=6;
const SN=['单枪','双枪','三枪','大招','小防','大防','能量','反弹','清零'];
const SC=[1,2,2,3,0,1,0,1,2];
let busy=false;

const $=id=>document.getElementById(id);

async function api(url,body){
  const r=await fetch(url,{method:'POST',
    headers:body?{'Content-Type':'application/json'}:{},
    body:body?JSON.stringify(body):undefined});
  return r.json();
}

async function init(){const s=await api('/init');draw(s)}
async function move(sk){
  if(busy)return; busy=true; disableGrid();
  draw(await api('/move',{skill:sk})); busy=false;
}
$('nbtn').addEventListener('click',async()=>draw(await api('/new_game')));

function draw(s){
  $('wh').textContent=s.wins['人类'];
  $('wa').textContent=s.wins['AI'];
  $('wd').textContent=s.wins['平局'];
  $('badge').textContent=`第${s.game_idx}局 · 你是P${s.human_player} · 第${s.rnd}回合`;

  const[pH,pE,aH,aE]=s.human_is_p1?[s.php,s.pen,s.ahp,s.aen]:[s.ahp,s.aen,s.php,s.pen];
  setCard('pl',pH,pE); setCard('ai',aH,aE);

  if(!s.done){
    $('over').style.display='none';
    renderGrid(s.legal_actions);
  } else {
    $('grid').innerHTML='';
    $('over').style.display='block';
    const t=$('over-t');
    if(s.result==='人类获胜'){t.textContent='🎉 你赢了！';t.className='over-title w';}
    else if(s.result==='AI获胜'){t.textContent='😅 AI 赢了';t.className='over-title l';}
    else{t.textContent='🤝 平局';t.className='over-title d';}
  }

  const logEl=$('log');
  const entries=s.log.slice().reverse();
  logEl.innerHTML=entries.length
    ?entries.map((e,i)=>`<div class="li">${e}</div>`).join('')
    :'<div style="color:#333;text-align:center;padding:4px">战斗记录</div>';
}

function setCard(id,hp,en){
  $(id+'-ht').textContent=`HP ${hp}/${MH}`;
  $(id+'-hb').style.width=`${hp/MH*100}%`;
  const el=$(id+'-en');
  el.innerHTML=Array.from({length:ME},(_,i)=>`<div class="dot${i<en?' on':''}"></div>`).join('');
}

function renderGrid(legal){
  $('grid').innerHTML=SN.map((n,i)=>{
    const ok=legal.includes(i);
    return `<button class="sk${ok?' legal':''}"${ok?'':' disabled'} onclick="move(${i})">
      ${n}<span class="cost">${SC[i]?SC[i]+'能':'免费'}</span></button>`;
  }).join('');
}

function disableGrid(){$('grid').querySelectorAll('.sk').forEach(b=>b.disabled=true);}

init();
</script>
</body>
</html>'''


# ── 启动 ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = '127.0.0.1'

    print(f"\n{'='*46}")
    print(f"  掌心战争 Nash AI — Web 版")
    print(f"  本机访问:  http://127.0.0.1:5000")
    print(f"  手机访问:  http://{local_ip}:5000")
    print(f"  (手机和电脑需在同一 WiFi 下)")
    print(f"{'='*46}\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
