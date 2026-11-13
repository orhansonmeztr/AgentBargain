/* ============================================================
   AgentBargain Dashboard — Main Application Logic
   ============================================================ */

'use strict';

// ---- State -------------------------------------------------------
const state = {
    negotiations: new Map(),   // negotiationId → { id, status, rounds, price, messages, buyerId, sellerId }
    activeTab: null,
    ws: null,
    priceChart: null,
    chartDatasets: {},         // negotiationId → { label, data, color }
};

const COLORS = ['#4f8cff', '#a78bfa', '#2dd4bf', '#f59e0b', '#ef4444', '#10b981'];
let colorIdx = 0;

// ---- DOM helpers -------------------------------------------------
const $ = id => document.getElementById(id);
const esc = str => {
    if (str == null) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/'/g, '&#39;')
        .replace(/"/g, '&quot;');
};

function logEvent(msg, type = 'info') {
    const log = $('event-log');
    const el = document.createElement('div');
    el.className = `log-entry ${type}`;
    const ts = new Date().toLocaleTimeString('en-GB', { hour12: false });
    el.innerHTML = `<span class="log-ts">[${ts}]</span>${esc(msg)}`;
    log.appendChild(el);
    log.scrollTop = log.scrollHeight;
    // Keep log trimmed
    while (log.children.length > 200) log.removeChild(log.firstChild);
}

function clearLog() { $('event-log').innerHTML = ''; }

// ---- API key -----------------------------------------------------
// When API_KEYS is configured server-side every /api/v1 call needs a key.
// The game-theory and aggregate panels additionally need an ADMIN_API_KEYS key.
// Stored in localStorage so it survives a reload; note that this makes the
// secret readable by any script on this origin. That is inherent to a
// browser-held shared secret, not something this dashboard can fix.
const API_KEY_STORAGE = 'agentbargain.apiKey';

function apiKey() {
    return localStorage.getItem(API_KEY_STORAGE) || '';
}

function setApiKey(value) {
    if (value) localStorage.setItem(API_KEY_STORAGE, value);
    else localStorage.removeItem(API_KEY_STORAGE);
    // Reconnect at once with the new credential; connectWebSocket() closes the
    // old socket (without it scheduling a reconnect) and cancels any pending one.
    reconnectFailures = 0;
    connectWebSocket();
    refreshAll();
}

function apiHeaders(extra = {}) {
    const key = apiKey();
    return key ? { 'X-API-Key': key, ...extra } : { ...extra };
}

/** fetch() that always carries the API key. Use instead of bare fetch. */
function apiFetch(url, options = {}) {
    return fetch(url, { ...options, headers: apiHeaders(options.headers) });
}

function saveApiKey() {
    const input = $('api-key');
    const value = input.value.trim();
    setApiKey(value);
    logEvent(value ? 'API key saved — reconnecting.' : 'API key cleared — reconnecting.', 'system');
}

function restoreApiKey() {
    const saved = apiKey();
    if (saved) $('api-key').value = saved;
}

// ---- WebSocket ---------------------------------------------------
// At most one live socket and one pending reconnect. Closing a socket fires
// its onclose, which used to schedule a reconnect even when the close was
// ours (a new key) — and timers left over from failed attempts fired too, so
// the dashboard ended up with two connections and every event twice.
let reconnectTimer = null;
// Consecutive failed attempts. A refused handshake (wrong key) looks exactly
// like a server that is down — the browser reports 1006 for both — so retries
// back off (4s, 8s, ... 60s) instead of hammering the server every 4 seconds.
let reconnectFailures = 0;

function scheduleReconnect() {
    if (reconnectTimer !== null) return;
    const delay = Math.min(60_000, 4000 * 2 ** reconnectFailures);
    reconnectFailures += 1;
    logEvent(`Disconnected — reconnecting in ${Math.round(delay / 1000)}s…`, 'error');
    reconnectTimer = setTimeout(() => {
        reconnectTimer = null;
        connectWebSocket();
    }, delay);
}

function closeWebSocket() {
    if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
    }
    const old = state.ws;
    state.ws = null;
    if (old) {
        // Detach first: a deliberate close must not schedule a reconnect.
        old.onopen = old.onmessage = old.onclose = old.onerror = null;
        try { old.close(); } catch { /* already closed */ }
    }
}

function connectWebSocket() {
    closeWebSocket();
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    let url = `${proto}//${location.host}/api/v1/ws/events`;
    // Browsers cannot set headers on new WebSocket(), so the key has to ride in
    // the query string. It ends up in server access logs — the weaker transport,
    // and the reason the server also accepts the X-API-Key header.
    const key = apiKey();
    if (key) url += `?api_key=${encodeURIComponent(key)}`;

    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
        reconnectFailures = 0;
        $('ws-dot').className = 'ws-dot connected';
        $('ws-label').textContent = 'Live';
        logEvent('Connected to event stream.', 'success');
        refreshAll();
    };

    ws.onmessage = evt => {
        try { handleEvent(JSON.parse(evt.data)); }
        catch { logEvent(`Raw: ${evt.data}`, 'highlight'); }
    };

    ws.onclose = evt => {
        if (state.ws !== ws) return; // superseded by a newer socket
        state.ws = null;
        $('ws-dot').className = 'ws-dot disconnected';
        $('ws-label').textContent = apiKey() ? 'Disconnected' : 'API key required?';
        scheduleReconnect();
    };

    ws.onerror = () => logEvent('WebSocket error.', 'error');
}

// ---- Event handling ----------------------------------------------
function handleEvent(data) {
    if (data.type === 'agent_activity') {
        const { negotiation_id, agent_id, role: msgRole, action: actionField, price: priceField, sla, reasoning: reasonField, content, round: roundField } = data;

        // Register negotiation
        if (!state.negotiations.has(negotiation_id)) {
            const color = COLORS[colorIdx++ % COLORS.length];
            state.negotiations.set(negotiation_id, {
                id: negotiation_id,
                status: 'active',
                rounds: 0,
                price: null,
                messages: [],
                buyerId: null,
                sellerId: null,
                color,
            });
            addTab(negotiation_id);
            updateChartDataset(negotiation_id);
        }

        const neg = state.negotiations.get(negotiation_id);

        // Prefer structured fields; fall back to regex parsing of content string for
        // compatibility with older payloads that only have the raw action string.
        const rawContent = content || actionField || '';
        let role, actionType, price, reason;

        if (msgRole) {
            // Structured payload (new format)
            role       = msgRole;
            actionType = actionField ? actionField.toLowerCase() : null;
            price      = (priceField !== undefined && priceField !== null) ? priceField : null;
            reason     = reasonField || '';
        } else {
            // Legacy regex fallback
            const isBuyer  = rawContent.includes('BUYER:');
            const isSeller = rawContent.includes('SELLER:');
            role = isBuyer ? 'buyer' : isSeller ? 'seller' : 'system';

            const actionMatch = rawContent.match(/:\s*(ACCEPT|COUNTER|REJECT)/i);
            const priceMatch  = rawContent.match(/Price:\s*\$?([\d.]+)/i);
            const reasonMatch = rawContent.match(/Reasoning:\s*(.+)/i);

            actionType = actionMatch ? actionMatch[1].toLowerCase() : null;
            price      = priceMatch ? parseFloat(priceMatch[1]) : null;
            reason     = reasonMatch ? reasonMatch[1] : '';
        }

        if (role === 'buyer'  && !neg.buyerId)  neg.buyerId  = agent_id;
        if (role === 'seller' && !neg.sellerId) neg.sellerId = agent_id;

        // The server stamps each turn with the round it happened in. Deriving
        // this from the terminal negotiation_update event instead left every
        // live point plotted at round 0, collapsing the whole price timeline
        // onto a single x value.
        if (typeof roundField === 'number') {
            neg.rounds = roundField;
        }

        if (price !== null) {
            neg.price = price;
            // Update chart
            pushPricePoint(negotiation_id, price, neg.rounds);
        }

        // Add chat bubble
        neg.messages.push({ role, actionType, price, reason, raw: rawContent });

        if (state.activeTab === negotiation_id) {
            appendChatBubble({ role, actionType, price, reason });
        }

        logEvent(`[${negotiation_id}] ${agent_id}: ${rawContent.substring(0, 80)}`, 'info');
        updateNegotiationTab(negotiation_id);

    } else if (data.type === 'negotiation_update') {
        // The outcome event carries no score: evaluation is a separate, slower
        // step that arrives as `evaluation_update`. Keeping them apart is what
        // lets the deal show up immediately instead of after a 15-40s LLM call.
        const { negotiation_id, status, rounds, final_price } = data;

        const neg = state.negotiations.get(negotiation_id);
        if (neg) {
            neg.status = status;
            neg.rounds = rounds;
            neg.price  = final_price;
        }

        updateNegotiationTab(negotiation_id);
        logEvent(
            `Negotiation ${negotiation_id} → ${status.toUpperCase()}` +
            (final_price != null ? ` @ $${final_price}` : ''),
            status === 'accepted' ? 'success' : status === 'rejected' || status === 'failed' ? 'error' : 'warning'
        );

        if (status === 'accepted' || status === 'rejected' || status === 'timeout' || status === 'failed') {
            // Add system message to chat
            if (neg) {
                const sysMsg = status === 'accepted'
                    ? `✅ Deal agreed at $${final_price}`
                    : status === 'rejected'
                        ? '❌ Negotiation rejected'
                        : status === 'failed'
                            ? '⚠ Negotiation failed'
                            : '⏱ Negotiation timed out';
                neg.messages.push({ role: 'system', raw: sysMsg });
                if (state.activeTab === negotiation_id) {
                    appendChatBubble({ role: 'system', raw: sysMsg });
                }
            }
            // The outcome is already durable when this event fires, so the
            // analysis can be fetched straight away — no settling delay.
            if (status === 'accepted') {
                fetchGameTheoryAnalysis(negotiation_id);
            }
        }

        updateHeaderStats();
        refreshAggregateStats();

    } else if (data.type === 'evaluation_update') {
        const { negotiation_id, seller_score, buyer_score, seller_comments, buyer_comments } = data;

        const neg = state.negotiations.get(negotiation_id);
        if (neg) {
            neg.sellerScore = seller_score;
            neg.buyerScore  = buyer_score;
            const sysMsg =
                `⭐ Evaluation — seller ${seller_score.toFixed(1)}/5, ` +
                `buyer ${buyer_score.toFixed(1)}/5`;
            neg.messages.push({ role: 'system', raw: sysMsg });
            if (state.activeTab === negotiation_id) {
                appendChatBubble({ role: 'system', raw: sysMsg });
            }
        }

        logEvent(
            `Evaluation ${negotiation_id}: seller ${seller_score}/5, buyer ${buyer_score}/5`,
            'highlight'
        );
        if (seller_comments || buyer_comments) {
            logEvent(`  seller: ${seller_comments || '—'}`, 'info');
            logEvent(`  buyer:  ${buyer_comments || '—'}`, 'info');
        }
        // Reputation scores changed, so the leaderboard is stale.
        refreshLeaderboard();
    }
}

// ---- Chat rendering ----------------------------------------------
function switchTab(negId) {
    state.activeTab = negId;
    document.querySelectorAll('.tab').forEach(t => {
        t.classList.toggle('active', t.dataset.negId === negId);
    });
    renderChatView(negId);
    highlightChartDataset(negId);
}

function addTab(negId) {
    const tabBar = $('neg-tabs');
    const tab = document.createElement('button');
    tab.className = 'tab';
    tab.dataset.negId = negId;
    tab.textContent = negId;
    tab.onclick = () => switchTab(negId);
    tabBar.appendChild(tab);
    if (!state.activeTab) switchTab(negId);
}

function updateNegotiationTab(negId) {
    const neg = state.negotiations.get(negId);
    if (!neg) return;
    const tab = document.querySelector(`.tab[data-neg-id="${negId}"]`);
    if (!tab) return;
    // Color-code by status
    tab.style.borderColor = neg.status === 'accepted'
        ? 'var(--success)'
        : neg.status === 'rejected' || neg.status === 'timeout' || neg.status === 'failed'
            ? 'var(--danger)'
            : '';
}

function renderChatView(negId) {
    const neg = state.negotiations.get(negId);
    const chatView = $('chat-view');
    const empty = $('chat-empty');

    if (!neg || neg.messages.length === 0) {
        chatView.innerHTML = '';
        chatView.appendChild(empty);
        empty.style.display = 'flex';
        return;
    }

    empty.style.display = 'none';
    chatView.innerHTML = '';
    neg.messages.forEach(m => appendChatBubble(m, false));
    chatView.scrollTop = chatView.scrollHeight;

    // Update status bar
    const statusEl = $('chat-status-text');
    statusEl.textContent = `${negId} · Round ${neg.rounds} · ${neg.status.toUpperCase()}` +
        (neg.price ? ` · $${neg.price}` : '');
}

function appendChatBubble(msg, animate = true) {
    const chatView = $('chat-view');
    const empty = $('chat-empty');
    if (empty) empty.style.display = 'none';

    const bubble = document.createElement('div');
    // Only known roles become CSS classes, so an odd payload cannot pick
    // arbitrary styling; anything else renders as a plain system line.
    const role = ['buyer', 'seller', 'system'].includes(msg.role) ? msg.role : 'system';
    bubble.className = `chat-bubble ${role}`;
    if (!animate) bubble.style.animation = 'none';

    if (role === 'system') {
        bubble.textContent = msg.raw || '';
    } else {
        // Every interpolated field is escaped: action text comes from strategy
        // code (built-in or third-party), not from a fixed set.
        const action = esc(msg.actionType);
        const actionBadge = msg.actionType
            ? `<span class="bubble-badge ${action}">${action}</span>`
            : '';
        const price = Number(msg.price);
        const priceStr = msg.price != null && Number.isFinite(price)
            ? `<span class="bubble-price">$${price.toFixed(2)}</span>`
            : '';
        const reasonStr = msg.reason ? `<p class="bubble-reason">${esc(msg.reason)}</p>` : '';

        bubble.innerHTML = `
            <div class="bubble-header">
                <span class="bubble-role ${role}">${role.toUpperCase()}</span>
                ${actionBadge}
            </div>
            ${priceStr}
            ${reasonStr}
        `;
    }

    chatView.appendChild(bubble);
    chatView.scrollTop = chatView.scrollHeight;

    // Update status bar
    const neg = state.activeTab ? state.negotiations.get(state.activeTab) : null;
    if (neg) {
        $('chat-status-text').textContent =
            `${state.activeTab} · Round ${neg.rounds} · ${neg.status.toUpperCase()}` +
            (neg.price ? ` · $${neg.price}` : '');
    }
}

// ---- Price Chart -------------------------------------------------
function initChart() {
    const ctx = $('price-chart').getContext('2d');
    state.priceChart = new Chart(ctx, {
        type: 'line',
        // No labels array: the x axis is a numeric round, so points carry their
        // own x. A category scale rebuilt from max(round) collapsed the two
        // turns of a single round onto one slot.
        data: { datasets: [] },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: { duration: 400 },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#0d1626',
                    borderColor: 'rgba(99,140,255,0.3)',
                    borderWidth: 1,
                    titleColor: '#f0f4ff',
                    bodyColor: '#8899bb',
                    callbacks: {
                        title: items => `Round ${items[0].parsed.x}`,
                        label: ctx => `${ctx.dataset.label}: $${ctx.parsed.y.toFixed(2)}`,
                    },
                },
            },
            scales: {
                x: {
                    type: 'linear',
                    beginAtZero: true,
                    grid: { color: 'rgba(99,140,255,0.06)' },
                    ticks: { color: '#4a5a7a', font: { size: 10 }, precision: 0, stepSize: 1 },
                    title: { display: true, text: 'Round', color: '#4a5a7a', font: { size: 10 } },
                },
                y: {
                    grid: { color: 'rgba(99,140,255,0.06)' },
                    ticks: {
                        color: '#4a5a7a',
                        font: { size: 10 },
                        callback: v => `$${v}`,
                    },
                },
            },
        },
    });
}

function updateChartDataset(negId) {
    const neg = state.negotiations.get(negId);
    if (!neg || !state.priceChart) return;

    const dataset = {
        label: negId,
        data: [],
        borderColor: neg.color,
        backgroundColor: neg.color + '22',
        borderWidth: 2,
        pointRadius: 4,
        pointHoverRadius: 6,
        tension: 0.4,
        fill: false,
    };
    state.chartDatasets[negId] = dataset;
    state.priceChart.data.datasets.push(dataset);
    state.priceChart.update('none');
    renderChartLegend();
}

function renderChartLegend() {
    // Chart.js' own legend is disabled, so without this there is no way to tell
    // which colour belongs to which negotiation once more than one is plotted.
    const legend = $('chart-legend');
    if (!legend || !state.priceChart) return;

    legend.innerHTML = '';
    state.priceChart.data.datasets.forEach(ds => {
        const neg = state.negotiations.get(ds.label);
        const item = document.createElement('button');
        item.className = 'chart-legend-item';
        item.type = 'button';
        item.title = `Show ${ds.label}`;
        item.onclick = () => switchTab(ds.label);

        const swatch = document.createElement('span');
        swatch.className = 'legend-dot';
        swatch.style.background = neg?.color || ds.borderColor;

        const text = document.createElement('span');
        text.textContent = ds.label;

        item.append(swatch, text);
        legend.appendChild(item);
    });
}

function pushPricePoint(negId, price, round) {
    if (!state.priceChart) return;
    const ds = state.priceChart.data.datasets.find(d => d.label === negId);
    if (!ds) return;
    ds.data.push({ x: round, y: price });
    state.priceChart.update('none');
}

function highlightChartDataset(negId) {
    if (!state.priceChart) return;
    state.priceChart.data.datasets.forEach(ds => {
        ds.borderWidth = ds.label === negId ? 3 : 1.5;
        ds.borderColor = ds.label === negId
            ? (state.negotiations.get(negId)?.color || '#4f8cff')
            : '#2a3a5a';
    });
    state.priceChart.update('none');
}

// ---- Negotiation Launch ------------------------------------------
async function postJson(url, payload) {
    const res = await apiFetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
    });
    let data = null;
    try { data = await res.json(); } catch { /* non-JSON error body */ }
    return { ok: res.ok, status: res.status, data };
}

async function launchNegotiation(evt) {
    evt.preventDefault();
    const btn = $('launch-btn');
    const btnText = $('launch-btn-text');
    btn.disabled = true;
    btnText.textContent = '⏳ Launching…';

    const sellerId  = $('seller-id').value.trim();
    const serviceId = $('service-id').value.trim();
    const engine    = $('engine').value;

    try {
        // Step 1: the seller publishes the listing, including their own floor.
        // The floor lives here and nowhere else — the negotiation endpoint no
        // longer accepts it, so a buyer cannot declare the seller's walk-away.
        const listing = await postJson('/api/v1/marketplace/services', {
            service_id:  serviceId,
            seller_id:   sellerId,
            title:       $('service-title').value.trim(),
            base_price:  parseFloat($('base-price').value),
            floor_price: parseFloat($('floor-price').value),
            sla_terms:   'Standard',
        });
        if (!listing.ok) {
            logEvent(`❌ Listing failed (${listing.status}): ${JSON.stringify(listing.data)}`, 'error');
            return;
        }

        // Step 2: open the negotiation. No initial_price — it defaults to the
        // listing's asking price, which is the seller's number too.
        const startBody = {
            buyer_id:   $('buyer-id').value.trim(),
            seller_id:  sellerId,
            service_id: serviceId,
            max_budget: parseFloat($('max-budget').value),
            engine,
        };
        if (engine === 'strategy') {
            startBody.buyer_strategy  = $('buyer-strategy').value;
            startBody.seller_strategy = $('seller-strategy').value;
        }

        const started = await postJson('/api/v1/negotiations/start', startBody);
        if (started.ok) {
            logEvent(`✅ Negotiation ${started.data.negotiation_id} started!`, 'success');
        } else {
            logEvent(`❌ Error (${started.status}): ${JSON.stringify(started.data)}`, 'error');
        }
    } catch (e) {
        logEvent(`❌ Network error: ${e.message}`, 'error');
    } finally {
        btn.disabled = false;
        btnText.textContent = '⚡ Launch Negotiation';
    }
}

// Demo: launches with preset values
function startDemo() {
    const demos = [
        { buyer: 'agent_gemini', seller: 'agent_ollama',  title: 'ML Training Pipeline API', svc: 'svc_ml_training',   budget: 800, floor: 200, base: 650 },
        { buyer: 'agent_gpt',    seller: 'agent_claude',  title: 'Vector DB Hosting Service', svc: 'svc_vector_db',    budget: 300, floor: 80,  base: 280 },
        { buyer: 'agent_phi4',   seller: 'agent_mistral', title: 'RAG Inference Endpoint',   svc: 'svc_rag_inference', budget: 150, floor: 50,  base: 130 },
    ];
    const demo = demos[Math.floor(Math.random() * demos.length)];
    $('buyer-id').value    = demo.buyer;
    $('seller-id').value   = demo.seller;
    $('service-title').value = demo.title;
    $('service-id').value  = demo.svc;
    $('max-budget').value  = demo.budget;
    $('floor-price').value = demo.floor;
    $('base-price').value  = demo.base;
    $('launch-form').dispatchEvent(new Event('submit', { cancelable: true }));
}

// ---- Game Theory Analysis ----------------------------------------
async function fetchGameTheoryAnalysis(negId) {
    try {
        const res = await apiFetch(`/api/v1/analytics/${negId}/game-theory`);
        if (!res.ok) return;
        const d = await res.json();
        renderGameTheory(d);
        logEvent(`📐 Game theory analysis computed for ${negId}`, 'system');
    } catch (e) {
        logEvent(`Game theory fetch error: ${e.message}`, 'error');
    }
}

function renderGameTheory(d) {
    $('gt-empty').style.display = 'none';
    $('gt-content').style.display = 'flex';

    // ZOPA bar
    const zopa = d.zopa;
    const totalRange = zopa.upper_bound - zopa.lower_bound;
    $('zopa-low-label').textContent  = `$${zopa.lower_bound}`;
    $('zopa-high-label').textContent = `$${zopa.upper_bound}`;

    if (zopa.exists && totalRange > 0) {
        $('zopa-fill').style.cssText = 'left:0; right:0;';

        // Nash marker position (midpoint)
        const nashPct = ((d.nash.nash_price - zopa.lower_bound) / totalRange * 100).toFixed(1);
        $('nash-marker').style.left = `${Math.min(95, Math.max(5, nashPct))}%`;

        // Final price marker position
        if (d.final_price != null) {
            const pricePct = ((d.final_price - zopa.lower_bound) / totalRange * 100).toFixed(1);
            $('price-marker').style.left = `${Math.min(95, Math.max(5, pricePct))}%`;
            $('price-marker').style.display = 'flex';
        }
    } else {
        $('zopa-fill').style.cssText = 'width:0;';
        $('price-marker').style.display = 'none';
    }

    // Metrics
    $('gt-nash').textContent = d.nash.nash_price != null ? `$${d.nash.nash_price}` : '—';
    $('gt-pareto').textContent = d.pareto.is_efficient ? '✅ Yes' : '❌ No';
    $('gt-efficiency').textContent = d.efficiency_score != null ? (d.efficiency_score * 100).toFixed(0) + '%' : '—';
    $('gt-fairness').textContent   = d.fairness_score != null   ? (d.fairness_score   * 100).toFixed(0) + '%' : '—';

    // Surplus bar
    const total = d.pareto.total_surplus || 1;
    const buyerPct  = ((d.pareto.buyer_surplus  / total) * 100).toFixed(1);
    const sellerPct = ((d.pareto.seller_surplus / total) * 100).toFixed(1);
    $('surplus-buyer-bar').style.width  = `${buyerPct}%`;
    $('surplus-seller-bar').style.width = `${sellerPct}%`;
    $('surplus-buyer-val').textContent  = `$${d.pareto.buyer_surplus}`;
    $('surplus-seller-val').textContent = `$${d.pareto.seller_surplus}`;

    // BATNA
    const bBatna = d.batna.buyer_batna_distance;
    const sBatna = d.batna.seller_batna_distance;
    const batnaEl = $('batna-buyer');
    batnaEl.textContent = bBatna >= 0 ? `+$${bBatna}` : `-$${Math.abs(bBatna)}`;
    batnaEl.className = `batna-val ${bBatna >= 0 ? 'positive' : 'negative'}`;
    const sellerEl = $('batna-seller');
    sellerEl.textContent = sBatna >= 0 ? `+$${sBatna}` : `-$${Math.abs(sBatna)}`;
    sellerEl.className = `batna-val ${sBatna >= 0 ? 'positive' : 'negative'}`;
}

// ---- Leaderboard -------------------------------------------------
async function refreshLeaderboard() {
    try {
        const res = await apiFetch('/api/v1/marketplace/agents?limit=10');
        const data = await res.json();
        renderLeaderboard(data.agents || []);
    } catch {}
}

function renderLeaderboard(agents) {
    const list = $('leaderboard-list');
    if (!agents.length) {
        list.innerHTML = '<li class="leaderboard-empty">No agents yet.</li>';
        return;
    }
    list.innerHTML = '';
    agents.forEach((agent, i) => {
        const rankClass = i === 0 ? 'gold' : i === 1 ? 'silver' : i === 2 ? 'bronze' : '';
        const score = agent.reputation_score ?? 0;
        const stars = Array.from({ length: 5 }, (_, j) =>
            `<span class="star ${j < Math.round(score) ? 'filled' : 'empty'}">★</span>`
        ).join('');

        const li = document.createElement('li');
        li.className = 'leaderboard-item';
        li.innerHTML = `
            <span class="rank-badge ${rankClass}">${i + 1}</span>
            <span class="agent-name" title="${esc(agent.agent_id)}">${esc(agent.agent_id)}</span>
            <div class="star-bar">${stars}</div>
            <span class="agent-score">${score.toFixed(2)}</span>
        `;
        list.appendChild(li);
    });
}

// ---- Strategies --------------------------------------------------
async function loadStrategies() {
    try {
        const res = await apiFetch('/api/v1/analytics/strategies');
        const data = await res.json();
        renderStrategies(data.strategies || []);
    } catch {}
}

function renderStrategies(strategies) {
    const list = $('strategies-list');
    // The same registry feeds the launch form, so the strategy engine is
    // actually reachable from the dashboard instead of API-only.
    populateStrategySelects(strategies);

    if (!strategies.length) {
        list.innerHTML = '<li class="strategy-loading">No strategies registered.</li>';
        return;
    }
    list.innerHTML = '';
    strategies.forEach(s => {
        const li = document.createElement('li');
        li.className = 'strategy-item';
        li.innerHTML = `
            <span class="strategy-tag">${esc(s.name)}</span>
            <span class="strategy-desc">${esc(s.description) || esc(s.class)}</span>
        `;
        list.appendChild(li);
    });
}

function populateStrategySelects(strategies) {
    const presets = { 'buyer-strategy': 'cooperative', 'seller-strategy': 'aggressive' };

    Object.entries(presets).forEach(([id, preferred]) => {
        const select = $(id);
        if (!select) return;

        select.innerHTML = '';
        strategies.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.name;
            opt.textContent = s.name;
            opt.title = s.description || '';
            select.appendChild(opt);
        });

        const names = strategies.map(s => s.name);
        select.value = names.includes(preferred) ? preferred : (names[0] || '');
    });
}

function syncStrategyVisibility() {
    const row = $('strategy-row');
    if (row) row.hidden = $('engine').value !== 'strategy';
}

// ---- Aggregate Stats ---------------------------------------------
async function refreshAggregateStats() {
    try {
        const res = await apiFetch('/api/v1/analytics/aggregate');
        // 403 = a regular (non-admin) key: analytics stays blank, not broken.
        if (!res.ok) return;
        const d = await res.json();
        $('agg-total').textContent      = d.total_negotiations ?? '—';
        $('agg-rate').textContent       = d.acceptance_rate != null ? (d.acceptance_rate * 100).toFixed(0) + '%' : '—';
        $('agg-rounds').textContent     = d.avg_rounds != null ? d.avg_rounds.toFixed(1) : '—';
        $('agg-price').textContent      = d.avg_final_price ? `$${d.avg_final_price.toFixed(0)}` : '—';
        $('agg-efficiency').textContent = d.avg_efficiency != null ? (d.avg_efficiency * 100).toFixed(0) + '%' : '—';
        $('agg-fairness').textContent   = d.avg_fairness != null   ? (d.avg_fairness   * 100).toFixed(0) + '%' : '—';
    } catch {}
}

// ---- Header stats ------------------------------------------------
function updateHeaderStats() {
    const negs = [...state.negotiations.values()];
    $('total-count').textContent   = negs.length;
    const accepted = negs.filter(n => n.status === 'accepted').length;
    $('accepted-count').textContent = accepted;
    const rate = negs.length ? (accepted / negs.length * 100).toFixed(0) + '%' : '—';
    $('accept-rate').textContent = rate;
}

// ---- Init --------------------------------------------------------
async function refreshAll() {
    await Promise.all([
        refreshLeaderboard(),
        refreshAggregateStats(),
        loadStrategies(),
    ]);
}

window.addEventListener('DOMContentLoaded', () => {
    initChart();
    syncStrategyVisibility();
    $('engine').addEventListener('change', syncStrategyVisibility);
    // Before connecting: the stored key has to be in the WebSocket URL.
    restoreApiKey();
    $('api-key').addEventListener('keydown', e => {
        if (e.key === 'Enter') { e.preventDefault(); saveApiKey(); }
    });
    connectWebSocket();
    // Also load on DOM ready: refreshAll() otherwise only runs once the
    // WebSocket opens, so a failed connection left the panels empty forever.
    refreshAll();
    // Poll aggregate stats every 30s
    setInterval(refreshAggregateStats, 30_000);
    setInterval(refreshLeaderboard, 60_000);
});

// Expose globals for onclick handlers
window.launchNegotiation = launchNegotiation;
window.startDemo = startDemo;
window.clearLog = clearLog;
window.refreshLeaderboard = refreshLeaderboard;
window.saveApiKey = saveApiKey;
