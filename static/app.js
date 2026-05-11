"use strict";

const $  = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);

const els = {
    query:        $("#query"),
    topK:         $("#topK"),
    topKValue:    $("#topKValue"),
    filterV:      $("#filterViolation"),
    filterS:      $("#filterSanction"),
    searchBtn:    $("#searchBtn"),
    answerBtn:    $("#answerBtn"),
    loading:      $("#loading"),
    loadingText:  $("#loadingText"),
    error:        $("#error"),
    answer:       $("#answer"),
    answerBody:   $("#answerBody"),
    answerMeta:   $("#answerMeta"),
    sources:      $("#sources"),
    sourcesList:  $("#sourcesList"),
    sourcesMeta:  $("#sourcesMeta"),
    status:       $("#status"),
    statusText:   $("#statusText"),
};

// ───────── 헬스 체크 ─────────
async function checkHealth() {
    try {
        const r = await fetch("/health");
        if (!r.ok) throw new Error(r.status);
        const d = await r.json();
        els.status.classList.add("ok");
        els.status.classList.remove("error");
        const chunks = (d.chunk_count || 0).toLocaleString();
        const llm = d.llm_loaded ? "LLM ready" : "LLM offline";
        els.statusText.textContent = `정상 · ${chunks}청크 · ${llm}`;
    } catch (err) {
        els.status.classList.add("error");
        els.status.classList.remove("ok");
        els.statusText.textContent = "서버 연결 실패";
    }
}

// ───────── UI 헬퍼 ─────────
function show(el)        { el && el.classList.remove("hidden"); }
function hide(el)        { el && el.classList.add("hidden"); }
function setLoading(on, text) {
    if (on) {
        els.loadingText.textContent = text || "처리 중…";
        show(els.loading);
        hide(els.answer);
        hide(els.sources);
        hide(els.error);
        els.searchBtn.disabled = true;
        els.answerBtn.disabled = true;
    } else {
        hide(els.loading);
        els.searchBtn.disabled = false;
        els.answerBtn.disabled = false;
    }
}
function showError(msg) {
    els.error.textContent = msg;
    show(els.error);
}

// ───────── 요청 빌더 ─────────
function buildSearchRequest() {
    const q = els.query.value.trim();
    if (!q) { els.query.focus(); return null; }

    const body = {
        query: q,
        top_k: parseInt(els.topK.value, 10),
    };
    const fv = els.filterV.value.trim();
    const fs = els.filterS.value.trim();
    if (fv) body["filter_위반유형"] = [fv];
    if (fs) body["filter_조치유형"] = [fs];
    return body;
}

function buildPredictRequest() {
    const q = els.query.value.trim();
    if (!q) { els.query.focus(); return null; }
    return {
        id: "ui_" + Date.now(),
        question: q,
    };
}

// ───────── 안전 HTML ─────────
function escapeHtml(s) {
    return (s || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}
function renderAnswer(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    return html;
}

// ───────── 출처 렌더러 ─────────
function renderSources(hits) {
    els.sourcesList.innerHTML = "";
    hits.forEach((h, i) => {
        const item = document.createElement("div");
        item.className = "source-item";

        const headerHtml = `
            <div class="source-header">
                <div class="source-title">${escapeHtml(h.title || h.chunk_id)}</div>
                <div class="source-rank">#${i + 1} · ${(h.score || 0).toFixed(3)}</div>
            </div>
        `;

        const tags = [
            h.section ? `<span class="tag tag-section">${escapeHtml(h.section)}</span>` : "",
            ...(h["피심인"] || []).slice(0, 3).map(p => `<span class="tag tag-party">${escapeHtml(p)}</span>`),
            ...(h["위반유형"] || []).slice(0, 3).map(v => `<span class="tag tag-violation">${escapeHtml(v)}</span>`),
            ...(h["조치유형"] || []).slice(0, 2).map(s => `<span class="tag tag-sanction">${escapeHtml(s)}</span>`),
            `<span class="tag tag-id">${escapeHtml(h.chunk_id)}</span>`,
        ].filter(Boolean).join("");

        const preview = h.text && h.text.length > 320 ? h.text.slice(0, 320) + "…" : (h.text || "");
        item.innerHTML = `
            ${headerHtml}
            <div class="source-meta">${tags}</div>
            <div class="source-text">${escapeHtml(preview)}</div>
        `;
        els.sourcesList.appendChild(item);
    });
}

// ───────── 출처 (predict 응답 형식: chunk_ids 만 있고 hit detail 없음) ─────────
async function fetchHitDetails(chunkIds) {
    // /predict 는 chunk_ids 만 반환. 상세는 /search 로 별도 조회.
    // 여기서는 빠르게 ids 만 표시하기 위해 더미 hit 생성.
    return chunkIds.map((cid, i) => ({
        chunk_id: cid,
        title: cid,
        section: "",
        text: "",
        score: 0,
    }));
}

// ───────── API ─────────
async function callApi(path, body) {
    const r = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json; charset=utf-8" },
        body: JSON.stringify(body),
    });
    const data = await r.json();
    if (!r.ok) {
        throw new Error(data.detail || `HTTP ${r.status}`);
    }
    return data;
}

async function handleSearch() {
    const body = buildSearchRequest();
    if (!body) return;

    setLoading(true, "검색 중…");
    try {
        const t0 = performance.now();
        const data = await callApi("/search", body);
        const elapsed = (performance.now() - t0).toFixed(0);

        if (!data.hits || !data.hits.length) {
            showError("관련 의결서를 찾지 못했습니다. 질문을 다르게 표현해 보세요.");
            return;
        }

        els.sourcesMeta.textContent =
            `${data.hits.length}건 · 서버 ${(data.elapsed_ms || 0).toFixed(0)}ms · 왕복 ${elapsed}ms`;
        renderSources(data.hits);
        show(els.sources);
    } catch (err) {
        showError(`검색 실패: ${err.message}`);
    } finally {
        setLoading(false);
    }
}

async function handleAnswer() {
    // /predict 호출 → 답변 + chunk_ids
    const predBody = buildPredictRequest();
    if (!predBody) return;
    const searchBody = buildSearchRequest();
    if (!searchBody) return;

    setLoading(true, "검색 + AI 답변 생성 중… (수 초 소요)");
    try {
        const t0 = performance.now();
        // predict + search 병렬 (search는 hits detail 표시용)
        const [predData, searchData] = await Promise.all([
            callApi("/predict", predBody),
            callApi("/search", searchBody).catch(() => null),
        ]);
        const elapsed = (performance.now() - t0).toFixed(0);

        const chunkIds = predData.retrieved_chunk_ids || predData.chunk_ids || [];
        if (!chunkIds.length) {
            showError("관련 의결서를 찾지 못했습니다. 질문을 다르게 표현해 보세요.");
            return;
        }

        els.answerMeta.textContent =
            `Qwen2.5-7B · ${chunkIds.length} 청크 · 서버 ${((predData.elapsed_ms || 0)/1000).toFixed(1)}s · 왕복 ${(elapsed/1000).toFixed(1)}s`;
        els.answerBody.innerHTML = renderAnswer(predData.answer || "");
        show(els.answer);

        const hits = (searchData && searchData.hits) ? searchData.hits :
                     await fetchHitDetails(chunkIds);
        els.sourcesMeta.textContent = `${hits.length}건`;
        renderSources(hits);
        show(els.sources);
    } catch (err) {
        showError(`답변 생성 실패: ${err.message}`);
    } finally {
        setLoading(false);
    }
}

// ───────── 이벤트 바인딩 ─────────
if (els.topK) {
    els.topK.addEventListener("input", () => { els.topKValue.textContent = els.topK.value; });
}
els.searchBtn.addEventListener("click", handleSearch);
els.answerBtn.addEventListener("click", handleAnswer);

els.query.addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleAnswer();
    }
});

$$(".chip").forEach(chip => {
    chip.addEventListener("click", () => {
        els.query.value = chip.dataset.q;
        els.query.focus();
        // 부드러운 포커스 효과
        els.query.scrollIntoView({ behavior: "smooth", block: "center" });
    });
});

// ───────── 초기화 ─────────
checkHealth();
setInterval(checkHealth, 30000);
