const API = "";
const TOKEN = localStorage.getItem("token") || "";

function $(id) { return document.getElementById(id); }

function api(path, opts = {}) {
  return fetch(API + path, {
    ...opts,
    headers: { "Authorization": "Bearer " + TOKEN, ...(opts.headers || {}) },
  }).then(r => r.json());
}

function renderLogin() {
  $("app").innerHTML = `
    <div class="min-h-screen flex items-center justify-center p-4">
      <div class="card w-full max-w-sm">
        <h1 class="text-2xl font-bold mb-4">个人 RAG 知识库</h1>
        <input id="tokenInput" type="password" class="input" placeholder="Bearer Token" value="${TOKEN}">
        <button onclick="saveToken()" class="btn w-full">进入</button>
      </div>
    </div>`;
}

function saveToken() {
  localStorage.setItem("token", $("tokenInput").value);
  location.reload();
}

const TABS = [
  { name: "home", label: "知识库", icon: "🏠" },
  { name: "create", label: "录入", icon: "➕" },
  { name: "fav", label: "收藏夹", icon: "📺" },
  { name: "jobs", label: "任务", icon: "📥" },
  { name: "todos", label: "待办", icon: "✅" },
  { name: "graph", label: "图谱", icon: "🧠" },
];

let currentView = "home";

function renderApp() {
  $("app").innerHTML = `
    <div class="min-h-screen flex flex-col md:flex-row">
      <aside class="hidden md:block bg-white border-b md:border-b-0 md:border-r md:w-64 p-4">
        <h1 class="text-xl font-bold mb-6">RAG 知识库</h1>
        <nav class="space-y-2">
          ${TABS.map(t => `<button onclick="showView('${t.name}')" class="nav-btn ${currentView === t.name ? 'bg-blue-50 text-blue-600 font-medium' : ''}">${t.icon} ${t.label}</button>`).join("")}
        </nav>
      </aside>
      <main id="main" class="flex-1 p-4 overflow-auto"></main>
      <nav class="bottom-nav md:hidden">
        ${TABS.map(t => `<button onclick="showView('${t.name}')" class="${currentView === t.name ? 'active' : ''}"><span class="icon">${t.icon}</span><span>${t.label}</span></button>`).join("")}
      </nav>
    </div>`;
  showView(currentView);
}

function showView(name) {
  currentView = name;
  const main = $("main");
  if (name === "home") renderHome(main);
  else if (name === "create") renderCreate(main);
  else if (name === "fav") renderFav(main);
  else if (name === "jobs") renderJobs(main);
  else if (name === "graph") renderGraph(main);
  else if (name === "todos") renderTodos(main);
}

async function renderHome(main) {
  main.innerHTML = `<div class="mb-4"><div class="card"><div class="flex gap-2"><input id="search" class="input mb-0" placeholder="搜索知识库" onkeydown="if(event.key==='Enter') searchEntries()"><button onclick="searchEntries()" class="btn btn-sm">搜索</button></div></div></div><div id="homeBody"><div class="card skeleton h-24 mb-3"></div><div class="card skeleton h-24 mb-3"></div></div>`;
  const [entries, todos] = await Promise.all([api("/entries"), api("/todos")]);
  const undone = todos.filter(t => !t.done).slice(0, 3);
  $("homeBody").innerHTML = `
    <div class="grid gap-3">
      ${entries.map(e => `
        <div class="card active:scale-[0.99] transition-transform" onclick="showEntry('${e.id}')">
          <div class="font-bold text-lg">${escapeHtml(e.title)}</div>
          <div class="text-sm text-gray-500 mt-1">${e.type} · ${new Date(e.created_at).toLocaleString()}</div>
          <div class="mt-2 flex flex-wrap gap-2">${(e.tags||[]).map(t => `<span class="tag">${escapeHtml(t)}</span>`).join("")}</div>
        </div>
      `).join("") || "<p class='text-gray-400 text-center py-8'>暂无知识，点击底部“录入”添加</p>"}
    </div>
    ${undone.length ? `<div class="mt-6"><h2 class="font-bold mb-2">待办速览</h2>${undone.map(t => `<div class="card text-sm flex items-center gap-2"><span class="w-2 h-2 rounded-full bg-orange-400"></span>${escapeHtml(t.content)}</div>`).join("")}</div>` : ""}`;
}

async function searchEntries() {
  const q = $("search").value;
  if (!q) return;
  const res = await api("/query", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({query: q}) });
  $("main").innerHTML = `<div class="card"><h2 class="font-bold mb-2">搜索结果</h2><p>${escapeHtml(res.answer)}</p><pre class="text-xs mt-2 bg-gray-50 p-2 rounded">${escapeHtml(JSON.stringify(res.sources, null, 2))}</pre></div>`;
}

let createType = "bili";

function renderCreate(main) {
  main.innerHTML = `
    <div class="card">
      <h2 class="text-xl font-bold mb-4">新建录入</h2>
      <div class="grid grid-cols-3 gap-2 mb-4">
        ${["bili", "note", "web", "doc", "media"].map(t => {
          const labels = {bili:"B站", doc:"文档", media:"音视频", web:"网页", note:"笔记"};
          return `<button onclick="setCreateForm('${t}')" id="type-${t}" class="btn-outline ${createType===t?'ring-2 ring-blue-500 bg-blue-50':''}">${labels[t]}</button>`;
        }).join("")}
      </div>
      <div id="createForm"></div>
    </div>`;
  setCreateForm(createType);
}

function setCreateForm(type) {
  createType = type;
  document.querySelectorAll('[id^="type-"]').forEach(b => b.classList.remove("ring-2", "ring-blue-500", "bg-blue-50"));
  const active = $(`type-${type}`);
  if (active) active.classList.add("ring-2", "ring-blue-500", "bg-blue-50");
  const c = $("createForm");
  const langSel = `<select id="lang${type[0].toUpperCase()+type.slice(1)}" class="input"><option value="auto">自动</option><option value="zh">中文</option><option value="en">英文</option></select>`;
  if (type === "bili") {
    c.innerHTML = `<input id="bvid" class="input" placeholder="BV号 或 完整链接" inputmode="url"><div class="flex gap-2"><input id="biliPage" type="number" min="1" class="input" placeholder="分P（留空=全部）"><button onclidk="clearBiliPage()" type="button" class="btn btn-sm">清空</button></div>${langSel}<button onclick="submitBili()" class="btn">提取</button><div class="text-sm text-gray-500 mt-2">多P视频：留空采集全部分P并合并；填数字只采集该分P。粘贴 ?p=2 链接会自动识别。</div><div id="biliProgress" class="hidden"><div class="text-sm text-blue-600 mb-1 flex justify-between"><span>采集中...</span><span id="biliProgressText">0%</span></div><div class="progress-wrap"><div id="biliProgressBar" class="progress-bar" style="width:0%"></div></div></div><div id="biliError" class="text-red-500 mt-2"></div>`;
  } else if (type === "doc") {
    c.innerHTML = `<input id="docFile" type="file" class="input">${langSel}<button onclidk="submitDoc()" class="btn">上传</button><div id="docProgress" class="hidden progress-wrap mt-2"><div id="docProgressBar" class="progress-bar" style="width:0%"></div></div>`;
  } else if (type === "media") {
    c.innerHTML = `<input id="mediaFile" type="file" accept="audio/*,video/*" class="input">${langSel}<button onclick="submitMedia()" class="btn">转写</button><div id="mediaProgress" class="hidden progress-wrap mt-2"><div id="mediaProgressBar" class="progress-bar" style="width:0%"></div></div>`;
  } else if (type === "web") {
    c.innerHTML = `<input id="webUrl" class="input" placeholder="链接" inputmode="url">${langSel}<button onclick="submitWeb()" class="btn">抓取</button><div id="webManual" class="hidden mt-4"><textarea id="webText" class="input h-32" placeholder="复制全文粘贴到这里"></textarea><button onclick="submitWebManual()" class="btn mt-2">提交全文</button></div>`;
  } else if (type === "note") {
    c.innerHTML = `<input id="noteTitle" class="input" placeholder="标题"><textarea id="noteContent" class="input h-48" placeholder="内容..."></textarea><input id="noteTags" class="input" placeholder="标签用空格分隔"><button onclidk="submitNote()" class="btn">保存</button>`;
  }
}

function clearBiliPage() { $("biliPage").value = ""; }

async function submitBili() {
  let bvid = $("bvid").value.trim();
  const m = bvid.match(/(BV[0-9A-Za-z]{10})/);
  const pMatch = bvid.match(/[?&]p=(\d+)/);
  bvid = m ? m[1] : bvid;
  let page = $("biliPage").value ? parseInt($("biliPage").value) : (pMatch ? parseInt(pMatch[1]) : null);
  const lang = $("langBili").value;
  const progress = $("biliProgress"), bar = $("biliProgressBar"), text = $("biliProgressText");
  $("biliError").innerHTML = "";
  if (progress) progress.classList.remove("hidden");
  // 模拟进度动画：多P采集阶段主要是后端等B站，这里给 15s 缓冲进度
  let p = 0;
  const timer = setInterval(() => {
    if (p < 85) { p += Math.random() * 8; if (p > 85) p = 85; }
    if (bar) bar.style.width = p + "%";
    if (text) text.innerText = Math.round(p) + "%";
  }, 1000);
  try {
    const res = await api("/ingest/bilibili", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({bvid, language: lang, page}) });
    clearInterval(timer);
    if (bar) bar.style.width = "100%";
    if (text) text.innerText = "100%";
    if (res.status === "no_subtitle" || res.status === "api_error") {
      $("biliError").innerHTML = `${res.message} ${res.status === "no_subtitle" ? `<button onclick="forceBili('${bvid}')" class="btn-secondary text-sm mt-2">强制转写</button>` : ""}`;
      return;
    }
    setTimeout(() => showEntry(res.entry.id), 300);
  } catch (e) {
    clearInterval(timer);
    $("biliError").innerHTML = "网络错误，请重试";
  }
}

async function forceBili(bvid) {
  alert("请上传该视频的音频文件走网盘音视频转写兜底");
}

async function submitDoc() {
  const file = $("docFile").files[0];
  if (!file) return alert("请选择文件");
  const form = new FormData();
  form.append("file", file);
  const lang = $("langDoc").value;
  const progress = $("docProgress"), bar = $("docProgressBar");
  if (progress) progress.classList.remove("hidden");
  // 模拟上传/解析进度
  let p = 0;
  const timer = setInterval(() => { p += Math.random() * 10; if (p > 90) p = 90; if (bar) bar.style.width = p + "%"; }, 600);
  try {
    const res = await api(`/ingest/upload-doc?language=${lang}`, { method: "POST", body: form });
    clearInterval(timer); if (bar) bar.style.width = "100%";
    showEntry(res.entry.id);
  } catch (e) { clearInterval(timer); alert("上传失败"); }
}

async function submitMedia() {
  const file = $("mediaFile").files[0];
  if (!file) return alert("请选择音视频文件");
  const form = new FormData();
  form.append("file", file);
  const lang = $("langMedia").value;
  const progress = $("mediaProgress"), bar = $("mediaProgressBar");
  if (progress) progress.classList.remove("hidden");
  let p = 0;
  const timer = setInterval(() => { p += Math.random() * 5; if (p > 95) p = 95; if (bar) bar.style.width = p + "%"; }, 1500);
  try {
    const res = await api(`/ingest/upload-media?language=${lang}`, { method: "POST", body: form });
    clearInterval(timer); if (bar) bar.style.width = "100%";
    showEntry(res.entry.id);
  } catch (e) { clearInterval(timer); alert("转写失败"); }
}

async function submitWeb() {
  const url = $("webUrl").value;
  const lang = $("langWeb").value;
  const res = await api("/ingest/web", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({url, language: lang}) });
  if (res.status === "fetch_error") {
    $("webManual").classList.remove("hidden");
    return;
  }
  showEntry(res.entry.id);
}

async function submitWebManual() {
  const title = $("webUrl").value;
  const content = $("webText").value;
  const res = await api("/ingest/note", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({title, content, tags: ["网页"]}) });
  showEntry(res.entry.id);
}

async function submitNote() {
  const title = $("noteTitle").value;
  const content = $("noteContent").value;
  const tags = $("noteTags").value.split(/\s+/).filter(Boolean);
  const res = await api("/ingest/note", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({title, content, tags}) });
  showEntry(res.entry.id);
}

async function showEntry(id) {
  const e = await api(`/entries/${id}`);
  const main = $("main");
  main.innerHTML = `
    <div class="card">
      <h2 class="text-xl font-bold">${escapeHtml(e.title)}</h2>
      <div class="text-sm text-gray-500 mb-2">${e.type} · ${new Date(e.created_at).toLocaleString()}</div>
      <div class="tabs flex gap-2 mb-4 flex-wrap">
        <button onclick="showEntryTab('${id}', 'text')" class="tab">文稿</button>
        <button onclick="showEntryTab('${id}', 'analysis')" class="tab">分析</button>
        <button onclick="showEntryTab('${id}', 'viz')" class="tab">导图</button>
        <button onclick="showEntryTab('${id}', 'tts')" class="tab">朗读</button>
      </div>
      <div id="entryBody"></div>
      <div class="mt-4"><button onclick="createTodoFromEntry('${id}')" class="btn-secondary">转为待办</button></div>
    </div>`;
  window._entry = e;
  showEntysTab(id, "text");
}

function showEntryTab(id, name) {
  const e = window._entry;
  const body = $("entryBody");
  if (name === "text") {
    body.innerHTML = `
      <textarea id="editor" class="input h-64">${escapeHtml(e.corrected_text)}</textarea>
      <button onclik="saveEdit(('${id}')" class="btn mt-2">保存并重新入库</button>
      <div id="editResult"></div>`;
  } else if (name === "analysis") {
    body.innerHTML = `<div class="result-block"><b>总结</b><p>${escapeHtml(e.analysis?.summary||"")}</p><b>知识点</b><ul>${(e.analysis?.knowledge||[]).map(k=>`<li>${escapeHtml(k)}</li>`).join("")}</ul><b>扩展</b><ul>${(e.analysis?.expansion||[]).map(x=>`<li>${escapeHtml(x)}</li>`).join("")}</ul></div>`;
  } else if (name === "viz") {
    body.innerHTML = `<div class="result-block"><div class="markmap">${escapeHtml(e.visualization?.mindmap||"")}</div></div><div class="result-block"><pre class="mermaid">${escapeHtml(e.visualization?.flowchart||"")}</pre></div>`;
    if (window.markmap) window.markmap.autoLoader.renderAll();
    mermaid.init(undefined, document.querySelectorAll(".mermaid"));
  } else if (name === "tts") {
    body.innerHTML = `<button onclick="speakEntry('${id}')" class="btn">播放</button><audio id="player" controls class="mt-2 hidden w-full"></audio>`;
  }
}

async function saveEdit(id) {
  const text = $("editor").value;
  const e = window._entry;
  const res = await api("/ingest/manual", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({...e, raw_text: text, corrected_text: text})
  });
  if (res.status === "ok") {
    window._entry = res.entry;
    showEntry(id);
  } else {
    $("editResult").innerHTML = `<p class="text-red-500">${escapeHtml(res.message || "保存失败")}</p>`;
  }
}

async function speakEntry(id) {
  const e = window._entry;
  const text = e.corrected_text.slice(0, 500);
  const res = await fetch(API + "/tts/speak?text=" + encodeURIComponent(text) + "&language=" + e.language, { headers: { "Authorization": "Bearer " + TOKEN } });
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const player = $("player");
  player.src = url;
  player.classList.remove("hidden");
  player.play();
}

async function createTodoFromEntry(id) {
  const content = prompt("待办内容");
  if (!content) return;
  await api("/todos", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({content, entry_id: id}) });
  alert("已创建待办");
}

async function renderTodos(main) {
  const todos = await api("/todos");
  main.innerHTML = `
    <div class="card">
      <h2 class="text-xl font-bold mb-4">待办</h2>
      <div class="flex gap-2 mb-4"><input id="todoInput" class="input flex-1" placeholder="新待办"><button onclick="addTodo()" class="btn">添加</button></div>
      <div class="space-y-2">${todos.map(t => `
        <div class="flex items-center gap-2 card ${t.done?'opacity-50':''}">
          <input type="checkbox" ${t.done?'checked':''} onchange="toggleTodo('${t.id}')">
          <span class="flex-1 ${t.done?'line-through':''}">${escapeHtml(t.content)}</span>
          <button onclick="deleteTodo('${t.id}')" class="text-red-500">删</button>
        </div>
      `).join("")}</div>
    </div>`;
}

async function addTodo() {
  const content = $("todoInput").value;
  if (!content) return;
  await api("/todos", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify({content}) });
  showView("todos");
}

async function toggleTodo(id) {
  await api(`/todos/${id}/toggle`, { method: "POST" });
  showView("todos");
}

async function deleteTodo(id) {
  await api(`/todos/${id}`, { method: "DELETE" });
  showView("todos");
}

function escapeHtml(s) {
  return (s || "").replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;"}[c]));
}

// ---- B站收藏夹采集页 ----
async function renderFav(main) {
  main.innerHTML = `
    <div class="card">
      <h2 class="text-xl font-bold mb-4">B站收藏夹批量采集</h2>
      <button onclidk="loadFavs()" class="btn">加载我的收藏夹</button>
      <div id="favList" class="mt-4"></div>
    </div>`;
  loadFavs();
}

async function loadFavs() {
  const box = $("favList");
  if (!box) return;
  box.innerHTML = `<div class="card skeleton h-16 mb-2"></div><div class="card skeleton h-16 mb-2"></div>`;
  try {
    const favs = await api("/collections");
    box.innerHTML = favs.map(f => `
      <div class="card mt-2 flex items-center justify-between gap-3 active:scale-[0.99] transition-transform">
        <div class="min-w-0">
          <div class="font-medium truncate">${escapeHtml(f.title)}</div>
          <div class="text-xs text-gray-500">${f.media_count || 0} 个视频</div>
        </div>
        <button onclick="syncFav(${f.id})" class="btn-secondary btn-sm shrink-0">同步</button>
      </div>`).join("") || "<p class='text-gray-400 text-center py-6'>暂无收藏夹，检查 BILIBILI_SESSDATA</p>";
  } catch (e) {
    box.innerHTML = `<p class="text-red-500">加载失败，请确认 BILIBILI_SESSDATA 已配置</p>`;
  }
}

async function syncFav(id) {
  await api(`/collections/${id}/sync`, { method: "POST" });
  showView("jobs");
}

// ---- 采集任务页 ----
let jobsTimer = null;

async function renderJobs(main) {
  main.innerHTML = `<div class="card">
    <h2 class="text-xl font-bold mb-4">采集任务</h2>
    <div id="jobsBody"><p class="text-gray-400">加载中...</p></div>
    <div class="mt-4 text-xs text-gray-400">同步收藏夹或上传大文件时会生成任务，自动轮询进度。</div>
  </div>`;
  await updateJobs();
  if (jobsTimer) clearInterval(jobsTimer);
  jobsTimer = setInterval(updateJobs, 2000);
}

async function updateJobs() {
  if (currentView !== "jobs") { if (jobsTimer) clearInterval(jobsTimer); return; }
  const box = $("jobsBody");
  if (!box) return;
  const jobs = await api("/jobs");
  box.innerHTML = jobs.map(j => `
    <div class="mb-4 card bg-gray-50">
      <div class="flex justify-between text-sm font-medium"><span>${escapeHtml(j.kind)}</span><span id="job-status-${j.id}" class="${j.status==='done'?'text-green-600':j.status==='failed'?'text-red-500':'text-blue-600'}">${escapeHtml(j.status)} ${j.progress}%</span></div>
      <div class="progress-wrap"><div id="job-bar-${j.id}" class="progress-bar" style="width:${j.progress}%"></div></div>
      <div class="text-xs text-gray-500 mt-1">共 ${j.total || '?'} 个 · 成功 ${j.success || 0} · 失败 ${(j.failed||[]).length}</div>
      ${j.failed && j.failed.length ? `<div class="text-red-500 text-xs mt-1">失败: ${escapeHtml(j.failed.slice(0,3).join(', '))}${j.failed.length>3?' 等':''}</div>` : ""}
    </div>`).join("") || "<p class='text-gray-400 text-center py-8'>暂无任务</p>";
  // 全部完成停止轮询
  if (jobs.length && jobs.every(j => j.status === "done" || j.status === "failed")) {
    if (jobsTimer) clearInterval(jobsTimer);
  }
}

function stopJobsTimer() { if (jobsTimer) clearInterval(jobsTimer); }

// ---- 知识图谱 / 知识树页 ----
async function renderGraph(main) {
  const nodes = await api("/knowledge/nodes");
  main.innerHTML = `<div class="card">
    <h2 class="text-xl font-bold mb-4">知识图谱</h2>
    <p class="text-sm text-gray-500 mb-3">点击知识点展开知识树</p>
    <div class="flex flex-wrap gap-2">${nodes.map(n => `<button onclick="showTree('${escapeHtml(n.name)}')" class="tag">${escapeHtml(n.name)}</button>`).join("") || "<p>暂无知识点</p>"}</div>
    <div id="treeBox" class="mt-4"></div>
  </div>`;
}

async function showTree(name) {
  const tree = await api(`/knowledge/tree?name=${encodeURIComponent(name)}`);
  let md = `# ${tree.name}\n`;
  (function walk(node, indent) {
    for (const child of node.children || []) {
      md += `${indent}## ${child.name} (${child.relation})\n`;
      walk(child, indent + "## ");
    }
  })(tree, "");
  $("treeBox").innerHTML = `<div class="result-block markmap">${escapeHtml(md)}</div>`;
  if (window.markmap) window.markmap.autoLoader.renderAll();
}

if ("serviceWorker" in navigator) {
  navigator.serviceWorker.register("sw.js").catch(() => {});
}
if (!TOKEN) renderLogin();
else renderApp();
