// LLMQwen test console — plain JS, no build step, no framework.
// This page is served as its own standalone static-file service (see
// serve.sh's frontend-svc window), on a different port than the backend —
// so every call below is cross-origin to the search service, which already
// allows any origin (CORSMiddleware(allow_origins=["*"]) in services/search/app.py).
//
// The backend URL is a user-editable field, not a hardcoded constant: over
// SSH port-forwarding (e.g. VS Code Remote) the "Forwarded Address" for a
// port often does NOT match the real port number (every port can get
// remapped to a different local one when the natural number is already
// taken locally) — so the actual reachable address has to come from
// whatever the Ports panel currently shows, not a guess baked into the code.
const BACKEND_URL_STORAGE_KEY = "llmqwen-backend-url";
const backendUrlInput = document.getElementById("backend-url");

function getBackendUrl() {
  return (backendUrlInput.value.trim() || backendUrlInput.placeholder).replace(/\/+$/, "");
}

function loadBackendUrl() {
  backendUrlInput.value = localStorage.getItem(BACKEND_URL_STORAGE_KEY) || "";
}

backendUrlInput.addEventListener("change", () => {
  localStorage.setItem(BACKEND_URL_STORAGE_KEY, backendUrlInput.value.trim());
  loadAgents();
});

const chatEl = document.getElementById("chat");
const agentSelect = document.getElementById("agent-select");
const agentInfoEl = document.getElementById("agent-info");
const temperatureInput = document.getElementById("temperature");
const temperatureValueEl = document.getElementById("temperature-value");
const composeForm = document.getElementById("compose-form");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const stopBtn = document.getElementById("stop-btn");
const attachBtn = document.getElementById("attach-btn");
const fileInput = document.getElementById("file-input");
const attachmentsEl = document.getElementById("attachments");
const reloadAgentsBtn = document.getElementById("reload-agents-btn");
const newConversationBtn = document.getElementById("new-conversation-btn");

let agents = [];
let history = []; // [{role: "user"|"assistant", content: string}, ...]
let attachedDocs = []; // [{filename, text}, ...]
let abortController = null;

// ---------------------------------------------------------------------------
// Agents
// ---------------------------------------------------------------------------

async function loadAgents() {
  try {
    const res = await fetch(`${getBackendUrl()}/agents`);
    const data = await res.json();
    agents = data.agents || [];
    agentSelect.innerHTML = "";
    for (const agent of agents) {
      const opt = document.createElement("option");
      opt.value = agent.id;
      opt.textContent = `${agent.id} — ${agent.name}`;
      agentSelect.appendChild(opt);
    }
    if (agents.length > 0) {
      agentSelect.value = agents.some((a) => a.id === "TSP-0") ? "TSP-0" : agents[0].id;
    }
    updateAgentInfo();
  } catch (err) {
    agentSelect.innerHTML = '<option value="TSP-0">TSP-0 (failed to load list)</option>';
    showToast(`Failed to load agents: ${err.message}`, true);
  }
}

function updateAgentInfo() {
  const agent = agents.find((a) => a.id === agentSelect.value);
  if (!agent) {
    agentInfoEl.textContent = "";
    return;
  }
  const webSearchClass = agent.web_search ? "web-search-on" : "web-search-off";
  const webSearchLabel = agent.web_search ? "web search ON" : "web search off";
  agentInfoEl.innerHTML =
    `<b>${escapeHtml(agent.name)}</b> — ${escapeHtml(agent.profession)}` +
    (agent.department ? ` · ${escapeHtml(agent.department)}` : "") +
    ` &nbsp;<span class="${webSearchClass}">[${webSearchLabel}]</span>`;
}

agentSelect.addEventListener("change", updateAgentInfo);

reloadAgentsBtn.addEventListener("click", async () => {
  reloadAgentsBtn.disabled = true;
  try {
    await fetch(`${getBackendUrl()}/agents/reload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    await loadAgents();
    showToast("Agent caches reloaded (search + llm)");
  } catch (err) {
    showToast(`Reload failed: ${err.message}`, true);
  } finally {
    reloadAgentsBtn.disabled = false;
  }
});

// ---------------------------------------------------------------------------
// Temperature
// ---------------------------------------------------------------------------

temperatureInput.addEventListener("input", () => {
  temperatureValueEl.textContent = parseFloat(temperatureInput.value).toFixed(2);
});

// ---------------------------------------------------------------------------
// Message rendering
// ---------------------------------------------------------------------------

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function formatMessageText(raw) {
  let html = escapeHtml(raw);
  // Linkify bare URLs (escaped text has no HTML, so this is safe)
  html = html.replace(/(https?:\/\/[^\s)]+)/g, '<a href="$1" target="_blank" rel="noopener">$1</a>');
  // Minimal **bold** support
  html = html.replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  return html;
}

function appendMessage(role, text, { pending = false, error = false } = {}) {
  const el = document.createElement("div");
  el.className = `msg ${role}` + (pending ? " pending" : "") + (error ? " error" : "");
  const tag = document.createElement("span");
  tag.className = "role-tag";
  tag.textContent = role;
  el.appendChild(tag);
  const body = document.createElement("span");
  body.className = "body";
  body.innerHTML = formatMessageText(text);
  el.appendChild(body);
  chatEl.appendChild(el);
  chatEl.scrollTop = chatEl.scrollHeight;
  return { el, body };
}

function setMessageSources(el, sources) {
  if (!sources || sources.length === 0) return;
  const div = document.createElement("div");
  div.className = "sources";
  div.appendChild(document.createTextNode("Sources: "));
  for (const src of sources) {
    const a = document.createElement("a");
    a.href = src;
    a.target = "_blank";
    a.rel = "noopener";
    a.textContent = src;
    div.appendChild(a);
  }
  el.appendChild(div);
}

// ---------------------------------------------------------------------------
// SSE parsing (manual — EventSource doesn't support POST bodies)
// ---------------------------------------------------------------------------

async function streamQuery(payload, callbacks) {
  abortController = new AbortController();
  const res = await fetch(`${getBackendUrl()}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: abortController.signal,
  });

  if (!res.ok || !res.body) {
    let message = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      message = err.detail || message;
    } catch {
      // not JSON, keep status-based message
    }
    callbacks.onError(message);
    return;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const rawEvent = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      const line = rawEvent.split("\n").find((l) => l.startsWith("data: "));
      if (!line) continue;
      let payload;
      try {
        payload = JSON.parse(line.slice(6));
      } catch {
        continue;
      }
      if (payload.type === "meta") callbacks.onMeta(payload);
      else if (payload.type === "chunk") callbacks.onChunk(payload.text);
      else if (payload.type === "done") callbacks.onDone();
      else if (payload.type === "error") callbacks.onError(payload.error);
    }
  }
}

// ---------------------------------------------------------------------------
// Sending
// ---------------------------------------------------------------------------

function setBusy(busy) {
  sendBtn.hidden = busy;
  stopBtn.hidden = !busy;
  messageInput.disabled = busy;
  agentSelect.disabled = busy;
}

async function sendMessage(query) {
  const documents = attachedDocs.length > 0 ? attachedDocs : null;
  appendMessage("user", query);
  const historyForThisTurn = history.slice(); // snapshot before this turn

  const { el: assistantEl, body: assistantBody } = appendMessage("assistant", "", { pending: true });
  let accumulated = "";
  let sources = [];
  setBusy(true);

  try {
    await streamQuery(
      {
        query,
        documents,
        temperature: parseFloat(temperatureInput.value),
        agent_id: agentSelect.value,
        history: historyForThisTurn,
      },
      {
        onMeta: (meta) => {
          sources = meta.sources || [];
        },
        onChunk: (text) => {
          accumulated += text;
          assistantBody.innerHTML = formatMessageText(accumulated);
          chatEl.scrollTop = chatEl.scrollHeight;
        },
        onDone: () => {
          assistantEl.classList.remove("pending");
          setMessageSources(assistantEl, sources);
          history.push({ role: "user", content: query });
          history.push({ role: "assistant", content: accumulated });
        },
        onError: (message) => {
          assistantEl.classList.remove("pending");
          assistantEl.classList.add("error");
          assistantBody.innerHTML = formatMessageText(accumulated ? `${accumulated}\n\n[Error: ${message}]` : `[Error: ${message}]`);
        },
      }
    );
  } catch (err) {
    if (err.name === "AbortError") {
      assistantEl.classList.remove("pending");
      if (!accumulated) assistantBody.textContent = "[stopped]";
    } else {
      assistantEl.classList.remove("pending");
      assistantEl.classList.add("error");
      assistantBody.textContent = `[Error: ${err.message}]`;
    }
  } finally {
    setBusy(false);
    abortController = null;
  }
}

composeForm.addEventListener("submit", (e) => {
  e.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;
  messageInput.value = "";
  autoResize();
  sendMessage(text);
});

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    composeForm.requestSubmit();
  }
});

function autoResize() {
  messageInput.style.height = "auto";
  messageInput.style.height = `${Math.min(messageInput.scrollHeight, 160)}px`;
}
messageInput.addEventListener("input", autoResize);

stopBtn.addEventListener("click", () => {
  if (abortController) abortController.abort();
});

// ---------------------------------------------------------------------------
// Attachments (POST /extract)
// ---------------------------------------------------------------------------

attachBtn.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", async () => {
  const file = fileInput.files[0];
  fileInput.value = "";
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);
  attachBtn.disabled = true;
  try {
    const res = await fetch(`${getBackendUrl()}/extract`, { method: "POST", body: formData });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    attachedDocs.push({ filename: data.filename, text: data.text });
    renderAttachments();
  } catch (err) {
    showToast(`Failed to extract ${file.name}: ${err.message}`, true);
  } finally {
    attachBtn.disabled = false;
  }
});

function renderAttachments() {
  attachmentsEl.innerHTML = "";
  attachedDocs.forEach((doc, i) => {
    const chip = document.createElement("div");
    chip.className = "attachment-chip";
    const label = document.createElement("span");
    label.textContent = doc.filename;
    chip.appendChild(label);
    const removeBtn = document.createElement("button");
    removeBtn.textContent = "✕";
    removeBtn.type = "button";
    removeBtn.addEventListener("click", () => {
      attachedDocs.splice(i, 1);
      renderAttachments();
    });
    chip.appendChild(removeBtn);
    attachmentsEl.appendChild(chip);
  });
}

// ---------------------------------------------------------------------------
// New conversation
// ---------------------------------------------------------------------------

newConversationBtn.addEventListener("click", () => {
  history = [];
  attachedDocs = [];
  renderAttachments();
  chatEl.innerHTML = "";
});

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------

let toastTimer = null;
function showToast(message, isError = false) {
  let toast = document.querySelector(".status-toast");
  if (!toast) {
    toast = document.createElement("div");
    toast.className = "status-toast";
    document.body.appendChild(toast);
  }
  toast.textContent = message;
  toast.classList.toggle("error", isError);
  toast.classList.add("visible");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove("visible"), 3000);
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

loadBackendUrl();
loadAgents();
