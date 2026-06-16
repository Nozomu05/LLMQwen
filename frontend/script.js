const askButton = document.getElementById('ask')
const questionInput = document.getElementById('question')
const statusEl = document.getElementById('status')
const chatEl = document.getElementById('chat')
const fileInput = document.getElementById('fileInput')
const attachLabel = document.getElementById('attachLabel')
const attachmentsEl = document.getElementById('attachments')

const MAX_DOCS = 5
const MAX_FILE_BYTES = 20 * 1024 * 1024  // 20 MB per file

const attachedDocs = []  // [{filename, text, chars}]

const setStatus = (text, level) => {
  statusEl.textContent = text
  statusEl.className = 'status' + (level ? ` status-${level}` : '')
}
const setLoading = (loading) => { askButton.disabled = loading }

// --- Queue polling ---
let _queuePollTimer = null

function startQueuePolling() {
  stopQueuePolling()
  _pollQueue()
  _queuePollTimer = setInterval(_pollQueue, 3000)
}

function stopQueuePolling() {
  if (_queuePollTimer !== null) {
    clearInterval(_queuePollTimer)
    _queuePollTimer = null
  }
}

async function _pollQueue() {
  try {
    const res = await fetch('/api/queue')
    if (!res.ok) return
    const data = await res.json()

    if (data.backend === 'transformers') return  // single-user mode, no queue to show

    if (data.backend === 'vllm_unreachable') {
      setStatus('vLLM server unreachable — check that serve.sh is running', 'error')
      return
    }

    const { waiting, running } = data
    if (waiting === 0 && running === 0) {
      setStatus('Sending to model...', 'busy')
    } else if (waiting === 0) {
      setStatus(`Generating... (${running} request${running !== 1 ? 's' : ''} running)`, 'busy')
    } else {
      const est = Math.round(waiting * 35)  // rough 35s per slot
      setStatus(`Queued — ${waiting} ahead of you (~${est}s wait)`, 'queue')
    }
  } catch (_) {
    // network error — stay silent, the main fetch will catch it
  }
}


const formatAnswer = (text) => {
  while (text.indexOf('**') !== -1) {
    text = text.split('**').join('')
  }
  while (text.indexOf('*') !== -1) {
    text = text.split('*').join('')
  }

  const lines = text.split('\n')
  const formatted = []
  let inList = false

  for (let line of lines) {
    line = line.trim()
    if (!line) {
      if (inList) formatted.push('</ul>')
      formatted.push('<p>&nbsp;</p>')
      inList = false
      continue
    }

    if (line.startsWith('#### ')) {
      if (inList) formatted.push('</ul>')
      formatted.push(`<h2>${escapeHtml(line.substring(5))}</h2>`)
      inList = false
    } else if (line.startsWith('### ')) {
      if (inList) formatted.push('</ul>')
      formatted.push(`<h3>${escapeHtml(line.substring(4))}</h3>`)
      inList = false
    } else if (line.startsWith('## ')) {
      if (inList) formatted.push('</ul>')
      formatted.push(`<h2>${escapeHtml(line.substring(3))}</h2>`)
      inList = false
    } else if (line.startsWith('# ')) {
      if (inList) formatted.push('</ul>')
      formatted.push(`<h2>${escapeHtml(line.substring(2))}</h2>`)
      inList = false
    } else if (line.startsWith('- ')) {
      if (!inList) formatted.push('<ul>')
      formatted.push(`<li>${escapeHtml(line.substring(2))}</li>`)
      inList = true
    } else {
      if (inList) formatted.push('</ul>')
      formatted.push(`<p>${escapeHtml(line)}</p>`)
      inList = false
    }
  }

  if (inList) formatted.push('</ul>')
  return formatted.join('')
}

const escapeHtml = (text) => {
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }
  return text.replace(/[&<>"']/g, (ch) => map[ch])
}

const appendMessage = (role, text, meta) => {
  const bubble = document.createElement('div')
  bubble.className = `message ${role}`

  if (role === 'assistant') {
    bubble.innerHTML = formatAnswer(text)
  } else {
    bubble.textContent = text
  }

  if (meta && role === 'assistant') {
    const metaEl = document.createElement('div')
    metaEl.className = 'meta'
    metaEl.textContent = meta
    bubble.appendChild(metaEl)
  }

  chatEl.appendChild(bubble)
  chatEl.scrollTop = chatEl.scrollHeight
}

const renderAttachments = () => {
  attachmentsEl.innerHTML = ''
  attachedDocs.forEach((doc, i) => {
    const chip = document.createElement('div')
    chip.className = 'chip'
    const kchars = doc.chars >= 1000 ? `${(doc.chars / 1000).toFixed(1)}k` : doc.chars
    chip.innerHTML =
      `<span title="${escapeHtml(doc.filename)}">📄 ${escapeHtml(doc.filename)} <em>(${kchars} chars)</em></span>` +
      `<button class="chip-remove" data-idx="${i}" title="Remove">×</button>`
    attachmentsEl.appendChild(chip)
  })
  // Disable attach button when limit reached
  if (attachedDocs.length >= MAX_DOCS) {
    attachLabel.classList.add('disabled')
    fileInput.disabled = true
  } else {
    attachLabel.classList.remove('disabled')
    fileInput.disabled = false
  }
}

attachmentsEl.addEventListener('click', (e) => {
  if (e.target.classList.contains('chip-remove')) {
    const idx = parseInt(e.target.dataset.idx, 10)
    attachedDocs.splice(idx, 1)
    renderAttachments()
    if (attachedDocs.length === 0) setStatus('Ready')
    else setStatus(`${attachedDocs.length} document${attachedDocs.length > 1 ? 's' : ''} attached`)
  }
})

const readFileAsDataURL = (file) => new Promise((resolve, reject) => {
  const reader = new FileReader()
  reader.onload = (e) => resolve(e.target.result)
  reader.onerror = reject
  reader.readAsDataURL(file)
})

fileInput.addEventListener('change', async (e) => {
  const files = Array.from(e.target.files)
  fileInput.value = ''  // reset so same file can be re-selected

  for (const file of files) {
    if (attachedDocs.length >= MAX_DOCS) {
      setStatus(`Maximum ${MAX_DOCS} documents allowed`)
      break
    }
    if (file.size > MAX_FILE_BYTES) {
      setStatus(`"${file.name}" exceeds 20 MB limit — skipped`)
      continue
    }
    setStatus(`Uploading ${file.name}...`)
    try {
      const dataUrl = await readFileAsDataURL(file)
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: file.name, data: dataUrl })
      })
      const result = await res.json()
      if (!res.ok) throw new Error(result.error || 'Upload failed')
      attachedDocs.push({ filename: result.filename, text: result.text, chars: result.chars })
      renderAttachments()
      setStatus(`${attachedDocs.length} document${attachedDocs.length > 1 ? 's' : ''} attached`)
    } catch (err) {
      setStatus(`Upload error for "${file.name}": ${err.message}`)
    }
  }
})

const ask = async () => {
  const question = questionInput.value.trim()
  if (!question) {
    setStatus('Please enter a question')
    return
  }
  if (askButton.disabled) return
  setLoading(true)
  setStatus('Sending request...', 'busy')
  appendMessage('user', question)
  questionInput.value = ''
  startQueuePolling()

  let bubble = null
  let streamDiv = null
  let fullText = ''
  let metaData = null

  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question,
        documents: attachedDocs.map(d => ({ filename: d.filename, text: d.text }))
      })
    })

    if (!response.ok) {
      let errMsg = 'Unknown error'
      try { errMsg = (await response.json()).error || errMsg } catch (_) {}
      throw new Error(errMsg)
    }

    // Create a bubble for streaming output
    bubble = document.createElement('div')
    bubble.className = 'message assistant'
    streamDiv = document.createElement('div')
    bubble.appendChild(streamDiv)
    chatEl.appendChild(bubble)

    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let sseBuffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      sseBuffer += decoder.decode(value, { stream: true })

      // SSE events are separated by double newlines
      const parts = sseBuffer.split('\n\n')
      sseBuffer = parts.pop()  // keep any incomplete trailing chunk

      for (const part of parts) {
        if (!part.startsWith('data: ')) continue
        let event
        try { event = JSON.parse(part.slice(6)) } catch (_) { continue }

        if (event.type === 'error') {
          throw new Error(event.error)
        } else if (event.type === 'meta') {
          metaData = event
          fullText = event.header
          streamDiv.textContent = fullText
          stopQueuePolling()
          setStatus('Generating...', 'busy')
          chatEl.scrollTop = chatEl.scrollHeight
        } else if (event.type === 'chunk') {
          fullText += event.text
          streamDiv.textContent = fullText
          chatEl.scrollTop = chatEl.scrollHeight
        } else if (event.type === 'done') {
          // Replace raw text with fully formatted HTML
          bubble.innerHTML = formatAnswer(fullText)
          if (metaData && metaData.sources && metaData.sources.length) {
            const metaEl = document.createElement('div')
            metaEl.className = 'meta'
            metaEl.textContent = `Sources:\n${metaData.sources.join('\n')}`
            bubble.appendChild(metaEl)
          }
          chatEl.scrollTop = chatEl.scrollHeight
        }
      }
    }

    setStatus(attachedDocs.length > 0
      ? `Done — ${attachedDocs.length} document${attachedDocs.length > 1 ? 's' : ''} attached`
      : 'Done')
  } catch (err) {
    console.error('Fetch error:', err)
    const msg = err.message || 'Unknown error'
    if (bubble) {
      bubble.innerHTML = `<p>Error: ${escapeHtml(msg)}</p>`
    } else {
      appendMessage('assistant', 'Error: ' + msg)
    }
    setStatus('Error: ' + msg, 'error')
  } finally {
    stopQueuePolling()
    setLoading(false)
    questionInput.focus()
  }
}

if (askButton && questionInput) {
  askButton.addEventListener('click', (e) => {
    e.preventDefault()
    ask()
  })
  questionInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      ask()
    }
  })
}
