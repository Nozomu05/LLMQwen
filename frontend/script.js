const askButton = document.getElementById('ask')
const questionInput = document.getElementById('question')
const statusEl = document.getElementById('status')
const chatEl = document.getElementById('chat')

const setStatus = (text) => { statusEl.textContent = text }
const setLoading = (loading) => { askButton.disabled = loading }


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

const ask = async () => {
  const question = questionInput.value.trim()
  if (!question) {
    setStatus('Please enter a question')
    return
  }
  if (askButton.disabled) return
  setLoading(true)
  setStatus('Retrieving and generating...')
  appendMessage('user', question)
  questionInput.value = ''
  try {
    const response = await fetch('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question })
    })
    const data = await response.json()
    if (!response.ok) {
      throw new Error(data.error || 'Unknown error')
    }
    const sourcesText = data.sources.length ? `Sources:\n${data.sources.join('\n')}` : 'Sources: none'
    appendMessage('assistant', `Model: ${data.model}\n\n${data.answer}`, sourcesText)
    setStatus('Done')
  } catch (err) {
    console.error('Fetch error:', err)
    appendMessage('assistant', 'Error: ' + err.message)
    setStatus('Error')
  } finally {
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
