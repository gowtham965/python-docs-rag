import { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const ERROR_MESSAGE =
  "Something went wrong answering that question — the LLM provider may be temporarily rate-limited or unavailable. Please try again in a moment."

async function* streamChatEvents(question) {
  const response = await fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!response.ok || !response.body) {
    throw new Error('request failed')
  }

  const reader = response.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    buffer += decoder.decode(value, { stream: true })

    let boundary = buffer.indexOf('\n\n')
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary)
      buffer = buffer.slice(boundary + 2)
      if (rawEvent.startsWith('data: ')) {
        yield JSON.parse(rawEvent.slice('data: '.length))
      }
      boundary = buffer.indexOf('\n\n')
    }
  }
}

function SourcesList({ sources }) {
  if (!sources || sources.length === 0) return null
  return (
    <details className="sources">
      <summary>Sources used</summary>
      <ul>
        {sources.map((source, index) => (
          <li key={index}>
            <strong>{source.section_title}</strong> ({source.source_file}) — score{' '}
            {source.score.toFixed(2)}
          </li>
        ))}
      </ul>
    </details>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [question, setQuestion] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)

  function updateLastMessage(updater) {
    setMessages((prev) => {
      const next = [...prev]
      next[next.length - 1] = updater(next[next.length - 1])
      return next
    })
  }

  async function handleSubmit(event) {
    event.preventDefault()
    const trimmed = question.trim()
    if (!trimmed || isStreaming) return

    setQuestion('')
    setIsStreaming(true)
    setMessages((prev) => [
      ...prev,
      { role: 'user', text: trimmed },
      { role: 'assistant', text: '', sources: [] },
    ])

    try {
      for await (const streamEvent of streamChatEvents(trimmed)) {
        if (streamEvent.type === 'token') {
          updateLastMessage((msg) => ({ ...msg, text: msg.text + streamEvent.text }))
        } else if (streamEvent.type === 'done') {
          updateLastMessage((msg) => ({ ...msg, sources: streamEvent.sources }))
        } else if (streamEvent.type === 'error') {
          updateLastMessage((msg) => ({ ...msg, text: ERROR_MESSAGE, sources: [] }))
        }
      }
    } catch {
      updateLastMessage((msg) => ({ ...msg, text: ERROR_MESSAGE, sources: [] }))
    } finally {
      setIsStreaming(false)
    }
  }

  return (
    <div className="app">
      <h1>🐍 Python Docs Q&amp;A</h1>
      <div className="messages">
        {messages.map((message, index) => (
          <div key={index} className={`message ${message.role}`}>
            <p>{message.text}</p>
            {message.role === 'assistant' && <SourcesList sources={message.sources} />}
          </div>
        ))}
      </div>
      <form onSubmit={handleSubmit} className="composer">
        <input
          type="text"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="Ask a question about the Python standard library..."
          disabled={isStreaming}
        />
        <button type="submit" disabled={isStreaming || !question.trim()}>
          Send
        </button>
      </form>
    </div>
  )
}
