// The S1 client.
//
// Three rules the whole UI depends on:
//
// 1. **Every permission decision is the server's.** This module never hides a
//    control because of a role; it sends the request and renders the refusal. A UI
//    that guessed would eventually guess differently from the service, and the
//    disagreement would be invisible.
// 2. **Every mutation carries an idempotency key.** A double-clicked button must
//    not approve twice, and the key makes the second request return the first
//    answer rather than doing the work again.
// 3. **Nothing is cached across a mutation.** The views re-fetch, because the read
//    models are projections and the projection is what the operator must see.

const TOKEN_KEY = 'agent-s1-token'

export function token() {
  try {
    return sessionStorage.getItem(TOKEN_KEY) || ''
  } catch {
    // A private window with storage blocked: the session is this page load.
    return ''
  }
}

export function setToken(value) {
  try {
    if (value) sessionStorage.setItem(TOKEN_KEY, value)
    else sessionStorage.removeItem(TOKEN_KEY)
  } catch {
    /* ignored: the token lives for this page load only */
  }
}

/** A key that is stable for one intent and different for the next. */
export function idempotencyKey(intent) {
  const random =
    globalThis.crypto && globalThis.crypto.randomUUID
      ? globalThis.crypto.randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `${intent}:${random}`
}

async function request(method, path, body) {
  const headers = { accept: 'application/json' }
  const current = token()
  if (current) headers.authorization = `Bearer ${current}`
  if (body) headers['content-type'] = 'application/json'
  const response = await fetch(path, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  })
  const text = await response.text()
  const parsed = text ? JSON.parse(text) : null
  if (!response.ok) {
    const error = new Error(
      (parsed && parsed.error && parsed.error.detail) || `${method} ${path} failed`,
    )
    error.status = response.status
    // The service says whether a retry can help; the UI repeats that rather than
    // deciding.
    error.retryable = Boolean(parsed && parsed.error && parsed.error.retryable)
    error.code = parsed && parsed.error && parsed.error.code
    throw error
  }
  return parsed
}

export const get = (path) => request('GET', path)
export const post = (path, body) => request('POST', path, body)
export const del = (path) => request('DELETE', path)

/** A mutation: the key is added here so no caller can forget it. */
export function mutate(path, intent, body = {}) {
  return post(path, { idempotency_key: idempotencyKey(intent), ...body })
}

export const health = () => get('/healthz')
export const deployments = () => get('/deployments')
export const members = (root) => get(`/deployments/${encodeURIComponent(root)}/members`)
export const functions = (root) => get(`/deployments/${encodeURIComponent(root)}/functions`)
export const promotions = () => get('/promotions')
export const policy = () => get('/policy')
export const budgets = () => get('/budgets')
export const jobs = () => get('/jobs')
export const experiments = () => get('/experiments')
export const killswitches = () => get('/killswitch')

export const approve = (id, note) =>
  mutate(`/promotions/${encodeURIComponent(id)}/approve`, `approve:${id}`, { note })
export const reject = (id, note) =>
  mutate(`/promotions/${encodeURIComponent(id)}/reject`, `reject:${id}`, { note })
export const issuePolicy = (deployments, capacity) =>
  mutate('/policy', 'policy', { deployments, capacity })
export const engage = (scope, mode, reason) =>
  mutate('/killswitch', 'killswitch', { ...scope, mode, reason })
export const release = (id) => del(`/killswitch/${encodeURIComponent(id)}`)
export const pauseExperiment = (id) =>
  mutate(`/experiments/${encodeURIComponent(id)}/pause`, `pause:${id}`)
export const resumeExperiment = (id) =>
  mutate(`/experiments/${encodeURIComponent(id)}/resume`, `resume:${id}`)
export const cancelExperiment = (id, reason) =>
  mutate(`/experiments/${encodeURIComponent(id)}/cancel`, `cancel:${id}`, { reason })

/** A live event feed. SSE, so the browser reconnects on its own. */
export function subscribe(onEvent, onError) {
  const source = new EventSource('/events/subscribe')
  source.onmessage = (message) => {
    try {
      onEvent(JSON.parse(message.data))
    } catch (error) {
      if (onError) onError(error)
    }
  }
  source.onerror = (error) => {
    if (onError) onError(error)
  }
  return () => source.close()
}

/** Short hash, for a table. The full one is always in a title attribute. */
export function short(hash) {
  if (!hash) return '—'
  const hex = String(hash).split(':').pop()
  return hex.slice(0, 10)
}

export function pct(value) {
  return value == null ? '—' : `${(value * 100).toFixed(1)}%`
}

export function money(value) {
  return value == null ? '—' : `$${Number(value).toFixed(4)}`
}

export function ms(value) {
  if (value == null) return '—'
  return value >= 1000 ? `${(value / 1000).toFixed(2)}s` : `${Math.round(value)}ms`
}
