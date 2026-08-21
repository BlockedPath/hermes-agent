// Regression tests for #19: close() during 'connecting' must settle the
// in-flight connect() promise immediately (with the closed error) and must
// NOT leave the handshake timer armed to flap closed→error afterwards.

import { JsonRpcGatewayClient } from '@hermes/shared'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

/** A socket that never opens (and never errors) — handshake hangs until
 * timeout/close. Records state-relevant callbacks for assertions. */
class NeverOpeningSocket {
  static OPEN = 1
  readyState = 0
  openHandlers: Array<() => void> = []
  addEventListener = vi.fn((type: string, handler: () => void) => {
    if (type === 'open') {
      this.openHandlers.push(handler)
    }
  })
  removeEventListener = vi.fn()
  close = vi.fn()
  send = vi.fn()
}

describe('JsonRpcGatewayClient close() during connecting (#19)', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('WebSocket', NeverOpeningSocket)
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('rejects connect() promptly with the closed error', async () => {
    const client = new JsonRpcGatewayClient({
      socketFactory: () => new NeverOpeningSocket() as unknown as WebSocket,
      // Long timeout so only close() could settle the promise.
      connectTimeoutMs: 60_000
    })

    let rejection: Error | null = null
    const pending = client.connect('ws://127.0.0.1:1234/api/ws').catch((e: Error) => {
      rejection = e
      return undefined
    })

    expect(client.connectionState).toBe('connecting')
    client.close()
    await pending

    expect(rejection).toBeInstanceOf(Error)
    expect(rejection!.message).toMatch(/closed/i)
    expect(client.connectionState).toBe('closed')
  })

  it('does not flap closed→error when the handshake timer later fires', async () => {
    const states: string[] = []
    const client = new JsonRpcGatewayClient({
      socketFactory: () => new NeverOpeningSocket() as unknown as WebSocket,
      connectTimeoutMs: 60_000
    })
    client.onState(s => states.push(s))

    const pending = client.connect('ws://127.0.0.1:1234/api/ws').catch(() => undefined)
    client.close()
    await pending

    // Advance well past the original 60s handshake timeout: the timer was
    // cleared, so no spurious 'error' state may appear after 'closed'.
    vi.advanceTimersByTime(120_000)

    // onState fires immediately with the initial state ('idle'), then
    // 'connecting' on connect(), and exactly 'closed' on close() — no
    // spurious 'error' may appear even after advancing past the handshake
    // timeout.
    expect(states).toEqual(['idle', 'connecting', 'closed'])
    expect(client.connectionState).toBe('closed')
  })

  it('clears the handshake timer (no post-close timer work)', async () => {
    const socket = new NeverOpeningSocket()
    const client = new JsonRpcGatewayClient({
      socketFactory: () => socket as unknown as WebSocket,
      connectTimeoutMs: 60_000
    })

    const pending = client.connect('ws://127.0.0.1:1234/api/ws').catch(() => undefined)
    client.close()
    await pending
    vi.advanceTimersByTime(120_000)

    // removeEventListener called for both once-listeners during abort cleanup.
    expect(socket.removeEventListener).toHaveBeenCalledWith('open', expect.any(Function))
    expect(socket.removeEventListener).toHaveBeenCalledWith('error', expect.any(Function))
  })

  it('a fresh connect() works after closing mid-handshake', async () => {
    const first = new NeverOpeningSocket()
    const second = new NeverOpeningSocket()
    const sockets = [first, second]
    const client = new JsonRpcGatewayClient({
      socketFactory: () => (sockets.shift() ?? new NeverOpeningSocket()) as unknown as WebSocket,
      connectTimeoutMs: 60_000
    })

    const stale = client.connect('ws://127.0.0.1:1234/api/ws').catch(() => undefined)
    client.close()
    await stale

    const reopened = client.connect('ws://127.0.0.1:1234/api/ws')
    // Open the second socket manually.
    for (const handler of second.openHandlers) {
      handler()
    }
    await expect(reopened).resolves.toBeUndefined()
    expect(client.connectionState).toBe('open')
  })
})
