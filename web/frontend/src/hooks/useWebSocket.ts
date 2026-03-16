import { useEffect, useRef, useState, useCallback } from 'react'

interface WsMessage {
  type: 'history' | 'line' | 'status' | 'metric'
  lines?: string[]
  line?: string
  status?: string
  generations_done?: number
  total_generations?: number
  efficiency?: number
  asr?: number
}

interface ChartPoint {
  gen: number
  efficiency: number
  asr: number
}

export function useLogSocket() {
  const [lines, setLines] = useState<string[]>([])
  const [wsStatus, setWsStatus] = useState<string>('idle')
  const [genDone, setGenDone] = useState(0)
  const [genTotal, setGenTotal] = useState(0)
  const [history, setHistory] = useState<ChartPoint[]>([])
  const wsRef = useRef<WebSocket | null>(null)
  const retryRef = useRef<ReturnType<typeof setTimeout>>()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = import.meta.env.DEV ? 'localhost:8765' : window.location.host
    const ws = new WebSocket(`${protocol}://${host}/ws/logs`)
    wsRef.current = ws

    ws.onmessage = (e) => {
      const msg: WsMessage = JSON.parse(e.data)
      if (msg.type === 'history' && msg.lines) {
        setLines(msg.lines)
      } else if (msg.type === 'line' && msg.line) {
        setLines((prev) => [...prev.slice(-1999), msg.line!])
      } else if (msg.type === 'status') {
        if (msg.status) setWsStatus(msg.status)
        if (msg.generations_done !== undefined) setGenDone(msg.generations_done)
        if (msg.total_generations !== undefined) setGenTotal(msg.total_generations)
      } else if (msg.type === 'metric' && msg.generations_done !== undefined) {
        setHistory((prev) => [
          ...prev,
          { gen: msg.generations_done!, efficiency: msg.efficiency ?? 0, asr: msg.asr ?? 0 },
        ])
      }
    }

    ws.onclose = () => {
      retryRef.current = setTimeout(connect, 2000)
    }
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryRef.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { lines, wsStatus, genDone, genTotal, history }
}
