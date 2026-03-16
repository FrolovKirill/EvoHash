import { useState, useCallback, useEffect } from 'react'

const BASE = import.meta.env.DEV ? 'http://localhost:8765' : ''

export async function apiFetch(path: string, opts?: RequestInit) {
  const res = await fetch(BASE + path, opts)
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`)
  return res.json()
}

export function useStatus() {
  const [data, setData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  const fetch_ = useCallback(() => {
    apiFetch('/api/status').then(setData).catch(console.error).finally(() => setLoading(false))
  }, [])

  useEffect(() => {
    fetch_()
    const id = setInterval(fetch_, 1500)
    return () => clearInterval(id)
  }, [fetch_])

  return { status: data, loading, refetch: fetch_ }
}

export function useEnvCheck() {
  const [checks, setChecks] = useState<any>(null)

  useEffect(() => {
    apiFetch('/api/check-env').then(setChecks).catch(console.error)
  }, [])

  return checks
}

export function useMetrics(phf: string) {
  const [metrics, setMetrics] = useState<any>({})

  useEffect(() => {
    const fetch_ = () =>
      apiFetch(`/api/metrics?phf=${phf}`).then((d) => setMetrics(d.metrics)).catch(() => {})
    fetch_()
    const id = setInterval(fetch_, 3000)
    return () => clearInterval(id)
  }, [phf])

  return metrics
}

export function useLatestMetrics(phf: string) {
  const [metrics, setMetrics] = useState<any>({})

  useEffect(() => {
    const fetch_ = () =>
      apiFetch(`/api/metrics-latest?phf=${phf}`).then((d) => setMetrics(d.metrics)).catch(() => {})
    fetch_()
    const id = setInterval(fetch_, 2000)
    return () => clearInterval(id)
  }, [phf])

  return metrics
}

export function useMetricsHistory(phf: string) {
  const [history, setHistory] = useState<any[]>([])

  useEffect(() => {
    const fetch_ = () =>
      apiFetch(`/api/metrics-history?phf=${phf}`).then((d) => setHistory(d.history)).catch(() => {})
    fetch_()
    const id = setInterval(fetch_, 3000)
    return () => clearInterval(id)
  }, [phf])

  return history
}

export function usePrograms(phf: string) {
  const [programs, setPrograms] = useState<any[]>([])
  const [loading, setLoading] = useState(false)

  const fetch_ = useCallback(() => {
    setLoading(true)
    apiFetch(`/api/programs?phf=${phf}`)
      .then((d) => setPrograms(d.programs))
      .catch(() => {})
      .finally(() => setLoading(false))
  }, [phf])

  useEffect(() => {
    fetch_()
  }, [fetch_])

  return { programs, loading, refetch: fetch_ }
}
