import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts'
import { useLogSocket } from '../hooks/useWebSocket'
import { useStatus, useMetrics, useLatestMetrics, useMetricsHistory } from '../hooks/useApi'
import LogViewer from '../components/LogViewer'
import MetricCard from '../components/MetricCard'
import StatusBadge from '../components/StatusBadge'

const BASE = import.meta.env.DEV ? 'http://localhost:8765' : ''

export default function Monitor() {
  const { status } = useStatus()
  const phf = status?.phf || 'phash'
  const bestMetrics = useMetrics(phf)
  const latestMetrics = useLatestMetrics(phf)
  const { lines } = useLogSocket()
  const chartData = useMetricsHistory(phf)

  const [tab, setTab] = useState<'logs' | 'chart' | 'latest' | 'best'>('logs')
  const [latestTs, setLatestTs] = useState(Date.now())
  const [bestTs, setBestTs] = useState(Date.now())
  const [latestError, setLatestError] = useState(false)
  const [bestError, setBestError] = useState(false)

  useEffect(() => {
    if (tab !== 'latest') return
    const id = setInterval(() => { setLatestTs(Date.now()); setLatestError(false) }, 2000)
    return () => clearInterval(id)
  }, [tab])

  useEffect(() => {
    if (tab !== 'best') return
    const id = setInterval(() => { setBestTs(Date.now()); setBestError(false) }, 2000)
    return () => clearInterval(id)
  }, [tab])

  return (
    <div className="p-8 max-w-5xl">
      <div className="flex items-center gap-3 mb-8">
        <h1 className="text-2xl font-bold text-white">Live Monitor</h1>
        {status && <StatusBadge status={status.status} />}
      </div>

      {/* Best program metrics */}
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Лучшая программа</div>
      <div className="grid grid-cols-4 gap-4 mb-4">
        <MetricCard label="Efficiency" value={bestMetrics?.efficiency ?? '—'} />
        <MetricCard label="ASR" value={bestMetrics?.asr ?? '—'} color="text-green-400" />
        <MetricCard label="L2 Distance" value={bestMetrics?.l2 ?? '—'} color="text-yellow-400" />
        <MetricCard label="Queries" value={bestMetrics?.n_queries ?? '—'} color="text-purple-400" />
      </div>

      {/* Latest program metrics */}
      <div className="text-xs text-gray-500 uppercase tracking-wider mb-2">Последняя программа</div>
      <div className="grid grid-cols-4 gap-4 mb-6">
        <MetricCard label="Efficiency" value={latestMetrics?.efficiency ?? '—'} />
        <MetricCard label="ASR" value={latestMetrics?.asr ?? '—'} color="text-green-400" />
        <MetricCard label="L2 Distance" value={latestMetrics?.l2 ?? '—'} color="text-yellow-400" />
        <MetricCard label="Queries" value={latestMetrics?.n_queries ?? '—'} color="text-purple-400" />
      </div>

      {/* Progress */}
      {status?.total_generations > 0 && (
        <div className="mb-6">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Поколение {status.generations_done} / {status.total_generations}</span>
            <span>{Math.round((status.generations_done / status.total_generations) * 100)}%</span>
          </div>
          <div className="h-2 bg-dark-2 rounded-full overflow-hidden">
            <div
              className="h-full bg-brand transition-all duration-500 rounded-full"
              style={{ width: `${Math.round((status.generations_done / status.total_generations) * 100)}%` }}
            />
          </div>
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 mb-4">
        {([['logs', 'Логи'], ['chart', 'График'], ['latest', 'Последняя'], ['best', 'Лучшая']] as const).map(([t, label]) => (
          <button
            key={t}
            onClick={() => setTab(t as typeof tab)}
            className={`px-4 py-1.5 rounded text-sm transition-all ${
              tab === t ? 'bg-brand text-black font-bold' : 'text-gray-400 hover:text-white bg-dark-1'
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      {tab === 'logs' ? (
        <LogViewer lines={lines} height="h-[500px]" />
      ) : tab === 'chart' ? (
        <div className="bg-dark-1 rounded border border-gray-800 p-4">
          {chartData.length < 2 ? (
            <div className="h-72 flex items-center justify-center text-gray-600 text-sm">
              Данные появятся после запуска эволюции...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
                <XAxis dataKey="gen" stroke="#6b7280" tick={{ fontSize: 11 }} label={{ value: 'Поколение', position: 'insideBottom', offset: -5, fill: '#6b7280', fontSize: 11 }} />
                <YAxis stroke="#6b7280" tick={{ fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#111', border: '1px solid #374151', borderRadius: 6, fontSize: 12 }}
                  labelStyle={{ color: '#9ca3af' }}
                />
                <Line type="monotone" dataKey="efficiency" stroke="#39ff14" strokeWidth={2} dot={false} name="Efficiency" />
                <Line type="monotone" dataKey="asr" stroke="#22d3ee" strokeWidth={1.5} dot={false} name="ASR" />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      ) : tab === 'latest' ? (
        <div className="bg-dark-1 rounded border border-gray-800 p-4">
          {latestError ? (
            <div className="h-72 flex items-center justify-center text-gray-600 text-sm">
              Изображения появятся после запуска эволюции...
            </div>
          ) : (
            <img
              src={`${BASE}/api/grid-image?phf=${phf}&variant=latest&t=${latestTs}`}
              alt="Latest attack examples"
              className="max-w-full rounded"
              onError={() => setLatestError(true)}
              onLoad={() => setLatestError(false)}
            />
          )}
        </div>
      ) : (
        <div className="bg-dark-1 rounded border border-gray-800 p-4">
          {bestError ? (
            <div className="h-72 flex items-center justify-center text-gray-600 text-sm">
              Изображения появятся после запуска эволюции...
            </div>
          ) : (
            <img
              src={`${BASE}/api/grid-image?phf=${phf}&variant=best&t=${bestTs}`}
              alt="Best attack examples"
              className="max-w-full rounded"
              onError={() => setBestError(true)}
              onLoad={() => setBestError(false)}
            />
          )}
        </div>
      )}
    </div>
  )
}