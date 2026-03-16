import { useState, useEffect } from 'react'
import { Play, RefreshCw } from 'lucide-react'
import { apiFetch } from '../hooks/useApi'

interface BaselineRow {
  attack: string
  phf: string
  asr: string | number
  l2: string | number
  queries: string | number
  efficiency: string | number
}

function n(v: string | number) { return typeof v === 'string' ? parseFloat(v) : v }

export default function Baselines() {
  const [rows, setRows] = useState<BaselineRow[]>([])
  const [loading, setLoading] = useState(false)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState('')

  function fetchBaselines() {
    setLoading(true)
    apiFetch('/api/baselines')
      .then((d) => setRows(d.rows || []))
      .catch(() => setRows([]))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchBaselines() }, [])

  async function handleRun() {
    setRunning(true)
    setError('')
    try {
      await apiFetch('/api/run-baselines', { method: 'POST' })
      setTimeout(fetchBaselines, 3000)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setRunning(false)
    }
  }

  // Group by PHF
  const byPhf: Record<string, BaselineRow[]> = {}
  for (const row of rows) {
    if (!byPhf[row.phf]) byPhf[row.phf] = []
    byPhf[row.phf].push(row)
  }

  return (
    <div className="p-8 max-w-5xl">
      <div className="flex items-center gap-3 mb-8">
        <h1 className="text-2xl font-bold text-white">Бейзлайны</h1>
        <button
          onClick={fetchBaselines}
          disabled={loading}
          className="ml-auto flex items-center gap-2 px-3 py-1.5 bg-dark-1 border border-gray-700 rounded text-sm text-gray-400 hover:text-white transition-all"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Обновить
        </button>
        <button
          onClick={handleRun}
          disabled={running}
          className="flex items-center gap-2 px-4 py-1.5 bg-brand text-black font-bold rounded text-sm hover:bg-brand/90 disabled:opacity-50 transition-all"
        >
          <Play size={14} />
          {running ? 'Запуск...' : 'Запустить все'}
        </button>
      </div>

      {error && (
        <div className="mb-4 text-sm text-red-400 bg-red-900/20 border border-red-800 rounded p-3">{error}</div>
      )}

      {rows.length === 0 ? (
        <div className="bg-dark-1 rounded border border-gray-800 p-12 flex items-center justify-center">
          <span className="text-gray-600 text-sm">
            {loading ? 'Загрузка...' : 'Нажми «Запустить все» чтобы оценить бейзлайны'}
          </span>
        </div>
      ) : (
        <div className="flex flex-col gap-8">
          {Object.entries(byPhf).map(([phf, phfRows]) => (
            <div key={phf}>
              <div className="text-xs text-gray-500 uppercase tracking-wider mb-3 font-mono">{phf}</div>
              <div className="bg-dark-1 rounded border border-gray-800 overflow-hidden">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-wider">
                      <th className="text-left px-4 py-3">Атака</th>
                      <th className="text-right px-4 py-3">Efficiency</th>
                      <th className="text-right px-4 py-3">ASR</th>
                      <th className="text-right px-4 py-3">L2</th>
                      <th className="text-right px-4 py-3">Queries</th>
                    </tr>
                  </thead>
                  <tbody>
                    {phfRows.sort((a, b) => n(b.efficiency) - n(a.efficiency)).map((row, i) => (
                      <tr key={i} className="border-b border-gray-800/50 hover:bg-dark-2 transition-all">
                        <td className="px-4 py-3 text-white font-mono">{row.attack}</td>
                        <td className="px-4 py-3 text-right font-mono text-brand font-bold">
                          {n(row.efficiency).toFixed(4)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-green-400">
                          {(n(row.asr) * 100).toFixed(0)}%
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-yellow-400">
                          {n(row.l2).toFixed(2)}
                        </td>
                        <td className="px-4 py-3 text-right font-mono text-gray-400">
                          {Math.round(n(row.queries))}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
