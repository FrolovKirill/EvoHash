import { useState } from 'react'
import { Download, RefreshCw } from 'lucide-react'
import { useStatus, usePrograms } from '../hooks/useApi'
import CodeViewer from '../components/CodeViewer'

export default function Results() {
  const { status } = useStatus()
  const phf = status?.phf || 'phash'
  const { programs, loading, refetch } = usePrograms(phf)
  const [selected, setSelected] = useState<number | null>(null)

  function downloadProgram(prog: any, idx: number) {
    const blob = new Blob([prog.code], { type: 'text/plain' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `evohash_${phf}_rank${idx + 1}.py`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="p-8 max-w-5xl">
      <div className="flex items-center gap-3 mb-8">
        <h1 className="text-2xl font-bold text-white">Результаты</h1>
        <span className="text-xs text-gray-500 font-mono uppercase border border-gray-700 rounded px-2 py-0.5">{phf}</span>
        <button
          onClick={refetch}
          disabled={loading}
          className="ml-auto flex items-center gap-2 px-3 py-1.5 bg-dark-1 border border-gray-700 rounded text-sm text-gray-400 hover:text-white transition-all"
        >
          <RefreshCw size={14} className={loading ? 'animate-spin' : ''} />
          Обновить
        </button>
      </div>

      {programs.length === 0 ? (
        <div className="bg-dark-1 rounded border border-gray-800 p-12 flex items-center justify-center">
          <span className="text-gray-600 text-sm">
            {loading ? 'Загрузка...' : 'Результаты появятся после завершения запуска'}
          </span>
        </div>
      ) : (
        <div className="flex gap-6">
          {/* Programs table */}
          <div className="flex-1 min-w-0">
            <div className="bg-dark-1 rounded border border-gray-800 overflow-hidden">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-gray-800 text-gray-500 text-xs uppercase tracking-wider">
                    <th className="text-left px-4 py-3">#</th>
                    <th className="text-right px-4 py-3">Efficiency</th>
                    <th className="text-right px-4 py-3">ASR</th>
                    <th className="text-right px-4 py-3">L2</th>
                    <th className="text-right px-4 py-3">Queries</th>
                    <th className="px-4 py-3"></th>
                  </tr>
                </thead>
                <tbody>
                  {programs.map((prog, idx) => (
                    <tr
                      key={idx}
                      onClick={() => setSelected(selected === idx ? null : idx)}
                      className={`border-b border-gray-800/50 cursor-pointer transition-all ${
                        selected === idx ? 'bg-brand/5' : 'hover:bg-dark-2'
                      }`}
                    >
                      <td className="px-4 py-3 text-gray-500 font-mono">{idx + 1}</td>
                      <td className="px-4 py-3 text-right font-mono text-brand font-bold">
                        {typeof prog.efficiency === 'number' ? prog.efficiency.toFixed(3) : '—'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-green-400">
                        {typeof prog.asr === 'number' ? `${(prog.asr * 100).toFixed(1)}%` : '—'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-yellow-400">
                        {typeof prog.l2 === 'number' ? prog.l2.toFixed(3) : '—'}
                      </td>
                      <td className="px-4 py-3 text-right font-mono text-gray-400">
                        {prog.n_queries ?? '—'}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <button
                          onClick={(e) => { e.stopPropagation(); downloadProgram(prog, idx) }}
                          className="text-gray-600 hover:text-brand transition-all"
                          title="Скачать .py"
                        >
                          <Download size={14} />
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Code panel */}
          {selected !== null && programs[selected]?.code && (
            <div className="w-96 shrink-0">
              <div className="bg-dark-1 rounded border border-gray-800 overflow-hidden">
                <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
                  <span className="text-xs text-gray-500">Программа #{selected + 1}</span>
                  <button
                    onClick={() => downloadProgram(programs[selected], selected)}
                    className="flex items-center gap-1 text-xs text-gray-500 hover:text-brand transition-all"
                  >
                    <Download size={12} /> .py
                  </button>
                </div>
                <CodeViewer code={programs[selected].code} />
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
