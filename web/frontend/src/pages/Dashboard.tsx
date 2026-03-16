import { useState } from 'react'
import { Play, Square, CheckCircle, XCircle, AlertCircle } from 'lucide-react'
import { apiFetch, useStatus, useEnvCheck } from '../hooks/useApi'
import StatusBadge from '../components/StatusBadge'

const PHF_OPTIONS = [
  { value: 'phash', label: 'pHash', desc: '64-bit, Hamming ≤ 12' },
  { value: 'pdq', label: 'PDQ', desc: '256-bit, Hamming ≤ 92' },
  { value: 'neuralhash', label: 'NeuralHash', desc: '96-bit, Hamming ≤ 17 (macOS)' },
]

const LLM_OPTIONS = [
  { value: 'openrouter_gpt_oss', label: 'GPT-OSS 120B (OpenRouter)' },
]

export default function Dashboard() {
  const { status } = useStatus()
  const checks = useEnvCheck()

  const [phf, setPhf] = useState('phash')
  const [generations, setGenerations] = useState(50)
  const [llm, setLlm] = useState('openrouter_gpt_oss')
  const [nPairs, setNPairs] = useState(10)
  const [starting, setStarting] = useState(false)

  const isRunning = status?.status === 'running' || status?.status === 'downloading' || status?.status === 'evaluating'

  async function handleStart() {
    setStarting(true)
    try {
      await apiFetch('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ phf, generations, llm_config: llm, n_pairs: nPairs }),
      })
    } finally {
      setStarting(false)
    }
  }

  async function handleStop() {
    await apiFetch('/api/stop', { method: 'POST' })
  }

  const progress = status?.total_generations
    ? Math.round((status.generations_done / status.total_generations) * 100)
    : 0

  return (
    <div className="p-8 max-w-4xl">
      <div className="flex items-center gap-3 mb-8">
        <h1 className="text-2xl font-bold text-white">Запуск эволюции</h1>
        {status && <StatusBadge status={status.status} />}
      </div>

      {/* Env checks */}
      {checks && (
        <div className="mb-8 bg-dark-1 rounded border border-gray-800 p-4">
          <div className="text-xs text-gray-500 uppercase tracking-wider mb-3">Проверка окружения</div>
          <div className="grid grid-cols-2 gap-2 text-sm">
            <EnvRow ok={checks.gigaevo_core} label="gigaevo-core" hint="Клонируй: git clone .../gigaevo-core" />
            <EnvRow ok={checks.dataset} label={`Датасет (${checks.dataset_count} изображений)`} hint="Будет скачан автоматически" />
            <EnvRow ok={checks.env_file} label=".env файл" hint="Скопируй .env.example → .env" />
            <EnvRow ok={checks.api_key} label="OPENROUTER_API_KEY" hint="Укажи в .env" />
          </div>
        </div>
      )}

      {/* Config */}
      <div className="bg-dark-1 rounded border border-gray-800 p-6 mb-6">
        <div className="text-xs text-gray-500 uppercase tracking-wider mb-4">Конфигурация</div>

        <div className="grid grid-cols-2 gap-6">
          {/* PHF */}
          <div>
            <label className="block text-sm text-gray-400 mb-2">Хэш-функция (PHF)</label>
            <div className="flex flex-col gap-2">
              {PHF_OPTIONS.map((opt) => (
                <label key={opt.value} className={`flex items-start gap-3 p-3 rounded border cursor-pointer transition-all ${
                  phf === opt.value ? 'border-brand bg-brand/5' : 'border-gray-700 hover:border-gray-500'
                }`}>
                  <input type="radio" name="phf" value={opt.value} checked={phf === opt.value}
                    onChange={() => setPhf(opt.value)} className="mt-0.5 accent-brand" />
                  <div>
                    <div className="text-white font-medium">{opt.label}</div>
                    <div className="text-gray-500 text-xs">{opt.desc}</div>
                  </div>
                </label>
              ))}
            </div>
          </div>

          {/* Params */}
          <div className="flex flex-col gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Поколения</label>
              <input
                type="number" min={1} max={1000} value={generations}
                onChange={(e) => setGenerations(+e.target.value)}
                className="w-full bg-dark-2 border border-gray-700 rounded px-3 py-2 text-white text-sm focus:border-brand outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">Пар изображений</label>
              <input
                type="number" min={1} max={100} value={nPairs}
                onChange={(e) => setNPairs(+e.target.value)}
                className="w-full bg-dark-2 border border-gray-700 rounded px-3 py-2 text-white text-sm focus:border-brand outline-none"
              />
            </div>
            <div>
              <label className="block text-sm text-gray-400 mb-1">LLM</label>
              <select
                value={llm} onChange={(e) => setLlm(e.target.value)}
                className="w-full bg-dark-2 border border-gray-700 rounded px-3 py-2 text-white text-sm focus:border-brand outline-none"
              >
                {LLM_OPTIONS.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Progress */}
      {isRunning && (
        <div className="mb-6">
          <div className="flex justify-between text-xs text-gray-500 mb-1">
            <span>Поколение {status?.generations_done} / {status?.total_generations}</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 bg-dark-2 rounded-full overflow-hidden">
            <div
              className="h-full bg-brand transition-all duration-500 rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>
      )}

      {/* Buttons */}
      <div className="flex gap-3">
        {!isRunning ? (
          <button
            onClick={handleStart}
            disabled={starting || !checks?.gigaevo_core}
            className="flex items-center gap-2 px-6 py-3 bg-brand text-black font-bold rounded hover:bg-brand/90 disabled:opacity-40 disabled:cursor-not-allowed transition-all text-sm"
          >
            <Play size={16} />
            {starting ? 'Запуск...' : 'СТАРТ'}
          </button>
        ) : (
          <button
            onClick={handleStop}
            className="flex items-center gap-2 px-6 py-3 bg-red-600 text-white font-bold rounded hover:bg-red-500 transition-all text-sm"
          >
            <Square size={16} />
            СТОП
          </button>
        )}
        {status?.status === 'error' && (
          <div className="flex items-center gap-2 text-red-400 text-sm">
            <AlertCircle size={16} />
            {status.error || 'Ошибка выполнения'}
          </div>
        )}
      </div>
    </div>
  )
}

function EnvRow({ ok, label, hint }: { ok: boolean; label: string; hint: string }) {
  return (
    <div className="flex items-start gap-2">
      {ok
        ? <CheckCircle size={14} className="text-green-400 mt-0.5 shrink-0" />
        : <XCircle size={14} className="text-red-400 mt-0.5 shrink-0" />}
      <div>
        <div className={`text-sm ${ok ? 'text-white' : 'text-gray-400'}`}>{label}</div>
        {!ok && <div className="text-xs text-gray-600">{hint}</div>}
      </div>
    </div>
  )
}
