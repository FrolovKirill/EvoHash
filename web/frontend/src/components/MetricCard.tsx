interface Props {
  label: string
  value: string | number | null | undefined
  unit?: string
  color?: string
}

function formatMetric(v: number): string {
  if (!Number.isFinite(v)) return '—'
  if (v === 0) return '0'
  const abs = Math.abs(v)
  if (abs >= 1) return v.toFixed(3)
  if (abs >= 0.001) return v.toFixed(4)
  return v.toExponential(2)
}

export default function MetricCard({ label, value, unit, color = 'text-brand' }: Props) {
  return (
    <div className="bg-dark-2 rounded border border-gray-800 p-4">
      <div className="text-gray-500 text-xs uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {value == null ? '—' : typeof value === 'number' ? formatMetric(value) : value}
        {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
      </div>
    </div>
  )
}
