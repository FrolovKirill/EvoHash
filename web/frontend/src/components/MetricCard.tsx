interface Props {
  label: string
  value: string | number
  unit?: string
  color?: string
}

export default function MetricCard({ label, value, unit, color = 'text-brand' }: Props) {
  return (
    <div className="bg-dark-2 rounded border border-gray-800 p-4">
      <div className="text-gray-500 text-xs uppercase tracking-wider mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {typeof value === 'number' ? (Number.isFinite(value) ? value.toFixed(3) : '—') : value}
        {unit && <span className="text-sm text-gray-500 ml-1">{unit}</span>}
      </div>
    </div>
  )
}
