export default function StatusBadge({ status }: { status: string }) {
  const map: Record<string, { label: string; cls: string }> = {
    idle:        { label: 'Ожидание',   cls: 'bg-gray-800 text-gray-400 border-gray-600' },
    downloading: { label: 'Скачиваю',  cls: 'bg-yellow-900/40 text-yellow-400 border-yellow-600 animate-pulse' },
    running:     { label: 'Запущено',  cls: 'bg-green-900/40 text-green-400 border-green-600 animate-pulse' },
    evaluating:  { label: 'Оценка',    cls: 'bg-blue-900/40 text-blue-400 border-blue-600 animate-pulse' },
    done:        { label: 'Готово',    cls: 'bg-brand/10 text-brand border-brand/50' },
    error:       { label: 'Ошибка',    cls: 'bg-red-900/40 text-red-400 border-red-600' },
  }
  const { label, cls } = map[status] ?? map.idle
  return (
    <span className={`text-xs px-2 py-0.5 rounded border font-mono uppercase tracking-wider ${cls}`}>
      {label}
    </span>
  )
}
