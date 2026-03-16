import { useEffect, useRef } from 'react'

interface Props {
  lines: string[]
  height?: string
}

function colorize(line: string): string {
  if (line.includes('[web]')) return 'text-brand'
  if (line.includes('ERROR') || line.includes('error') || line.includes('Error')) return 'text-red-400'
  if (line.includes('WARNING') || line.includes('warning')) return 'text-yellow-400'
  if (line.includes('efficiency') || line.includes('asr') || line.includes('ASR')) return 'text-cyan-400'
  if (line.includes('generation') || line.includes('Generation')) return 'text-purple-400'
  if (line.startsWith('[')) return 'text-gray-300'
  return 'text-gray-500'
}

export default function LogViewer({ lines, height = 'h-96' }: Props) {
  const bottomRef = useRef<HTMLDivElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const atBottomRef = useRef(true)

  const checkScroll = () => {
    const el = containerRef.current
    if (!el) return
    atBottomRef.current = el.scrollHeight - el.scrollTop - el.clientHeight < 40
  }

  useEffect(() => {
    if (atBottomRef.current) {
      bottomRef.current?.scrollIntoView({ behavior: 'instant' })
    }
  }, [lines])

  if (lines.length === 0) {
    return (
      <div className={`${height} bg-dark-1 rounded border border-gray-800 flex items-center justify-center`}>
        <span className="text-gray-600 text-sm">Логи появятся здесь после запуска...</span>
      </div>
    )
  }

  return (
    <div
      ref={containerRef}
      onScroll={checkScroll}
      className={`${height} overflow-auto bg-dark-1 rounded border border-gray-800 p-3 text-xs font-mono`}
    >
      {lines.map((line, i) => (
        <div key={i} className={`leading-5 whitespace-pre-wrap break-all ${colorize(line)}`}>
          {line}
        </div>
      ))}
      <div ref={bottomRef} />
    </div>
  )
}
