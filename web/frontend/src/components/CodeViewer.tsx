import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

export default function CodeViewer({ code }: { code: string }) {
  return (
    <SyntaxHighlighter
      language="python"
      style={vscDarkPlus}
      customStyle={{ margin: 0, borderRadius: '6px', fontSize: '12px', maxHeight: '400px' }}
      showLineNumbers
    >
      {code}
    </SyntaxHighlighter>
  )
}
