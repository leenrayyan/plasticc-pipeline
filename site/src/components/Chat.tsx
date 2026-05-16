import { useEffect, useRef, useState } from 'react'
import { askGemini, hasKey, type ChatMsg } from '../lib/gemini'

const STARTERS = [
  'why does XGBoost beat the deep models?',
  'what is recall@50 and why is T1 = 0?',
  'how is T3 different from T2?',
  'why ASTROMER underperformed',
]

export default function Chat({
  onRequireKey,
}: {
  onRequireKey: () => void
}) {
  const [open, setOpen] = useState(false)
  const [history, setHistory] = useState<ChatMsg[]>([
    {
      role: 'model',
      text: 'HERALD CLI ready. ask me anything about the paper — models, results, prioritization strategies, kilonovae… anything.',
    },
  ])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const scrollRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight
    }
  }, [history, busy])

  const send = async (message?: string) => {
    const text = (message ?? draft).trim()
    if (!text || busy) return
    if (!hasKey()) {
      onRequireKey()
      return
    }
    setErr(null)
    setDraft('')
    setHistory((h) => [...h, { role: 'user', text }])
    setBusy(true)
    try {
      const reply = await askGemini(history, text)
      setHistory((h) => [...h, { role: 'model', text: reply }])
    } catch (e: any) {
      setErr(e?.message ?? 'request failed')
    } finally {
      setBusy(false)
    }
  }

  return (
    <>
      <button
        onClick={() => setOpen((o) => !o)}
        className={`fixed bottom-5 right-5 z-[150] w-14 h-14 rounded-full panel glow-teal text-teal text-glow flex items-center justify-center text-xl font-bold hover:scale-105 transition-transform ${
          open ? 'rotate-45' : ''
        }`}
        aria-label="Open HERALD chat"
        title="Ask HERALD"
      >
        {open ? '×' : '>_'}
      </button>

      {open && (
        <div className="fixed bottom-24 right-5 z-[150] w-[min(420px,calc(100vw-2.5rem))] h-[min(560px,70vh)] panel glow-teal flex flex-col overflow-hidden">
          <div className="px-4 py-2.5 border-b border-teal/30 flex items-center justify-between bg-ink-2/60">
            <div>
              <p className="text-teal text-glow text-sm font-bold">
                HERALD CLI · gemini-2.5-flash
              </p>
              <p className="text-mint/40 text-[10px] tracking-wider">
                primed with full paper context
              </p>
            </div>
            <button
              onClick={() => setOpen(false)}
              className="text-mint/60 hover:text-teal text-lg leading-none"
              aria-label="Close chat"
            >
              ×
            </button>
          </div>

          <div
            ref={scrollRef}
            className="flex-1 overflow-y-auto px-4 py-3 space-y-3 text-xs sm:text-sm"
          >
            {history.map((m, i) => (
              <div
                key={i}
                className={
                  m.role === 'user' ? 'text-mint/90' : 'text-teal'
                }
              >
                <span className="text-mint/40">
                  {m.role === 'user' ? '$ user :' : '> herald:'}{' '}
                </span>
                <span className="whitespace-pre-wrap">{m.text}</span>
              </div>
            ))}
            {busy && (
              <div className="text-teal/70 italic cursor">
                <span className="text-mint/40">&gt; herald:</span> generating
              </div>
            )}
            {err && (
              <div className="text-orange text-xs">err :: {err}</div>
            )}
            {history.length === 1 && !busy && (
              <div className="pt-2 space-y-1">
                <p className="text-mint/40 text-[10px] tracking-wider">
                  // try
                </p>
                {STARTERS.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="block w-full text-left text-mint/70 hover:text-teal text-[11px] border border-mint/15 hover:border-teal/50 rounded-sm px-2 py-1"
                  >
                    › {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <form
            onSubmit={(e) => {
              e.preventDefault()
              send()
            }}
            className="border-t border-teal/30 p-2 bg-ink-2/60 flex items-center gap-2"
          >
            <span className="text-teal text-sm">$</span>
            <input
              value={draft}
              onChange={(e) => setDraft(e.target.value)}
              placeholder="ask anything about HERALD…"
              className="flex-1 bg-transparent text-mint placeholder:text-mint/30 text-xs sm:text-sm focus:outline-none"
            />
            <button
              type="submit"
              disabled={busy || !draft.trim()}
              className="text-teal text-xs tracking-widest border border-teal/50 rounded-sm px-2 py-1 hover:bg-teal/10 disabled:opacity-40"
            >
              send
            </button>
          </form>
        </div>
      )}
    </>
  )
}
