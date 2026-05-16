import { useEffect, useState } from 'react'
import { clearStoredKey, getStoredKey, setStoredKey } from '../lib/gemini'

export default function KeyModal({
  open,
  onClose,
}: {
  open: boolean
  onClose: () => void
}) {
  const [value, setValue] = useState('')

  useEffect(() => {
    if (open) setValue(getStoredKey())
  }, [open])

  if (!open) return null

  return (
    <div
      className="fixed inset-0 z-[200] bg-black/70 backdrop-blur-sm flex items-center justify-center px-4"
      onClick={onClose}
    >
      <div
        className="panel glow-teal max-w-lg w-full p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <p className="text-mint/40 text-xs tracking-widest">
          // ai_credentials
        </p>
        <h3 className="text-teal text-glow text-2xl font-bold mt-1">
          paste your Gemini API key
        </h3>
        <p className="text-mint/70 text-sm mt-2 leading-relaxed">
          HERALD's chat &amp; per-alert explanations call Google's Gemini API
          directly from your browser. The key is stored only in your{' '}
          <code className="text-teal">localStorage</code> — never uploaded
          anywhere. Get a free key at{' '}
          <a
            href="https://aistudio.google.com/apikey"
            target="_blank"
            rel="noreferrer noopener"
            className="text-teal underline decoration-dotted hover:text-glow"
          >
            aistudio.google.com/apikey
          </a>
          .
        </p>

        <input
          type="password"
          value={value}
          onChange={(e) => setValue(e.target.value)}
          placeholder="AIza..."
          className="w-full mt-4 px-3 py-2 bg-ink-2 border border-teal/40 text-mint font-mono text-sm rounded-sm focus:outline-none focus:border-teal focus:glow-teal"
        />

        <div className="flex gap-2 mt-4">
          <button
            onClick={() => {
              setStoredKey(value)
              onClose()
            }}
            disabled={value.length < 10}
            className="flex-1 px-3 py-2 border border-teal/60 text-teal rounded-sm text-sm tracking-widest hover:bg-teal/10 disabled:opacity-40"
          >
            save &amp; activate
          </button>
          <button
            onClick={() => {
              clearStoredKey()
              setValue('')
            }}
            className="px-3 py-2 border border-orange/50 text-orange rounded-sm text-sm tracking-widest hover:bg-orange/10"
          >
            clear
          </button>
          <button
            onClick={onClose}
            className="px-3 py-2 border border-mint/30 text-mint/70 rounded-sm text-sm tracking-widest hover:bg-mint/5"
          >
            cancel
          </button>
        </div>

        <p className="text-mint/40 text-[10px] mt-3 tracking-wider">
          ⚠ never paste a key you intend to keep secret — anyone with browser
          devtools on this device could read it.
        </p>
      </div>
    </div>
  )
}
