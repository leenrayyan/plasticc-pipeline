import { GoogleGenAI } from '@google/genai'
import { PAPER_ABSTRACT } from '../data/paper'

const KEY_STORAGE = 'herald.geminiKey'

export function getStoredKey(): string {
  if (typeof window === 'undefined') return ''
  return window.localStorage.getItem(KEY_STORAGE) ?? ''
}

export function setStoredKey(key: string): void {
  window.localStorage.setItem(KEY_STORAGE, key.trim())
}

export function clearStoredKey(): void {
  window.localStorage.removeItem(KEY_STORAGE)
}

export function hasKey(): boolean {
  return getStoredKey().length > 8
}

export const SYSTEM_PROMPT = `You are HERALD, an in-terminal AI assistant for the paper "HERALD: Hierarchical Early-Alert Ranking for Astrophysical Latent-event Detection" by Leen Rayyan and Retal Emad (Princess Sumaya University for Technology, 2026).

Use the paper context below to answer questions about the work. Be concise (1–4 short sentences unless the user asks for more), technically precise, and friendly. Stay in a terminal-style voice — plain text, no markdown headers, no emoji. If you don't know something from the paper, say so plainly.

PAPER CONTEXT:
${PAPER_ABSTRACT}

Key numbers to keep straight:
- Recall@50: T1 = 0% at every observation fraction. T2 = 42% at f=1.0, 11% at f=0.1. T3 = 41.2% on XGBoost.
- XGBoost macro-F1 = 0.754. Per-class: KN F1=0.81 (precision 0.94), TDE F1=0.75, SNIax F1=0.09 (only 36 training samples).
- Deep models underperform because PLAsTiCC only has 7,848 objects. ATAT on ELAsTiCC (230× larger) reaches F1=0.829.
- The fix is objective alignment, not a better classifier. Same XGBoost gives 0% under T1 and 42% under T2.`

export interface ChatMsg {
  role: 'user' | 'model'
  text: string
}

export async function askGemini(
  history: ChatMsg[],
  userMessage: string
): Promise<string> {
  const key = getStoredKey()
  if (!key) throw new Error('No Gemini API key set.')

  const ai = new GoogleGenAI({ apiKey: key })

  const contents = [
    ...history.map((m) => ({
      role: m.role,
      parts: [{ text: m.text }],
    })),
    { role: 'user' as const, parts: [{ text: userMessage }] },
  ]

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents,
    config: {
      systemInstruction: SYSTEM_PROMPT,
      temperature: 0.6,
      maxOutputTokens: 600,
    },
  })

  return response.text ?? ''
}

export async function explainAlert(opts: {
  classKey: string
  className: string
  rare: boolean
  probs: { className: string; p: number }[]
  t1Score: number
  t2Score: number
  t3Score: number
  fraction: number
}): Promise<string> {
  const key = getStoredKey()
  if (!key) throw new Error('No Gemini API key set.')

  const ai = new GoogleGenAI({ apiKey: key })

  const top3 = [...opts.probs].sort((a, b) => b.p - a.p).slice(0, 3)
  const topStr = top3
    .map((p) => `${p.className}=${p.p.toFixed(2)}`)
    .join(', ')

  const prompt = `An LSST-style alert just arrived. True class: ${opts.className} (${opts.classKey}${opts.rare ? ', RARE' : ''}). Truncation fraction f=${opts.fraction.toFixed(2)}.
Classifier outputs (top-3): ${topStr}.
Prioritization scores: T1=${opts.t1Score.toFixed(3)} (max-prob), T2=${opts.t2Score.toFixed(3)} (P(KN)+P(TDE)), T3=${opts.t3Score.toFixed(3)} (uncertainty-weighted).

In 2–3 short sentences, explain whether this alert is worth a spectroscopic follow-up tonight and why T2/T3 rank it differently from T1. Speak like a terminal log line — terse, no fluff.`

  const response = await ai.models.generateContent({
    model: 'gemini-2.5-flash',
    contents: [{ role: 'user', parts: [{ text: prompt }] }],
    config: {
      systemInstruction: SYSTEM_PROMPT,
      temperature: 0.5,
      maxOutputTokens: 220,
    },
  })

  return response.text ?? ''
}
