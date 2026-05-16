import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CLASSES, type TransientClass } from '../data/paper'
import { generateAlert, type Alert } from '../lib/simulator'
import { explainAlert, hasKey } from '../lib/gemini'
import { SectionHeader } from './Models'

const MAX_BUFFER = 80
const TICK_MS = 1200

type Strategy = 't1' | 't2' | 't3'

interface RankColProps {
  title: string
  subtitle: string
  strategy: Strategy
  alerts: Alert[]
  budget: number
  accent: 'grey' | 'teal' | 'purple'
  selectedId: string | null
  onSelect: (id: string) => void
}

const ACC: Record<
  RankColProps['accent'],
  { text: string; border: string; bar: string; chipBg: string }
> = {
  grey: {
    text: 'text-mint/80',
    border: 'border-mint/20',
    bar: 'bg-mint/40',
    chipBg: 'bg-mint/5',
  },
  teal: {
    text: 'text-teal text-glow',
    border: 'border-teal/40',
    bar: 'bg-teal',
    chipBg: 'bg-teal/10',
  },
  purple: {
    text: 'text-purple text-glow-purple',
    border: 'border-purple/40',
    bar: 'bg-purple',
    chipBg: 'bg-purple/10',
  },
}

function score(a: Alert, s: Strategy): number {
  if (s === 't1') return a.t1
  if (s === 't2') return a.t2
  return a.t3
}

function RankColumn({
  title,
  subtitle,
  strategy,
  alerts,
  budget,
  accent,
  selectedId,
  onSelect,
}: RankColProps) {
  const ranked = useMemo(
    () =>
      [...alerts]
        .sort((a, b) => score(b, strategy) - score(a, strategy))
        .slice(0, budget),
    [alerts, strategy, budget]
  )

  const rareHit = ranked.filter((a) => CLASSES[a.trueClass].rare).length
  const rareInBuffer = alerts.filter((a) => CLASSES[a.trueClass].rare).length
  const recall = rareInBuffer === 0 ? 0 : rareHit / rareInBuffer

  const acc = ACC[accent]

  return (
    <div className={`panel p-3 sm:p-4 flex flex-col`}>
      <div className="flex justify-between items-baseline mb-2">
        <div>
          <h4 className={`text-sm sm:text-base font-bold ${acc.text}`}>
            {title}
          </h4>
          <p className="text-mint/40 text-[10px] tracking-wider">{subtitle}</p>
        </div>
        <div className="text-right">
          <p className="text-[10px] text-mint/40 tracking-wider">recall@{budget}</p>
          <p className={`text-base font-bold ${acc.text}`}>
            {(recall * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      <div className={`flex-1 overflow-y-auto max-h-[420px] border-t ${acc.border} pt-2 space-y-1`}>
        {ranked.length === 0 && (
          <p className="text-mint/40 text-xs italic">awaiting alerts…</p>
        )}
        {ranked.map((a, i) => {
          const cls = CLASSES[a.trueClass]
          const isSelected = selectedId === a.id
          return (
            <button
              key={a.id}
              onClick={() => onSelect(a.id)}
              className={`w-full text-left px-2 py-1.5 rounded-sm row-in border ${
                isSelected
                  ? `${acc.chipBg} ${acc.border}`
                  : 'border-transparent hover:border-mint/15'
              } flex items-center gap-2 text-[11px] sm:text-xs`}
              title={`${a.id} · true: ${cls.fullName} · pred: ${CLASSES[a.predictedClass].fullName}`}
            >
              <span className="text-mint/40 w-5 text-right">
                {String(i + 1).padStart(2, '0')}
              </span>
              <span
                className="w-2 h-2 rounded-full shrink-0"
                style={{ background: cls.color, boxShadow: cls.rare ? `0 0 8px ${cls.color}` : 'none' }}
              />
              <span className="font-mono text-mint/80 truncate flex-1">
                {a.id}
              </span>
              <span className={`shrink-0 ${cls.rare ? (cls.key === 'KN' ? 'text-yellow text-glow-yellow' : 'text-orange') : 'text-mint/60'}`}>
                {a.trueClass}
              </span>
              <span className="text-mint/40 shrink-0 w-10 text-right">
                {score(a, strategy).toFixed(2)}
              </span>
            </button>
          )
        })}
      </div>
    </div>
  )
}

function ProbBars({ alert }: { alert: Alert }) {
  const top = (Object.entries(alert.probs) as [TransientClass, number][])
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)

  return (
    <div className="space-y-1.5">
      {top.map(([k, p]) => {
        const cls = CLASSES[k]
        const isTrue = k === alert.trueClass
        return (
          <div key={k} className="text-[11px] sm:text-xs">
            <div className="flex justify-between mb-0.5">
              <span
                className={
                  isTrue
                    ? cls.rare
                      ? cls.key === 'KN'
                        ? 'text-yellow text-glow-yellow font-bold'
                        : 'text-orange font-bold'
                      : 'text-teal text-glow font-bold'
                    : 'text-mint/70'
                }
              >
                {isTrue && '› '}
                {cls.fullName}
              </span>
              <span className="text-mint/60 font-mono">
                {(p * 100).toFixed(1)}%
              </span>
            </div>
            <div className="h-1.5 bg-ink-2 border border-mint/10 rounded-sm overflow-hidden">
              <div
                className="h-full transition-all duration-700"
                style={{ width: `${p * 100}%`, background: cls.color }}
              />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function ExplanationPanel({
  alert,
  fraction,
  onRequireKey,
}: {
  alert: Alert | null
  fraction: number
  onRequireKey: () => void
}) {
  const [text, setText] = useState<string>('')
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  useEffect(() => {
    setText('')
    setErr(null)
  }, [alert?.id])

  const run = useCallback(async () => {
    if (!alert) return
    if (!hasKey()) {
      onRequireKey()
      return
    }
    setLoading(true)
    setErr(null)
    try {
      const cls = CLASSES[alert.trueClass]
      const probs = (Object.entries(alert.probs) as [TransientClass, number][]).map(
        ([k, p]) => ({ className: CLASSES[k].fullName, p })
      )
      const out = await explainAlert({
        classKey: alert.trueClass,
        className: cls.fullName,
        rare: cls.rare,
        probs,
        t1Score: alert.t1,
        t2Score: alert.t2,
        t3Score: alert.t3,
        fraction,
      })
      setText(out)
    } catch (e: any) {
      setErr(e?.message ?? 'Gemini request failed.')
    } finally {
      setLoading(false)
    }
  }, [alert, fraction, onRequireKey])

  if (!alert) {
    return (
      <div className="panel p-4 h-full text-mint/40 text-xs italic">
        // click any alert to inspect
      </div>
    )
  }

  const cls = CLASSES[alert.trueClass]
  const predCls = CLASSES[alert.predictedClass]
  const correct = alert.predictedClass === alert.trueClass

  return (
    <div
      className={`panel p-4 h-full flex flex-col gap-3 ${
        cls.rare ? (cls.key === 'KN' ? 'glow-yellow' : 'glow-orange') : 'glow-teal'
      }`}
    >
      <div>
        <p className="text-mint/40 text-[10px] tracking-widest">
          // inspecting :: {alert.id}
        </p>
        <h4
          className={`text-lg sm:text-xl font-bold ${
            cls.rare
              ? cls.key === 'KN'
                ? 'text-yellow text-glow-yellow'
                : 'text-orange'
              : 'text-teal text-glow'
          }`}
        >
          {cls.fullName}
          {cls.rare && (
            <span className="ml-2 text-[10px] tracking-widest border border-current px-1.5 py-0.5 rounded-sm align-middle">
              RARE
            </span>
          )}
        </h4>
        <p className="text-mint/70 text-xs mt-1">{cls.description}</p>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <div>
          <p className="text-mint/40">RA</p>
          <p className="text-mint">{alert.ra}°</p>
        </div>
        <div>
          <p className="text-mint/40">Dec</p>
          <p className="text-mint">{alert.dec}°</p>
        </div>
        <div>
          <p className="text-mint/40">mag</p>
          <p className="text-mint">{alert.mag}</p>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-2 text-[11px]">
        <ScoreChip label="T1" value={alert.t1} tone="grey" />
        <ScoreChip label="T2" value={alert.t2} tone="teal" />
        <ScoreChip label="T3" value={alert.t3} tone="purple" />
      </div>

      <div className="border-t border-mint/15 pt-3">
        <p className="text-mint/40 text-[10px] tracking-widest mb-2">
          // classifier outputs
        </p>
        <ProbBars alert={alert} />
        <p
          className={`text-[11px] mt-2 ${
            correct ? 'text-teal' : 'text-orange'
          }`}
        >
          predicted: {predCls.fullName}{' '}
          {correct ? '✓ matches truth' : `× truth = ${cls.fullName}`}
        </p>
      </div>

      <div className="border-t border-mint/15 pt-3 mt-auto">
        <div className="flex items-center justify-between mb-2">
          <p className="text-mint/40 text-[10px] tracking-widest">
            // ai_explanation
          </p>
          <button
            onClick={run}
            disabled={loading}
            className="text-[10px] tracking-widest border border-teal/50 text-teal px-2 py-1 rounded-sm hover:bg-teal/10 disabled:opacity-50"
          >
            {loading ? 'thinking…' : text ? 'regenerate' : 'explain'}
          </button>
        </div>
        {err && <p className="text-orange text-[11px]">err: {err}</p>}
        {!text && !err && !loading && (
          <p className="text-mint/40 text-[11px] italic">
            press 'explain' for a 2-sentence AI rationale (Gemini).
          </p>
        )}
        {text && (
          <p className="text-mint/90 text-[12px] leading-relaxed whitespace-pre-wrap">
            {text}
          </p>
        )}
      </div>
    </div>
  )
}

function ScoreChip({
  label,
  value,
  tone,
}: {
  label: string
  value: number
  tone: 'grey' | 'teal' | 'purple'
}) {
  const map = {
    grey: 'text-mint/70 border-mint/20',
    teal: 'text-teal border-teal/50',
    purple: 'text-purple border-purple/50',
  }[tone]
  return (
    <div className={`border ${map} rounded-sm px-2 py-1`}>
      <p className="text-[9px] tracking-widest opacity-70">{label}</p>
      <p className="font-mono text-sm">{value.toFixed(3)}</p>
    </div>
  )
}

export default function Demo({
  onRequireKey,
}: {
  onRequireKey: () => void
}) {
  const [alerts, setAlerts] = useState<Alert[]>([])
  const [running, setRunning] = useState(true)
  const [fraction, setFraction] = useState(1.0)
  const [budget, setBudget] = useState(10)
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const fractionRef = useRef(fraction)
  fractionRef.current = fraction

  // Seed with a handful of alerts so the UI isn't empty.
  useEffect(() => {
    const initial = Array.from({ length: 25 }, () => generateAlert(fraction, 4))
    setAlerts(initial)
    setSelectedId(initial[0]?.id ?? null)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!running) return
    const id = window.setInterval(() => {
      const next = generateAlert(fractionRef.current, 4)
      setAlerts((prev) => {
        const merged = [next, ...prev]
        return merged.slice(0, MAX_BUFFER)
      })
    }, TICK_MS)
    return () => window.clearInterval(id)
  }, [running])

  const selectedAlert = useMemo(
    () => alerts.find((a) => a.id === selectedId) ?? alerts[0] ?? null,
    [alerts, selectedId]
  )

  const rareInBuffer = alerts.filter((a) => CLASSES[a.trueClass].rare).length

  return (
    <section id="demo" className="py-20 px-4 sm:px-6 max-w-7xl mx-auto">
      <SectionHeader
        label="// section_04"
        title="LIVE DEMO"
        subtitle="A simulated LSST alert stream. Adjust the observation fraction and follow-up budget to watch T1/T2/T3 recover rare events in real time."
      />

      <div className="mt-8 panel p-4 sm:p-5 flex flex-wrap items-center gap-4 sm:gap-6">
        <button
          onClick={() => setRunning((r) => !r)}
          className={`px-3 py-1.5 rounded-sm text-xs tracking-widest border ${
            running
              ? 'border-orange/60 text-orange hover:bg-orange/10'
              : 'border-teal/60 text-teal hover:bg-teal/10'
          }`}
        >
          {running ? '◼ pause stream' : '▶ resume'}
        </button>

        <div className="flex-1 min-w-[200px]">
          <div className="flex justify-between text-[11px] tracking-wider mb-1">
            <span className="text-mint/60">
              observation fraction (f)
            </span>
            <span className="text-teal text-glow font-bold">
              {fraction.toFixed(2)}
            </span>
          </div>
          <input
            type="range"
            min={0.1}
            max={1.0}
            step={0.05}
            value={fraction}
            onChange={(e) => setFraction(parseFloat(e.target.value))}
            className="w-full accent-teal"
          />
          <div className="flex justify-between text-[9px] text-mint/40 mt-0.5 tracking-wider">
            <span>f=0.10 · ~3–5 nights</span>
            <span>f=1.00 · full light curve</span>
          </div>
        </div>

        <div className="flex-1 min-w-[200px]">
          <div className="flex justify-between text-[11px] tracking-wider mb-1">
            <span className="text-mint/60">follow-up budget (K)</span>
            <span className="text-teal text-glow font-bold">{budget}</span>
          </div>
          <input
            type="range"
            min={1}
            max={30}
            step={1}
            value={budget}
            onChange={(e) => setBudget(parseInt(e.target.value))}
            className="w-full accent-teal"
          />
          <div className="flex justify-between text-[9px] text-mint/40 mt-0.5 tracking-wider">
            <span>K=1</span>
            <span>nightly = 50</span>
          </div>
        </div>

        <div className="text-right">
          <p className="text-[10px] text-mint/40 tracking-widest">
            buffer · rare
          </p>
          <p className="font-mono text-mint">
            {alerts.length}
            <span className="text-mint/40"> / </span>
            <span className="text-yellow text-glow-yellow">
              {rareInBuffer}
            </span>
          </p>
        </div>
      </div>

      <div className="grid lg:grid-cols-[1fr_1fr_1fr_1.1fr] gap-3 sm:gap-4 mt-4">
        <RankColumn
          title="T1 · max-prob"
          subtitle="current broker standard"
          strategy="t1"
          alerts={alerts}
          budget={budget}
          accent="grey"
          selectedId={selectedId}
          onSelect={setSelectedId}
        />
        <RankColumn
          title="T2 · P(KN) + P(TDE)"
          subtitle="goal-aligned"
          strategy="t2"
          alerts={alerts}
          budget={budget}
          accent="teal"
          selectedId={selectedId}
          onSelect={setSelectedId}
        />
        <RankColumn
          title="T3 · uncertainty-weighted"
          subtitle="T2 / (1 + H(y|x))"
          strategy="t3"
          alerts={alerts}
          budget={budget}
          accent="purple"
          selectedId={selectedId}
          onSelect={setSelectedId}
        />
        <ExplanationPanel
          alert={selectedAlert}
          fraction={fraction}
          onRequireKey={onRequireKey}
        />
      </div>

      <p className="text-mint/40 text-[11px] mt-4 leading-relaxed">
        // simulator note :: classifier probabilities are sampled from a noise
        model that follows the paper's empirical behavior (rare-class
        suppression under common-class dominance; recall degradation at low f).
        T1/T2/T3 are computed from those probabilities exactly as in §V.
      </p>
    </section>
  )
}
