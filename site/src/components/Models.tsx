import { useEffect, useRef, useState } from 'react'
import { MODELS, type ModelInfo } from '../data/paper'

const ACCENT: Record<
  ModelInfo['accent'],
  { fill: string; text: string; glow: string; border: string }
> = {
  teal: {
    fill: 'bg-teal',
    text: 'text-teal text-glow',
    glow: 'glow-teal',
    border: 'border-teal/60',
  },
  purple: {
    fill: 'bg-purple',
    text: 'text-purple text-glow-purple',
    glow: 'glow-purple',
    border: 'border-purple/60',
  },
  yellow: {
    fill: 'bg-yellow',
    text: 'text-yellow text-glow-yellow',
    glow: 'glow-yellow',
    border: 'border-yellow/60',
  },
  orange: {
    fill: 'bg-orange',
    text: 'text-orange',
    glow: 'glow-orange',
    border: 'border-orange/60',
  },
}

function Bar({
  label,
  value,
  cap = 1.0,
  accent,
  delay,
}: {
  label: string
  value: number
  cap?: number
  accent: ModelInfo['accent']
  delay: number
}) {
  const [w, setW] = useState(0)
  const ref = useRef<HTMLDivElement | null>(null)
  const fired = useRef(false)

  useEffect(() => {
    if (!ref.current) return
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting && !fired.current) {
            fired.current = true
            setTimeout(() => setW(Math.min(1, value / cap) * 100), delay)
          }
        })
      },
      { threshold: 0.3 }
    )
    obs.observe(ref.current)
    return () => obs.disconnect()
  }, [value, cap, delay])

  const a = ACCENT[accent]
  return (
    <div ref={ref}>
      <div className="flex justify-between text-[11px] sm:text-xs tracking-wider text-mint/60 mb-1">
        <span>{label}</span>
        <span className={a.text}>{value.toFixed(3)}</span>
      </div>
      <div className="h-2.5 bg-ink-2 border border-mint/10 overflow-hidden rounded-sm relative">
        <div
          className={`h-full ${a.fill} transition-[width] duration-[1100ms] ease-out`}
          style={{ width: `${w}%` }}
        />
      </div>
    </div>
  )
}

function ModelCard({ m, rank }: { m: ModelInfo; rank: number }) {
  const a = ACCENT[m.accent]
  return (
    <div className={`panel ${a.glow} p-5 sm:p-6 flex flex-col gap-4 h-full`}>
      <div className="flex justify-between items-start">
        <div>
          <p className="text-mint/40 text-[10px] tracking-widest">
            // model_{rank.toString().padStart(2, '0')}
          </p>
          <h3 className={`text-2xl sm:text-3xl font-bold ${a.text}`}>
            {m.name}
          </h3>
          <p className="text-mint/60 text-xs sm:text-sm mt-1">{m.subtitle}</p>
        </div>
        {rank === 1 && (
          <span className="text-[10px] tracking-widest text-teal border border-teal/50 px-2 py-0.5 rounded-sm">
            CHAMPION
          </span>
        )}
      </div>

      <p className="text-mint/80 text-xs sm:text-sm leading-relaxed">
        {m.blurb}
      </p>

      <div className="flex flex-col gap-3 mt-auto">
        <Bar label="macro-F1" value={m.f1} accent={m.accent} delay={0} />
        <Bar
          label="PR-AUC"
          value={m.prAuc}
          accent={m.accent}
          delay={120}
        />
        <Bar
          label="ECE  (lower = better calibrated)"
          value={m.ece}
          cap={0.2}
          accent={m.accent}
          delay={240}
        />
      </div>
    </div>
  )
}

export default function Models() {
  // Display in F1-descending order, and mark rank.
  const sorted = [...MODELS].sort((a, b) => b.f1 - a.f1)

  return (
    <section id="models" className="py-20 px-4 sm:px-6 max-w-7xl mx-auto">
      <SectionHeader
        label="// section_02"
        title="MODELS"
        subtitle="Four classifiers, one finding: at PLAsTiCC scale, physics-informed feature engineering beats every deep model by ~5×."
      />

      <div className="grid md:grid-cols-2 gap-4 sm:gap-6 mt-10">
        {sorted.map((m, i) => (
          <ModelCard key={m.key} m={m} rank={i + 1} />
        ))}
      </div>

      <div className="mt-8 panel p-4 text-xs sm:text-sm text-mint/70 leading-relaxed">
        <span className="text-teal">$</span> note ::{' '}
        <span className="text-mint">
          deep-model underperformance is a data-scale issue, not an architecture
          one.
        </span>{' '}
        ATAT (Cabrera-Vives 2024) reaches F1 = 0.829 on ELAsTiCC — a dataset{' '}
        <span className="text-yellow">230× larger</span> than PLAsTiCC. The
        ranking finding (T1 = 0%, T2 = 42%) is independent of which classifier
        sits in front.
      </div>
    </section>
  )
}

export function SectionHeader({
  label,
  title,
  subtitle,
}: {
  label: string
  title: string
  subtitle: string
}) {
  return (
    <div>
      <p className="text-mint/40 text-xs tracking-widest">{label}</p>
      <h2
        className="text-3xl sm:text-5xl font-bold tracking-tight mt-1 bg-clip-text text-transparent"
        style={{
          backgroundImage:
            'linear-gradient(110deg, #00c9a7 0%, #b07cf0 60%, #845ec2 100%)',
          filter:
            'drop-shadow(0 0 12px rgba(176,124,240,0.45)) drop-shadow(0 0 16px rgba(0,201,167,0.25))',
        }}
      >
        {title}
      </h2>
      <div className="h-px bg-gradient-to-r from-teal/60 via-purple/50 to-transparent mt-3 mb-4" />
      <p className="text-mint/70 max-w-3xl text-sm sm:text-base">{subtitle}</p>
    </div>
  )
}
