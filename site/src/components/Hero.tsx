import { useEffect, useRef, useState } from 'react'
import { STATS } from '../data/paper'

const INTRO_LINES = [
  '> initiating HERALD broker...',
  '> loading PLAsTiCC :: 7,848 objects / 14 classes / 6 bands',
  '> classifier := XGBoost(macro-F1 = 0.754)',
  '> prioritizer := T2 :: argsort P(KN|x) + P(TDE|x)',
  '> ready.',
]

function useTypewriter(lines: string[], speed = 22) {
  const [out, setOut] = useState<string[]>([])
  const [done, setDone] = useState(false)

  useEffect(() => {
    let cancelled = false
    setOut([])
    setDone(false)
    const work = async () => {
      for (let i = 0; i < lines.length; i++) {
        for (let c = 1; c <= lines[i].length; c++) {
          if (cancelled) return
          await new Promise((r) => setTimeout(r, speed))
          setOut((prev) => {
            const next = [...prev]
            next[i] = lines[i].slice(0, c)
            return next
          })
        }
        await new Promise((r) => setTimeout(r, 140))
      }
      if (!cancelled) setDone(true)
    }
    work()
    return () => {
      cancelled = true
    }
  }, [])

  return { out, done }
}

function Counter({
  end,
  duration = 1200,
  format = (n: number) => n.toLocaleString(),
}: {
  end: number
  duration?: number
  format?: (n: number) => string
}) {
  const [value, setValue] = useState(0)
  const ref = useRef<HTMLSpanElement | null>(null)
  const fired = useRef(false)

  useEffect(() => {
    if (!ref.current) return
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting && !fired.current) {
            fired.current = true
            const start = performance.now()
            const tick = (now: number) => {
              const t = Math.min(1, (now - start) / duration)
              const eased = 1 - Math.pow(1 - t, 3)
              setValue(end * eased)
              if (t < 1) requestAnimationFrame(tick)
              else setValue(end)
            }
            requestAnimationFrame(tick)
          }
        })
      },
      { threshold: 0.4 }
    )
    obs.observe(ref.current)
    return () => obs.disconnect()
  }, [end, duration])

  return <span ref={ref}>{format(Math.round(value))}</span>
}

export default function Hero() {
  const { out, done } = useTypewriter(INTRO_LINES, 18)

  return (
    <section
      id="abstract"
      className="pt-28 pb-24 px-4 sm:px-6 max-w-7xl mx-auto"
    >
      <div className="grid lg:grid-cols-[1.1fr_1fr] gap-10 items-center">
        <div>
          <p className="text-mint/50 text-xs mb-3 tracking-widest">
            // IJSPC 2026 · psut.dsai · rayyan &amp; emad
          </p>
          <h1
            className="text-5xl sm:text-7xl font-bold tracking-tight leading-none bg-clip-text text-transparent"
            style={{
              backgroundImage:
                'linear-gradient(120deg, #00c9a7 0%, #6cdfff 30%, #b07cf0 70%, #845ec2 100%)',
              filter:
                'drop-shadow(0 0 18px rgba(176,124,240,0.55)) drop-shadow(0 0 28px rgba(0,201,167,0.35))',
            }}
          >
            HERALD
          </h1>
          <p className="text-mint/80 mt-3 text-sm sm:text-base tracking-wide">
            Hierarchical Early-Alert Ranking for
            <br />
            Astrophysical Latent-event Detection
          </p>

          <div className="mt-8 panel p-4 sm:p-6 font-mono text-xs sm:text-sm">
            {out.map((line, i) => {
              const isLast = i === out.length - 1
              return (
                <div
                  key={i}
                  className={`whitespace-pre-wrap ${
                    isLast && !done ? 'cursor' : ''
                  } ${
                    line?.includes('ready')
                      ? 'text-teal text-glow'
                      : 'text-mint/85'
                  }`}
                >
                  {line ?? ''}
                </div>
              )
            })}
            {done && (
              <div className="text-mint/40 mt-2 cursor">
                user@psut:~/herald$ &nbsp;
              </div>
            )}
          </div>

          <p className="text-mint/85 mt-8 leading-relaxed text-sm sm:text-base">
            The Rubin Observatory's LSST will fire{' '}
            <span className="text-teal text-glow">
              ~10 million photometric alerts per night
            </span>{' '}
            — but spectroscopic follow-up can only handle ~1,000. Production
            brokers rank by generic classifier confidence and recover{' '}
            <span className="text-orange">0% of rare events</span>{' '}
            (kilonovae, TDEs) in the top-50 budget. HERALD swaps the ranking
            objective for{' '}
            <span className="text-yellow text-glow-yellow">
              P(KN|x) + P(TDE|x)
            </span>{' '}
            and recovers <span className="text-teal text-glow">42%</span> using
            the same classifier. The fix is the question — not the model.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-3 sm:gap-4">
          <StatTile
            label="T1 baseline recall@50"
            value="0%"
            sub="generic confidence"
            tone="orange"
          />
          <StatTile
            label="T2 goal-aligned recall@50"
            value="42%"
            sub="P(KN) + P(TDE)"
            tone="teal"
          />
          <StatTile
            label="XGBoost macro-F1"
            value={<Counter end={754} format={(n) => `0.${n}`} />}
            sub="41 physics-informed features"
            tone="mint"
          />
          <StatTile
            label="P precision · kilonovae"
            value={<Counter end={94} format={(n) => `${n}%`} />}
            sub="when KN is the model's top class"
            tone="yellow"
          />
          <StatTile
            label="LSST alerts / night"
            value={
              <Counter
                end={STATS.alertsPerNight}
                format={(n) => `${(n / 1_000_000).toFixed(0)}M`}
              />
            }
            sub="vs ~1k follow-up budget"
            tone="purple"
          />
          <StatTile
            label="Rare events in test set"
            value={<Counter end={STATS.rareObjects} />}
            sub="50 KN · 69 TDE / 7,848 total"
            tone="mint"
          />
        </div>
      </div>
    </section>
  )
}

function StatTile({
  label,
  value,
  sub,
  tone,
}: {
  label: string
  value: React.ReactNode
  sub?: string
  tone: 'teal' | 'orange' | 'purple' | 'yellow' | 'mint'
}) {
  const map = {
    teal: 'glow-teal text-teal text-glow',
    orange: 'glow-orange text-orange',
    purple: 'glow-purple text-purple',
    yellow: 'glow-yellow text-yellow text-glow-yellow',
    mint: 'glow-teal text-mint',
  } as const
  const cls = map[tone]
  return (
    <div className={`panel p-3 sm:p-4 ${cls.split(' ')[0]}`}>
      <p className="text-mint/50 text-[10px] sm:text-xs tracking-widest uppercase">
        {label}
      </p>
      <p className={`text-2xl sm:text-4xl font-bold mt-1 ${cls.split(' ').slice(1).join(' ')}`}>
        {value}
      </p>
      {sub && (
        <p className="text-mint/40 text-[10px] sm:text-xs mt-1 tracking-wider">
          {sub}
        </p>
      )}
    </div>
  )
}
