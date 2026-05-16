import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import {
  FEATURE_IMPORTANCE,
  RECALL_AT_K,
  RECALL_VS_FRACTION,
} from '../data/paper'
import { SectionHeader } from './Models'

const tealColor = '#00C9A7'
const purpleColor = '#845EC2'
const yellowColor = '#FFD93D'
const orangeColor = '#FF6B35'
const mintColor = '#C4FCEF'

const axisStyle = { fill: '#c4fcef', fontSize: 10, fontFamily: 'JetBrains Mono' }
const gridStyle = { stroke: 'rgba(0,201,167,0.12)' }

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="panel px-3 py-2 text-xs font-mono">
      <p className="text-mint/60">{label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.color }}>
          {p.name}: {typeof p.value === 'number' ? p.value.toFixed(3) : p.value}
        </p>
      ))}
    </div>
  )
}

function RecallAtKChart() {
  return (
    <div className="panel glow-teal p-4 sm:p-6">
      <p className="text-mint/40 text-xs tracking-widest">// fig_01</p>
      <h3 className="text-teal text-glow text-xl sm:text-2xl font-bold mt-1">
        recall@K vs follow-up budget
      </h3>
      <p className="text-mint/70 text-xs sm:text-sm mt-2 mb-4">
        At full observation (f = 1.0). T1 (grey) is the current broker
        standard — flat at ≈0 across the entire nightly budget. T2 / T3
        converge from K = 10 onward. Orange dashed = K = 50 nightly
        spectroscopic capacity.
      </p>
      <div className="h-72">
        <ResponsiveContainer>
          <LineChart data={RECALL_AT_K} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
            <CartesianGrid {...gridStyle} />
            <XAxis dataKey="k" tick={axisStyle} stroke="#4d8076" label={{ value: 'K (follow-up budget)', fill: mintColor, fontSize: 11, dy: 14 }} />
            <YAxis tick={axisStyle} stroke="#4d8076" domain={[0, 1]} tickFormatter={(v) => v.toFixed(1)} />
            <Tooltip content={<ChartTooltip />} cursor={{ stroke: tealColor, strokeOpacity: 0.3 }} />
            <Legend wrapperStyle={{ fontSize: 11, color: mintColor }} />
            <ReferenceLine x={50} stroke={orangeColor} strokeDasharray="4 4" label={{ value: 'K=50 budget', fill: orangeColor, fontSize: 10, position: 'top' }} />
            <Line type="monotone" dataKey="T1" name="T1 max-prob" stroke="#7a8a92" strokeWidth={2} strokeDasharray="6 4" dot={false} />
            <Line type="monotone" dataKey="T2" name="T2 P(KN)+P(TDE)" stroke={tealColor} strokeWidth={2.5} dot={{ r: 3, fill: tealColor }} />
            <Line type="monotone" dataKey="T3" name="T3 uncertainty-weighted" stroke={purpleColor} strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function RecallVsFractionChart() {
  return (
    <div className="panel glow-purple p-4 sm:p-6">
      <p className="text-mint/40 text-xs tracking-widest">// fig_02</p>
      <h3 className="text-purple text-glow-purple text-xl sm:text-2xl font-bold mt-1">
        recall@50 vs observation fraction
      </h3>
      <p className="text-mint/70 text-xs sm:text-sm mt-2 mb-4">
        How rare-event recovery degrades as we shrink the available light
        curve. T2/T3 hold up gracefully: 42% at full data → 11% with only ~3–5
        nights of observations. T1 stays at 0 — not a data problem.
      </p>
      <div className="h-72">
        <ResponsiveContainer>
          <BarChart data={RECALL_VS_FRACTION} margin={{ top: 8, right: 12, left: -8, bottom: 0 }}>
            <CartesianGrid {...gridStyle} vertical={false} />
            <XAxis dataKey="label" tick={axisStyle} stroke="#4d8076" />
            <YAxis tick={axisStyle} stroke="#4d8076" domain={[0, 0.5]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(0,201,167,0.08)' }} />
            <Legend wrapperStyle={{ fontSize: 11, color: mintColor }} />
            <Bar dataKey="T1" name="T1" fill="#7a8a92" radius={[2, 2, 0, 0]} />
            <Bar dataKey="T2" name="T2" fill={tealColor} radius={[2, 2, 0, 0]} />
            <Bar dataKey="T3" name="T3" fill={purpleColor} radius={[2, 2, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

function FeatureImportanceChart() {
  // Sort descending and reverse for horizontal display (top of chart = most important).
  const data = [...FEATURE_IMPORTANCE].sort((a, b) => a.gain - b.gain)
  return (
    <div className="panel glow-yellow p-4 sm:p-6 xl:col-span-2">
      <p className="text-mint/40 text-xs tracking-widest">// fig_03</p>
      <h3 className="text-yellow text-glow-yellow text-xl sm:text-2xl font-bold mt-1">
        XGBoost top-20 feature importances (gain)
      </h3>
      <p className="text-mint/70 text-xs sm:text-sm mt-2 mb-4">
        Yellow = host-galaxy / metadata features. Teal = photometric. The two
        most important features (distmod, hostgal_photoz) are{' '}
        <span className="text-yellow">not</span> light-curve features —
        confirming transient classification is fundamentally multimodal.
      </p>
      <div className="h-[460px]">
        <ResponsiveContainer>
          <BarChart data={data} layout="vertical" margin={{ top: 4, right: 16, left: 4, bottom: 0 }}>
            <CartesianGrid {...gridStyle} horizontal={false} />
            <XAxis type="number" tick={axisStyle} stroke="#4d8076" tickFormatter={(v) => v.toFixed(2)} />
            <YAxis dataKey="feature" type="category" tick={{ ...axisStyle, fontSize: 11 }} stroke="#4d8076" width={130} />
            <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,217,61,0.06)' }} />
            <Bar dataKey="gain" radius={[0, 2, 2, 0]}>
              {data.map((d, i) => (
                <Cell key={i} fill={d.kind === 'meta' ? yellowColor : tealColor} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export default function Results() {
  return (
    <section id="results" className="py-20 px-4 sm:px-6 max-w-7xl mx-auto">
      <SectionHeader
        label="// section_03"
        title="RESULTS"
        subtitle="Three plots, one thesis: the limiting factor is the ranking objective, not the classifier."
      />

      <div className="grid xl:grid-cols-2 gap-4 sm:gap-6 mt-10">
        <RecallAtKChart />
        <RecallVsFractionChart />
        <FeatureImportanceChart />
      </div>
    </section>
  )
}
