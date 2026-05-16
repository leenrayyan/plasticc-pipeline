import { CLASSES, type TransientClass, RARE_CLASSES } from '../data/paper'

const ALL_CLASSES = Object.keys(CLASSES) as TransientClass[]

function sampleClass(rareBoost = 1): TransientClass {
  // Optionally over-sample rare classes for an interesting demo stream.
  const weights = ALL_CLASSES.map((k) =>
    CLASSES[k].rare ? CLASSES[k].prior * rareBoost : CLASSES[k].prior
  )
  const total = weights.reduce((s, w) => s + w, 0)
  let r = Math.random() * total
  for (let i = 0; i < ALL_CLASSES.length; i++) {
    r -= weights[i]
    if (r <= 0) return ALL_CLASSES[i]
  }
  return ALL_CLASSES[0]
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits)
  const exps = logits.map((l) => Math.exp(l - max))
  const sum = exps.reduce((s, e) => s + e, 0)
  return exps.map((e) => e / sum)
}

// Generate per-class probabilities that *look* like a real XGBoost output:
// strongly peaked on the true class at f=1.0, increasingly noisy as f → 0.1.
// Common classes leak probability mass into rare-class predictions when f is small.
function simulateProbs(
  trueClass: TransientClass,
  fraction: number
): Record<TransientClass, number> {
  const noiseScale = (1 - fraction) * 3.2 + 0.3
  const peakStrength = 2.8 + fraction * 4.2 // 3.1 at f=0.1, 7.0 at f=1.0

  const logits = ALL_CLASSES.map((k) => {
    let l = (Math.random() - 0.5) * noiseScale
    if (k === trueClass) {
      l += peakStrength
    }
    // model confuses TDE <-> AGN, KN <-> fast SNe
    if (trueClass === 'TDE' && k === 'AGN') l += 0.9
    if (trueClass === 'AGN' && k === 'TDE') l += 0.4
    if (trueClass === 'KN' && k === 'SNIax') l += 0.6
    if (CLASSES[k].rare && !CLASSES[trueClass].rare) {
      // suppress rare-class probability for common-true alerts (this is the
      // mechanism that drives T1's failure mode in the paper).
      l -= 0.8 + noiseScale * 0.4
    }
    return l
  })

  // Penalize KN/TDE at very low fraction (early truncation hurts rare classes
  // disproportionately — paper's T2 drops from 42% → 11%).
  if (fraction <= 0.3 && CLASSES[trueClass].rare) {
    const trueIdx = ALL_CLASSES.indexOf(trueClass)
    logits[trueIdx] -= (0.3 - fraction) * 4
  }

  const probs = softmax(logits)
  const out = {} as Record<TransientClass, number>
  ALL_CLASSES.forEach((k, i) => (out[k] = probs[i]))
  return out
}

export interface Alert {
  id: string
  timestamp: number
  trueClass: TransientClass
  probs: Record<TransientClass, number>
  t1: number
  t2: number
  t3: number
  predictedClass: TransientClass
  entropy: number
  // mock display metadata
  ra: number
  dec: number
  mag: number
}

function entropy(probs: number[]): number {
  let h = 0
  for (const p of probs) {
    if (p > 1e-9) h -= p * Math.log(p)
  }
  return h
}

function uid(): string {
  return Math.random().toString(36).slice(2, 10).toUpperCase()
}

export function generateAlert(fraction: number, rareBoost = 4): Alert {
  const trueClass = sampleClass(rareBoost)
  const probs = simulateProbs(trueClass, fraction)
  const values = ALL_CLASSES.map((k) => probs[k])

  const t1 = Math.max(...values) // generic confidence
  const t2 = (probs.KN ?? 0) + (probs.TDE ?? 0) // goal-aligned
  const h = entropy(values)
  const t3 = t2 / (1 + h) // uncertainty-weighted

  let predicted = trueClass
  let best = -Infinity
  ALL_CLASSES.forEach((k) => {
    if (probs[k] > best) {
      best = probs[k]
      predicted = k
    }
  })

  return {
    id: `LSST-${uid()}`,
    timestamp: Date.now(),
    trueClass,
    probs,
    t1,
    t2,
    t3,
    predictedClass: predicted,
    entropy: h,
    ra: +(Math.random() * 360).toFixed(2),
    dec: +(Math.random() * 180 - 90).toFixed(2),
    mag: +(Math.random() * 4 + 19).toFixed(2),
  }
}

export { RARE_CLASSES }
