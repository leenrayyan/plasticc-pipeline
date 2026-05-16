import { useEffect, useState } from 'react'

const SECTIONS = [
  { id: 'abstract', label: 'ABSTRACT' },
  { id: 'models', label: 'MODELS' },
  { id: 'results', label: 'RESULTS' },
  { id: 'demo', label: 'DEMO' },
  { id: 'team', label: 'TEAM' },
] as const

export default function Navbar() {
  const [active, setActive] = useState<string>('abstract')

  useEffect(() => {
    const obs = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) setActive(e.target.id)
        })
      },
      { rootMargin: '-40% 0px -55% 0px', threshold: 0 }
    )
    SECTIONS.forEach((s) => {
      const el = document.getElementById(s.id)
      if (el) obs.observe(el)
    })
    return () => obs.disconnect()
  }, [])

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 backdrop-blur-md bg-[rgba(4,3,13,0.72)] border-b border-purple/30">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4">
        <a
          href="#abstract"
          className="font-bold tracking-wider text-sm sm:text-base whitespace-nowrap"
        >
          <span className="text-teal text-glow">[HERALD</span>
          <span className="text-purple text-glow-purple">@psut</span>
          <span className="text-teal text-glow">]</span>
          <span className="text-mint/40 hidden sm:inline">
            {' '}
            :~$ ./broker --live
          </span>
        </a>
        <ul className="flex items-center gap-1 sm:gap-2 text-xs sm:text-sm">
          {SECTIONS.map((s) => {
            // alternate teal and purple highlights across the nav
            const i = SECTIONS.findIndex((x) => x.id === s.id)
            const isPurple = i % 2 === 1
            const activeCls = isPurple
              ? 'text-purple text-glow-purple border-purple/60 bg-purple/10'
              : 'text-teal text-glow border-teal/60 bg-teal/5'
            const hoverCls = isPurple
              ? 'hover:text-purple hover:border-purple/40'
              : 'hover:text-teal hover:border-teal/30'
            return (
              <li key={s.id}>
                <a
                  href={`#${s.id}`}
                  className={`px-2 sm:px-3 py-1.5 rounded transition-all border ${
                    active === s.id
                      ? activeCls
                      : `text-mint/70 border-transparent ${hoverCls}`
                  }`}
                >
                  {s.label}
                </a>
              </li>
            )
          })}
        </ul>
      </div>
    </nav>
  )
}
