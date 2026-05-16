import { SectionHeader } from './Models'

interface Member {
  name: string
  role: string
  email: string
  avatar: string
  accent: 'teal' | 'purple'
}

const MEMBERS: Member[] = [
  {
    name: 'Leen Rayyan',
    role: 'co-author · data science & AI',
    email: 'leenrayyan@ieee.org',
    avatar: 'LR',
    accent: 'teal',
  },
  {
    name: 'Retal Emad',
    role: 'co-author · data science & AI',
    email: 'ret20230381@std.psut.edu.jo',
    avatar: 'RE',
    accent: 'purple',
  },
]

function MemberCard({ m }: { m: Member }) {
  const cls =
    m.accent === 'teal'
      ? {
          glow: 'glow-teal',
          text: 'text-teal text-glow',
          border: 'border-teal/50',
        }
      : {
          glow: 'glow-purple',
          text: 'text-purple text-glow-purple',
          border: 'border-purple/50',
        }
  return (
    <div className={`panel ${cls.glow} p-6 flex items-center gap-5`}>
      <div
        className={`w-20 h-20 rounded-sm border ${cls.border} ${cls.text} font-bold text-3xl flex items-center justify-center shrink-0`}
      >
        {m.avatar}
      </div>
      <div className="flex-1 min-w-0">
        <p className={`text-xl sm:text-2xl font-bold ${cls.text}`}>{m.name}</p>
        <p className="text-mint/70 text-xs sm:text-sm">{m.role}</p>
        <a
          href={`mailto:${m.email}`}
          className="text-mint/60 hover:text-teal text-xs sm:text-sm font-mono break-all"
        >
          {m.email}
        </a>
      </div>
    </div>
  )
}

export default function Team() {
  return (
    <section id="team" className="py-20 px-4 sm:px-6 max-w-7xl mx-auto">
      <SectionHeader
        label="// section_05"
        title="TEAM"
        subtitle="Department of Data Science &amp; Artificial Intelligence · Princess Sumaya University for Technology, Amman, Jordan."
      />

      <div className="grid md:grid-cols-2 gap-4 sm:gap-6 mt-10">
        {MEMBERS.map((m) => (
          <MemberCard key={m.email} m={m} />
        ))}
      </div>

      <div className="mt-10 panel p-5 text-mint/80 text-sm">
        <p className="text-mint/40 text-[10px] tracking-widest mb-2">
          // citation
        </p>
        <code className="block text-mint/85 text-[12px] leading-relaxed whitespace-pre-wrap">
{`@inproceedings{rayyan2026herald,
  title  = {HERALD: Hierarchical Early-Alert Ranking for
            Astrophysical Latent-event Detection},
  author = {Rayyan, Leen and Emad, Retal},
  booktitle = {IJSPC},
  year   = {2026},
}`}
        </code>
      </div>

      <footer className="mt-10 flex flex-wrap items-center justify-between gap-4 text-mint/50 text-xs border-t border-teal/20 pt-6">
        <span>
          [HERALD@psut] :: built with vite · react · recharts · gemini
        </span>
        <a
          href="https://github.com/leenrayyan/plasticc-pipeline"
          target="_blank"
          rel="noreferrer noopener"
          className="text-teal hover:text-glow border border-teal/40 hover:border-teal px-3 py-1 rounded-sm tracking-widest"
        >
          → github.com/leenrayyan/plasticc-pipeline
        </a>
      </footer>
    </section>
  )
}
