import { useState } from 'react'
import './App.css'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import Models from './components/Models'
import Results from './components/Results'
import Demo from './components/Demo'
import Team from './components/Team'
import Chat from './components/Chat'
import KeyModal from './components/KeyModal'
import Cosmos from './components/Cosmos'
import { hasKey } from './lib/gemini'

function App() {
  const [keyModal, setKeyModal] = useState(false)

  const requireKey = () => {
    if (!hasKey()) setKeyModal(true)
  }

  return (
    <>
      <Cosmos />
      <Navbar />
      <main className="relative z-10">
        <Hero />
        <Divider label="// next :: classifier zoo" />
        <Models />
        <Divider label="// next :: empirical results" />
        <Results />
        <Divider label="// next :: live alert stream" />
        <Demo onRequireKey={requireKey} />
        <Divider label="// next :: authors" />
        <Team />
      </main>
      <Chat onRequireKey={requireKey} />
      <KeyModal open={keyModal} onClose={() => setKeyModal(false)} />

      <button
        onClick={() => setKeyModal(true)}
        className="fixed bottom-5 left-5 z-[150] panel px-3 py-2 text-[10px] tracking-widest text-mint/70 hover:text-teal hover:border-teal/60"
        title="Set or update your Gemini API key"
      >
        {hasKey() ? 'update ai key' : 'set ai key'}
      </button>
    </>
  )
}

function Divider({ label }: { label: string }) {
  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 flex items-center gap-3 text-mint/40 text-[10px] tracking-widest">
      <span className="h-px bg-teal/20 flex-1" />
      <span>{label}</span>
      <span className="h-px bg-teal/20 flex-1" />
    </div>
  )
}

export default App
