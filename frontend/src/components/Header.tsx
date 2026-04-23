import { Bell, PlayCircle, ShieldCheck, Sparkles } from 'lucide-react'

type HeaderProps = {
  developerMode: boolean
  onToggleDeveloperMode: () => void
}

export function Header({ developerMode, onToggleDeveloperMode }: HeaderProps) {
  return (
    <header className="glass-card flex flex-wrap items-center justify-between gap-4 px-6 py-5">
      <div>
        <h1 className="text-2xl font-semibold text-white">AI Command Center</h1>
        <p className="text-sm text-slate-400">Traceable intelligence workflow for triage and prioritization</p>
      </div>
      <div className="flex items-center gap-3">
        <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/30 bg-emerald-500/10 px-3 py-1 text-xs text-emerald-300">
          <ShieldCheck size={14} /> System Healthy
        </span>
        <span className="inline-flex items-center gap-2 rounded-full border border-indigo-400/30 bg-indigo-500/10 px-3 py-1 text-xs text-indigo-300">
          <Sparkles size={14} /> Model: XGBoost + RAG
        </span>
        <button className="rounded-lg border border-slate-700 bg-slate-900/70 p-2 text-slate-300 hover:text-white">
          <Bell size={16} />
        </button>
        <button className="inline-flex items-center gap-2 rounded-lg bg-gradient-to-r from-accentBlue to-accentPurple px-4 py-2 text-sm font-medium text-white">
          <PlayCircle size={16} /> Run Pipeline
        </button>
        <button
          onClick={onToggleDeveloperMode}
          className={`rounded-lg border px-3 py-2 text-xs ${
            developerMode
              ? 'border-cyan-400/50 bg-cyan-500/20 text-cyan-200'
              : 'border-slate-700 bg-slate-900/50 text-slate-300'
          }`}
        >
          Dev Mode {developerMode ? 'On' : 'Off'}
        </button>
      </div>
    </header>
  )
}
