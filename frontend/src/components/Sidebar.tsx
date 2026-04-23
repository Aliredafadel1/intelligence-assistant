import { Activity, BarChart3, Bot, Cpu, Database, FlaskConical, Home, Search } from 'lucide-react'

const navItems = [
  { label: 'Overview', icon: Home, active: true },
  { label: 'Pipeline', icon: Activity },
  { label: 'Retrieval', icon: Search },
  { label: 'Models', icon: Cpu },
  { label: 'Analytics', icon: BarChart3 },
  { label: 'Knowledge Base', icon: Database },
  { label: 'Experiments', icon: FlaskConical },
  { label: 'Assistant', icon: Bot },
]

export function Sidebar() {
  return (
    <aside className="sticky top-0 h-screen w-72 border-r border-slate-800/90 bg-slate-950/70 px-5 py-6 backdrop-blur-xl">
      <div className="mb-8 rounded-xl border border-slate-700/60 bg-slate-900/70 p-4">
        <p className="text-xs uppercase tracking-[0.3em] text-slate-400">AI Command Center</p>
        <p className="mt-2 text-lg font-semibold text-white">Decision Intelligence</p>
      </div>
      <nav className="space-y-2">
        {navItems.map(({ label, icon: Icon, active }) => (
          <button
            key={label}
            className={`flex w-full items-center gap-3 rounded-xl px-3 py-2 text-left transition ${
              active
                ? 'bg-gradient-to-r from-accentBlue/20 to-accentPurple/20 text-white shadow-neon'
                : 'text-slate-300 hover:bg-slate-800/60'
            }`}
          >
            <Icon size={16} />
            <span className="text-sm">{label}</span>
          </button>
        ))}
      </nav>
    </aside>
  )
}
