import { FileUp, Wand2 } from 'lucide-react'

type InputPanelProps = {
  value: string
  isLoading: boolean
  onChange: (value: string) => void
  onRun: () => void
  onUseSample: () => void
}

export function InputPanel({ value, isLoading, onChange, onRun, onUseSample }: InputPanelProps) {
  return (
    <section className="glass-card p-5">
      <p className="section-title">Input Control</p>
      <div className="mt-3 grid gap-3 md:grid-cols-[1fr_auto]">
        <textarea
          rows={3}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full rounded-xl border border-slate-700 bg-slate-950/70 p-3 text-sm text-slate-100 outline-none ring-0 placeholder:text-slate-500 focus:border-accentBlue"
        />
        <div className="flex gap-2 md:flex-col">
          <button className="inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-slate-900/70 px-3 py-2 text-sm text-slate-200">
            <FileUp size={16} /> Upload Ticket
          </button>
          <button
            onClick={onUseSample}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-accentBlue to-accentPurple px-3 py-2 text-sm text-white"
          >
            <Wand2 size={16} /> Use Sample
          </button>
          <button
            onClick={onRun}
            disabled={isLoading || !value.trim()}
            className="inline-flex items-center justify-center rounded-xl border border-indigo-400/40 bg-indigo-500/20 px-3 py-2 text-sm text-indigo-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isLoading ? 'Running...' : 'Run Inference'}
          </button>
        </div>
      </div>
    </section>
  )
}
