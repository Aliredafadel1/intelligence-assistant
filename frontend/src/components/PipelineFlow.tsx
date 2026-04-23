import { motion } from 'framer-motion'
import type { PipelineStep } from '../data/mockData'

type PipelineFlowProps = {
  steps: PipelineStep[]
  onSelectStep: (step: PipelineStep) => void
}

const statusStyles: Record<PipelineStep['status'], string> = {
  completed: 'border-emerald-400/40 bg-emerald-500/10 text-emerald-300',
  active: 'border-indigo-400/50 bg-indigo-500/20 text-indigo-200',
  queued: 'border-slate-700 bg-slate-900/60 text-slate-300',
}

export function PipelineFlow({ steps, onSelectStep }: PipelineFlowProps) {
  return (
    <section className="glass-card p-5">
      <p className="section-title">Pipeline Intelligence Flow</p>
      <div className="mt-4 grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {steps.map((step, idx) => (
          <motion.button
            key={step.id}
            onClick={() => onSelectStep(step)}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: idx * 0.05 }}
            className={`rounded-xl border p-4 text-left transition hover:scale-[1.02] ${statusStyles[step.status]}`}
          >
            <p className="text-xs uppercase tracking-[0.2em]">{step.subtitle}</p>
            <p className="mt-2 text-lg font-semibold">{step.title}</p>
            <p className="mt-2 text-sm opacity-90">{step.detail}</p>
          </motion.button>
        ))}
      </div>
    </section>
  )
}
