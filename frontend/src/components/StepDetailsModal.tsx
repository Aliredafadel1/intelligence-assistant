import { AnimatePresence, motion } from 'framer-motion'
import { X } from 'lucide-react'
import type { PipelineStep } from '../data/mockData'

type StepDetailsModalProps = {
  step: PipelineStep | null
  onClose: () => void
}

export function StepDetailsModal({ step, onClose }: StepDetailsModalProps) {
  return (
    <AnimatePresence>
      {step && (
        <motion.div
          className="fixed inset-0 z-40 grid place-items-center bg-slate-950/70 p-4 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            className="glass-card w-full max-w-xl p-6"
            initial={{ y: 16, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 16, opacity: 0 }}
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.2em] text-slate-400">{step.subtitle}</p>
                <h3 className="mt-1 text-xl font-semibold text-white">{step.title}</h3>
              </div>
              <button onClick={onClose} className="rounded-md border border-slate-700 p-1 text-slate-300">
                <X size={16} />
              </button>
            </div>
            <p className="mt-4 text-sm text-slate-300">{step.detail}</p>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
