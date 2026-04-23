type StatCardProps = {
  title: string
  value: string
  delta: string
  trend: string
}

export function StatCard({ title, value, delta, trend }: StatCardProps) {
  return (
    <article className="glass-card p-4">
      <p className="text-xs uppercase tracking-[0.2em] text-slate-400">{title}</p>
      <p className="mt-2 text-2xl font-semibold text-white">{value}</p>
      <div className="mt-3 flex items-center justify-between text-xs">
        <span className="text-slate-300">{delta}</span>
        <span className="rounded-full bg-slate-800 px-2 py-1 text-indigo-300">{trend}</span>
      </div>
    </article>
  )
}
