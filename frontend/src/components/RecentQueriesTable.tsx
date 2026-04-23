type QueryRow = {
  id: string
  query: string
  priority: string
  latency: string
  status: string
}

type RecentQueriesTableProps = {
  rows: QueryRow[]
}

export function RecentQueriesTable({ rows }: RecentQueriesTableProps) {
  return (
    <section className="glass-card p-5">
      <p className="section-title">Recent Queries</p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead className="text-xs uppercase tracking-[0.12em] text-slate-400">
            <tr>
              <th className="pb-2">ID</th>
              <th className="pb-2">Query</th>
              <th className="pb-2">Priority</th>
              <th className="pb-2">Latency</th>
              <th className="pb-2">Status</th>
            </tr>
          </thead>
          <tbody className="text-slate-200">
            {rows.map((row) => (
              <tr key={row.id} className="border-t border-slate-800">
                <td className="py-3">{row.id}</td>
                <td className="py-3">{row.query}</td>
                <td className="py-3">{row.priority}</td>
                <td className="py-3">{row.latency}</td>
                <td className="py-3">{row.status}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
