import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

type InsightsChartsProps = {
  latencyData: { name: string; value: number }[]
  confidenceData: { t: string; confidence: number }[]
}

export function InsightsCharts({ latencyData, confidenceData }: InsightsChartsProps) {
  return (
    <section className="grid gap-4 xl:grid-cols-2">
      <article className="glass-card p-4">
        <p className="section-title">Latency Breakdown (ms)</p>
        <div className="mt-3 h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={latencyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#24304e" />
              <XAxis dataKey="name" stroke="#94a3b8" />
              <YAxis stroke="#94a3b8" />
              <Tooltip />
              <Bar dataKey="value" fill="#4f8cff" radius={[6, 6, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </article>
      <article className="glass-card p-4">
        <p className="section-title">Confidence Trend</p>
        <div className="mt-3 h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={confidenceData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#24304e" />
              <XAxis dataKey="t" stroke="#94a3b8" />
              <YAxis domain={[0.7, 1]} stroke="#94a3b8" />
              <Tooltip />
              <Line type="monotone" dataKey="confidence" stroke="#a855f7" strokeWidth={3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </article>
    </section>
  )
}
