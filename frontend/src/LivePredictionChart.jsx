import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';

import { getLatestPrediction, getTimeline } from './api';

function formatXAxisTick(t) {
  const n = Number(t);
  if (!Number.isFinite(n)) return String(t);
  // Backend stores timestamps as ms since epoch.
  if (n > 1e11) return new Date(n).toLocaleTimeString();
  return `#${t}`;
}

export default function LivePredictionChart({
  enabled = true,
  maxPoints = 100,
  pollIntervalMs = 2000,
}) {
  const [points, setPoints] = useState([]);
  const lastTimestampRef = useRef(null);

  const chartData = useMemo(() => points, [points]);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;

    const bootstrap = async () => {
      try {
        const timeline = await getTimeline();
        const mapped = (timeline || []).map((row) => ({
          timestamp: row.timestamp,
          actual: row.actual_cpu,
          predicted: row.predicted_cpu,
        }));

        const tail = mapped.slice(-maxPoints);
        if (cancelled) return;
        setPoints(tail);
        if (tail.length) lastTimestampRef.current = tail[tail.length - 1].timestamp;
      } catch (e) {
        console.error('Failed to bootstrap timeline for LivePredictionChart', e);
      }
    };

    bootstrap();

    const intervalId = setInterval(async () => {
      try {
        const latest = await getLatestPrediction();
        if (cancelled) return;

        const ts = latest?.timestamp;
        if (ts == null) return;

        // Avoid appending duplicates when a new step hasn't been produced yet.
        if (lastTimestampRef.current === ts) return;
        lastTimestampRef.current = ts;

        setPoints((prev) => {
          const next = [
            ...prev,
            { timestamp: ts, actual: latest.actual, predicted: latest.predicted },
          ];
          return next.slice(-maxPoints);
        });
      } catch (e) {
        // Keep the chart alive; backend might temporarily be busy training.
        console.error('Failed to fetch latest prediction', e);
      }
    }, pollIntervalMs);

    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, [enabled, maxPoints, pollIntervalMs]);

  return (
    <div className="w-full">
      <h3 className="text-xl font-bold mb-4 text-gray-200">
        Actual vs Predicted CPU Usage (Live)
      </h3>
      <div className="h-80 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart
            data={chartData}
            margin={{ top: 5, right: 20, bottom: 5, left: 0 }}
            isAnimationActive={true}
            animationDuration={400}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
            <XAxis
              dataKey="timestamp"
              tickFormatter={formatXAxisTick}
              stroke="#6B7280"
              label={{ value: 'Time / Index', position: 'insideBottom', offset: -5 }}
            />
            <YAxis
              stroke="#6B7280"
              domain={[0, 100]}
              label={{ value: 'CPU Usage (%)', angle: -90, position: 'insideLeft' }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#161B22',
                borderColor: '#30363D',
                color: '#fff',
              }}
              labelFormatter={(label) => formatXAxisTick(label)}
            />
            <Legend />
            <Line
              type="monotone"
              dataKey="actual"
              name="Actual CPU Usage"
              stroke="#4F46E5"
              strokeWidth={2}
              dot={false}
              isAnimationActive={true}
            />
            <Line
              type="monotone"
              dataKey="predicted"
              name="Predicted CPU Usage (TCN + Attention Ensemble)"
              stroke="#F59E0B"
              strokeWidth={2}
              dot={false}
              strokeDasharray="5 5"
              isAnimationActive={true}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <p className="mt-4 text-sm text-gray-400 leading-relaxed">
        Each point represents one simulation interval.
        The orange line is the model’s prediction of the next-step CPU usage from the TCN + Attention ensemble.
        The blue line shows the actual CPU usage used by the simulator at that step.
        When the two lines overlap, predictions are close; larger gaps indicate higher prediction error.
      </p>
    </div>
  );
}

