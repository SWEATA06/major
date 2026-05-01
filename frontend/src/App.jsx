import { useCallback, useEffect, useState } from 'react';
import { getStatus, getCurrentMetrics, getTimeline, runSimulationStep, trainModels } from './api';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, ComposedChart, Bar
} from 'recharts';
import { Play, Activity, Server, AlertTriangle, CheckCircle, ShieldAlert, DollarSign } from 'lucide-react';
import LivePredictionChart from './LivePredictionChart.jsx';

function App() {
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);
  const [simRunning, setSimRunning] = useState(false);
  const [staticData, setStaticData] = useState([]);

  // Load static CSV served from public/static_chart_data.csv
  const loadStaticCsv = useCallback(async () => {
    try {
      const res = await fetch('/static_chart_data.csv');
      const text = await res.text();
      const lines = text.trim().split('\n');
      const headers = lines[0].split(',').map(h => h.trim());
      const rows = lines.slice(1).map(line => {
        const parts = line.split(',');
        const obj = {};
        headers.forEach((h, i) => {
          const v = parts[i];
          // convert numeric-looking fields to numbers
          obj[h] = isNaN(Number(v)) ? v : Number(v);
        });
        return obj;
      });
      setStaticData(rows);
    } catch (e) {
      console.error('Failed to load static CSV', e);
    }
  }, []);

  const fetchData = useCallback(async () => {
    try {
      const s = await getStatus();
      setStatus(s);
      if (s.models_loaded) {
        const m = await getCurrentMetrics();
        setMetrics(m);
        const t = await getTimeline();
        setTimeline(t);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, []);

  const triggerStep = useCallback(async () => {
    try {
      await runSimulationStep();
      await fetchData();
    } catch (err) {
      console.error("Simulation ended or error", err);
      setSimRunning(false);
    }
  }, [fetchData]);

  useEffect(() => {
    // Schedule the initial load after the effect finishes to avoid sync state updates.
    const kickoffId = setTimeout(() => {
      fetchData();
      loadStaticCsv();
    }, 0);
    const interval = setInterval(() => {
      if (simRunning) triggerStep();
      else fetchData();
    }, 2000);
    return () => {
      clearInterval(interval);
      clearTimeout(kickoffId);
    };
  }, [simRunning, fetchData, triggerStep]);

  const handleTrain = async () => {
    try {
      await trainModels();
      alert("Training started in the background!");
    } catch (err) {
      alert("Training failed to start: " + err);
    }
  };

  if (loading) return <div className="flex h-screen items-center justify-center text-xl">Loading System State...</div>;

  return (
    <div className="min-h-screen p-6 lg:p-10 flex flex-col gap-8">
      <header className="flex justify-between items-center bg-card/40 backdrop-blur pb-4 border-b border-border/50">
        <div>
          <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-400 to-primary bg-clip-text text-transparent flex items-center gap-3">
            <Activity className="text-primary" /> AutoScale Intelligence AI
          </h1>
          <p className="text-gray-400 mt-2 text-sm">Predictive Analytics & Cost-Aware Cloud Provisioning</p>
        </div>
        <div className="flex gap-4 items-center">
          <div className={`px-4 py-1.5 rounded-full text-sm font-medium flex items-center gap-2 ${status?.models_loaded ? 'bg-success/20 text-success border border-success/30' : 'bg-warning/20 text-warning border border-warning/30'}`}>
            {status?.models_loaded ? <CheckCircle size={16}/> : <AlertTriangle size={16}/>}
            {status?.models_loaded ? 'AI Core Online' : 'Models Offline'}
          </div>
          {!status?.models_loaded && (
            <button onClick={handleTrain} className="bg-primary hover:bg-primary/80 transition-colors px-6 py-2 rounded-lg font-medium shadow-[0_0_15px_rgba(79,70,229,0.5)]">
              Train Models
            </button>
          )}
          {status?.models_loaded && (
            <button 
              onClick={() => setSimRunning(!simRunning)} 
              className={`flex items-center gap-2 px-6 py-2 rounded-lg font-medium transition-all ${simRunning ? 'bg-danger text-white hover:bg-danger/80 shadow-[0_0_15px_rgba(239,68,68,0.5)]' : 'bg-primary text-white hover:bg-primary/80 shadow-[0_0_15px_rgba(79,70,229,0.5)]'}`}
            >
              <Play size={18} /> {simRunning ? 'Stop Simulation' : 'Auto Simulate'}
            </button>
          )}
          {status?.models_loaded && !simRunning && (
             <button onClick={triggerStep} className="bg-card hover:bg-card/80 border border-border transition-colors px-6 py-2 rounded-lg font-medium text-gray-300">
               Step Forward
             </button>
          )}
        </div>
      </header>

      {status?.models_loaded && metrics && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-6">
            {/* KPI Cards */}
            <div className="glass-card p-6 flex flex-col justify-between">
              <div className="flex justify-between items-start text-gray-400 mb-2 font-medium">
                Active Instances
                <Server size={20} className="text-blue-400"/>
              </div>
              <div className="text-4xl font-bold">{metrics.instances}</div>
              <div className="mt-4 text-xs text-primary bg-primary/10 px-2 py-1 rounded inline-flex self-start">Current Scale</div>
            </div>

            <div className="glass-card p-6 flex flex-col justify-between">
              <div className="flex justify-between items-start text-gray-400 mb-2 font-medium">
                Predicted CPU
                <Activity size={20} className="text-orange-400"/>
              </div>
              <div className="text-4xl font-bold">{metrics.predicted_cpu?.toFixed(1)}%</div>
              <div className="mt-4 text-xs text-gray-400 border border-border px-2 py-1 rounded inline-flex self-start">Actual: {metrics.actual_cpu?.toFixed(1)}%</div>
            </div>

            <div className="glass-card p-6 flex flex-col justify-between relative overflow-hidden">
              <div className={`absolute top-0 left-0 w-1 h-full ${metrics.failure_prob > 0.7 ? 'bg-danger' : 'bg-success'}`}></div>
              <div className="flex justify-between items-start text-gray-400 mb-2 font-medium">
                Failure Risk
                <ShieldAlert size={20} className={metrics.failure_prob > 0.7 ? 'text-danger' : 'text-success'}/>
              </div>
              <div className="text-4xl font-bold">{(metrics.failure_prob * 100)?.toFixed(1)}%</div>
              <div className="mt-4 text-xs text-gray-400">XGBoost Ensemble</div>
            </div>

            <div className="glass-card p-6 flex flex-col justify-between">
              <div className="flex justify-between items-start text-gray-400 mb-2 font-medium">
                Decision Cost
                <DollarSign size={20} className="text-green-400"/>
              </div>
              <div className="text-4xl font-bold">${metrics.cost?.toFixed(2)}</div>
              <div className="mt-4 text-xs text-gray-400">Per Interval</div>
            </div>

            <div className="glass-card p-6 flex flex-col justify-between relative">
              {metrics.drift_detected && <div className="absolute -top-2 -right-2 w-4 h-4 bg-danger rounded-full animate-ping"></div>}
              <div className="flex justify-between items-start text-gray-400 mb-2 font-medium">
                Decision Logic
              </div>
              <div className="text-2xl font-bold uppercase tracking-wider bg-clip-text text-transparent bg-gradient-to-r from-gray-100 to-gray-400">{metrics.action.replace('_', ' ')}</div>
              <div className={`mt-4 text-xs px-2 py-1 rounded inline-flex self-start font-medium ${metrics.drift_detected ? 'bg-danger/20 text-danger border border-danger/30' : 'bg-success/20 text-success border border-success/30'}`}>
                {metrics.drift_detected ? 'Model Drift Detected' : 'Optimal Path'}
              </div>
            </div>
          </div>

          <div className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-xl font-bold text-gray-200">Static Actual vs Predicted CPU Usage (From Sheet)</h3>
              <div className="flex gap-4">
                <a href="/static_chart_data.csv" download className="text-sm text-primary underline">Download CSV</a>
                <a href="/static_chart_data.xlsx" download className="text-sm text-primary underline">Download Excel (.xlsx)</a>
              </div>
            </div>
            <div className="h-80 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart data={staticData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
                  <XAxis dataKey="timestamp" tickFormatter={(t) => new Date(Number(t)).toLocaleTimeString()} stroke="#6B7280" />
                  <YAxis yAxisId="left" stroke="#6B7280" domain={[0, 100]} />
                  <Tooltip contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363D', color: '#fff' }} />
                  <Legend />
                  <Line yAxisId="left" type="monotone" dataKey="actual_cpu" stroke="#4F46E5" strokeWidth={2} dot={false} name="Actual CPU" />
                  <Line yAxisId="left" type="monotone" dataKey="predicted_cpu" stroke="#F59E0B" strokeWidth={2} dot={false} strokeDasharray="5 5" name="Predicted CPU" />
                  <Bar dataKey="instances" fill="#3B82F6" opacity={0.25} name="Instances" />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
            <p className="mt-4 text-sm text-gray-400">This chart is static and generated from the provided sheet. Use the download link to open the source in Excel.</p>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
