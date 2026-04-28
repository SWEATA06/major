import React, { useEffect, useState } from 'react';
import { getStatus, getCurrentMetrics, getTimeline, runSimulationStep, trainModels } from './api';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area, ComposedChart, Bar
} from 'recharts';
import { Play, Activity, Server, AlertTriangle, CheckCircle, ShieldAlert, DollarSign } from 'lucide-react';

function App() {
  const [status, setStatus] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [loading, setLoading] = useState(true);
  const [simRunning, setSimRunning] = useState(false);

  useEffect(() => {
    fetchData();
    const interval = setInterval(() => {
      if (simRunning) triggerStep();
      else fetchData();
    }, 2000);
    return () => clearInterval(interval);
  }, [simRunning]);

  const fetchData = async () => {
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
  };

  const triggerStep = async () => {
    try {
      await runSimulationStep();
      await fetchData();
    } catch (err) {
      console.error("Simulation ended or error", err);
      setSimRunning(false);
    }
  };

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

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="glass-card p-6">
              <h3 className="text-xl font-bold mb-6 text-gray-200">Workload & Instance Tracking</h3>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={timeline} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
                    <XAxis dataKey="timestamp" tickFormatter={(t) => new Date(t).toLocaleTimeString()} stroke="#6B7280" />
                    <YAxis yAxisId="left" stroke="#6B7280" domain={[0, 100]} />
                    <YAxis yAxisId="right" orientation="right" stroke="#6B7280" />
                    <Tooltip contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363D', color: '#fff' }} />
                    <Legend />
                    <Area yAxisId="left" type="monotone" dataKey="actual_cpu" fill="#4F46E5" fillOpacity={0.1} stroke="#4F46E5" strokeWidth={2} name="Actual Load (%)" />
                    <Line yAxisId="left" type="monotone" strokeDasharray="5 5" dataKey="predicted_cpu" stroke="#F59E0B" strokeWidth={2} name="Predicted Load (%)" />
                    <Bar yAxisId="right" dataKey="instances" fill="#3B82F6" opacity={0.3} name="Instances" barSize={20} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="glass-card p-6">
              <h3 className="text-xl font-bold mb-6 text-gray-200">Failure Risk & Uncertainty</h3>
              <div className="h-80 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={timeline} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#30363D" vertical={false} />
                    <XAxis dataKey="timestamp" tickFormatter={(t) => new Date(t).toLocaleTimeString()} stroke="#6B7280" />
                    <YAxis stroke="#6B7280" />
                    <Tooltip contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363D', color: '#fff' }} />
                    <Legend />
                    <Line type="monotone" dataKey="failure_prob" stroke="#EF4444" strokeWidth={2} name="Failure Risk (0-1)" />
                    <Line type="monotone" dataKey="uncertainty" stroke="#10B981" strokeWidth={2} name="Prediction Uncertainty" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

export default App;
