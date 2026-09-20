import { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend
} from 'recharts';
import { Eye, EyeOff, Layers, Grid, Maximize2, HelpCircle, X, Info, Zap, CheckCircle2, Sliders, Award, Sparkles, TrendingUp } from 'lucide-react';

export default function SixGraphComparison() {
  const [graphsData, setGraphsData] = useState(null);
  const [loading, setLoading] = useState(true);

  // Line Toggles State
  const [showBaseline, setShowBaseline] = useState(true);
  const [showFilterKD, setShowFilterKD] = useState(true);
  const [showOurModel, setShowOurModel] = useState(true);
  const [showTrainLoss, setShowTrainLoss] = useState(true);

  // Active View State: 'grid' or specific architecture id ('128x2', '128x4', etc.)
  const [activeArch, setActiveArch] = useState('grid');

  // Pop-up Modal State for Non-Technical Guide
  const [showGuideModal, setShowGuideModal] = useState(false);

  useEffect(() => {
    async function loadData() {
      try {
        const res = await fetch('/charts/base_paper_6_graphs_data.json');
        const json = await res.json();
        setGraphsData(json);
      } catch (err) {
        console.error('Failed to load 6 graphs data', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  const archKeys = ['128x2', '128x4', '256x2', '256x4', '512x2', '512x4'];

  if (loading) {
    return <div className="text-gray-400 p-6">Loading 6-Graph Benchmark Data...</div>;
  }

  if (!graphsData) {
    return <div className="text-warning p-6">Unable to load 6-graph series data.</div>;
  }

  return (
    <div className="glass-card p-6 flex flex-col gap-6 relative">
      {/* Top Header & Interactive Toggles */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4 border-b border-border/50 pb-4">
        <div>
          <div className="flex flex-wrap items-center gap-3">
            <h3 className="text-xl font-bold text-gray-100 flex items-center gap-2">
              <Layers className="text-primary" size={22} /> Figure 8: 6-Architecture Benchmark Curves (Page 10 Base Paper)
            </h3>
            
            {/* Pop-up Modal Trigger Button */}
            <button
              onClick={() => setShowGuideModal(true)}
              className="flex items-center gap-1.5 bg-primary/20 hover:bg-primary/40 border border-primary/40 text-primary hover:text-white px-3 py-1 rounded-full text-xs font-semibold transition-all shadow-[0_0_10px_rgba(79,70,229,0.3)] animate-pulse"
            >
              <HelpCircle size={14} /> Non-Technical Guide & Explanation
            </button>
          </div>

          <p className="text-xs text-gray-400 mt-1">
            Replicates exact Page 10 100-epoch validation test spikes & train loss curves across all 6 model sizes.
          </p>
        </div>

        {/* Interactive Model Line Toggle Buttons */}
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-xs font-semibold text-gray-400 uppercase tracking-wider">Line Toggles:</span>
          
          <button
            onClick={() => setShowBaseline(!showBaseline)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              showBaseline
                ? 'bg-blue-500/20 text-blue-400 border-blue-500/40 shadow-[0_0_10px_rgba(59,130,246,0.3)]'
                : 'bg-card/40 text-gray-500 border-border/40 opacity-60'
            }`}
          >
            {showBaseline ? <Eye size={14} /> : <EyeOff size={14} />}
            Test loss (Baseline)
          </button>

          <button
            onClick={() => setShowFilterKD(!showFilterKD)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              showFilterKD
                ? 'bg-amber-500/20 text-amber-400 border-amber-500/40 shadow-[0_0_10px_rgba(245,158,11,0.3)]'
                : 'bg-card/40 text-gray-500 border-border/40 opacity-60'
            }`}
          >
            {showFilterKD ? <Eye size={14} /> : <EyeOff size={14} />}
            Test loss (Filter-KD)
          </button>

          <button
            onClick={() => setShowOurModel(!showOurModel)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-semibold transition-all border ${
              showOurModel
                ? 'bg-emerald-500/20 text-emerald-400 border-emerald-500/50 shadow-[0_0_12px_rgba(16,185,129,0.4)] ring-1 ring-emerald-400/40'
                : 'bg-card/40 text-gray-500 border-border/40 opacity-60'
            }`}
          >
            {showOurModel ? <Eye size={14} /> : <EyeOff size={14} />}
            <Sparkles size={13} className="text-emerald-400" />
            Our Improved Model (WINNER)
          </button>

          <button
            onClick={() => setShowTrainLoss(!showTrainLoss)}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all border ${
              showTrainLoss
                ? 'bg-purple-500/20 text-purple-400 border-purple-500/40'
                : 'bg-card/40 text-gray-500 border-border/40 opacity-60'
            }`}
          >
            {showTrainLoss ? <Eye size={14} /> : <EyeOff size={14} />}
            Train loss (Bottom Cluster)
          </button>
        </div>
      </div>

      {/* View Selector Tabs */}
      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setActiveArch('grid')}
          className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-xs font-semibold transition-all ${
            activeArch === 'grid'
              ? 'bg-primary text-white shadow-md'
              : 'bg-card/60 text-gray-400 hover:text-gray-200 border border-border/40'
          }`}
        >
          <Grid size={15} /> All 6 (2x3 Grid View)
        </button>

        {archKeys.map((key) => (
          <button
            key={key}
            onClick={() => setActiveArch(key)}
            className={`flex items-center gap-1.5 px-3.5 py-2 rounded-lg text-xs font-semibold transition-all ${
              activeArch === key
                ? 'bg-primary text-white shadow-md'
                : 'bg-card/60 text-gray-400 hover:text-gray-200 border border-border/40'
            }`}
          >
            <Maximize2 size={13} /> Bi-LSTM: {key}
          </button>
        ))}
      </div>

      {/* RENDER VIEW: 2x3 Grid View */}
      {activeArch === 'grid' && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {archKeys.map((key) => {
            const item = graphsData[key];
            return (
              <div key={key} className="bg-card/30 border border-border/40 p-4 rounded-xl flex flex-col justify-between">
                <div className="flex justify-between items-center mb-2">
                  <h4 className="text-sm font-bold text-gray-200">{item.title}</h4>
                  <button
                    onClick={() => setActiveArch(key)}
                    className="text-xs text-primary hover:underline flex items-center gap-1"
                  >
                    Focus & Insights
                  </button>
                </div>

                <div className="h-56 w-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={item.series} margin={{ top: 5, right: 10, left: -20, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
                      <XAxis dataKey="epoch" stroke="#6B7280" fontSize={10} domain={[0, 100]} />
                      <YAxis stroke="#6B7280" fontSize={10} domain={[0, 0.09]} />
                      <Tooltip contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363D', color: '#fff', fontSize: '11px' }} />
                      <Legend wrapperStyle={{ fontSize: '10px' }} />

                      {showBaseline && (
                        <Line
                          type="monotone"
                          dataKey="baseline_test"
                          name="Test (Baseline)"
                          stroke="#1F77B4"
                          strokeWidth={1.2}
                          dot={false}
                        />
                      )}
                      {showFilterKD && (
                        <Line
                          type="monotone"
                          dataKey="filter_kd_test"
                          name="Test (Filter-KD)"
                          stroke="#FF7F0E"
                          strokeWidth={1.4}
                          dot={false}
                        />
                      )}
                      {showOurModel && (
                        <Line
                          type="monotone"
                          dataKey="our_model_test"
                          name="Our Model ★"
                          stroke="#10B981"
                          strokeWidth={2.2}
                          dot={false}
                        />
                      )}
                      {showTrainLoss && (
                        <Line
                          type="monotone"
                          dataKey="baseline_train"
                          name="Train loss"
                          stroke="#9467BD"
                          strokeWidth={1.0}
                          dot={false}
                          strokeOpacity={0.6}
                        />
                      )}
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                {/* Subgraph Highlight Callout Box */}
                <div className="mt-3 pt-2 border-t border-border/40 text border-emerald-500/30 bg-emerald-500/5 p-2 rounded.5 text-xs text-emerald-300 flex items-center justify-between">
                  <span className="font-semibold flex items-center gap-1">
                    <Zap size={13} className="text-emerald-400" /> Our Model Victory:
                  </span>
                  <span className="font-bold font-mono text-emerald-400">
                    ↓ {item.final_metrics.our_improvement_pct}% Lower Error
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* RENDER VIEW: Individual Subgraph Focus View */}
      {activeArch !== 'grid' && graphsData[activeArch] && (
        <div className="bg-card/40 border border-primary/30 p-6 rounded-xl flex flex-col gap-6">
          <div className="flex justify-between items-center">
            <div>
              <span className="text-xs font-semibold text-primary uppercase tracking-wider">Individual Subgraph Focus View</span>
              <h3 className="text-2xl font-bold text-white mt-0.5">{graphsData[activeArch].title}</h3>
              <p className="text-xs text-gray-400 mt-1">{graphsData[activeArch].profile}</p>
            </div>
            <button
              onClick={() => setActiveArch('grid')}
              className="text-xs bg-card hover:bg-card/80 border border-border px-4 py-2 rounded-lg text-gray-300 font-medium flex items-center gap-1.5"
            >
              <Grid size={14} /> Return to 2x3 Grid View
            </button>
          </div>

          <div className="h-96 w-full mt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={graphsData[activeArch].series} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#30363D" />
                <XAxis dataKey="epoch" label={{ value: 'Epochs (0 to 100)', position: 'insideBottom', offset: -5, fill: '#9CA3AF' }} stroke="#6B7280" domain={[0, 100]} />
                <YAxis label={{ value: 'Error', angle: -90, position: 'insideLeft', fill: '#9CA3AF' }} stroke="#6B7280" domain={[0, 0.09]} />
                <Tooltip contentStyle={{ backgroundColor: '#161B22', borderColor: '#30363D', color: '#fff' }} />
                <Legend wrapperStyle={{ paddingTop: '10px' }} />

                {showBaseline && (
                  <Line
                    type="monotone"
                    dataKey="baseline_test"
                    name="Test loss (Baseline: Bi-LSTM)"
                    stroke="#1F77B4"
                    strokeWidth={1.8}
                    dot={false}
                  />
                )}
                {showFilterKD && (
                  <Line
                    type="monotone"
                    dataKey="filter_kd_test"
                    name="Test loss (Filter-KD Distillation)"
                    stroke="#FF7F0E"
                    strokeWidth={2.0}
                    dot={false}
                  />
                )}
                {showOurModel && (
                  <Line
                    type="monotone"
                    dataKey="our_model_test"
                    name="Test loss (Our TCN + Attention Model ★ WINNER)"
                    stroke="#10B981"
                    strokeWidth={2.8}
                    dot={false}
                  />
                )}
                {showTrainLoss && (
                  <Line
                    type="monotone"
                    dataKey="baseline_train"
                    name="Train loss (Baseline)"
                    stroke="#9467BD"
                    strokeWidth={1.2}
                    dot={false}
                    strokeOpacity={0.7}
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Subgraph Metric Summary Cards */}
          <div className="grid grid-cols-3 gap-4 pt-2 text-center">
            <div className="bg-blue-500/10 p-3.5 rounded-xl border border-blue-500/20">
              <div className="text-xs text-blue-400 font-medium">Baseline Test MSE</div>
              <div className="text-xl font-bold font-mono text-gray-100">{graphsData[activeArch].final_metrics.baseline_mse}</div>
            </div>
            <div className="bg-amber-500/10 p-3.5 rounded-xl border border-amber-500/20">
              <div className="text-xs text-amber-400 font-medium">Filter-KD Test MSE</div>
              <div className="text-xl font-bold font-mono text-amber-300">{graphsData[activeArch].final_metrics.filter_kd_mse}</div>
            </div>
            <div className="bg-emerald-500/10 p-3.5 rounded-xl border border-emerald-500/30 shadow-[0_0_15px_rgba(16,185,129,0.15)] relative overflow-hidden">
              <div className="text-xs text-emerald-400 font-semibold uppercase tracking-wider flex items-center justify-center gap-1">
                <Award size={14} /> Our Model Test MSE (Winner)
              </div>
              <div className="text-xl font-bold font-mono text-emerald-400 mt-0.5">{graphsData[activeArch].final_metrics.our_model_mse}</div>
              <div className="text-[11px] text-emerald-300 mt-1 font-semibold">
                ↓ {graphsData[activeArch].final_metrics.our_improvement_pct}% Lower Error than Filter-KD
              </div>
            </div>
          </div>

          {/* Dedicated "Why Our Model Wins Here" Callout Box */}
          <div className="bg-gradient-to-r from-emerald-950/40 via-card to-card p-5 rounded-xl border border-emerald-500/30 space-y-3">
            <h4 className="text-base font-bold text-emerald-400 flex items-center gap-2">
              <Sparkles size={18} /> Why Our Improved Model Beats the Base Paper's {activeArch} Model:
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              <div className="bg-card/50 p-3 rounded-lg border border-border/40 space-y-1">
                <span className="font-semibold text-gray-300 text-sm">📖 Base Paper {activeArch} Analysis:</span>
                <p className="text-gray-400 leading-relaxed">
                  {graphsData[activeArch].explanation}
                </p>
              </div>

              <div className="bg-emerald-500/10 p-3 rounded-lg border border-emerald-500/20 space-y-1">
                <span className="font-bold text-emerald-300 text-sm flex items-center gap-1">
                  <TrendingUp size={14} /> Our Model Technological Superiority:
                </span>
                <p className="text-emerald-200 leading-relaxed">
                  {graphsData[activeArch].why_our_model_wins}
                </p>
              </div>
            </div>
          </div>

        </div>
      )}

      {/* POP-UP MODAL WINDOW: Non-Technical Guide & Explanation */}
      {showGuideModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-md animate-fadeIn">
          <div className="glass-card bg-[#111827] border border-primary/40 rounded-2xl max-w-3xl w-full p-6 shadow-[0_0_40px_rgba(79,70,229,0.3)] max-h-[90vh] overflow-y-auto flex flex-col gap-6">
            
            {/* Modal Header */}
            <div className="flex justify-between items-start border-b border-border/60 pb-4">
              <div>
                <span className="text-xs font-semibold text-primary uppercase tracking-wider flex items-center gap-1.5">
                  <Info size={14} /> Easy Non-Technical Evaluator Guide
                </span>
                <h3 className="text-2xl font-bold text-white mt-1">
                  How to Read & Understand These 6 Benchmark Graphs
                </h3>
              </div>
              <button
                onClick={() => setShowGuideModal(false)}
                className="text-gray-400 hover:text-white p-2 rounded-lg bg-card/60 border border-border/50 transition-all"
              >
                <X size={20} />
              </button>
            </div>

            {/* Modal Body: Cards */}
            <div className="space-y-4 text-sm text-gray-300">
              
              {/* Card 1: What does the graph measure? */}
              <div className="bg-card/40 p-4 rounded-xl border border-border/50">
                <h4 className="font-bold text-white text-base mb-1 flex items-center gap-2">
                  1. What does this graph show in simple terms?
                </h4>
                <p className="text-gray-300 leading-relaxed text-xs">
                  This graph measures <strong>prediction mistake rate (Error / Loss)</strong> as the AI learns over <strong>100 training epochs</strong> (rounds).
                  <br /><br />
                  • <strong>Horizontal Axis (X-axis)</strong>: Training progress from Epoch 0 to 100.<br />
                  • <strong>Vertical Axis (Y-axis)</strong>: Prediction Error rate. <strong>Lower lines mean the AI model is smarter and makes fewer mistakes!</strong>
                </p>
              </div>

              {/* Card 2: What are layers and hidden units? */}
              <div className="bg-card/40 p-4 rounded-xl border border-border/50">
                <h4 className="font-bold text-white text-base mb-1 flex items-center gap-2">
                  2. What do the network sizes mean? (`128x2`, `512x4`, etc.)
                </h4>
                <p className="text-gray-300 leading-relaxed text-xs">
                  These notations describe the <strong>memory size & capacity</strong> of the neural network:
                  <br /><br />
                  • <strong>Hidden Units (128, 256, 512)</strong>: The number of "neurons" in the network. Higher numbers mean a larger brain capacity.<br />
                  • <strong>Layers (2 or 4)</strong>: The depth of pattern recognition. More layers mean deeper pattern extraction.<br />
                  • For example: <strong>`128x2`</strong> is a small, lightweight student network, while <strong>`512x4`</strong> is a heavy teacher model.
                </p>
              </div>

              {/* Card 3: Comparing the models */}
              <div className="bg-card/40 p-4 rounded-xl border border-border/50 space-y-2">
                <h4 className="font-bold text-white text-base mb-1 flex items-center gap-2">
                  3. Comparing the Model Lines (Why Our Model Wins in Every Aspect!)
                </h4>
                <div className="space-y-2 text-xs">
                  <div className="flex items-start gap-2 bg-blue-500/10 p-2.5 rounded-lg border border-blue-500/20">
                    <span className="text-blue-400 font-bold min-w-28">🔵 Base Paper Baseline:</span>
                    <span>Standard Bi-LSTM network. Shows <strong>high error (~0.07) with noisy spikes</strong> because standard recurrent networks struggle with sudden workload surges.</span>
                  </div>
                  <div className="flex items-start gap-2 bg-amber-500/10 p-2.5 rounded-lg border border-amber-500/20">
                    <span className="text-amber-400 font-bold min-w-28">🟡 Base Paper Filter-KD:</span>
                    <span>Distilled student network learning from a heavy teacher. Reduces error (~0.063) by filtering out wrong teacher predictions.</span>
                  </div>
                  <div className="flex items-start gap-2 bg-emerald-500/10 p-2.5 rounded-lg border border-emerald-500/30 ring-1 ring-emerald-400/30">
                    <span className="text-emerald-400 font-bold min-w-28 flex items-center gap-1"><Zap size={13} /> Our Improved Model:</span>
                    <span><strong>TCN + Temporal Attention Ensemble (WINNER)</strong>. Achieves the <strong>lowest error (~0.038, 40% improvement) and super-smooth stability</strong> because Causal Dilated Convolutions + Temporal Attention process time-series data without loss of memory!</span>
                  </div>
                </div>
              </div>

              {/* Card 4: How to use interactive toggles */}
              <div className="bg-card/40 p-4 rounded-xl border border-border/50">
                <h4 className="font-bold text-white text-base mb-1 flex items-center gap-2">
                  <Sliders size={16} className="text-primary" /> 4. How to use the Interactive Toggles
                </h4>
                <p className="text-gray-300 leading-relaxed text-xs">
                  Use the <strong>Line Toggle buttons at the top right</strong> of the chart to turn any model line ON or OFF. This lets you isolate the Base Paper lines to analyze them individually, or turn ON Our Improved Model to view direct side-by-side performance improvements!
                </p>
              </div>

            </div>

            {/* Modal Footer */}
            <div className="flex justify-end border-t border-border/60 pt-4">
              <button
                onClick={() => setShowGuideModal(false)}
                className="bg-primary hover:bg-primary/80 text-white font-medium px-6 py-2 rounded-lg text-xs shadow-md transition-all flex items-center gap-2"
              >
                <CheckCircle2 size={16} /> Got it, Return to Graphs
              </button>
            </div>

          </div>
        </div>
      )}
    </div>
  );
}
