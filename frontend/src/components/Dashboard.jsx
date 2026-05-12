"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import {
  Activity,
  AlertTriangle,
  Wifi,
  WifiOff,
  TrendingUp,
  TrendingDown,
  ServerCrash,
  ArrowDownToLine,
} from "lucide-react";

const SYMBOL = "CRASH500";

export default function Dashboard() {
  const [ticks, setTicks] = useState([]);
  const [probability, setProbability] = useState(0);
  const [signalLevel, setSignalLevel] = useState("idle");
  const [recommendedAction, setRecommendedAction] = useState("WAIT");
  const [alerts, setAlerts] = useState([]);
  const [warnings60, setWarnings60] = useState([]);

  const [derivConnected, setDerivConnected] = useState(false);
  const [backendConnected, setBackendConnected] = useState(true);

  const playedSoundRef = useRef(false);
  const workerRef = useRef(null);

  useEffect(() => {
    workerRef.current = new Worker("/derivWorker.js");

    workerRef.current.onmessage = (e) => {
      const { type, payload } = e.data;
      if (type === "STATUS") {
        setDerivConnected(payload === "CONNECTED");
      } else if (type === "TICK") {
        setDerivConnected(true);
        const newTick = {
          ...payload,
          time: new Date(payload.timestamp * 1000).toLocaleTimeString(),
        };
        setTicks((prev) => {
          const updated = [...prev, newTick];
          // Fenêtre glissante de ~60 ticks max
          return updated.length > 60 ? updated.slice(updated.length - 60) : updated;
        });
      }
    };

    workerRef.current.postMessage({ type: "START" });

    return () => {
      workerRef.current.postMessage({ type: "STOP" });
      workerRef.current.terminate();
    };
  }, []);

  // Appel backend à chaque nouveau tick (assez de data pour la vélocité)
  useEffect(() => {
    if (ticks.length < 2) return;

    const sendPrediction = async () => {
      try {
        const payload = {
          symbol: SYMBOL,
          ticks: ticks.map((t) => ({ timestamp: t.timestamp, price: t.price })),
        };

        const res = await fetch("http://127.0.0.1:8000/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!res.ok) throw new Error("Backend error");
        const data = await res.json();

        setProbability(data.crash_probability);
        setSignalLevel(data.signal_level ?? "idle");
        setRecommendedAction(data.recommended_action ?? "WAIT");
        setBackendConnected(true);

        const latestPrice = ticks[ticks.length - 1].price;

        // Prémice : zone 60-79 % (avertissement précoce de crash)
        if (data.crash_probability >= 60 && data.crash_probability < 80) {
          pushWarning60(data.crash_probability, latestPrice);
        }

        // Alerte forte : >= 80 % — son + historique haut risque
        if (data.crash_probability >= 80) {
          if (!playedSoundRef.current) {
            playedSoundRef.current = true;
            const audio = new Audio("/son.wav");
            audio.play().catch((e) => {
              console.warn("Son bloqué : interaction utilisateur requise.", e.message);
            });
          }
        } else {
          playedSoundRef.current = false;
        }

        if (data.alert) {
          pushAlert(data.crash_probability, latestPrice);
        }
      } catch (err) {
        console.error("Failed to fetch prediction:", err);
        setBackendConnected(false);
      }
    };

    sendPrediction();
  }, [ticks]);

  const pushAlert = (prob, price) => {
    const newAlert = {
      id: Date.now(),
      time: new Date().toLocaleTimeString(),
      probability: prob,
      price: price,
    };

    setAlerts((prev) => {
      if (prev.length > 0 && newAlert.id - prev[0].id < 5000) return prev;
      return [newAlert, ...prev].slice(0, 10);
    });
  };

  const pushWarning60 = (prob, price) => {
    const newWarning = {
      id: Date.now(),
      time: new Date().toLocaleTimeString(),
      probability: prob,
      price: price,
    };

    setWarnings60((prev) => {
      if (prev.length > 0 && newWarning.id - prev[0].id < 5000) return prev;
      return [newWarning, ...prev].slice(0, 10);
    });
  };

  const isHighRisk = probability >= 80;
  const isMediumRisk = probability >= 50 && probability < 80;

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100 p-4 md:p-8 font-sans selection:bg-red-500/30">
      {/* Header */}
      <header className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between pb-8 border-b border-zinc-800/50 mb-8">
        <div className="flex items-center gap-3">
          <div className="p-3 bg-zinc-900 rounded-2xl border border-zinc-800 shadow-xl">
            <ArrowDownToLine className="w-8 h-8 text-red-400" />
          </div>
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-zinc-100 to-zinc-400 bg-clip-text text-transparent">
              La Tour de Contrôle
            </h1>
            <p className="text-zinc-500 text-sm font-medium">Flux IA en temps réel — CRASH 500</p>
          </div>
        </div>

        <div className="flex items-center gap-6 mt-4 md:mt-0">
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-400">Deriv API</span>
            {derivConnected ? (
              <Wifi className="w-5 h-5 text-emerald-400 animate-pulse" />
            ) : (
              <WifiOff className="w-5 h-5 text-red-500" />
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="text-sm text-zinc-400">IA Backend</span>
            {backendConnected ? (
              <ServerCrash className="w-5 h-5 text-emerald-400" />
            ) : (
              <ServerCrash className="w-5 h-5 text-red-500 animate-pulse" />
            )}
          </div>
        </div>
      </header>

      {/* Main Grid */}
      <main className="max-w-7xl mx-auto grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column (Probability & Status) */}
        <div className="flex flex-col gap-6 lg:col-span-1">
          {/* Probability Card */}
          <div
            className={`p-8 rounded-3xl border shadow-2xl backdrop-blur-xl relative overflow-hidden transition-all duration-500 ${
              isHighRisk
                ? "bg-red-500/10 border-red-500/30 shadow-red-500/20"
                : isMediumRisk
                ? "bg-yellow-500/5 border-yellow-500/20 shadow-yellow-500/10"
                : "bg-zinc-900/50 border-zinc-800"
            }`}
          >
            <div className="absolute top-0 right-0 w-32 h-32 bg-gradient-to-br from-white/5 to-transparent rounded-full blur-2xl -mt-10 -mr-10"></div>

            <h2 className="text-lg font-medium text-zinc-400 mb-2">Probabilité de Crash</h2>
            <div className="flex items-baseline gap-2">
              <span
                className={`text-7xl font-sans font-bold tracking-tight transition-colors duration-300 ${
                  isHighRisk ? "text-red-400" : isMediumRisk ? "text-yellow-400" : "text-emerald-400"
                }`}
              >
                {probability.toFixed(1)}
              </span>
              <span className="text-3xl text-zinc-500">%</span>
            </div>

            <div className="mt-4 text-xs uppercase tracking-wider text-zinc-500">
              Signal : <span className="text-zinc-300 font-semibold">{signalLevel}</span>
              <span className="mx-2">·</span>
              Action : <span className="text-zinc-300 font-semibold">{recommendedAction}</span>
            </div>

            {isHighRisk && (
              <div className="mt-6 flex items-center gap-2 text-red-400 bg-red-400/10 px-4 py-2 rounded-lg border border-red-400/20 animate-pulse">
                <AlertTriangle className="w-5 h-5" />
                <span className="font-semibold text-sm">CRASH IMMINENT — SELL</span>
              </div>
            )}
          </div>

          {/* Current Market Info */}
          <div className="p-6 rounded-3xl bg-zinc-900/50 border border-zinc-800 shadow-xl backdrop-blur-xl flex-1">
            <h2 className="text-lg font-medium text-zinc-400 mb-6">Snapshot du Marché</h2>
            <div className="space-y-4">
              <div>
                <p className="text-sm text-zinc-500">Prix Actuel</p>
                <p className="text-3xl font-medium font-mono text-zinc-100 flex items-center gap-2">
                  {ticks.length > 0 ? ticks[ticks.length - 1].price.toFixed(3) : "---"}
                  {ticks.length > 1 && ticks[ticks.length - 1].price > ticks[ticks.length - 2].price && (
                    <TrendingUp className="w-5 h-5 text-emerald-400" />
                  )}
                  {ticks.length > 1 && ticks[ticks.length - 1].price < ticks[ticks.length - 2].price && (
                    <TrendingDown className="w-5 h-5 text-red-400" />
                  )}
                </p>
              </div>
              <div>
                <p className="text-sm text-zinc-500">Ticks Analysés (Fenêtre)</p>
                <p className="text-xl font-medium text-zinc-200">{Math.min(ticks.length, 50)} / 50</p>
              </div>
            </div>
          </div>
        </div>

        {/* Right Column (Chart & Alerts) */}
        <div className="flex flex-col gap-6 lg:col-span-2">
          {/* Chart */}
          <div className="h-96 p-6 rounded-3xl bg-zinc-900/50 border border-zinc-800 shadow-xl backdrop-blur-xl">
            <h2 className="text-lg font-medium text-zinc-400 mb-4 flex items-center justify-between">
              <span>Flux Tick par Tick</span>
              <span className="text-xs px-2 py-1 bg-zinc-800 rounded text-zinc-300">{SYMBOL}</span>
            </h2>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={ticks}>
                  <defs>
                    <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f87171" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="#f87171" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="#27272a" vertical={false} />
                  <XAxis dataKey="time" stroke="#52525b" fontSize={12} tickMargin={10} minTickGap={30} />
                  <YAxis
                    stroke="#52525b"
                    domain={["auto", "auto"]}
                    fontSize={12}
                    tickFormatter={(val) => val.toFixed(1)}
                    width={60}
                  />
                  <Tooltip
                    contentStyle={{ backgroundColor: "#18181b", borderColor: "#27272a", borderRadius: "12px" }}
                    itemStyle={{ color: "#e4e4e7" }}
                  />
                  <Line
                    type="monotone"
                    dataKey="price"
                    stroke="#f87171"
                    strokeWidth={2}
                    dot={false}
                    activeDot={{ r: 6, fill: "#ef4444", stroke: "#b91c1c" }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Alert History */}
          <div className="flex-1 p-6 rounded-3xl bg-zinc-900/50 border border-zinc-800 shadow-xl backdrop-blur-xl">
            <h2 className="text-lg font-medium text-zinc-400 mb-4">Historique des Alertes (&gt;80%)</h2>
            {alerts.length === 0 ? (
              <div className="h-40 flex items-center justify-center text-zinc-600 italic">
                Aucune alerte de crash détectée pour le moment.
              </div>
            ) : (
              <div className="space-y-3">
                {alerts.map((alert) => (
                  <div
                    key={alert.id}
                    className="flex items-center justify-between p-4 bg-zinc-800/30 rounded-xl border border-zinc-800/80 hover:bg-zinc-800/50 transition-colors"
                  >
                    <div className="flex items-center gap-4">
                      <div className="p-2 bg-red-500/10 rounded-lg">
                        <AlertTriangle className="w-5 h-5 text-red-400" />
                      </div>
                      <div>
                        <p className="font-semibold text-zinc-200">{alert.time}</p>
                        <p className="text-sm text-zinc-500 font-mono">Prix : {alert.price.toFixed(3)}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-lg font-bold text-red-400">{alert.probability.toFixed(1)}%</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* Historique 60% (Full Width Bottom) */}
        <div className="lg:col-span-3 p-6 rounded-3xl bg-zinc-900/50 border border-zinc-800 shadow-xl backdrop-blur-xl mt-2">
          <h2 className="text-lg font-medium text-zinc-400 mb-4">Historique des Prémices (60-79%)</h2>
          {warnings60.length === 0 ? (
            <div className="h-20 flex items-center justify-center text-zinc-600 italic">
              Aucun prémice détecté pour le moment.
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
              {warnings60.map((w) => (
                <div
                  key={w.id}
                  className="flex flex-col p-4 bg-yellow-500/5 rounded-xl border border-yellow-500/20 hover:bg-yellow-500/10 transition-colors"
                >
                  <div className="flex items-center gap-2 mb-2">
                    <Activity className="w-4 h-4 text-yellow-400" />
                    <span className="font-semibold text-zinc-200">{w.time}</span>
                  </div>
                  <div className="flex justify-between items-end">
                    <span className="text-sm text-zinc-500 font-mono">{w.price.toFixed(3)}</span>
                    <span className="text-lg font-bold text-yellow-400">{w.probability.toFixed(1)}%</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
