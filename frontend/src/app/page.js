"use client";

import dynamic from "next/dynamic";

// Dashboard contient un Web Worker + Recharts (besoin du DOM),
// on évite donc le rendu SSR pour ne pas polluer le build avec des warnings de taille.
const Dashboard = dynamic(() => import("@/components/Dashboard"), { ssr: false });

export default function Home() {
  return (
    <main className="min-h-screen bg-zinc-950">
      <Dashboard />
    </main>
  );
}
