import { DetectorDashboard } from "@/components/detector-dashboard";
import { TopNav } from "@/components/top-nav";

export default function HomePage() {
  return (
    <div className="min-h-screen px-4 py-6">
      <div className="mx-auto max-w-[1280px] overflow-hidden rounded-2xl border border-cida-line bg-cida-panel shadow-panel">
        <TopNav />
        <DetectorDashboard />
      </div>
    </div>
  );
}

