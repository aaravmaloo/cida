import Link from "next/link";

export function TopNav() {
  return (
    <header className="flex items-center justify-between border-b border-cida-line px-6 py-4">
      <div className="flex items-center gap-3">
        <div className="grid h-10 w-10 place-items-center rounded-xl bg-cida-accent text-lg font-bold text-white">C</div>
        <div>
          <p className="font-heading text-xl font-semibold">Cida</p>
          <p className="text-xs text-cida-mute">AI Analytics Platform</p>
        </div>
      </div>
      <nav className="flex items-center gap-2 rounded-xl bg-cida-panel p-1 text-sm">
        <Link className="rounded-lg bg-white px-4 py-2 font-semibold text-cida-accent shadow-sm" href="/">
          AI Detector
        </Link>
        <Link className="rounded-lg px-4 py-2 text-cida-mute hover:bg-white" href="/humanizer">
          Humanizer
        </Link>
        <Link className="rounded-lg px-4 py-2 text-cida-mute hover:bg-white" href="/admin">
          Admin
        </Link>
      </nav>
      <div className="text-right">
        <p className="text-sm font-semibold">Public Mode</p>
        <p className="text-xs text-cida-mute">No signup required</p>
      </div>
    </header>
  );
}

