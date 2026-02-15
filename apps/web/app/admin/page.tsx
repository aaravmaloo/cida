"use client";

import { useMutation, useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { adminLogin, fetchAnalytics } from "@/lib/api";

export default function AdminPage() {
  const [passkey, setPasskey] = useState("");
  const [token, setToken] = useState<string>("");

  const login = useMutation({
    mutationFn: () => adminLogin(passkey),
    onSuccess: (data) => setToken(data.access_token),
  });

  const analytics = useQuery({
    queryKey: ["analytics", token],
    queryFn: () => fetchAnalytics(token),
    enabled: Boolean(token),
  });

  return (
    <main className="mx-auto my-8 max-w-4xl rounded-2xl border border-cida-line bg-cida-panel p-8 shadow-panel">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="font-heading text-3xl font-semibold">Admin Dashboard</h1>
        <Link href="/" className="rounded-lg border border-cida-line px-3 py-2 text-sm hover:bg-white">
          Back to Detector
        </Link>
      </div>

      {!token ? (
        <div className="max-w-md space-y-3 rounded-xl border border-cida-line bg-white p-4">
          <p className="text-sm text-cida-mute">Enter admin passkey</p>
          <input
            type="password"
            value={passkey}
            onChange={(event) => setPasskey(event.target.value)}
            className="w-full rounded-lg border border-cida-line p-2"
          />
          <button
            onClick={() => login.mutate()}
            disabled={login.isPending}
            className="rounded-xl bg-cida-accent px-4 py-2 font-semibold text-white"
          >
            {login.isPending ? "Authenticating..." : "Login"}
          </button>
          {login.error && <p className="text-sm text-red-600">{(login.error as Error).message}</p>}
        </div>
      ) : (
        <div className="rounded-xl border border-cida-line bg-white p-4">
          <p className="mb-4 text-sm text-cida-mute">Authenticated</p>

          {analytics.isLoading && <p>Loading analytics...</p>}
          {analytics.error && <p className="text-red-600">{(analytics.error as Error).message}</p>}

          {analytics.data && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
              <div className="rounded-lg border border-cida-line p-3">
                <p className="text-xs text-cida-mute">Total Analyses</p>
                <p className="text-2xl font-bold">{analytics.data.total_analyses}</p>
              </div>
              <div className="rounded-lg border border-cida-line p-3">
                <p className="text-xs text-cida-mute">Total Humanizations</p>
                <p className="text-2xl font-bold">{analytics.data.total_humanizations}</p>
              </div>
              <div className="rounded-lg border border-cida-line p-3">
                <p className="text-xs text-cida-mute">P95 Latency</p>
                <p className="text-2xl font-bold">{analytics.data.p95_latency_ms} ms</p>
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  );
}

