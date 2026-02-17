"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { AnalyticsSummary, adminLogin, fetchAnalytics } from "@/lib/api";

export default function AdminPage() {
  const [passkey, setPasskey] = useState("");
  const [token, setToken] = useState<string>("");
  const [isAuthenticating, setIsAuthenticating] = useState(false);
  const [authError, setAuthError] = useState("");
  const [analytics, setAnalytics] = useState<AnalyticsSummary | null>(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);
  const [analyticsError, setAnalyticsError] = useState("");

  const loadAnalytics = useCallback(async (authToken: string) => {
    setAnalyticsLoading(true);
    setAnalyticsError("");
    try {
      const data = await fetchAnalytics(authToken);
      setAnalytics(data);
    } catch (err) {
      setAnalyticsError(err instanceof Error ? err.message : "Failed to load analytics");
    } finally {
      setAnalyticsLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!token) {
      setAnalytics(null);
      setAnalyticsError("");
      setAnalyticsLoading(false);
      return;
    }
    void loadAnalytics(token);
  }, [token, loadAnalytics]);

  const handleLogin = async () => {
    setIsAuthenticating(true);
    setAuthError("");
    try {
      const data = await adminLogin(passkey);
      setToken(data.access_token);
    } catch (err) {
      setAuthError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setIsAuthenticating(false);
    }
  };

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
            onClick={handleLogin}
            disabled={isAuthenticating}
            className="rounded-xl bg-cida-accent px-4 py-2 font-semibold text-white"
          >
            {isAuthenticating ? "Authenticating..." : "Login"}
          </button>
          {authError && <p className="text-sm text-red-600">{authError}</p>}
        </div>
      ) : (
        <div className="rounded-xl border border-cida-line bg-white p-4">
          <div className="mb-4 flex items-center justify-between">
            <p className="text-sm text-cida-mute">Authenticated</p>
            <button
              type="button"
              onClick={() => void loadAnalytics(token)}
              disabled={analyticsLoading}
              className="rounded-lg border border-cida-line px-3 py-1 text-sm hover:bg-cida-panel disabled:opacity-60"
            >
              {analyticsLoading ? "Refreshing..." : "Refresh"}
            </button>
          </div>

          {analyticsLoading && <p>Loading analytics...</p>}
          {analyticsError && <p className="text-red-600">{analyticsError}</p>}

          {analytics && (
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
              <div className="rounded-lg border border-cida-line p-3">
                <p className="text-xs text-cida-mute">Total Analyses</p>
                <p className="text-2xl font-bold">{analytics.total_analyses}</p>
              </div>
              <div className="rounded-lg border border-cida-line p-3">
                <p className="text-xs text-cida-mute">P95 Latency</p>
                <p className="text-2xl font-bold">{analytics.p95_latency_ms} ms</p>
              </div>
            </div>
          )}
        </div>
      )}
    </main>
  );
}

