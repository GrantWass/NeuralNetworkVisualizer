"use client";

import { useState } from "react";
import { X, Trophy, Star } from "lucide-react";
import useStore from "@/components/network/lib/store";
import { LeaderboardEntry } from "@/components/network/lib/store";

const DATASET_LABELS: Record<string, string> = {
  xor: "XOR",
  iris: "Iris",
  auto_mpg: "Auto MPG",
  mnist: "MNIST",
};

const DATASET_ORDER = ["xor", "iris", "auto_mpg", "mnist"];

function formatScore(dataset: string, score: number): string {
  if (dataset === "xor") return `${score} epochs`;
  if (dataset === "auto_mpg") return score.toFixed(3);
  return `${score.toFixed(1)}%`;
}

function formatDate(ts: number): string {
  const d = new Date(ts * 1000);
  const now = Date.now();
  const diff = now - d.getTime();
  const days = Math.floor(diff / 86400000);
  if (days === 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days}d ago`;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function SkeletonRow() {
  return (
    <tr>
      {[40, 80, 60, 50].map((w, i) => (
        <td key={i} className="px-3 py-2">
          <div className="h-3 bg-gray-100 rounded animate-pulse" style={{ width: w }} />
        </td>
      ))}
    </tr>
  );
}

export default function LeaderboardPanel() {
  const {
    dataset,
    leaderboard,
    leaderboardLoading,
    leaderboardSubmitting,
    submittableScore,
    xorEpochsTo100,
    fetchLeaderboard,
    submitLeaderboardScore,
    computeQualification,
    setLeaderboardOpen,
  } = useStore();

  const [activeTab, setActiveTab] = useState(dataset);
  const [username, setUsername] = useState("");
  const [submitted, setSubmitted] = useState<{ rank: number } | null>(null);
  const [submitError, setSubmitError] = useState("");

  const entries: LeaderboardEntry[] = leaderboard[activeTab] ?? [];

  const handleTabChange = (tab: string) => {
    setActiveTab(tab);
    setSubmitted(null);
    setSubmitError("");
    if (!leaderboard[tab]) fetchLeaderboard(tab);
  };

  const isCurrentDataset = activeTab === dataset;
  const score = dataset === "xor" ? xorEpochsTo100 : submittableScore;
  const { qualifies, rank: projectedRank } = isCurrentDataset && score !== null
    ? computeQualification()
    : { qualifies: false, rank: null };

  const metricInfo: Record<string, { display: string; cap: number | null }> = {
    xor:      { display: "Fewest epochs to 100%", cap: null },
    iris:     { display: "Accuracy at epoch 100", cap: 100 },
    auto_mpg: { display: "MAE at epoch 200", cap: 200 },
    mnist:    { display: "Accuracy at epoch 300", cap: 300 },
  };

  const { display: metricDisplay, cap: epochCap } = metricInfo[activeTab] ?? { display: "", cap: null };

  const handleSubmit = async () => {
    setSubmitError("");
    if (!username.trim() || !/^[a-zA-Z0-9_-]+$/.test(username.trim())) {
      setSubmitError("Letters, digits, _ and - only (1–32 chars)");
      return;
    }
    const result = await submitLeaderboardScore(username.trim());
    if (result?.accepted && result.rank !== null) {
      setSubmitted({ rank: result.rank });
    } else if (result && !result.accepted) {
      setSubmitError("Score didn't make the top 10 this time!");
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/30 p-4">
      <div className="bg-white rounded-2xl shadow-2xl w-full max-w-lg flex flex-col max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-gray-100">
          <div className="flex items-center gap-2">
            <Trophy size={16} className="text-amber-500" />
            <span className="font-semibold text-gray-900 text-sm">Leaderboard</span>
          </div>
          <button
            onClick={() => setLeaderboardOpen(false)}
            className="text-gray-400 hover:text-gray-600 transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Dataset tabs */}
        <div className="flex gap-1 px-5 pt-3 pb-1">
          {DATASET_ORDER.map((ds) => (
            <button
              key={ds}
              onClick={() => handleTabChange(ds)}
              className={`px-3 py-1 rounded-full text-xs font-medium transition-colors ${
                activeTab === ds
                  ? "bg-gray-900 text-white"
                  : "bg-gray-100 text-gray-500 hover:bg-gray-200"
              }`}
            >
              {DATASET_LABELS[ds]}
            </button>
          ))}
        </div>

        {/* Metric subtitle */}
        <div className="px-5 pb-2 pt-1">
          <p className="text-[11px] text-gray-400">
            {metricDisplay}
            {epochCap !== null && (
              <span className="ml-1 text-gray-300">· train to epoch {epochCap} to lock in score</span>
            )}
          </p>
        </div>

        {/* Table */}
        <div className="overflow-y-auto flex-1 px-2">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-gray-400 border-b border-gray-100">
                <th className="px-3 py-2 font-medium w-8">#</th>
                <th className="px-3 py-2 font-medium">Username</th>
                <th className="px-3 py-2 font-medium text-right">Score</th>
                <th className="px-3 py-2 font-medium text-right">Date</th>
              </tr>
            </thead>
            <tbody>
              {leaderboardLoading ? (
                [1, 2, 3].map((i) => <SkeletonRow key={i} />)
              ) : entries.length === 0 ? (
                <tr>
                  <td colSpan={4} className="px-3 py-8 text-center text-gray-400 text-xs">
                    No entries yet — be the first!
                  </td>
                </tr>
              ) : (
                entries.map((entry) => (
                  <tr key={entry.rank} className="border-b border-gray-50 hover:bg-gray-50 transition-colors">
                    <td className="px-3 py-2 text-gray-500 font-mono">
                      {entry.rank === 1 ? (
                        <Star size={12} className="text-amber-400 fill-amber-400" />
                      ) : (
                        entry.rank
                      )}
                    </td>
                    <td className="px-3 py-2 font-medium text-gray-800">{entry.username}</td>
                    <td className="px-3 py-2 text-right font-mono text-gray-700">
                      {formatScore(activeTab, entry.score)}
                    </td>
                    <td className="px-3 py-2 text-right text-gray-400">{formatDate(entry.submitted_at)}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Submit section */}
        {isCurrentDataset && score !== null && !submitted && (
          <div className="border-t border-gray-100 px-5 py-4">
            {qualifies ? (
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-semibold text-emerald-600">You qualify!</span>
                  <span className="text-xs text-gray-500">
                    Your score: <span className="font-mono font-medium">{formatScore(activeTab, score)}</span>
                    {projectedRank !== null && (
                      <span className="ml-1 text-gray-400">· Rank #{projectedRank}</span>
                    )}
                  </span>
                </div>
                <div className="flex gap-2">
                  <input
                    type="text"
                    maxLength={32}
                    placeholder="Enter username"
                    value={username}
                    onChange={(e) => setUsername(e.target.value)}
                    onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
                    className="flex-1 text-xs border border-gray-200 rounded-lg px-3 py-1.5 outline-none focus:ring-1 focus:ring-gray-400"
                  />
                  <button
                    onClick={handleSubmit}
                    disabled={leaderboardSubmitting || !username.trim()}
                    className="text-xs font-semibold bg-gray-900 text-white px-4 py-1.5 rounded-lg hover:bg-gray-700 disabled:opacity-50 transition-colors whitespace-nowrap"
                  >
                    {leaderboardSubmitting ? "…" : "Submit"}
                  </button>
                </div>
                {submitError && <p className="text-[11px] text-red-500">{submitError}</p>}
              </div>
            ) : (
              <p className="text-[11px] text-gray-400 text-center">
                Your score ({formatScore(activeTab, score)}) didn&apos;t make the top 10 this time.
              </p>
            )}
          </div>
        )}

        {/* Success state */}
        {submitted && (
          <div className="border-t border-gray-100 px-5 py-4 text-center">
            <p className="text-sm font-semibold text-emerald-600">
              🎉 You&apos;re ranked #{submitted.rank}!
            </p>
            <p className="text-xs text-gray-400 mt-0.5">Score submitted as &ldquo;{username}&rdquo;</p>
          </div>
        )}
      </div>
    </div>
  );
}
