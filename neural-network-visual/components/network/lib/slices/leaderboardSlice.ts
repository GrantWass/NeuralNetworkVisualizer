import type { StateCreator } from "zustand";
import { toast } from "sonner";
import { fetchWithTimeout } from "../apiUtils";

export interface LeaderboardEntry {
  rank: number;
  username: string;
  score: number;
  epoch: number;
  submitted_at: number;
}

export interface LeaderboardSubmitResponse {
  accepted: boolean;
  rank: number | null;
  entries: LeaderboardEntry[];
}

const EPOCH_CAPS: Record<string, number | null> = {
  xor: null, iris: 100, auto_mpg: 200, mnist: 300,
};

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export interface LeaderboardSlice {
  leaderboardOpen: boolean;
  leaderboard: Record<string, LeaderboardEntry[]>;
  leaderboardLoading: boolean;
  leaderboardSubmitting: boolean;
  setLeaderboardOpen: (open: boolean) => void;
  fetchLeaderboard: (dataset: string) => Promise<void>;
  submitLeaderboardScore: (username: string) => Promise<LeaderboardSubmitResponse | null>;
  computeQualification: () => { qualifies: boolean; rank: number | null };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const createLeaderboardSlice: StateCreator<any, [], [], LeaderboardSlice> = (set, get) => ({
  leaderboardOpen: false,
  leaderboard: {},
  leaderboardLoading: false,
  leaderboardSubmitting: false,

  setLeaderboardOpen: (open) => {
    set({ leaderboardOpen: open });
    if (open) get().fetchLeaderboard(get().dataset);
  },

  fetchLeaderboard: async (dataset: string) => {
    set({ leaderboardLoading: true });
    try {
      const res = await fetchWithTimeout(`${API_URL}/leaderboard/${dataset}`);
      const data = await res.json();
      if (res.ok) {
        set((state: LeaderboardSlice) => ({ leaderboard: { ...state.leaderboard, [dataset]: data.entries } }));
      }
    } catch (e) {
      console.error("Failed to fetch leaderboard:", e);
    } finally {
      set({ leaderboardLoading: false });
    }
  },

  submitLeaderboardScore: async (username: string) => {
    const { dataset, submittableScore, xorEpochsTo100, epoch } = get();
    const score = dataset === "xor" ? xorEpochsTo100 : submittableScore;
    if (score === null) return null;

    const cap = EPOCH_CAPS[dataset];
    const submittedEpoch = dataset === "xor"
      ? (xorEpochsTo100 ?? epoch)
      : cap !== null ? Math.min(epoch, cap) : epoch;

    set({ leaderboardSubmitting: true });
    try {
      const res = await fetchWithTimeout(`${API_URL}/leaderboard/submit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset, score, epoch: submittedEpoch, username }),
      });
      const data: LeaderboardSubmitResponse = await res.json();
      if (res.ok && data.accepted) {
        set((state: LeaderboardSlice) => ({ leaderboard: { ...state.leaderboard, [dataset]: data.entries } }));
      }
      return data;
    } catch (e) {
      console.error("Failed to submit score:", e);
      toast.error("Submission failed", { description: "Could not submit your score. Please try again." });
      return null;
    } finally {
      set({ leaderboardSubmitting: false });
    }
  },

  computeQualification: () => {
    const { dataset, submittableScore, xorEpochsTo100, leaderboard } = get();
    const score = dataset === "xor" ? xorEpochsTo100 : submittableScore;
    if (score === null) return { qualifies: false, rank: null };

    const entries = leaderboard[dataset] ?? [];
    const higherIsBetter = dataset === "iris" || dataset === "mnist";

    for (let i = 0; i < entries.length; i++) {
      const beats = higherIsBetter ? score > entries[i].score : score < entries[i].score;
      if (beats) return { qualifies: true, rank: i + 1 };
    }
    if (entries.length < 10) return { qualifies: true, rank: entries.length + 1 };
    return { qualifies: false, rank: null };
  },
});
