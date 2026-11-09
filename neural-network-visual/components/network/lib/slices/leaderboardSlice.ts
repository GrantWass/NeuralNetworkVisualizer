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

// Authoritative per-dataset display metadata from GET /leaderboard/{dataset}.
// Absent/undefined fields mean the server response didn't include them
// (older/cached payload) — callers must fall back to local labels.
export interface LeaderboardMeta {
  metricDisplay?: string | null;
  higherIsBetter?: boolean | null;
  epochCap?: number | null;
}

const EPOCH_CAPS: Record<string, number | null> = {
  xor: null, iris: 100, auto_mpg: 200, mnist: 300,
};

const API_URL = process.env.NEXT_PUBLIC_API_URL;

// How long a cached leaderboard stays fresh. While the user trains, ranking
// refreshes at most once per window so entries stay current without spamming
// the API on every training cycle.
const LEADERBOARD_TTL_MS = 15_000;

export interface LeaderboardSlice {
  leaderboardOpen: boolean;
  leaderboard: Record<string, LeaderboardEntry[]>;
  leaderboardMeta: Record<string, LeaderboardMeta>;
  leaderboardFetchedAt: Record<string, number>;
  leaderboardLoading: boolean;
  leaderboardSubmitting: boolean;
  setLeaderboardOpen: (open: boolean) => void;
  fetchLeaderboard: (dataset: string) => Promise<void>;
  maybeRefreshLeaderboard: (dataset: string) => void;
  submitLeaderboardScore: (username: string) => Promise<LeaderboardSubmitResponse | null>;
  computeQualification: () => { qualifies: boolean; rank: number | null };
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const createLeaderboardSlice: StateCreator<any, [], [], LeaderboardSlice> = (set, get) => ({
  leaderboardOpen: false,
  leaderboard: {},
  leaderboardMeta: {},
  leaderboardFetchedAt: {},
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
        // Persist the server's display metadata alongside entries, so Score
        // headers/subtitles render direction ("higher/lower wins") and caps
        // from the backend instead of duplicated local constants. Every
        // field is defensive: a missing/odd-typed value stays undefined and
        // callers fall back to the existing hardcoded labels.
        const meta: LeaderboardMeta = {
          metricDisplay: typeof data?.metric_display === "string" ? data.metric_display : undefined,
          higherIsBetter: typeof data?.higher_is_better === "boolean" ? data.higher_is_better : undefined,
          epochCap:
            typeof data?.epoch_cap === "number"
              ? data.epoch_cap
              : data?.epoch_cap === null
                ? null
                : undefined,
        };
        set((state: LeaderboardSlice) => ({
          leaderboard: { ...state.leaderboard, [dataset]: data.entries },
          leaderboardMeta: { ...state.leaderboardMeta, [dataset]: meta },
        }));
      } else {
        toast.error("Couldn't load the leaderboard", {
          description: data?.detail || `Server responded with ${res.status}. Please try again.`,
        });
      }
    } catch (e) {
      console.error("Failed to fetch leaderboard:", e);
      toast.error("Couldn't load the leaderboard", {
        description: "Could not reach the server. Check your connection and try again.",
      });
    } finally {
      // Stamp the attempt, not just the success: a failed fetch has to open the
      // TTL window too, or maybeRefreshLeaderboard sees an empty timestamp and
      // retries (plus toasts) on every epoch while the API is down.
      set((state: LeaderboardSlice) => ({
        leaderboardLoading: false,
        leaderboardFetchedAt: { ...state.leaderboardFetchedAt, [dataset]: Date.now() },
      }));
    }
  },

  // Refetch only when the cache is empty or older than LEADERBOARD_TTL_MS.
  // Called as training progresses so the inline ranking stays current;
  // the TTL bounds it to one request per window per dataset.
  maybeRefreshLeaderboard: (dataset: string) => {
    if (get().leaderboardLoading) return;
    const fetchedAt = get().leaderboardFetchedAt[dataset];
    const isStale = fetchedAt === undefined || Date.now() - fetchedAt >= LEADERBOARD_TTL_MS;
    if (isStale) get().fetchLeaderboard(dataset);
  },

  submitLeaderboardScore: async (username: string) => {
    const { dataset, submittableScore, xorEpochsTo100, epoch } = get();
    const score = dataset === "xor" ? xorEpochsTo100 : submittableScore;
    if (score === null) return null;

    // Prefer the server-provided cap when fetched; fall back to the local
    // constant so submit gating still works before the first fetch.
    const metaCap = get().leaderboardMeta?.[dataset]?.epochCap;
    const cap = metaCap !== undefined ? metaCap : (EPOCH_CAPS[dataset] ?? null);
    // Mid-run scores are provisional — only a cap-epoch score is submittable.
    if (cap !== null && epoch < cap) {
      toast.error("Keep training", {
        description: `Train to epoch ${cap} before submitting (at epoch ${epoch} now).`,
      });
      return null;
    }
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
      const data: LeaderboardSubmitResponse & { detail?: string } = await res.json();
      if (!res.ok) {
        // Don't report a server failure as a qualification miss
        toast.error("Submission failed", {
          description: data?.detail || `Server responded with ${res.status}. Please try again.`,
        });
        return null;
      }
      if (data.accepted) {
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
    const { dataset, submittableScore, xorEpochsTo100, leaderboard, epoch } = get();
    const score = dataset === "xor" ? xorEpochsTo100 : submittableScore;
    if (score === null) return { qualifies: false, rank: null };

    // A mid-run score is provisional, not a final result — no projected rank
    // until training reaches the cap epoch the leaderboard label refers to.
    const metaCapForQual = get().leaderboardMeta?.[dataset]?.epochCap;
    const cap = metaCapForQual !== undefined ? metaCapForQual : (EPOCH_CAPS[dataset] ?? null);
    if (cap !== null && epoch < cap) return { qualifies: false, rank: null };

    const entries = leaderboard[dataset] ?? [];
    // Prefer the server's direction when fetched; fall back to the
    // long-standing local rule so qualification still works offline.
    const metaHigher = get().leaderboardMeta?.[dataset]?.higherIsBetter;
    const higherIsBetter = typeof metaHigher === "boolean"
      ? metaHigher
      : dataset === "iris" || dataset === "mnist";

    for (let i = 0; i < entries.length; i++) {
      const beats = higherIsBetter ? score > entries[i].score : score < entries[i].score;
      if (beats) return { qualifies: true, rank: i + 1 };
    }
    if (entries.length < 10) return { qualifies: true, rank: entries.length + 1 };
    return { qualifies: false, rank: null };
  },
});
