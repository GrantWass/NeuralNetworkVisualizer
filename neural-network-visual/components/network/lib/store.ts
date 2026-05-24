import { create } from "zustand";
import { createTrainingSlice } from "./slices/trainingSlice";
import { createUISlice } from "./slices/uiSlice";
import { createLeaderboardSlice } from "./slices/leaderboardSlice";
import type { TrainingSlice } from "./slices/trainingSlice";
import type { UISlice } from "./slices/uiSlice";
import type { LeaderboardSlice } from "./slices/leaderboardSlice";

// Re-export helpers so existing import paths are unchanged
export { applyActivation, forwardPassSingle } from "./networkUtils";

// Re-export leaderboard types so existing import paths are unchanged
export type { LeaderboardEntry, LeaderboardSubmitResponse } from "./slices/leaderboardSlice";

export type StoreState = TrainingSlice & UISlice & LeaderboardSlice;

const useStore = create<StoreState>()((...a) => ({
  ...createTrainingSlice(...a),
  ...createUISlice(...a),
  ...createLeaderboardSlice(...a),
}));

export default useStore;
