import type { StateCreator } from "zustand";
import type { HoveredConnection, HoveredNode } from "@/components/network/static/types";

export interface UISlice {
  hoveredConnection: HoveredConnection | null;
  hoveredNode: HoveredNode | null;
  configOpen: boolean;
  stepLayerHighlight: number | null;
  tourActive: boolean;
  tourStep: number;
  setHoveredConnection: (hoveredConnection: HoveredConnection | null) => void;
  setHoveredNode: (hoveredNode: HoveredNode | null) => void;
  setConfigOpen: (configOpen: boolean) => void;
  setStepLayerHighlight: (index: number | null) => void;
  setTourActive: (active: boolean) => void;
  setTourStep: (step: number) => void;
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const createUISlice: StateCreator<any, [], [], UISlice> = (set, get) => ({
  hoveredConnection: null,
  hoveredNode: null,
  configOpen: true,
  stepLayerHighlight: null,
  tourActive: false,
  tourStep: 0,

  setHoveredConnection: (hoveredConnection) => {
    if (hoveredConnection) get().setHoveredNode(null);
    set({ hoveredConnection });
  },
  setHoveredNode: (hoveredNode) => {
    if (hoveredNode) get().setHoveredConnection(null);
    set({ hoveredNode });
  },
  setConfigOpen: (configOpen) => set({ configOpen }),
  setStepLayerHighlight: (stepLayerHighlight) => set({ stepLayerHighlight }),
  setTourActive: (tourActive) => set({ tourActive }),
  setTourStep: (tourStep) => set({ tourStep }),
});
