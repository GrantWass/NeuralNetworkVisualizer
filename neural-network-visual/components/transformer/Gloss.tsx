"use client";
import { useState, useRef } from "react";

type GlossProps = {
  term: string;
  children: React.ReactNode;
};

/**
 * Inline term with a hover/focus tooltip. Use inside prose to define
 * technical vocabulary without interrupting the reading flow.
 */
export function Gloss({ term, children }: GlossProps) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLSpanElement>(null);

  return (
    <span ref={ref} className="relative inline-block">
      <button
        type="button"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        className="border-b border-dashed border-indigo-400 font-semibold text-foreground cursor-help"
        aria-haspopup="true"
        aria-expanded={open}
      >
        {term}
      </button>
      {open && (
        <span
          role="tooltip"
          className="absolute top-full left-0 mt-2 z-50 w-72 p-3 rounded-lg border border-border bg-popover shadow-lg text-xs text-popover-foreground leading-relaxed pointer-events-none"
        >
          {children}
        </span>
      )}
    </span>
  );
}
