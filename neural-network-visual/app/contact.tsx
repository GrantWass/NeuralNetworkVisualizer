"use client"

import { useState } from "react"
import { Linkedin, Mail, Copy, Check, MessageSquare, Globe, X } from "lucide-react"
import { Button } from "@/components/ui/button"

export default function ContactInfo() {
  const [copied, setCopied] = useState(false)
  const [isOpen, setIsOpen] = useState(false)

  const copyEmail = () => {
    navigator.clipboard.writeText("grantmwasserman@gmail.com")
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <>
      <Button
        variant="outline"
        className="fixed bottom-4 right-4 rounded-full shadow-md z-50 px-4 py-2 flex items-center gap-2"
        onClick={() => setIsOpen((s) => !s)}
        aria-expanded={isOpen}
        aria-label={isOpen ? "Close contact" : "Connect with me"}
      >
        <MessageSquare className="h-4 w-4" />
        <span className="text-sm font-medium">Connect</span>
      </Button>

      {isOpen && (
        <>
          {/* backdrop (mobile only) */}
          <div
            className="fixed inset-0 z-40 sm:hidden"
            onClick={() => setIsOpen(false)}
          />

          <div className="fixed bottom-16 right-4 z-50 w-[calc(100vw-2rem)] max-w-xs rounded-xl border border-border bg-background shadow-lg">
            <div className="flex items-center justify-between border-b border-border px-4 py-3">
              <span className="text-sm font-semibold">Connect with me</span>
              <Button
                variant="ghost"
                size="icon"
                className="h-6 w-6 text-muted-foreground"
                onClick={() => setIsOpen(false)}
              >
                <X className="h-3.5 w-3.5" />
                <span className="sr-only">Close</span>
              </Button>
            </div>

            <div className="p-2 space-y-0.5">
              <a
                href="https://www.linkedin.com/in/grant-wass/"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-muted"
              >
                <Linkedin className="h-4 w-4 text-blue-500 shrink-0" />
                <span className="font-medium">LinkedIn</span>
              </a>

              <a
                href="https://grantwasserman.com"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm transition-colors hover:bg-muted"
              >
                <Globe className="h-4 w-4 text-green-500 shrink-0" />
                <span className="font-medium">Portfolio</span>
              </a>

              <div className="flex items-center gap-3 rounded-lg px-3 py-2 text-sm">
                <Mail className="h-4 w-4 text-muted-foreground shrink-0" />
                <span className="flex-1 text-muted-foreground truncate">grantmwasserman@gmail.com</span>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-6 w-6 shrink-0"
                  onClick={copyEmail}
                  title={copied ? "Copied!" : "Copy email"}
                >
                  {copied
                    ? <Check className="h-3.5 w-3.5 text-green-500" />
                    : <Copy className="h-3.5 w-3.5 text-muted-foreground" />}
                  <span className="sr-only">{copied ? "Copied" : "Copy email"}</span>
                </Button>
              </div>
            </div>
          </div>
        </>
      )}
    </>
  )
}
