"use client"

import { useState } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Linkedin, Mail, ExternalLink, Copy, Check } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export default function ContactInfo() {
  const [copied, setCopied] = useState(false)

  const copyEmail = () => {
    navigator.clipboard.writeText("grantmwasserman@gmail.com")
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  return (
    <Card
      className={cn(
        "fixed bottom-4 right-4 shadow-lg border-0 bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 transition-all duration-300 z-50 w-82"
      )}
    >
      <CardHeader className="pb-1 pt-3">
        <CardTitle className="mt-2 text-base font-bold text-gray-800 dark:text-gray-100">Connect with Me</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 pt-1 pb-4">
        <div className="space-y-2">
          <a
            href="https://www.linkedin.com/in/grant-wass/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 p-2 rounded-lg transition-all duration-300 hover:bg-gray-100 dark:hover:bg-gray-800 group"
          >
            <div className="flex items-center justify-center w-7 h-7 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
              <Linkedin className="w-3.5 h-3.5" />
            </div>
            <div className="flex-1">
              <p className="font-medium text-sm text-gray-800 dark:text-gray-200">LinkedIn</p>
            </div>
            <ExternalLink className="w-3 h-3 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" />
          </a>

          <div className="flex items-center gap-2 p-2 rounded-lg transition-all duration-300 hover:bg-gray-100 dark:hover:bg-gray-800 group">
            <div className="flex items-center justify-center w-7 h-7 rounded-full bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400">
              <Mail className="w-3.5 h-3.5" />
            </div>
            <div className="flex-1">
              <p className="font-medium text-sm text-gray-800 dark:text-gray-200 truncate">
                grantmwasserman@gmail.com
              </p>
            </div>
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6"
              onClick={copyEmail}
              title={copied ? "Copied!" : "Copy email"}
            >
              {copied ? <Check className="h-3 w-3 text-green-500" /> : <Copy className="h-3 w-3 text-gray-400" />}
              <span className="sr-only">{copied ? "Copied" : "Copy email"}</span>
            </Button>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
