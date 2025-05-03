"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Linkedin, Mail, ExternalLink, Copy, Check, MessageSquare } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export default function ContactInfo() {
  const [copied, setCopied] = useState(false)
  const [isMobile, setIsMobile] = useState(false)
  const [isOpen, setIsOpen] = useState(false)

  useEffect(() => {
    const checkMobile = () => {
      setIsMobile(window.innerWidth < 768)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const copyEmail = () => {
    navigator.clipboard.writeText("grantmwasserman@gmail.com")
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const contactContent = (
    <div className="space-y-2">
      <a
        href="https://www.linkedin.com/in/grant-wass/"
        target="_blank"
        rel="noopener noreferrer"
        className="flex items-center gap-2 p-2 rounded-lg transition-all duration-300 hover:bg-gray-100 dark:hover:bg-gray-800 group"
      >
        <div className="flex items-center justify-center w-7 h-7 rounded-full bg-blue-100 dark:bg-blue-900/30 text-blue-600 dark:text-blue-400">
          <Linkedin className="w-2.5 h-2.5" />
        </div>
        <div className="flex-1">
          <p className="font-medium text-sm text-gray-800 dark:text-gray-200">LinkedIn</p>
        </div>
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
  )

  if (isMobile) {
    return (
      <>
        <Button
          variant="outline"
          size="icon"
          className="fixed bottom-4 right-4 h-12 w-12 rounded-full shadow-lg"
          onClick={() => setIsOpen(true)}
        >
          <MessageSquare className="h-5 w-5" />
        </Button>
        {isOpen && (
          <div className="fixed inset-0 bg-gray-600 bg-opacity-50 flex justify-center items-center z-50">
            <Card className="w-[90%] max-w-md">
              <CardHeader className="pb-1 pt-3 flex justify-between flex-row">
                <CardTitle className="mt-2 text-base font-bold text-gray-800 dark:text-gray-100 w-[50%]">Connect with Me</CardTitle>
                <div className="flex justify-end mt-2">
                  <Button onClick={() => setIsOpen(false)}>Close</Button>
                </div>
              </CardHeader>
              <CardContent className="space-y-2 pt-1 pb-4">
                {contactContent}
              </CardContent>
            </Card>
          </div>
        )}
      </>
    )
  }

  return (
    <Card
      className={cn(
        "fixed bottom-4 right-4 shadow-lg border-0 bg-gradient-to-br from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 transition-all duration-300 z-50 w-82 p-0"
      )}
    >
      <CardHeader className="pb-1 pt-3">
        <CardTitle className="mt-2 text-base font-bold text-gray-800 dark:text-gray-100">Connect with Me</CardTitle>
      </CardHeader>
      <CardContent className="space-y-2 p-2">
        {contactContent}
      </CardContent>
    </Card>
  )
} 