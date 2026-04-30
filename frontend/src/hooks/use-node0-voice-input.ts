import { useEffect, useRef, useState } from "react";

export type Node0VoiceInputStatus =
  | "unsupported"
  | "idle"
  | "listening"
  | "transcribed"
  | "error";

interface SpeechRecognitionAlternativeLike {
  transcript: string;
}

interface SpeechRecognitionResultLike {
  readonly isFinal: boolean;
  readonly length: number;
  [index: number]: SpeechRecognitionAlternativeLike;
}

interface SpeechRecognitionResultListLike {
  readonly length: number;
  [index: number]: SpeechRecognitionResultLike;
}

interface SpeechRecognitionEventLike extends Event {
  readonly results: SpeechRecognitionResultListLike;
}

interface SpeechRecognitionErrorEventLike extends Event {
  readonly error: string;
  readonly message?: string;
}

interface SpeechRecognitionLike {
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEventLike) => void) | null;
  onerror: ((event: SpeechRecognitionErrorEventLike) => void) | null;
  onend: (() => void) | null;
  start: () => void;
  stop: () => void;
  abort: () => void;
}

type SpeechRecognitionConstructor = new () => SpeechRecognitionLike;

interface SpeechRecognitionWindow extends Window {
  SpeechRecognition?: SpeechRecognitionConstructor;
  webkitSpeechRecognition?: SpeechRecognitionConstructor;
}

function speechRecognitionConstructor(): SpeechRecognitionConstructor | null {
  if (typeof window === "undefined") {
    return null;
  }
  const speechWindow = window as SpeechRecognitionWindow;
  return speechWindow.SpeechRecognition ?? speechWindow.webkitSpeechRecognition ?? null;
}

function errorMessage(error: unknown, fallback: string): string {
  return error instanceof Error && error.message ? error.message : fallback;
}

export function useNode0VoiceInput() {
  const recognitionRef = useRef<SpeechRecognitionLike | null>(null);
  const [supported, setSupported] = useState(false);
  const [status, setStatus] = useState<Node0VoiceInputStatus>("unsupported");
  const [transcript, setTranscript] = useState("");
  const [interimTranscript, setInterimTranscript] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const available = speechRecognitionConstructor() !== null;
    setSupported(available);
    setStatus(available ? "idle" : "unsupported");

    return () => {
      recognitionRef.current?.abort();
      recognitionRef.current = null;
    };
  }, []);

  const start = () => {
    const Recognition = speechRecognitionConstructor();
    if (Recognition === null) {
      setSupported(false);
      setStatus("unsupported");
      setError("Browser speech recognition is unavailable");
      return;
    }

    const recognition = new Recognition();
    recognition.continuous = false;
    recognition.interimResults = true;
    recognition.lang = "en-US";
    recognition.onresult = (event) => {
      let finalText = "";
      let interimText = "";

      for (let index = 0; index < event.results.length; index += 1) {
        const result = event.results[index];
        const alternative = result[0];
        if (!alternative?.transcript) {
          continue;
        }
        if (result.isFinal) {
          finalText += alternative.transcript;
        } else {
          interimText += alternative.transcript;
        }
      }

      setTranscript(finalText.trim());
      setInterimTranscript(interimText.trim());
      setStatus(finalText.trim() ? "transcribed" : "listening");
    };
    recognition.onerror = (event) => {
      setError(event.message || event.error || "Voice input failed");
      setStatus("error");
    };
    recognition.onend = () => {
      setInterimTranscript("");
      setStatus((current) => (current === "listening" ? "idle" : current));
      recognitionRef.current = null;
    };

    recognitionRef.current?.abort();
    recognitionRef.current = recognition;
    setTranscript("");
    setInterimTranscript("");
    setError(null);
    setStatus("listening");
    try {
      recognition.start();
    } catch (error) {
      recognitionRef.current = null;
      setStatus("error");
      setError(errorMessage(error, "Voice input failed to start"));
    }
  };

  const stop = () => {
    recognitionRef.current?.stop();
    recognitionRef.current = null;
    setStatus((current) => (current === "listening" ? "idle" : current));
  };

  const clear = () => {
    setTranscript("");
    setInterimTranscript("");
    setError(null);
    setStatus(supported ? "idle" : "unsupported");
  };

  return {
    supported,
    status,
    transcript,
    interimTranscript,
    error,
    isListening: status === "listening",
    start,
    stop,
    clear,
  };
}
