"use client";

import { useEffect, useRef, useState } from "react";
import { parseReplayGames } from "../lib/replay";
import type { ReplayGame } from "../lib/replay";

const MAX_FILE_SIZE = 30 * 1024 * 1024;

export function useGameReplay() {
  const [games, setGames] = useState<ReplayGame[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [currentPly, setCurrentPly] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(false);
  const [loading, setLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);
  const uploadVersion = useRef(0);

  const game = games[selectedIndex] || null;
  let totalPlies = 0;
  if (game) {
    totalPlies = game.moves.length;
  }

  async function loadFile(file: File | undefined): Promise<void> {
    if (!file) {
      return;
    }
    const version = ++uploadVersion.current;
    setPlaying(false);
    setLoading(true);
    setError(null);

    if (file.size > MAX_FILE_SIZE) {
      setError("Tệp vượt quá 30 MB. Hãy chọn file JSON/JSONL nhỏ hơn.");
      setLoading(false);
      return;
    }

    try {
      const contents = await file.text();
      const parsed = parseReplayGames(contents, file.name);
      if (version !== uploadVersion.current) {
        return;
      }
      setGames(parsed);
      setSelectedIndex(0);
      setCurrentPly(0);
      setFileName(file.name);
    } catch (reason: unknown) {
      if (version !== uploadVersion.current) {
        return;
      }
      if (reason instanceof Error) {
        setError(reason.message);
      } else {
        setError("Không đọc được tệp dữ liệu.");
      }
    } finally {
      if (version === uploadVersion.current) {
        setLoading(false);
      }
    }
  }

  function chooseGame(index: number): void {
    if (index < 0 || index >= games.length) {
      return;
    }
    setSelectedIndex(index);
    setCurrentPly(0);
    setPlaying(false);
    setError(null);
  }

  function seek(ply: number): void {
    if (!game || !Number.isFinite(ply)) {
      return;
    }
    const next = Math.max(0, Math.min(totalPlies, Math.floor(ply)));
    setCurrentPly(next);
    setPlaying(false);
  }

  function previous(): void {
    seek(currentPly - 1);
  }

  function next(): void {
    seek(currentPly + 1);
  }

  function goToStart(): void {
    seek(0);
  }

  function goToEnd(): void {
    seek(totalPlies);
  }

  function togglePlayback(): void {
    if (!game || totalPlies === 0) {
      return;
    }
    if (playing) {
      setPlaying(false);
      return;
    }
    if (currentPly >= totalPlies) {
      setCurrentPly(0);
    }
    setPlaying(true);
  }

  useEffect(function autoplayEffect() {
    if (!playing || !game || currentPly >= totalPlies) {
      return;
    }

    const nextPly = currentPly + 1;
    const timer = window.setTimeout(function advanceReplay() {
      setCurrentPly(nextPly);
      if (nextPly === totalPlies) {
        setPlaying(false);
      }
    }, Math.round(850 / speed));

    return function clearReplayTimer() {
      window.clearTimeout(timer);
    };
  }, [playing, currentPly, totalPlies, game, speed]);

  return {
    games,
    game,
    selectedIndex,
    currentPly,
    totalPlies,
    playing,
    loading,
    speed,
    fileName,
    error,
    setSpeed,
    loadFile,
    chooseGame,
    seek,
    previous,
    next,
    goToStart,
    goToEnd,
    togglePlayback,
  };
}
