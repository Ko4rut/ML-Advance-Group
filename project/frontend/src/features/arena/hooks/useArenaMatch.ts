"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ApiError } from "../api/client";
import { createMatch as createServerMatch, getMatch, resetMatch } from "../api/matches";
import { makeMove, stepAgent } from "../api/moves";
import type { LegalMove, MatchSnapshot, Participant } from "../api/types";
import { PIECE_NAMES } from "../constants";
import { boardFromEngine, initialBoard, toDisplaySquare, toEngineSquare } from "../lib/board";
import type { AgentId, Side } from "../types";
import { useAgentCatalog } from "./useAgentCatalog";

function toParticipant(id: AgentId): Participant {
  if (id === "human") {
    return { type: "human" };
  }
  return { type: "agent", agent_id: id };
}

function getActiveParticipant(match: MatchSnapshot): Participant {
  if (match.side_to_move === "red") {
    return match.red;
  }
  return match.black;
}

function getLastMove(match: MatchSnapshot | null): LegalMove | null {
  if (!match || !match.last_move) {
    return null;
  }
  return {
    source: toDisplaySquare(match.last_move.source),
    target: toDisplaySquare(match.last_move.target),
  };
}

/**
 * Owns match state and user actions. The backend remains authoritative:
 * no local chess rule is calculated or submitted without a legal-move check.
 */
export function useArenaMatch() {
  const { available, connected, catalogError } = useAgentCatalog();
  const [top, setTop] = useState<AgentId>("human");
  const [bottom, setBottom] = useState<AgentId>("human");
  const [snapshot, setSnapshot] = useState<MatchSnapshot | null>(null);
  const [selection, setSelection] = useState<number | null>(null);
  const [notice, setNotice] = useState("Chọn hai đối thủ rồi nhấn Tạo trận đấu.");
  const [pending, setPending] = useState(false);
  const [running, setRunning] = useState(false);
  const [modal, setModal] = useState(false);
  const [lightMode, setLightMode] = useState(false);

  // Prevent duplicate requests and discard responses from obsolete matches.
  const busyRef = useRef(false);
  const generationRef = useRef(0);

  useEffect(function showCatalogErrorEffect() {
    if (catalogError) {
      setNotice(catalogError);
    }
  }, [catalogError]);

  useEffect(function closeDialogOnEscapeEffect() {
    function handleEscape(event: KeyboardEvent): void {
      if (event.key === "Escape") {
        setModal(false);
      }
    }

    window.addEventListener("keydown", handleEscape);
    return function removeEscapeListener() {
      window.removeEventListener("keydown", handleEscape);
    };
  }, []);

  const board = useMemo(function calculateBoard() {
    if (!snapshot) {
      return initialBoard();
    }
    return boardFromEngine(snapshot.board);
  }, [snapshot]);

  let canHumanMove = false;
  let canRun = false;
  if (snapshot && snapshot.status === "playing" && !pending) {
    const active = getActiveParticipant(snapshot);
    canHumanMove = active.type === "human" && !running;
    canRun = active.type === "agent";
  }

  const legalTargets = useMemo(
    function calculateLegalTargets() {
      const targets = new Set<number>();
      if (selection === null || !canHumanMove || !snapshot) {
        return targets;
      }

      for (const move of snapshot.legal_moves) {
        if (toDisplaySquare(move.source) === selection) {
          targets.add(toDisplaySquare(move.target));
        }
      }
      return targets;
    },
    [selection, canHumanMove, snapshot],
  );

  const lastMove = getLastMove(snapshot);

  const handleFailure = useCallback(async function handleApiFailure(
    error: unknown,
    current?: MatchSnapshot,
  ): Promise<void> {
    if (error instanceof Error) {
      setNotice(error.message);
    } else {
      setNotice("Có lỗi xảy ra khi gọi API.");
    }

    if (error instanceof ApiError && error.status === 409 && current) {
      try {
        // Recover from stale expected_ply by requesting the current snapshot.
        const latest = await getMatch(current.match_id);
        setSnapshot(latest);
        setSelection(null);
      } catch {
        // Keep the previous board visible; the notice explains the problem.
      }
    }
  }, []);

  function chooseAgent(side: Side, id: AgentId): void {
    if (busyRef.current || !available.has(id)) {
      return;
    }

    generationRef.current += 1;
    setRunning(false);
    setSnapshot(null);
    setSelection(null);

    if (side === "top") {
      setTop(id);
    } else {
      setBottom(id);
    }
    setNotice("Đã đổi đối thủ. Tạo trận mới để áp dụng lựa chọn.");
  }

  function chooseTopAgent(id: AgentId): void {
    chooseAgent("top", id);
  }

  function chooseBottomAgent(id: AgentId): void {
    chooseAgent("bottom", id);
  }

  async function createMatch(): Promise<void> {
    if (busyRef.current) {
      return;
    }
    if (!available.has(top) || !available.has(bottom)) {
      setNotice("Agent này chưa được backend triển khai hoặc API chưa kết nối.");
      return;
    }

    const generation = ++generationRef.current;
    busyRef.current = true;
    setPending(true);
    setRunning(false);
    setSelection(null);

    try {
      const created = await createServerMatch(toParticipant(top), toParticipant(bottom));
      if (generation === generationRef.current) {
        setSnapshot(created);
        setNotice("Đã tạo trận đấu. Quân đỏ đi trước.");
      }
    } catch (error: unknown) {
      if (generation === generationRef.current) {
        await handleFailure(error);
      }
    } finally {
      busyRef.current = false;
      setPending(false);
    }
  }

  async function resetBoard(): Promise<void> {
    if (busyRef.current) {
      return;
    }

    setRunning(false);
    setSelection(null);
    if (!snapshot) {
      setNotice("Chưa có trận đấu. Nhấn Tạo trận để bắt đầu.");
      return;
    }

    const previous = snapshot;
    const generation = ++generationRef.current;
    busyRef.current = true;
    setPending(true);

    try {
      const restarted = await resetMatch(previous.match_id);
      if (generation === generationRef.current) {
        setSnapshot(restarted);
        setNotice("Đã đặt lại trận đấu.");
      }
    } catch (error: unknown) {
      if (generation === generationRef.current) {
        await handleFailure(error, previous);
      }
    } finally {
      busyRef.current = false;
      setPending(false);
    }
  }

  async function moveHuman(
    current: MatchSnapshot,
    source: number,
    target: number,
  ): Promise<void> {
    if (busyRef.current) {
      return;
    }

    const generation = generationRef.current;
    busyRef.current = true;
    setPending(true);
    setSelection(null);

    try {
      const next = await makeMove(
        current.match_id,
        { source: toEngineSquare(source), target: toEngineSquare(target) },
        current.ply,
      );
      if (generation !== generationRef.current) {
        return;
      }

      setSnapshot(next);
      if (next.status === "finished") {
        setNotice("Trận đấu đã kết thúc.");
        setRunning(false);
        return;
      }

      setNotice("Đã thực hiện nước đi hợp lệ.");
      if (getActiveParticipant(next).type === "agent") {
        setRunning(true);
      }
    } catch (error: unknown) {
      if (generation === generationRef.current) {
        await handleFailure(error, current);
      }
    } finally {
      busyRef.current = false;
      setPending(false);
    }
  }

  function selectSquare(displayIndex: number): void {
    if (!snapshot) {
      setNotice("Hãy tạo trận đấu để xem nước đi hợp lệ.");
      return;
    }
    if (!canHumanMove) {
      setNotice("Chưa tới lượt người chơi hoặc đang xử lý nước đi.");
      return;
    }
    if (selection !== null && legalTargets.has(displayIndex)) {
      void moveHuman(snapshot, selection, displayIndex);
      return;
    }
    if (selection === displayIndex) {
      setSelection(null);
      return;
    }

    const piece = board[displayIndex];
    let expectedSide: Side = "bottom";
    if (snapshot.side_to_move === "red") {
      expectedSide = "top";
    }
    let legalCount = 0;
    for (const move of snapshot.legal_moves) {
      if (toDisplaySquare(move.source) === displayIndex) {
        legalCount += 1;
      }
    }

    if (piece && piece.side === expectedSide && legalCount > 0) {
      setSelection(displayIndex);
      setNotice(
        `Đã chọn ${PIECE_NAMES[piece.kind]} · ${legalCount} nước đi hợp lệ. ` +
          "Nhấn dấu chấm hoặc vòng tròn để di chuyển.",
      );
      return;
    }

    setSelection(null);
    setNotice("Chọn quân của phe đang đi để xem nước hợp lệ.");
  }

  function startBattle(): void {
    if (running) {
      setRunning(false);
      setNotice("Đã tạm dừng lượt AI.");
      return;
    }
    if (!snapshot) {
      setNotice("Hãy tạo trận đấu trước.");
      return;
    }
    if (snapshot.status === "finished") {
      setNotice("Ván đã kết thúc. Hãy đặt lại trận.");
      return;
    }
    if (getActiveParticipant(snapshot).type === "human") {
      setNotice("Đang tới lượt người chơi. Hãy chọn một quân.");
      return;
    }

    setRunning(true);
    setNotice("AI đang thực hiện các nước đi...");
  }

  // Client-paced autoplay: one HTTP step per state update, no concurrent AI requests.
  useEffect(function agentAutoplayEffect() {
    if (!running || !snapshot || pending) {
      return;
    }
    if (snapshot.status === "finished" || getActiveParticipant(snapshot).type === "human") {
      setRunning(false);
      return;
    }

    const current = snapshot;
    const generation = generationRef.current;

    async function advanceAgentTurn(): Promise<void> {
      if (busyRef.current || generation !== generationRef.current) {
        return;
      }

      busyRef.current = true;
      setPending(true);

      try {
        const next = await stepAgent(current.match_id, current.ply);
        if (generation === generationRef.current) {
          setSnapshot(next);
          if (next.status === "finished") {
            setNotice("Trận đấu đã kết thúc.");
            setRunning(false);
          } else {
            setNotice(`AI vừa đánh nước thứ ${next.ply}.`);
          }
        }
      } catch (error: unknown) {
        if (generation === generationRef.current) {
          setRunning(false);
          await handleFailure(error, current);
        }
      } finally {
        busyRef.current = false;
        setPending(false);
      }
    }

    function runScheduledTurn(): void {
      void advanceAgentTurn();
    }

    const timer = window.setTimeout(runScheduledTurn, 550);
    return function clearScheduledTurn() {
      window.clearTimeout(timer);
    };
  }, [running, snapshot, pending, handleFailure]);

  function openStatistics(): void {
    setModal(true);
  }

  function closeStatistics(): void {
    setModal(false);
  }

  function toggleTheme(): void {
    setLightMode(function invertTheme(previous) {
      return !previous;
    });
  }

  return {
    top,
    bottom,
    board,
    snapshot,
    selection,
    legalTargets,
    lastMove,
    notice,
    modal,
    lightMode,
    available,
    connected,
    pending,
    running,
    canHumanMove,
    canRun,
    chooseAgent,
    chooseTopAgent,
    chooseBottomAgent,
    createMatch,
    resetBoard,
    selectSquare,
    startBattle,
    openStatistics,
    closeStatistics,
    toggleTheme,
  };
}

