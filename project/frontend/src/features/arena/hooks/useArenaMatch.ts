"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ApiError, createMatch as createOnServer, getAgents, getMatch,
  makeMove, resetMatch, stepAgent,
  type MatchSnapshot, type Participant,
} from "../lib/api";
import { boardFromEngine, initialBoard, toDisplaySquare, toEngineSquare } from "../lib/board";
import { PIECE_NAMES } from "../constants";
import type { AgentId, Side } from "../types";

const participant = (id: AgentId): Participant =>
  id === "human" ? { type: "human" } : { type: "agent", agent_id: id };

export function useArenaMatch() {
  const [top, setTop] = useState<AgentId>("human");
  const [bottom, setBottom] = useState<AgentId>("human");
  const [available, setAvailable] = useState<Set<AgentId>>(() => new Set(["human"]));
  const [connected, setConnected] = useState(false);
  const [snapshot, setSnapshot] = useState<MatchSnapshot | null>(null);
  const [selection, setSelection] = useState<number | null>(null);
  const [notice, setNotice] = useState("Chọn hai đối thủ rồi nhấn Tạo trận đấu.");
  const [pending, setPending] = useState(false);
  const [running, setRunning] = useState(false);
  const [modal, setModal] = useState(false);
  const [lightMode, setLightMode] = useState(false);
  const busyRef = useRef(false);
  const generationRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    getAgents().then((catalog) => {
      if (cancelled) return;
      setAvailable(new Set<AgentId>(["human", ...catalog.filter((a) => a.available).map((a) => a.id as AgentId)]));
      setConnected(true);
    }).catch((error: unknown) => {
      if (!cancelled) { setConnected(false); setNotice(error instanceof Error ? error.message : "Không tải được danh sách agent."); }
    });
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    const onEscape = (event: KeyboardEvent) => { if (event.key === "Escape") setModal(false); };
    window.addEventListener("keydown", onEscape);
    return () => window.removeEventListener("keydown", onEscape);
  }, []);

  const board = useMemo(() => snapshot ? boardFromEngine(snapshot.board) : initialBoard(), [snapshot]);
  const activeParticipant = snapshot ? snapshot.side_to_move === "red" ? snapshot.red : snapshot.black : null;
  const canHumanMove = Boolean(snapshot && snapshot.status === "playing" && activeParticipant?.type === "human" && !pending && !running);

  const legalTargets = useMemo(() => new Set(
    selection === null || !canHumanMove || !snapshot ? [] :
      snapshot.legal_moves.filter((move) => toDisplaySquare(move.source) === selection)
        .map((move) => toDisplaySquare(move.target))
  ), [selection, canHumanMove, snapshot]);

  const lastMove = snapshot?.last_move
    ? { source: toDisplaySquare(snapshot.last_move.source), target: toDisplaySquare(snapshot.last_move.target) } : null;

  const failure = async (error: unknown, current?: MatchSnapshot) => {
    setNotice(error instanceof Error ? error.message : "Có lỗi xảy ra khi gọi API.");
    if (error instanceof ApiError && error.status === 409 && current) {
      try { setSnapshot(await getMatch(current.match_id)); setSelection(null); }
      catch { /* Keep old board visible while explaining the error. */ }
    }
  };

  const chooseAgent = (side: Side, id: AgentId) => {
    if (busyRef.current || !available.has(id)) return;
    generationRef.current += 1;
    setRunning(false);
    setSnapshot(null);
    setSelection(null);
    if (side === "top") setTop(id); else setBottom(id);
    setNotice("Đã đổi đối thủ. Tạo trận mới để áp dụng lựa chọn.");
  };

  const createMatch = async () => {
    if (busyRef.current) return;
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
      const result = await createOnServer(participant(top), participant(bottom));
      if (generation === generationRef.current) {
        setSnapshot(result);
        setNotice("Đã tạo trận đấu. Quân đỏ đi trước.");
      }
    } catch (error) { if (generation === generationRef.current) await failure(error); }
    finally { busyRef.current = false; setPending(false); }
  };

  const resetBoard = async () => {
    if (busyRef.current) return;
    setRunning(false);
    setSelection(null);
    if (!snapshot) { setNotice("Chưa có trận đấu. Nhấn Tạo trận để bắt đầu."); return; }
    const generation = ++generationRef.current;
    busyRef.current = true;
    setPending(true);
    try {
      const result = await resetMatch(snapshot.match_id);
      if (generation === generationRef.current) { setSnapshot(result); setNotice("Đã đặt lại trận đấu."); }
    } catch (error) { if (generation === generationRef.current) await failure(error, snapshot); }
    finally { busyRef.current = false; setPending(false); }
  };

  const moveHuman = async (current: MatchSnapshot, source: number, target: number) => {
    if (busyRef.current) return;
    const generation = generationRef.current;
    busyRef.current = true;
    setPending(true);
    setSelection(null);
    try {
      const result = await makeMove(current.match_id, {
        source: toEngineSquare(source),
        target: toEngineSquare(target),
      }, current.ply);
      if (generation !== generationRef.current) return;
      setSnapshot(result);
      setNotice(result.status === "finished" ? "Trận đấu đã kết thúc." : "Đã thực hiện nước đi hợp lệ.");
      const next = result.side_to_move === "red" ? result.red : result.black;
      if (next.type === "agent" && result.status === "playing") setRunning(true);
    } catch (error) { if (generation === generationRef.current) await failure(error, current); }
    finally { busyRef.current = false; setPending(false); }
  };

  const selectSquare = (displayIndex: number) => {
    if (!snapshot) { setNotice("Hãy tạo trận đấu để xem nước đi hợp lệ."); return; }
    if (!canHumanMove) { setNotice("Chưa tới lượt người chơi hoặc đang xử lý nước đi."); return; }
    if (selection !== null && legalTargets.has(displayIndex)) {
      void moveHuman(snapshot, selection, displayIndex);
      return;
    }
    if (selection === displayIndex) { setSelection(null); return; }
    const piece = board[displayIndex];
    const expectedSide = snapshot.side_to_move === "red" ? "top" : "bottom";
    if (piece?.side === expectedSide && snapshot.legal_moves.some((move) => toDisplaySquare(move.source) === displayIndex)) {
      setSelection(displayIndex);
      setNotice(`Đã chọn ${PIECE_NAMES[piece.kind]}. Nhấn vào dấu chấm hoặc vòng tròn để đi.`);
    } else {
      setSelection(null);
      setNotice("Chọn quân của phe đang đi để xem nước hợp lệ.");
    }
  };

  const startBattle = () => {
    if (running) { setRunning(false); setNotice("Đã tạm dừng lượt AI."); return; }
    if (!snapshot) { setNotice("Hãy tạo trận đấu trước."); return; }
    if (snapshot.status === "finished") { setNotice("Ván đã kết thúc. Hãy đặt lại trận."); return; }
    const actor = snapshot.side_to_move === "red" ? snapshot.red : snapshot.black;
    if (actor.type === "human") { setNotice("Đang tới lượt người chơi. Hãy chọn một quân."); return; }
    setRunning(true);
    setNotice("AI đang thực hiện các nước đi...");
  };

  useEffect(() => {
    if (!running || !snapshot || pending) return;
    if (snapshot.status === "finished") {
      setRunning(false);
      return;
    }
    const current = snapshot.side_to_move === "red" ? snapshot.red : snapshot.black;
    if (current.type === "human") {
      setRunning(false);
      return;
    }
    const generation = generationRef.current;
    const timer = window.setTimeout(async () => {
      if (busyRef.current || generation !== generationRef.current) return;
      busyRef.current = true;
      setPending(true);
      try {
        const result = await stepAgent(snapshot.match_id, snapshot.ply);
        if (generation === generationRef.current) {
          setSnapshot(result);
          if (result.status === "finished") setNotice("Trận đấu đã kết thúc.");
          else setNotice(`AI vừa đánh nước thứ ${result.ply}.`);
        }
      } catch (error) {
        if (generation === generationRef.current) {
          setRunning(false);
          await failure(error, snapshot);
        }
      } finally {
        busyRef.current = false;
        setPending(false);
      }
    }, 550);
    return () => window.clearTimeout(timer);
  }, [running, snapshot, pending]);

  return {
    top, bottom, board, snapshot, selection, legalTargets, lastMove, notice,
    modal, lightMode, setModal, setLightMode, available, connected,
    pending, running, canHumanMove, chooseAgent, createMatch, resetBoard, selectSquare, startBattle,
  };
}
