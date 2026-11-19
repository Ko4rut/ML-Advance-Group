import type { LegalMove } from "../../arena/api/types";

export type ReplaySide = "red" | "black";

export type ReplayMove = LegalMove & {
  ply: number;
  side: ReplaySide;
  notation: string;
  actor: string | null;
};

export type ReplayFrame = {
  board: number[];
  lastMove: ReplayMove | null;
};

export type ReplayGame = {
  id: string;
  label: string;
  initialSide: ReplaySide;
  moves: ReplayMove[];
  frames: ReplayFrame[];
  outcome: string | null;
};

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function initialEngineBoard(): number[] {
  const board = Array<number>(90).fill(0);
  const backRank = [5, 4, 3, 2, 1, 2, 3, 4, 5];

  for (let col = 0; col < 9; col += 1) {
    board[col] = -backRank[col];
    board[81 + col] = backRank[col];
  }
  for (const col of [1, 7]) {
    board[18 + col] = -6;
    board[63 + col] = 6;
  }
  for (const col of [0, 2, 4, 6, 8]) {
    board[27 + col] = -7;
    board[54 + col] = 7;
  }
  return board;
}

function parseBoard(value: unknown, context: string): number[] {
  if (!Array.isArray(value) || value.length !== 90) {
    throw new Error(context + ": bàn cờ phải có đúng 90 ô.");
  }

  const board: number[] = [];
  for (let index = 0; index < value.length; index += 1) {
    const piece = value[index];
    if (typeof piece !== "number" || !Number.isInteger(piece) || piece < -7 || piece > 7) {
      throw new Error(context + ": mã quân không hợp lệ tại ô " + index + ".");
    }
    board.push(piece);
  }
  return board;
}

function parseSquare(value: unknown, context: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 0 || value >= 90) {
    throw new Error(context + ": chỉ số ô phải là số nguyên trong khoảng 0–89.");
  }
  return value;
}

function parseCoordinateMove(value: string, context: string): LegalMove {
  const match = /^([a-i])([0-9])([a-i])([0-9])$/i.exec(value.trim());
  if (!match) {
    throw new Error(context + ": ký hiệu nước đi phải có dạng b2e2.");
  }

  const sourceCol = match[1].toLowerCase().charCodeAt(0) - 97;
  const targetCol = match[3].toLowerCase().charCodeAt(0) - 97;
  return {
    source: (9 - Number(match[2])) * 9 + sourceCol,
    target: (9 - Number(match[4])) * 9 + targetCol,
  };
}

function squareNotation(index: number): string {
  const col = String.fromCharCode(97 + (index % 9));
  const rank = 9 - Math.floor(index / 9);
  return col + String(rank);
}

function parseAction(value: unknown, context: string): LegalMove {
  if (typeof value === "string") {
    return parseCoordinateMove(value, context);
  }
  if (!isObject(value)) {
    throw new Error(context + ": thiếu đối tượng action hoặc ký hiệu nước đi.");
  }
  return {
    source: parseSquare(value.source, context + " source"),
    target: parseSquare(value.target, context + " target"),
  };
}

function parseSide(value: unknown): ReplaySide {
  if (value === undefined || value === null) {
    return "red";
  }
  if (typeof value === "string") {
    const normalized = value.toLowerCase();
    if (normalized === "red" || normalized === "black") {
      return normalized;
    }
  }
  throw new Error("initial_side phải là red hoặc black.");
}

function oppositeSide(side: ReplaySide): ReplaySide {
  if (side === "red") {
    return "black";
  }
  return "red";
}

function matchesBoard(expected: number[], actual: number[]): boolean {
  for (let index = 0; index < 90; index += 1) {
    if (expected[index] !== actual[index]) {
      return false;
    }
  }
  return true;
}

function assertSnapshot(snapshot: unknown, board: number[], context: string): void {
  if (snapshot === undefined || snapshot === null) {
    return;
  }
  const recorded = parseBoard(snapshot, context);
  if (!matchesBoard(recorded, board)) {
    throw new Error(context + ": trạng thái bàn cờ không khớp lịch sử nước đi.");
  }
}

function readOutcome(root: Record<string, unknown>): string | null {
  let result: unknown = root.outcome;
  if (!isObject(result) && isObject(root.summary)) {
    result = root.summary.result;
  }
  if (!isObject(result)) {
    return null;
  }

  const labels: string[] = [];
  if (typeof result.status === "string") {
    labels.push(result.status);
  }
  if (typeof result.winner === "string") {
    labels.push(result.winner);
  }
  if (typeof result.reason === "string") {
    labels.push(result.reason);
  }
  return labels.join(" · ") || null;
}

function parseGame(value: unknown, index: number): ReplayGame {
  if (!isObject(value) || !Array.isArray(value.moves)) {
    throw new Error("Ván " + (index + 1) + ": cần có mảng moves.");
  }
  const root = value;
  let label = "Ván " + (index + 1);
  if (typeof root.game_id === "string" && root.game_id.trim()) {
    label = root.game_id.trim();
  }
  const initialSide = parseSide(root.initial_side);
  let initialBoard = initialEngineBoard();
  if (root.initial_board !== undefined) {
    initialBoard = parseBoard(root.initial_board, label + " initial_board");
  }

  if (root.moves.length > 1000) {
    throw new Error(label + ": số nước đi vượt giới hạn 1.000.");
  }
  const moves: ReplayMove[] = [];
  const frames: ReplayFrame[] = [{ board: initialBoard.slice(), lastMove: null }];
  const working = initialBoard.slice();
  let currentSide = initialSide;

  for (let indexMove = 0; indexMove < root.moves.length; indexMove += 1) {
    const raw = root.moves[indexMove];
    const context = label + " · nước " + (indexMove + 1);
    let input: unknown = raw;
    let actor: string | null = null;

    if (isObject(raw)) {
      if (raw.action !== undefined) {
        input = raw.action;
      } else if (raw.move !== undefined) {
        input = raw.move;
      }
      if (typeof raw.actor === "string") {
        actor = raw.actor;
      } else if (typeof raw.agent === "string") {
        actor = raw.agent;
      }
      if (raw.ply !== undefined && raw.ply !== indexMove + 1) {
        throw new Error(context + ": số ply không liên tục.");
      }
      if (typeof raw.side === "string" && raw.side.toLowerCase() !== currentSide) {
        throw new Error(context + ": bên đi không khớp lịch sử.");
      }
      assertSnapshot(raw.board, working, context + " board");
    }

    const action = parseAction(input, context);
    if (action.source === action.target) {
      throw new Error(context + ": ô đi và ô đến trùng nhau.");
    }
    const piece = working[action.source];
    const targetPiece = working[action.target];
    if (piece === 0) {
      throw new Error(context + ": không có quân tại ô xuất phát.");
    }
    if ((currentSide === "red" && piece < 0) || (currentSide === "black" && piece > 0)) {
      throw new Error(context + ": quân di chuyển không đúng lượt.");
    }
    if (targetPiece !== 0 && Math.sign(targetPiece) === Math.sign(piece)) {
      throw new Error(context + ": không thể ăn quân cùng phe.");
    }

    const move: ReplayMove = {
      source: action.source,
      target: action.target,
      ply: indexMove + 1,
      side: currentSide,
      notation: squareNotation(action.source) + squareNotation(action.target),
      actor,
    };
    working[action.target] = piece;
    working[action.source] = 0;
    if (isObject(raw)) {
      assertSnapshot(raw.next_board, working, context + " next_board");
    }
    moves.push(move);
    frames.push({ board: working.slice(), lastMove: move });
    currentSide = oppositeSide(currentSide);
  }

  return {
    id: label + "-" + index,
    label,
    initialSide,
    moves,
    frames,
    outcome: readOutcome(root),
  };
}

/**
 * Parse the project's Alpha-Beta/Silver JSON, JSONL datasets and neural
 * evaluation move notation. Frames reconstruct piece placement, not full
 * Xiangqi rule legality; validated datasets should still pass backend replay.
 */
export function parseReplayGames(text: string, fileName: string): ReplayGame[] {
  if (!text.trim()) {
    throw new Error("Tệp trống. Hãy chọn file JSON chứa lịch sử nước đi.");
  }

  let root: unknown;
  if (fileName.toLowerCase().endsWith(".jsonl")) {
    const lines = text.split(/\r?\n/).filter(function nonEmpty(line) {
      return line.trim().length > 0;
    });
    root = lines.map(function parseLine(line, index) {
      try {
        return JSON.parse(line) as unknown;
      } catch {
        throw new Error("JSONL không hợp lệ tại dòng " + (index + 1) + ".");
      }
    });
  } else {
    try {
      root = JSON.parse(text) as unknown;
    } catch {
      throw new Error("JSON không hợp lệ. Hãy kiểm tra cú pháp file.");
    }
  }

  let records: unknown[];
  if (isObject(root) && Array.isArray(root.games)) {
    records = root.games;
  } else if (Array.isArray(root)) {
    if (root.length > 0 && isObject(root[0]) && Array.isArray(root[0].moves)) {
      records = root;
    } else {
      records = [{ moves: root }];
    }
  } else {
    records = [root];
  }

  if (records.length === 0) {
    throw new Error("Không tìm thấy ván cờ nào trong file.");
  }
  if (records.length > 1000) {
    throw new Error("File chứa quá 1.000 ván. Hãy chia nhỏ dữ liệu.");
  }
  return records.map(function parseRecord(record, index) {
    return parseGame(record, index);
  });
}
