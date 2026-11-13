// The review page's playable board: step through the task's moves, or play other moves to explore a side line.
// Every position change is announced as a "board:position" event, which the analysis panel follows.
import { Chess } from "/static/vendor/chess_js/chess.js"
import { Chessboard, COLOR, INPUT_EVENT_TYPE } from "/static/vendor/cm_chessboard/src/Chessboard.js"
import { Markers, MARKER_TYPE } from "/static/vendor/cm_chessboard/src/extensions/markers/Markers.js"
import { PromotionDialog, PROMOTION_DIALOG_RESULT_TYPE } from "/static/vendor/cm_chessboard/src/extensions/promotion-dialog/PromotionDialog.js"

const config = JSON.parse(document.getElementById("board_config").textContent)
const mainline = config.moves.slice()
let line = mainline.slice()
let ply = config.ply

const label = document.getElementById("board_move")
const sideNote = document.getElementById("board_side_line")
const buttons = {
    start: document.getElementById("board_start"),
    back: document.getElementById("board_back"),
    forward: document.getElementById("board_forward"),
    end: document.getElementById("board_end"),
    task: document.getElementById("board_task"),
}

const board = new Chessboard(document.getElementById("board"), {
    position: config.start_fen,
    orientation: config.orientation === "black" ? COLOR.black : COLOR.white,
    assetsUrl: "/static/vendor/cm_chessboard/assets/",
    style: { cssClass: "default", showCoordinates: true, animationDuration: 150 },
    extensions: [{ class: Markers, props: { autoMarkers: MARKER_TYPE.frame } }, { class: PromotionDialog }],
})

function gameAt(count) {
    const game = new Chess(config.start_fen)
    let last = null
    for (const san of line.slice(0, count)) {
        last = game.move(san)
    }
    return { game, last }
}

// Worded as review_app.board_line.position_label words the first label, which the page renders.
function moveLabel(game, last, atTask) {
    const toMove = `move ${game.moveNumber()}, ${game.turn() === "w" ? "White" : "Black"} to move`
    if (atTask) return `Task position · ${toMove}`
    if (!last) return toMove
    const number = game.moveNumber() - (last.color === "b" ? 1 : 0)
    return last.color === "w" ? `after ${number}.${last.san}` : `after ${number}...${last.san}`
}

function atTaskPosition() {
    return ply === config.task_ply && line.slice(0, ply).every((san, index) => san === mainline[index])
}

function isSideLine() {
    return line.length !== mainline.length || line.some((san, index) => san !== mainline[index])
}

function render(animated) {
    const { game, last } = gameAt(ply)
    board.setPosition(game.fen(), animated)
    board.removeMarkers(MARKER_TYPE.square)
    if (last) {
        board.addMarker(MARKER_TYPE.square, last.from)
        board.addMarker(MARKER_TYPE.square, last.to)
    }
    label.textContent = moveLabel(game, last, atTaskPosition())
    sideNote.hidden = !isSideLine()
    buttons.start.disabled = buttons.back.disabled = ply === 0
    buttons.end.disabled = buttons.forward.disabled = ply === line.length
    buttons.task.disabled = !isSideLine() && ply === config.task_ply
    window.dispatchEvent(new CustomEvent("board:position", { detail: { fen: game.fen(), label: label.textContent, over: game.isGameOver() } }))
}

function goTo(target) {
    const next = Math.max(0, Math.min(line.length, target))
    if (next === ply) return
    const animated = Math.abs(next - ply) === 1
    ply = next
    render(animated)
}

function play(move) {
    const { game } = gameAt(ply)
    const played = game.move(move)
    if (line[ply] !== played.san) {
        line = line.slice(0, ply).concat([played.san])
    }
    ply += 1
    render(false)
}

function onInput(event) {
    const { game } = gameAt(ply)
    if (event.type === INPUT_EVENT_TYPE.moveInputStarted) {
        const piece = game.get(event.squareFrom)
        return Boolean(piece) && piece.color === game.turn()
    }
    if (event.type === INPUT_EVENT_TYPE.validateMoveInput) {
        const legal = game.moves({ square: event.squareFrom, verbose: true }).filter((m) => m.to === event.squareTo)
        if (legal.length === 0) return false
        if (legal[0].promotion) {
            const color = game.turn() === "w" ? COLOR.white : COLOR.black
            board.showPromotionDialog(event.squareTo, color, (result) => {
                if (result.type === PROMOTION_DIALOG_RESULT_TYPE.pieceSelected) {
                    play({ from: event.squareFrom, to: event.squareTo, promotion: result.piece.charAt(1) })
                } else {
                    render(false)
                }
            })
            return true
        }
        play({ from: event.squareFrom, to: event.squareTo })
        return true
    }
    return true
}

board.enableMoveInput(onInput)

buttons.start.addEventListener("click", () => goTo(0))
buttons.back.addEventListener("click", () => goTo(ply - 1))
buttons.forward.addEventListener("click", () => goTo(ply + 1))
buttons.end.addEventListener("click", () => goTo(line.length))
buttons.task.addEventListener("click", () => {
    line = mainline.slice()
    ply = config.task_ply
    render(false)
})

// A line written in a claim: the page found it legal from start_ply and gives its moves in UCI.
document.addEventListener("click", (event) => {
    const link = event.target.closest("a.claim-line")
    if (!link) return
    event.preventDefault()
    const start = Number(link.dataset.startPly)
    line = mainline.slice(0, start)
    const { game } = gameAt(start)
    for (const uci of link.dataset.uci.split(" ")) {
        line.push(game.move({ from: uci.slice(0, 2), to: uci.slice(2, 4), promotion: uci.slice(4) || undefined }).san)
    }
    ply = start + 1
    render(false)
})

document.addEventListener("keydown", (event) => {
    if (["INPUT", "TEXTAREA", "SELECT"].includes(event.target.tagName) || document.querySelector("dialog[open]")) return
    const keys = { ArrowLeft: ply - 1, ArrowRight: ply + 1, Home: 0, End: line.length }
    if (event.key in keys) {
        event.preventDefault()
        goTo(keys[event.key])
    }
})

render(false)
