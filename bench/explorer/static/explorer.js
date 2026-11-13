// A task page's board: the task position, and any line of the page (an expert point's move or line, the move under discussion,
// an engine line) played from it with step back, forward and reset. Lines are checked when the site is built.
import { Chess } from "./vendor/chess_js/chess.js"
import { Chessboard, COLOR } from "./vendor/cm_chessboard/src/Chessboard.js"
import { Markers, MARKER_TYPE } from "./vendor/cm_chessboard/src/extensions/markers/Markers.js"

const config = JSON.parse(document.getElementById("board_config").textContent)
const host = document.getElementById("board")
const status = document.getElementById("board_status")
const moves = document.getElementById("board_moves")
const buttons = {
    reset: document.getElementById("board_reset"),
    back: document.getElementById("board_back"),
    forward: document.getElementById("board_forward"),
    flip: document.getElementById("board_flip"),
}
const lineButtons = Array.from(document.querySelectorAll("button.line[data-line]"))

host.replaceChildren()
const board = new Chessboard(host, {
    position: config.fen,
    orientation: config.orientation === "black" ? COLOR.black : COLOR.white,
    assetsUrl: new URL("./vendor/cm_chessboard/assets/", import.meta.url).href,
    style: { cssClass: "default", showCoordinates: true, animationDuration: 150 },
    extensions: [{ class: Markers, props: { autoMarkers: null } }],
})

let line = { sans: [], labels: [] }
let lineId = null
let ply = 0

function render(animated) {
    const game = new Chess(config.fen)
    let last = null
    for (const san of line.sans.slice(0, ply)) last = game.move(san)
    board.setPosition(game.fen(), animated)
    board.removeMarkers(MARKER_TYPE.square)
    if (last) {
        board.addMarker(MARKER_TYPE.square, last.from)
        board.addMarker(MARKER_TYPE.square, last.to)
    }
    const text = ply === 0 ? `Task position · ${config.side} to move` : `After ${line.labels[ply - 1]} · move ${ply} of ${line.sans.length}`
    status.textContent = text
    host.setAttribute("aria-label", `Chess board: ${text}`)
    buttons.reset.disabled = buttons.back.disabled = ply === 0
    buttons.forward.disabled = ply === line.sans.length
    moves.querySelectorAll("button").forEach((button, index) => button.setAttribute("aria-current", String(index + 1 === ply)))
}

function goTo(target) {
    const next = Math.max(0, Math.min(line.sans.length, target))
    if (next === ply) return
    const animated = Math.abs(next - ply) === 1
    ply = next
    render(animated)
}

function showLine(id) {
    lineId = id
    line = config.lines[id]
    moves.replaceChildren(...line.labels.map((label, index) => {
        const item = document.createElement("li")
        const button = document.createElement("button")
        button.type = "button"
        button.textContent = label
        button.addEventListener("click", () => goTo(index + 1))
        item.append(button)
        return item
    }))
    for (const button of lineButtons) button.setAttribute("aria-pressed", String(button.dataset.line === id))
    ply = 0
    goTo(line.start)
    if (host.getBoundingClientRect().bottom < 0 || host.getBoundingClientRect().top > window.innerHeight) {
        host.scrollIntoView({ behavior: "smooth", block: "start" })
    }
}

for (const button of lineButtons) button.addEventListener("click", () => showLine(button.dataset.line))
buttons.reset.addEventListener("click", () => goTo(0))
buttons.back.addEventListener("click", () => goTo(ply - 1))
buttons.forward.addEventListener("click", () => goTo(ply + 1))
buttons.flip.addEventListener("click", () => board.setOrientation(board.getOrientation() === COLOR.white ? COLOR.black : COLOR.white))
document.addEventListener("keydown", (event) => {
    if (event.altKey || event.ctrlKey || event.metaKey || event.target.closest("input, textarea, select")) return
    const actions = { ArrowLeft: () => goTo(ply - 1), ArrowRight: () => goTo(ply + 1), Home: () => goTo(0), End: () => goTo(line.sans.length), f: () => buttons.flip.click() }
    if (actions[event.key]) {
        event.preventDefault()
        actions[event.key]()
    }
})
render(false)
