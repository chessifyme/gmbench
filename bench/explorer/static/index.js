// The index page's filters: by family and by words of the task id or question. Without scripts every task shows.
const cards = Array.from(document.querySelectorAll("#cards .card"))
const buttons = Array.from(document.querySelectorAll(".seg button[data-family]"))
const search = document.getElementById("search")
const count = document.getElementById("count")
const empty = document.getElementById("empty")
let family = ""

function apply() {
    const words = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean)
    let shown = 0
    for (const card of cards) {
        const visible = (!family || card.dataset.family === family) && words.every((word) => card.dataset.search.includes(word))
        card.hidden = !visible
        if (visible) shown += 1
    }
    count.textContent = shown === cards.length ? `${cards.length} tasks` : `${shown} of ${cards.length} tasks`
    empty.hidden = shown > 0
}

for (const button of buttons) {
    button.addEventListener("click", () => {
        family = button.dataset.family
        for (const other of buttons) other.setAttribute("aria-pressed", String(other === button))
        apply()
    })
}
search.addEventListener("input", apply)
apply()
