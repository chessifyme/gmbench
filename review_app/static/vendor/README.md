Third-party browser libraries for the review app's playable board, copied unchanged from npm (no CDN at run time).

- `chess_js/`: chess.js 1.4.0 (BSD-2-Clause), `dist/esm/chess.js`. Move legality, SAN and FEN in the browser.
- `cm_chessboard/`: cm-chessboard 8.15.4 (MIT), the core `src/` with the markers and promotion-dialog extensions, and the assets they load. The folder names inside are the library's own, because its code imports them by path.

Each folder keeps its LICENSE.
