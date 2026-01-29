
<p align="center">
  <a href="https://github.com/zozaai/MazeLLM/actions/workflows/ci.yml">
    <img src="https://github.com/zozaai/MazeLLM/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://codecov.io/gh/zozaai/MazeLLM">
    <img src="https://codecov.io/gh/zozaai/MazeLLM/branch/main/graph/badge.svg" alt="codecov">
  </a>
</p>



<p align="center">
  <a href="https://excalidraw.com/#json=VrLbNlJTp0jjCZJ-t8U-3,yWYmD7WfJaijoeX5uEnBSg">
    <img src="docs/block-diagram.svg" alt="MazeLLM architecture diagram" width="70%" />
  </a>
</p>


**MazeLLM** is an experimental project that explores maze solving using classical algorithms and language-model–friendly abstractions.  
It combines maze generation, robot sensing and movement, and a live Textual-based visualization.


---

## Overview

MazeLLM consists of three core components:

- **Maze**  
  Generates a randomized maze using recursive backtracking.  
  The maze is guaranteed to have a valid path from **Start (S)** to **End (E)**.

- **Robot**  
  Operates inside the maze with:
  - directional sensors (how far it can move before hitting a wall)
  - multi-step movement validation
  - strict collision and boundary checking

- **Visualizer (Textual UI)**  
  A terminal UI that renders the maze grid and animates the robot step-by-step, with live logs.

---

## Maze Generation (Guaranteed Solvable)

Unlike naive generators, MazeLLM **guarantees reachability**:

- The maze is carved starting from `(0,0)`
- All reachable cells are discovered using BFS
- The **End (E)** is placed at the *farthest reachable cell* from **Start (S)**

This ensures:
- ✅ A valid path always exists
- ✅ Paths are typically long and interesting
- ❌ No more disconnected end states

---

## Path Finding

MazeLLM currently uses **Breadth-First Search (BFS)** to find the shortest path from **S → E**:

- 4-connected grid (up, down, left, right)
- Walls (`1`) are impassable
- Start (`S`) and End (`E`) are treated as walkable
- BFS guarantees the shortest path in number of steps

---

## Demo: Visualized Robot Navigation

Run the full demo with animation:

```bash
python -m mazellm.main_demo --n 20 --m 15 --interval 0.05
```
