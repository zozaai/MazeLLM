# MazeLLM
MazeLLM: Solving Mazes with Language Models

# Path Finding
<p align="center">
  <a href="https://excalidraw.com/#json=VrLbNlJTp0jjCZJ-t8U-3,yWYmD7WfJaijoeX5uEnBSg">
    <img src="docs/block-diagram.svg" alt="Maze LLM diagram" width="125%" />
  </a>
</p>



# how to run the maze with random next steps (typer visualization)
```bash
python -m mazellm.cli animate --n 4 \
"1,1" "1,2" "2,2" "2,1" "3,1" "3,2" "2,2" "2,3" \
"3,3" "4,3" "4,2" "3,2" "3,3" "3,4" "4,4" \
--interval 0.1
```

```bash
python -m mazellm.cli animate --n 8 \
"1,1" "1,2" "2,2" "2,3" "3,3" "3,2" "4,2" "4,3" "4,4" \
"3,4" "3,5" "4,5" "5,5" "5,4" "6,4" "6,5" "6,6" \
"5,6" "5,7" "6,7" "7,7" "7,6" "8,6" "8,7" "8,8" \
--interval 0.5

```
