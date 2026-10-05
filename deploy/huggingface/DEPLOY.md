# Publishing the demo on Hugging Face Spaces

The repository root holds a `Dockerfile` that runs the web table in demo mode on port 7860.

1. Create an account on https://huggingface.co and a new **Space** (SDK: **Docker**, hardware: free CPU basic).
2. Clone the empty Space next to this repository:
   `git clone https://huggingface.co/spaces/<your-user>/<space-name>`
3. Copy the project into it (without `.git`) and use the Space's front matter as its `README.md`:
   ```
   rsync -a --exclude .git --exclude .venv ./ ../<space-name>/     # or copy the files by hand on Windows
   cp deploy/huggingface/README.md ../<space-name>/README.md
   ```
4. Commit and push from inside the Space folder (`git add -A && git commit -m "Demo" && git push`).
   Hugging Face builds the image (a few minutes) and the demo appears at `https://<your-user>-<space-name>.hf.space`.

Notes
- The Space is public by default. The server caps simultaneous players (`--max-sessions`, 10 in the Dockerfile).
- A free Space goes to sleep after a period without visitors and wakes up on the next visit.
- Test the image locally first: `docker build -t trump . && docker run -p 7860:7860 trump`, then open http://127.0.0.1:7860.
