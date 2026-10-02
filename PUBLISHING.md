# Publishing to GitHub (open source)

Do this once. Replace `Ezeai` everywhere.

## 1. Prepare

1. Open `LICENSE` and put your name in the copyright line.
2. Check `.gitignore` is present. It keeps `user_data/` (your chats and settings) and `__pycache__/` out of the repo.
3. Run the tests: `python -m unittest discover -s tests -p "test_*.py"`
4. Search for anything private before publishing:
   ```bash
   grep -rn "Authorised User\|D:\\\\" --include=*.py --include=*.js --include=*.md --exclude-dir=tests .
   ```
   The code ships with no personal paths (your own settings live in the git-ignored `user_data/`).

## 2. Install Git and sign in

- Install Git from <https://git-scm.com>.
- Create a free account at <https://github.com>.
- One-time identity:
  ```bash
  git config --global user.name "Your Name"
  git config --global user.email "you@example.com"
  ```

## 3. Create the repository on GitHub

1. GitHub → **New repository**.
2. Name: `ComfyUI-EzeAi-Comfy-Agent`. Visibility: **Public**.
3. Do **not** tick "Add a README", .gitignore or license (they already exist locally).
4. Create it and copy the repo URL.

## 4. Push the code

Run inside the node folder (`ComfyUI/custom_nodes/ComfyUI-EzeAi-Comfy-Agent`):

```bash
git init -b main
git add .
git status          # check that user_data/ and __pycache__/ are NOT listed
git commit -m "Add EzeAi Comfy Agent"
git remote add origin https://github.com/Ezeai/ComfyUI-EzeAi-Comfy-Agent.git
git push -u origin main
```

GitHub will ask you to sign in in the browser the first time. If it asks for a password
use a personal access token (Settings → Developer settings → Tokens), not your password.

## 5. Make the page welcoming

- Repo **About** (gear icon): description, and topics `comfyui`, `comfyui-custom-node`,
  `llm`, `ollama`, `llama-cpp`, `ai-agent`, `local-ai`.
- Add 2 or 3 screenshots or a short GIF of the mission panel to a `docs/` folder and
  reference them near the top of `README.md`. Nothing sells a node like seeing it work.
- Edit the README clone URL (replace `Ezeai`).

## 6. Release a version

```bash
git tag v0.1.0
git push origin v0.1.0
```

Then on GitHub: **Releases → Draft a new release** → choose the tag → write short notes.

## 7. Get it into ComfyUI-Manager and the Registry (optional)

- **Comfy Registry** (official): create a publisher at <https://registry.comfy.org>, edit
  `pyproject.toml` (set `PublisherId`), then run `comfy node publish` with the
  comfy-cli, or use the GitHub Action described in the Registry docs.
- **ComfyUI-Manager list:** open a pull request adding your repo URL to
  `custom-node-list.json` in the ComfyUI-Manager repository.

## 8. Running an open-source project

- Turn on **Issues** and **Discussions** in the repo settings.
- Add labels such as `bug`, `enhancement`, `good first issue`.
- Ask bug reporters for: ComfyUI version, the model used, the console traceback and the
  Inspector's context manifest.
- Review pull requests against the tests: `python -m unittest discover -s tests`.
- Use semantic versions: `v0.1.1` fixes, `v0.2.0` features.

## Licensing notes

- The code is MIT. You may ask users to credit you; MIT requires keeping the license text.
- Do **not** commit model files (`.gguf`) or generated images. The `.gitignore` excludes `*.gguf`.
- Models people use with this node keep their own licenses. Say so in your README if you recommend specific ones.
