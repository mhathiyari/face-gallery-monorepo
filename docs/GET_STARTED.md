# Get Started - Your Next Steps

## What You Have Now

✅ Complete monorepo structure with:
- Backend (face-search-system)
- Frontend (face_search_viewer)
- Docker setup (GPU + CPU)
- Installation scripts
- Unified configuration system
- Full documentation

## Before You Can Run

### ⚠️ ONE Required Change

The `frontend/app.py` needs a small update to use the new config system.

**Option A: Quick manual edit (2 minutes)**

Edit `frontend/app.py`:

1. Add this import near the top (after other imports):
   ```python
   from config_loader import get_config
   ```

2. Replace the config loading section with:
   ```python
   config = get_config()
   ```

3. Update path references:
   - Change `config.get('face_search_root', '...')`
   - To: `config['paths']['backend_root']`

See [INTEGRATION_NOTES.md](INTEGRATION_NOTES.md) for details.

**Option B: Use search/replace**

```bash
cd face-gallery/frontend
# Backup first
cp app.py app.py.backup
# Then manually edit as described above
```

## Quick Start (After Integration)

### Method 1: Docker (Easiest)

```bash
cd face-gallery

# Setup
cp .env.example .env
nano .env  # Set PHOTOS_DIR to your photo folder

# Run
docker-compose up
```

Open http://localhost:5050

### Method 2: Manual Install

```bash
cd face-gallery

# Install
./scripts/install.sh

# Verify
./scripts/verify.sh

# Run
./scripts/run.sh
```

Open http://localhost:5050

## What's Included

```
face-gallery/
├── README.md              ← Full feature documentation
├── QUICKSTART.md          ← 5-minute quick start
├── SETUP_STEPS.md         ← Detailed setup guide
├── INTEGRATION_NOTES.md   ← Frontend integration (read this!)
│
├── backend/               ← Face recognition engine
├── frontend/              ← Web UI
├── docker/                ← Docker configs
├── scripts/               ← Helper scripts
│   ├── install.sh        ← Auto installer
│   ├── run.sh            ← Start app
│   └── verify.sh         ← Check setup
│
├── config/                ← Configuration
│   ├── config.example.json
│   └── README.md         ← Config documentation
│
└── docker-compose.yml     ← Docker setup (GPU)
```

## Recommended Path

1. **Verify setup:**
   ```bash
   cd face-gallery
   ./scripts/verify.sh
   ```

2. **Read integration notes:**
   ```bash
   cat INTEGRATION_NOTES.md
   ```

3. **Update frontend/app.py** (see above)

4. **Choose your method:**
   - Docker? Follow [QUICKSTART.md](QUICKSTART.md)
   - Manual? Run `./scripts/install.sh`

5. **Test with small photo set first!**

## Documentation

- **Quick start**: [QUICKSTART.md](QUICKSTART.md)
- **Full setup**: [SETUP_STEPS.md](SETUP_STEPS.md)
- **Installation**: [docs/INSTALLATION.md](docs/INSTALLATION.md)
- **Configuration**: [config/README.md](config/README.md)
- **Integration**: [INTEGRATION_NOTES.md](INTEGRATION_NOTES.md)

## Support

- Check documentation in `docs/`
- Run `./scripts/verify.sh` to diagnose issues
- See [README.md](README.md) troubleshooting section

## Summary

You're 95% ready to go! Just need to:

1. ✅ Update `frontend/app.py` to use new config (see INTEGRATION_NOTES.md)
2. ✅ Run `./scripts/verify.sh` to check setup
3. ✅ Choose Docker or manual install
4. ✅ Start organizing your photos!

**The monorepo is ready - happy sorting! 🎉**
